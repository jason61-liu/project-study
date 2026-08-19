# 第 12 周面试与阶段关卡：Serving 面试题、白板讲解与需求推导

> 目标：用「面试问答 + 白板讲解 + 反例分析 + 需求推导」四块，把本周六个知识点的碎片串成可输出的能力——既能在面试里讲清 KV Cache 的分页/前缀/分层三个层次，也能从一段 Agent 请求特征反推出该开哪些 serving 特性。

本文承接 [01-PagedAttention](01-PagedAttention-KV-Cache碎片与分页思想.md)、[02-vLLM](02-vLLM-Serving-Benchmark与缓存章节.md)、[03-SGLang](03-SGLang-Runtime-RadixCache与Serving概览.md)、[04-LMCache](04-LMCache-Quickstart与Integration-命中与传输度量.md)、[05-P/D 解耦](05-Mooncake-KV-centric架构与Dynamo-llm-d-PD解耦对照.md)、[06-TensorRT-LLM](06-TensorRT-LLM-ContinuousBatching-ChunkedPrefill-KV路由与PD解耦收益条件.md) 六篇。

---

## 1. 12 道 Serving 面试题（带参考答法）

> 每题按「一句话结论 → 机制 → 边界/代价」三层答，面试里先给结论再展开。

### Q1. PagedAttention 为什么能把 KV Cache 浪费从 60–80% 降到 ~4%？

- **一句话**：把 KV Cache 从「按 max 长度连续预分配」改成「按固定大小的物理块 + 逻辑块表」管理，消除了按最大序列长度预留造成的外部碎片。
- **机制**：每个序列不再申请一整块连续 `max_seq_len × num_layers × head_dim` 的显存，而是维护一张 **block table**（逻辑块号 → 物理块号）。prefill/decode 每产生一块 KV 就从空闲物理块池里领一块，用完放回。复制（beam search / 并行采样）只复制块表指针（**Copy-on-Write**），物理 KV 共享，改写时才真正复制被改的块。
- **边界/代价**：仍有 ~4% 浪费来自**内部碎片**（最后一个块没写满），以及 block table 的间接寻址开销（vLLM 用 kernel 内融合的间接访存吸收掉大部分）。但相比外部碎片（按最坏情况预分配、长尾请求占着大片不用），这是数量级的改善；论文报告吞吐提升 2–4×。

### Q2. Continuous batching 相比 static batching 为什么吞吐更高？代价是什么？

- **一句话**：static batching 是「整批同进同出、最慢者卡所有人」，continuous batching 把调度粒度降到**每步（iteration）**，请求完成即离队、新请求即补位。
- **机制**：decode 阶段各请求输出长度差异极大，static 下短请求早生成了却要等最长请求，GPU 空转。continuous 在每步 decode 后检查「谁结束了、谁可以进来」，动态重组 batch，GPU 每步都尽量满负荷。
- **代价**：单请求的「每步等待」变多（因为 batch 成员在变，单请求感知的调度抖动增大），这是吞吐与 TTFT/尾延迟之间的权衡（见 11-w/01）。另外调度器本身更复杂，有额外开销。

### Q3. vLLM 的 Prefix Cache 和 SGLang 的 Radix Cache 本质区别在哪？

- **一句话**：复用粒度不同——vLLM 按**固定块（如 16 token）**对齐，SGLang 按**任意 token 序列**对齐。
- **机制**：vLLM APC 用块 hash 识别「这个块算没算过」，命中以块为最小单位；SGLang 用 radix tree 做最长前缀匹配，命中能精确到任意 token 边界。比如共享了 37 个 token，vLLM 只能对齐到 32 或 48，SGLang 能只重算第 38 个起。
- **边界**：块粒度实现简单、命中查询便宜；token 粒度复用更彻底，但 radix tree 有维护/拆分/淘汰开销。对 system prompt、few-shot、多轮历史、tree-of-thought 这类「前缀高度重叠但边界不齐块」的负载，SGLang 收益更大（论文最高 ~4.4×）。

### Q4. radix tree 为什么用变长边？split 和 evict 什么时候发生？ref_count 是干嘛的？

- **一句话**：变长边把「连续一段 token」压成一个节点，减少树深和节点数，降低最长前缀匹配的遍历代价。
- **机制**：每个节点存一段连续 token 序列 + 对应 KV 位置；「根到节点」路径唯一代表一个前缀。**split** 发生在两个会话共享同一 system prompt 但后续分叉时，公共前缀留作一个节点、分叉处拆出子节点。**evict** 在显存不足时淘汰叶子：默认 LRU 淘汰最久未用的叶子（保留公共祖先），`priority` 模式优先淘汰低优先级叶子。
- **ref_count**：记录多少在途请求正在引用该节点，归零才可安全淘汰——防止把正在被 decode 引用的 KV 换掉导致出错。

### Q5. 分层 KV Cache（HiCache / LMCache 多级后端）解决什么问题？

- **一句话**：把「单层 GPU 显存缓存」变成「GPU → CPU/宿主内存 → NVMe → 远端」的多级缓存，用容量换命中，用 overlap 换延迟。
- **机制**：GPU 显存有限，长上下文/高并发下缓存频繁淘汰、复用率掉。HiCache 让 GPU 淘汰变成**降级到 CPU 内存**而非删除，真正删除发生在宿主内存也淘汰时；LMCache 更进一步，后端可选 DRAM/NVMe/Redis/P2P/Mooncake，做跨进程、跨节点、跨引擎的 KV 共享。
- **边界**：每一级「容量更大、延迟更高」。关键是把 CPU→GPU 的加载与计算**重叠**（overlap scheduler / layerwise 逐层搬运），否则命中反而被加载延迟拖慢——这正是 Q6 的陷阱。

### Q6. KV Cache 命中了，TTFT 就一定会降吗？什么时候命中反而更慢？

- **一句话**：不一定。命中省了 prefill 计算，但**要付 KV 从存储加载回 GPU 的延迟和带宽**，加载比 prefill 还慢时就负收益。
- **机制**：命中收益 = 省下的 prefill 计算 − 加载延迟 − 传输带宽代价。冷缓存命中在显存内，加载几乎零成本；一旦 KV 在 CPU/NVMe/远端，加载延迟可能超过对短输入重新 prefill 的成本。**CacheGen** 这类 KV 压缩就是为了压低传输字节，让远端命中的带宽代价可控。
- **边界**：短输入、低带宽网络、远端（跨节点）命中时最容易「命中反而更慢」。这是「缓存命中率很高但端到端更慢」陷阱的根因。

### Q7. Prefill 和 Decode 的资源特征有什么不同？为什么这导致 P/D 解耦？

- **一句话**：prefill 是**计算受限**（FLOPs 密集、瞬时拉高 GPU 利用率、决定 TTFT），decode 是**内存带宽受限**（逐 token 读 KV、单步算力低、决定 TPOT/吞吐）。
- **机制**：同构部署里两者共用一个 batch，长输入的 prefill 会抢占正在 decode 请求的算力，抬高别人的 TTFT 和 ITL；prefill 的 burst 性又让算力利用率不稳。拆开后 prefill/decode 各自独立扩缩、互不干扰，代价是多了一条 KV 从 prefill 节点传到 decode 节点的路径。
- **边界**：只有当错配足够大（长输入、高 ISL:OSL、MoE、需独立扩缩）且 KV 传输够快时解耦才划算（详见第 3 节）。

### Q8. P/D 解耦的 KV 传输有哪几种介质？带宽量级？

- **一句话**：TCP、RDMA（InfiniBand/RoCEv2/eRDMA/GPUDirect）、NVMe-oF；RDMA 量级可达 87–190 GB/s。
- **机制**：Mooncake Transfer Engine 统一搬 DRAM/VRAM/NVMe，实测 **87 GB/s**（4×200G RoCE）与 **190 GB/s**（8×400G RoCE），约为 TCP 的 2.4–4.6 倍；NVIDIA Dynamo / llm-d 用 **NIXL**（可下接 UCX/IB），TRT-LLM 的 `cache_transceiver_config` 支持 UCX/NIXL/LIBFABRIC/MPI。
- **边界**：带宽不是唯一条件——**延迟**（首块到达）和**多跳/拓扑**同样决定解耦是否划算，网络差时 TCP 级别的传输会吃光收益。

### Q9. 什么是 KV-aware routing？它解决什么问题？

- **一句话**：把请求路由到「**已持有其前缀 KV**」的 worker，避免跨节点重复 prefill。
- **机制**：在多 worker / disaggregated 部署里，router 跟踪每个 worker 持有哪些 KV 块（Dynamo 用事件 AddRequest/MarkPrefillCompleted/Free 同步并持久化到 NATS JetStream），把请求派到前缀命中最多的 worker。NVIDIA 在 DeepSeek V3.2 WideEP（2P+2D）实测约 44% 缓存命中率、57% 输入 token 来自共享前缀时，省掉大量冗余长上下文 prefill。
- **边界**：命中率低时，路由判断只是额外一跳。收益条件是「同一上下文被多个 worker 服务」（缓存局部性强）。

### Q10. 抢占（preemption）的 swap 和 recompute 怎么权衡？

- **一句话**：swap 省计算费带宽，recompute 省带宽费计算。
- **机制**：显存不够时，把在途序列换出腾 KV 块。**swap** 把 KV 搬到 CPU 内存、之后原样搬回，省了重算但占用带宽和宿主内存；**recompute** 直接丢弃 KV、之后重算 prefill，省带宽但多花算力。选择取决于「CPU↔GPU 带宽 vs prefill 算力」哪个更便宜，以及长序列重算成本是否可接受。
- **边界**：长序列重算贵（更倾向 swap），短序列重算便宜（更倾向 recompute）。

### Q11. Chunked prefill 解决什么？什么负载才值得开？

- **一句话**：把长 prefill 切成多个 chunk 与 decode 交错，解决「长 prefill 饿死短请求」。
- **机制**：一次性算 100k token 的 prefill 会阻塞所有在途 decode 很久、抬高别人 ITL。切块后长 prefill 的每一小块和别的请求 decode 轮流跑。
- **边界**：收益条件是「长输入与延迟敏感的 decode 共存」；全是短请求时切块只有额外开销。开错场景反而更慢。

### Q12. 从一段 Agent 请求特征推导 Serving 配置（阶段关卡题）

> 见第 4 节的完整推导框架，面试答法：先列特征，再逐项映射到「开/关哪个特性 + 为什么」。

---

## 2. 白板讲解：PagedAttention / Prefix Cache / Radix Cache / 分层 KV Cache 的关系与差异

> 白板上一句话总纲：**这四者是在同一个 KV Cache 上、不同层次、解决不同问题**——PagedAttention 管「KV 放哪」，Prefix/Radix Cache 管「什么 KV 值得复用」，分层 KV Cache 管「KV 能存多大、复用多远」。

### 2.1 一张表看清四者的层次

| | 解决什么问题 | 核心抽象 | 粒度 | 作用范围 | 代表实现 |
|---|---|---|---|---|---|
| **PagedAttention** | KV 显存怎么分配（消除碎片） | 固定物理块 + 逻辑块表 | 块（如 16 token） | 单进程单 GPU | vLLM、SGLang 底层 |
| **Prefix Cache** | 哪些块的 KV 值得留（复用共享前缀） | 块 hash → 命中判定 | **块** | 单进程单 GPU | vLLM APC |
| **Radix Cache** | 同上，但复用更精细 | radix tree + 最长前缀匹配 | **任意 token 序列** | 单进程单 GPU | SGLang RadixAttention |
| **分层 KV Cache** | KV 容量与共享范围（跨进程/节点/引擎） | GPU→CPU→NVMe→远端 多级后端 | 前缀/整段/跨实例 | 跨进程、跨节点、跨引擎 | SGLang HiCache、LMCache |

### 2.2 三者（PagedAttention / Prefix / Radix）的关系

- **PagedAttention 是地基**：它决定了 KV 以「块」为单位可独立分配、复制、换出——没有分页，前缀复用就无法「只换入共享的那几个块」。Prefix Cache 和 Radix Cache 都建立在分页 KV 之上。
- **Prefix Cache 与 Radix Cache 是同一目标（复用前缀）的两种精度**：vLLM 复用对齐到块边界，SGLang 对齐到 token 边界。所以它们是**并列的可替代方案**，不是上下层关系。差异只在「复用粒度」：token 粒度复用更彻底，但要多维护一棵 radix tree。
- **分层 KV Cache 是对前两者的外延**：它不改变「怎么分页」「怎么匹配前缀」，而是改变「KV 存哪里、能复用多远」。HiCache 把 Radix Cache 的 GPU 淘汰变成层级降级；LMCache 把单实例的 Prefix/Radix 复用扩展到跨实例/跨节点/跨引擎。

### 2.3 白板画法（从左到右三层）

```text
[共享前缀] ──命中──▶ Prefix Cache(块) / Radix Cache(token)  ← 什么值得复用（精度之争）
                           │ 建在
[分页 KV]  ◀────────── PagedAttention（块表 + 物理块）      ← 放哪（地基）
                           │ 淘汰/降级到
[多级存储] ◀────────── 分层 KV Cache（GPU→CPU→NVMe→远端）  ← 存多大/复用多远（外延）
```

讲的时候强调三点：**地基（分页）→ 精度之争（Prefix vs Radix）→ 范围外延（分层）**，以及「分层缓存命中不等于更快」（回 Q6）。

---

## 3. 何时 P/D 解耦或远端 KV Cache 反而更慢？

> 总纲：这两个「高级特性」都有一个**隐藏成本——KV 要跨介质/跨网络搬一次**。划算与否，取决于「省下的 prefill 计算」是否大于「这次搬运」的代价。判断就两条硬约束：**KV 传输的带宽/延迟** 和 **ISL:OSL / 缓存命中率**。

### 3.1 P/D 解耦：什么情况下反而更慢

**收益**（解耦赚了什么）：prefill/decode 独立扩缩、burst 不互相打扰、按 ISL:OSL 调 P:D 比例。
**成本**（解耦付了什么）：KV 从 prefill 节点传到 decode 节点，多一跳网络 + 传输延迟 + 传输带宽。

**反而更慢的条件（要能明确说出来）**：

| 条件 | 为什么亏 |
|---|---|
| **KV 传输带宽 < 本地 prefill 算力吞吐** | 传一次 KV 比直接重算 prefill 还慢，解耦纯属多一跳 |
| **网络延迟高 / 非 RDMA（TCP）** | 首块 KV 到达慢，TTFT 反而被抬高；llm-d 生产明确要求 RDMA |
| **短输入、低 ISL:OSL** | prefill 本来便宜，解耦的传输开销占比大；llm-d 明确「短 prompt（200/200）不建议解耦」 |
| **低并发 / 单请求** | 独立扩缩、路由调度这些固定开销摊不薄，没有错配可解 |
| **缓存命中率低 / 前缀不共享** | 传过来的 KV 没多少被复用，等于白传 |

**一句话记法**：解耦是用「一条 KV 传输路径」换「prefill 与 decode 互不干扰」。当**传输比 prefill 还慢**、或**没有足够资源错配可解**时，解耦就变成纯开销。llm-d 给的经验门槛是「长输入（如 10k ISL / 1k OSL）、稀疏 MoE 宽并行」才划算。

### 3.2 远端 KV Cache：什么情况下命中反而更慢

**收益**：跨实例/跨节点复用前缀，省 prefill 计算。
**成本**：KV 从 CPU/NVMe/远端**加载回 GPU** 的延迟 + 传输字节。

**反而更慢的条件**：

| 条件 | 为什么亏 |
|---|---|
| **加载延迟 > 对该前缀重算 prefill 的延迟** | 短输入时重算极快，远端加载的固定延迟反而更高 |
| **网络/存储带宽低** | 传输字节大（未压缩 KV），加载挤占带宽、拖慢在途 decode |
| **命中率虚高但整段命中少** | 只有共享 system prompt 的小前缀命中，省的计算抵不上连接/查表开销 |
| **未做 KV 压缩（无 CacheGen）** | 远端命中的带宽代价不可控，省的计算被传输字节吃掉 |

**量化判断法（呼应 04 的三个指标）**：不是看「有没有命中」，而是算净收益 = 省下的 prefill 计算 − 传输字节 × 单位带宽成本，再看命中后的 TTFT 是否真的下降。**命中率高但端到端更慢**，正是这条公式的负值。

### 3.3 两条硬约束的统一表达

> 无论 P/D 解耦还是远端 KV Cache，**判断标准都是「KV 跨介质/跨网络搬一次的代价 vs 本地重算的代价」**。前者赌「搬运比计算便宜 + 复用率高」，一旦 KV 传输带宽/延迟吃紧，或 ISL:OSL / 命中率低，高级特性就退化成「花钱搬了一堆没用的 KV」。

---

## 4. 阶段关卡：从 Agent 请求特征推导 Serving 需求

> 关卡通过标准：给一段 Agent 负载描述，能逐条把「请求特征」映射到「开/关哪个 serving 特性 + 为什么」。下面是可复用的推导框架。

### 4.1 推导框架：特征 → 需求 → 配置

先按四个维度给 Agent 请求打标签，再逐维映射：

| 请求特征维度 | 典型信号 | 推导出的 Serving 需求 | 具体配置 |
|---|---|---|---|
| **前缀共享度** | 固定 system prompt、few-shot、多轮历史 | 前缀复用 → 省重复 prefill | 开 Prefix/Radix Cache；命中率高 → 可考虑 KV-aware routing |
| **输入长度 / ISL:OSL** | 长文档、长历史（ISL≫OSL） | 长 prefill 会饿死 decode → 切块 + 解耦 | 开 chunked prefill；ISL:OSL 高 + MoE → 考虑 P/D 解耦 |
| **分支/树搜索** | tree-of-thought、多候选采样 | 大量共享前缀但边界不齐块 | 选 token 粒度 Radix Cache（SGLang）而非块粒度 |
| **并发与延迟敏感** | 多请求、要求低 TTFT/ITL | 连续批处理、抢占、overlap | continuous batching；长短混杂 → 开抢占；命中加载要 overlap |

### 4.2 三步推导法（面试口头答）

1. **标特征**：列这个 Agent 负载的前缀共享度、输入长度、输出长度、分支数、并发数、是否跨实例。
2. **找瓶颈**：判断是「算力浪费（重复 prefill）」还是「延迟敏感（长 prefill 饿死 decode）」还是「容量/共享范围不够（单实例装不下）」。
3. **配特性 + 说边界**：对应开 prefix/radix cache、chunked prefill、P/D 解耦、分层/远端 KV；但每开一个都要说清「什么条件下反而更慢」（回第 3 节）。

### 4.3 两个对照示例（通过关卡的关键）

**示例 A —— 多轮 Agent + 固定大 system prompt + 长历史、单实例**

- 特征：前缀共享度高、ISL 高、有树搜索分支、单节点。
- 推导：开 **Radix Cache**（token 粒度复用多轮历史 + 树搜索共享前缀）→ 长 prefill 开 **chunked prefill**（避免饿死 decode）→ 单实例**不需要** LMCache/远端 KV（没有跨实例可复用的前缀，徒增加载延迟）。
- 反例警醒：别看到「多轮」就上 P/D 解耦——单节点长历史若 ISL:OSL 错配不大，解耦的 KV 传输反而是新增瓶颈。

**示例 B —— 跨节点多副本、共享 system prompt、短输入短输出**

- 特征：跨节点前缀可复用，但输入短。
- 推导：跨节点共享 system prompt → 可考虑 **KV-aware routing / LMCache 远端 KV**；但**输入短**意味着重算 prefill 很便宜——要先用第 3.3 节公式算净收益，若「远端加载延迟 > 短 prefill 重算」，就**不要**上远端 KV，宁可各节点本地重算。
- 反例警醒：短输入（200/200）是最典型的「高级特性负收益」场景，llm-d 明确不建议解耦。

### 4.4 关卡自检清单

能对任意 Agent 负载回答以下五点，即视为通过：

- [ ] 前缀共享度 → 要不要 cache、选块粒度还是 token 粒度？
- [ ] ISL:OSL → 要不要 chunked prefill、要不要 P/D 解耦？
- [ ] 分支/树搜索 → 是否值得用 radix tree 而不是块 cache？
- [ ] 跨实例/跨节点 → 是否值得上远端 KV，命中加载延迟会不会反噬？
- [ ] 每条配置都答出「什么负载下反而更慢」（第 3 节的两条硬约束）

---

## 5. 本文结论

本周六个知识点可以收敛成**一条主线 + 三个层次 + 两条硬约束**：

- **一条主线**：KV Cache 是 serving 的核心资产，所有高级特性都是围绕「**怎么分、怎么复用、怎么存更大复用更远**」展开的。
- **三个层次**：PagedAttention 管「KV 放哪」（分页消碎片），Prefix/Radix Cache 管「什么值得复用」（块 vs token 精度之争），分层 KV Cache 管「KV 能存多大、复用多远」（GPU→CPU→NVMe→远端）。
- **两条硬约束**：凡是涉及「把 KV 搬一次」的特性（P/D 解耦、远端 KV Cache），划算与否都取决于 **KV 传输的带宽/延迟** 和 **ISL:OSL / 命中率**——传输比 prefill 还慢、或没有资源错配/复用可赚时，高级特性反而更慢。
- **阶段关卡能力**：能从一段 Agent 请求特征（前缀共享度、ISL:OSL、分支数、并发、是否跨实例）推导出该开哪些 serving 特性，并说清每条的反例条件——这既是面试题，也是第 12 周「编码与分析」实验的选题依据。

---

## 参考资料

- 本周六篇文档：[01](01-PagedAttention-KV-Cache碎片与分页思想.md) · [02](02-vLLM-Serving-Benchmark与缓存章节.md) · [03](03-SGLang-Runtime-RadixCache与Serving概览.md) · [04](04-LMCache-Quickstart与Integration-命中与传输度量.md) · [05](05-Mooncake-KV-centric架构与Dynamo-llm-d-PD解耦对照.md) · [06](06-TensorRT-LLM-ContinuousBatching-ChunkedPrefill-KV路由与PD解耦收益条件.md)
- [PagedAttention（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180)
- [SGLang（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104)
- [Mooncake（arXiv:2407.00079 / FAST'25）](https://arxiv.org/abs/2407.00079)
- [NVIDIA Dynamo — Disaggregated Serving / KV-Aware Routing](https://docs.dynamo.nvidia.com/)
- [llm-d — P/D Disaggregation 指南](https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/README.md)
- [LMCache Quickstart / Integration](https://docs.lmcache.ai/)
