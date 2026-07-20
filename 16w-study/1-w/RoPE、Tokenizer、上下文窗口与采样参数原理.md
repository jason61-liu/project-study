# RoPE、Tokenizer、上下文窗口与采样参数原理

## 1. 六个概念位于生成流程的什么位置

这六个概念不属于同一层次。它们分别解决输入表示、位置建模、长度限制和输出选择问题：

```text
文本
  ↓ Tokenizer：文本切成 token，并映射为 token ID
token ID 序列
  ↓ Context Window：限制本次请求可处理的 token 总量
Embedding + Transformer
  ↓ RoPE：向 Attention 的 Q、K 注入位置信息
下一个 token 的 Logits
  ↓ Temperature：调整概率分布的尖锐或平坦程度
  ↓ Top-k / Top-p：限制候选 token 集合
  ↓ Sampling：按最终概率抽取一个 token
新 token
  ↓ 重复 Decode，直到停止
```

先给出最短定义：

| 概念 | 解决的问题 | 直接影响 |
|---|---|---|
| Tokenizer | 文本如何变成模型可处理的离散编号 | token 数、词表、输入成本 |
| Context Window | 一次最多能处理多少 token | 可见历史、KV Cache、计算量 |
| RoPE | Attention 如何知道 token 的位置 | 顺序、距离与长上下文能力 |
| Temperature | 如何调整输出概率分布 | 确定性与随机性 |
| Top-k | 只允许概率最高的多少个候选 | 固定候选数量 |
| Top-p | 只保留累计概率达到阈值的最小候选集 | 动态候选数量 |

## 2. Tokenizer：文本与 token ID 之间的桥梁

### 2.1 模型为什么不能直接读取字符串

神经网络处理的是数值张量，而不是汉字、单词或 Unicode 字符串。Tokenizer（分词器）负责把文本转换成离散 token，再把每个 token 映射为词表中的整数编号：

```text
"我喜欢 Transformer"
        ↓ Tokenizer
["我", "喜欢", " Transformer"]
        ↓ Vocabulary
[314, 9821, 14621]
```

模型收到的是 token ID。Embedding 矩阵再将每个 ID 映射为 $D$ 维向量：

$$
E\in\mathbb{R}^{V\times D}
$$

$$
x_t=E[\operatorname{token\_id}_t]
$$

其中：

- $V$：词表大小；
- $D$：模型隐藏维度；
- $x_t$：第 $t$ 个 token 的初始隐藏表示。

如果 Batch Size 为 $B$、序列长度为 $T$，Embedding 输出形状为：

$$
[B,T]\rightarrow[B,T,D]
$$

### 2.2 token 不等于字、词或字符

一个 token 可能是：

- 一个完整单词；
- 单词的一部分；
- 一个汉字或多个汉字；
- 标点或空格；
- 一个字节片段；
- 特殊控制符。

同一段文本使用不同 Tokenizer，token 数和 token ID 都可能不同。例如，下面只是概念示意：

```text
"unbelievable"

Tokenizer A: ["un", "believ", "able"]      → 3 tokens
Tokenizer B: ["unbelievable"]              → 1 token
Tokenizer C: ["un", "bel", "iev", "able"] → 4 tokens
```

因此，字符数、单词数和 token 数不能互相直接替代。

### 2.3 为什么使用子词分词

如果把每个完整单词都放进词表，会遇到：

- 词形变化导致词表迅速膨胀；
- 人名、代码、拼写错误和新词难以覆盖；
- 低频词训练样本太少。

如果只使用单字符或单字节，词表较小，但序列会变长。子词 Tokenizer 在二者之间折中：高频片段使用较大的 token，低频词拆成更小片段。

常见方法包括：

- **BPE**：反复合并高频相邻符号；
- **WordPiece**：根据语言模型式目标选择子词合并；
- **Unigram**：从候选子词集合中选择概率较优的切分；
- **SentencePiece**：直接从原始文本训练，可实现 BPE 或 Unigram；
- **Byte-level / byte fallback**：退回到字节表示，减少真正的未知字符。

这些名字描述的是不同算法或工具体系，不能完全当作同义词。

### 2.4 词表和特殊 token

Tokenizer 的词表通常还包含：

- BOS：序列开始；
- EOS：序列结束；
- PAD：Batch 对齐填充；
- UNK：未知 token，现代字节回退方案可能很少使用；
- 模型或聊天模板使用的角色、分隔和工具调用 token。

特殊 token 是模型协议的一部分。聊天模型看到的通常不只是用户文字，而是套用 Chat Template 后的完整 token 序列。错误添加、遗漏或重复 BOS/EOS，可能改变模型行为。

### 2.5 编码和解码不一定是直观的一一对应

编码：

$$
\text{text}\xrightarrow{\text{Tokenizer}}\text{token IDs}
$$

解码：

$$
\text{token IDs}\xrightarrow{\text{Tokenizer}}\text{text}
$$

单个 token 单独解码时，结果可能不是完整合法字符；多个 token 拼接后才构成正确文本。空格也可能被编码进 token 本身。因此调试生成结果时，通常应解码完整 token 序列，而不是假设每个 token 都对应一个可独立显示的词。

### 2.6 Tokenizer 的实际影响

Tokenizer 会影响：

1. **上下文占用**：同一文本切出的 token 越多，占用的 Context Window 越大；
2. **费用**：按 token 计费的服务会受切分结果影响；
3. **延迟和显存**：更多 token 意味着更大的 Attention 计算量和 KV Cache；
4. **多语言效率**：不同语言在某个词表中的压缩效率可能不同；
5. **模型兼容性**：Checkpoint 必须使用与训练时匹配的 Tokenizer 和特殊 token 配置。

## 3. Context Window：模型一次能看到多少 token

### 3.1 基本定义

Context Window（上下文窗口）表示模型一次前向计算或一次生成请求中可以处理的 token 范围。对典型自回归生成，可以先建立下面的预算关系：

$$
T_{\text{prompt}}+T_{\text{generated}}\le T_{\text{context}}
$$

例如上下文窗口为 8192 token，Prompt 已占用 6000 token，那么理论剩余空间最多约为：

$$
8192-6000=2192\text{ tokens}
$$

实际 API 或推理引擎还可能分别设置最大输入长度、最大输出长度、保留空间和特殊 token，因此最终限制应以具体模型与服务配置为准。

### 3.2 Context 包含哪些内容

上下文不只是用户刚输入的问题，还可能包括：

- System Prompt；
- 历史对话；
- Chat Template 和角色标记；
- 检索到的文档；
- 工具定义与工具结果；
- 图片或音频被编码后占用的 token；
- 当前正在生成的输出；
- BOS、EOS 等特殊 token。

因此，界面上看到的文字长度不等于模型实际接收的 token 数。

### 3.3 上下文窗口不是“永久记忆”

Context Window 是本次推理可见的工作区，不代表模型永久记住其中内容：

- 内容被截断或移出窗口后，模型通常无法再直接访问；
- 新会话不会自动继承旧会话内容；
- 模型权重不会因为普通推理请求而更新；
- 长期记忆通常需要外部数据库、检索或状态系统。

Context Window、模型参数知识、KV Cache、Prompt Cache 和业务数据库是不同概念。

### 3.4 为什么长上下文消耗更多资源

对标准全注意力，Prefill 阶段的 Attention Score 矩阵形状为：

$$
[B,H,T,T]
$$

其 Attention 计算量随序列长度近似按：

$$
O(T^2)
$$

增长。使用 KV Cache 进行 Decode 时，不需要重新计算全部历史 K、V，但每个新 Query 仍要读取并匹配前 $T$ 个历史 token，所以每生成一个新 token 的 Attention 工作量近似为：

$$
O(T)
$$

KV Cache 容量则与上下文长度线性增长：

$$
\text{KV bytes}=2LBT H_{kv}D_hs
$$

因此，上下文变长会同时影响 Prefill 延迟、Decode 延迟和显存占用。

### 3.5 标称窗口、训练窗口与有效能力

“模型支持 128K”并不自动意味着它能在 128K 中同等准确地利用所有信息。需要区分：

- **标称窗口**：系统允许输入的最大 token 数；
- **训练长度分布**：模型训练时实际见过哪些长度；
- **位置编码外推能力**：超过训练长度后位置表示是否仍稳定；
- **有效上下文能力**：在检索、推理和生成任务中实际利用远距离信息的能力。

长上下文中还可能出现信息稀释或“中间信息较难被利用”等现象。因此应通过目标任务评测，而不能只看最大长度数字。

### 3.6 超出窗口后会发生什么

不同系统可能：

- 直接拒绝请求；
- 截断最早的 token；
- 截断某一部分 Prompt；
- 使用滑动窗口 Attention；
- 对历史内容进行摘要或检索；
- 使用位置缩放扩展可接受长度。

截断策略会影响哪些信息仍然可见，不能假设系统一定自动保留最重要内容。

## 4. RoPE：把位置信息旋转进 Q 和 K

### 4.1 为什么 Attention 需要位置信息

不加入位置机制时，Self-Attention 本身主要根据 token 内容计算关系，无法充分区分：

```text
猫追狗
狗追猫
```

两句话包含相似 token，但顺序不同，语义也不同。模型需要知道 token 位于第几个位置，以及两个 token 相隔多远。

RoPE（Rotary Position Embedding，旋转位置编码）通过按位置旋转 Attention 中的 Query 和 Key 向量，把位置信息直接融入点积。

### 4.2 二维旋转的直觉

先看二维向量：

$$
x=
\begin{bmatrix}
x_0\\
x_1
\end{bmatrix}
$$

在位置 $m$，将它旋转角度 $m\theta$：

$$
R(m\theta)=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}
$$

$$
\operatorname{RoPE}(x,m)=R(m\theta)x
$$

位置越靠后，旋转角度越大。旋转改变方向但保持向量范数：

$$
\|R(m\theta)x\|=\|x\|
$$

### 4.3 高维向量如何旋转

Head 维度通常是偶数。RoPE 将 $D_h$ 维向量按两个维度一组进行旋转：

```text
(x0, x1)   使用频率 θ0
(x2, x3)   使用频率 θ1
(x4, x5)   使用频率 θ2
...
```

常见频率形式为：

$$
\theta_i=\operatorname{base}^{-2i/D_h}
$$

不同维度对使用不同频率，有的变化快，有的变化慢，使模型能表示不同尺度的位置关系。具体维度配对方式、频率定义和缩放策略会因实现而异。

### 4.4 为什么 RoPE 能表示相对位置

设原始 Query、Key 为 $q,k$，分别位于位置 $m,n$：

$$
q_m=R(m)q,\qquad k_n=R(n)k
$$

它们的点积为：

$$
q_m^\top k_n
=q^\top R(m)^\top R(n)k
=q^\top R(n-m)k
$$

结果只显式依赖相对位置 $n-m$。这就是 RoPE 的关键性质：Q、K 分别根据绝对位置旋转，但 Attention 点积自然包含相对位置信息。

### 4.5 RoPE 作用于哪些张量

典型 Decoder Transformer 中：

$$
Q,K:[B,H,T,D_h]
$$

RoPE 按位置作用在每个 Head 的 Q、K 上：

$$
Q'=\operatorname{RoPE}(Q,\text{position})
$$

$$
K'=\operatorname{RoPE}(K,\text{position})
$$

然后计算：

$$
\operatorname{Attention}(Q',K',V)
=\operatorname{softmax}\left(
\frac{Q'K'^\top}{\sqrt{D_h}}+M
\right)V
$$

通常：

- RoPE 作用于 Q、K；
- 不作用于 V；
- 不改变张量形状；
- 不增加可训练的位置 Embedding 表；
- 使用 KV Cache 时，通常缓存已经完成 RoPE 的 K。

### 4.6 RoPE 与长上下文

RoPE 可以计算任意位置的旋转角度，但这不代表模型天然能可靠外推到任意长度。超过训练长度后，模型可能遇到未充分学习的频率模式。

常见长上下文扩展思路包括位置插值、频率缩放和分段或动态缩放。它们改变位置到旋转频率的映射，以减少超出训练范围后的失真。但扩展标称窗口仍需要长上下文训练或继续训练，以及实际任务验证。

### 4.7 RoPE 常见误区

1. **RoPE 旋转 token ID**：错误。它旋转的是每层 Attention 的 Q、K 向量。
2. **RoPE 会改变张量形状**：错误。旋转前后形状相同。
3. **RoPE 也必须旋转 V**：典型实现不旋转 V。
4. **能计算更大位置就等于能准确理解更长文本**：错误。数学可计算性不等于训练后的有效能力。
5. **RoPE 与 Tokenizer 是同一阶段**：错误。Tokenizer 在模型外将文本转成 ID，RoPE 在模型层内处理 Q、K。

## 5. 从 Logits 到概率

Transformer 对当前序列计算后，会为词表中每个候选 token 输出一个 Logit：

$$
z=[z_1,z_2,\ldots,z_V]
$$

Logit 是未归一化分数，不是概率。Softmax 将它转换为概率：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

接下来可以：

- 直接选择概率最大的 token，即 Greedy Decoding；
- 按概率随机采样；
- 在采样前应用 Temperature、Top-k、Top-p 等策略。

这些参数只改变如何从模型已经给出的 Logits 中选择 token，不会为模型增加新知识，也不会扩大 Context Window。

## 6. Temperature：调整概率分布的形状

### 6.1 公式

Temperature（温度）记为 $\tau$，通常在 Softmax 前缩放 Logits：

$$
p_i(\tau)
=\frac{e^{z_i/\tau}}
{\sum_{j=1}^{V}e^{z_j/\tau}}
$$

通常要求：

$$
\tau>0
$$

### 6.2 温度降低会发生什么

当 $0<\tau<1$ 时，Logit 差异被放大：

```text
较大的 Logit  → 概率变得更大
较小的 Logit  → 概率变得更小
```

分布更尖锐，输出更稳定、更接近最高概率选择。

### 6.3 温度升高会发生什么

当 $\tau>1$ 时，Logit 差异被缩小，概率分布更平坦，低概率 token 更容易被采样。输出通常更多样，但错误、跑题或不连贯的风险也可能增加。

当 $\tau\rightarrow\infty$ 时，分布趋向均匀；当 $\tau\rightarrow0^+$ 时，分布趋向把概率集中到最大 Logit。很多 API 将 `temperature=0` 特殊处理为 Greedy Decoding，而不是直接执行除以零。

### 6.4 数值例子

设三个 token 的 Logits 为：

$$
z=[2,1,0]
$$

近似概率为：

| 温度 | 概率分布 | 直觉 |
|---:|---|---|
| $\tau=0.5$ | $[0.867,0.117,0.016]$ | 很尖锐 |
| $\tau=1$ | $[0.665,0.245,0.090]$ | 原始 Softmax |
| $\tau=2$ | $[0.506,0.307,0.186]$ | 更平坦 |

Temperature 不会改变 token 的概率排序，因为所有 Logit 都除以同一个正数；它改变的是概率差距。

## 7. Top-k：固定保留概率最高的 k 个 token

### 7.1 算法

Top-k Sampling 的过程是：

1. 按 Logit 或概率从高到低排序；
2. 只保留最高的 $k$ 个 token；
3. 将其余 token 的 Logit 设为 $-\infty$；
4. 对保留项重新归一化；
5. 从中采样。

设原概率为：

$$
p=[0.40,0.30,0.15,0.10,0.05]
$$

当 $k=3$ 时，只保留前三个：

$$
[0.40,0.30,0.15,0,0]
$$

重新归一化后：

$$
p'=
\left[
\frac{0.40}{0.85},
\frac{0.30}{0.85},
\frac{0.15}{0.85},
0,
0
\right]
$$

$$
p'\approx[0.471,0.353,0.176,0,0]
$$

### 7.2 k 的影响

- $k=1$：只保留最高分 token，等价于 Greedy；
- 较小 $k$：输出稳定，但可能过于保守或重复；
- 较大 $k$：候选更多，但可能引入不合理的长尾 token；
- $k$ 大于等于词表大小：基本不进行过滤。

Top-k 的缺点是候选数量固定。无论当前分布非常确定还是非常不确定，都保留相同数量的 token。

## 8. Top-p：保留累计概率达到 p 的最小集合

### 8.1 算法

Top-p Sampling 又称 Nucleus Sampling。它按概率从高到低排序，选择累计概率达到阈值 $p$ 的最小候选集合，然后重新归一化并采样。

仍使用：

$$
p=[0.40,0.30,0.15,0.10,0.05]
$$

当 Top-p 为 $0.8$：

```text
第 1 个：累计概率 0.40
第 2 个：累计概率 0.70
第 3 个：累计概率 0.85，首次达到 0.80
```

因此保留前三个 token。一般会包含那个使累计概率首次达到或超过阈值的 token。

### 8.2 为什么候选数量是动态的

如果模型很确定：

$$
p=[0.92,0.03,0.02,0.02,0.01]
$$

当 Top-p 为 $0.9$ 时，只需第一个 token。

如果模型不确定：

$$
p=[0.25,0.22,0.20,0.18,0.15]
$$

同样使用 Top-p $=0.9$，则需要保留多个 token。Top-p 根据当前分布动态调整候选集大小，这也是它相比固定 Top-k 的主要特点。

### 8.3 p 的影响

- 较小 $p$：候选集更窄，结果更保守；
- 较大 $p$：保留更多概率质量，结果更多样；
- $p=1$：通常相当于不执行 Top-p 截断；
- 极小 $p$：通常仍至少保留最高概率 token，具体边界由实现决定。

## 9. Temperature、Top-k 和 Top-p 如何共同工作

一种常见流程为：

```text
原始 Logits
  ↓ 除以 Temperature
调整后的 Logits
  ↓ Top-k 过滤
  ↓ Top-p 过滤
候选 Logits
  ↓ Softmax 并重新归一化
  ↓ Sampling
下一个 token
```

但不同框架可能采用不同的过滤顺序、最少保留 token 数和数值处理方式，因此相同参数不保证在所有实现中产生完全相同的结果。

三者作用不同：

| 参数 | 操作对象 | 核心作用 | 候选数量 |
|---|---|---|---|
| Temperature | 全部 Logits | 调整相对概率差距 | 本身不删候选 |
| Top-k | 排名 | 只保留最高的 $k$ 个 | 固定上限 |
| Top-p | 累计概率 | 保留主要概率质量 | 随分布动态变化 |

同时启用 Top-k 和 Top-p 时，最终集合通常是连续过滤后的更小候选集。例如 Top-k 先限制最多 50 个，Top-p 再从这些候选中保留累计概率达到 0.9 的最小集合。

### 9.1 参数选择的思路

不存在适用于所有模型和任务的固定最优值。一般可以从目标出发：

- **事实问答、代码补全、结构化输出**：偏低随机性，方便稳定复现；
- **创意写作、头脑风暴**：可以适当提高多样性；
- **严格格式任务**：优先使用约束解码、Schema 或语法约束，不能只依赖低温度；
- **需要复现**：除采样参数外，还要固定随机种子、模型版本、Prompt 和推理实现；并行执行仍可能带来差异。

不要同时大幅提高 Temperature、Top-k 和 Top-p 后，再期待模型保持严格可靠。调参应逐项观察质量、多样性、重复率和格式成功率。

## 10. 六个概念之间的关键联系

### 10.1 Tokenizer 决定 Context Window 如何被消耗

同一段文本经过不同 Tokenizer 可能得到不同的 $T$。token 越多：

- 越快占满上下文窗口；
- Prefill Attention 计算越多；
- KV Cache 越大；
- Decode 读取历史 KV 的带宽需求越高。

### 10.2 Context Window 决定 RoPE 需要处理的位置范围

序列长度为 $T$ 时，RoPE 要为位置 $0$ 到 $T-1$ 的 Q、K 生成相应旋转。扩展 Context Window 时，必须考虑 RoPE 在更大位置上的频率行为，以及模型是否经过相应长度训练。

### 10.3 采样参数不改变上下文理解能力

Temperature、Top-k 和 Top-p 只处理当前步的输出 Logits：

- 不会让模型看到被截断的历史；
- 不会扩大 Context Window；
- 不会修改 RoPE；
- 不会改变 Tokenizer；
- 不会给模型增加知识。

提高 Temperature 可能让回答看起来更丰富，但不代表推理能力或事实准确率提高。

### 10.4 生成的 token 会继续占用窗口

每次采样得到的新 token 都会被追加到序列：

$$
T\leftarrow T+1
$$

然后新 token 的 K、V 被写入 KV Cache，下一步继续生成。因此生成越长，剩余 Context Window 越少，KV Cache 和单步 Attention 工作量也越大。

## 11. 常见误区

1. **一个 token 就是一个单词**：错误。token 可能是子词、字符、字节、标点或特殊符号。
2. **上下文窗口只计算用户输入**：错误。系统提示、历史、工具定义、模板和输出都会占用。
3. **上下文窗口等于长期记忆**：错误。窗口外内容不会自动成为模型永久记忆。
4. **RoPE 是一种 Tokenizer**：错误。Tokenizer 处理文本和 ID，RoPE 处理 Attention 的 Q、K。
5. **RoPE 给 V 也加入旋转**：典型 RoPE 只旋转 Q、K。
6. **Temperature 越高，模型越聪明**：错误。它只是让采样分布更平坦。
7. **Temperature 为 0 是普通公式中的合法除法**：错误。通常由框架特殊处理成 Greedy。
8. **Top-k=10 表示前十个 token 概率不变**：错误。过滤后还要重新归一化。
9. **Top-p=0.9 表示固定保留 90% 的词表**：错误。它保留累计概率达到 0.9 的动态候选集。
10. **Top-p 候选概率之和必须刚好等于 p**：错误。加入临界 token 后累计值通常会超过 p。
11. **采样参数可以修复模型不知道的事实**：错误。它们只改变选择方式。
12. **同样的参数在所有框架中完全等价**：错误。处理顺序和边界实现可能不同。

## 12. 自测题

### 12.1 题目

1. 为什么 token 数不能用字符数直接估算？
2. Tokenizer 输出 `[B,T]` 后，Embedding 为什么会得到 `[B,T,D]`？
3. 一个 8192 token 的窗口已经使用 7000 token，是否一定还能生成 1192 token？
4. RoPE 为什么旋转 Q、K 而通常不旋转 V？
5. 为什么 RoPE 的点积能够表示相对位置？
6. Temperature 是否会改变 token 的概率排名？
7. Top-k 和 Top-p 的候选集大小有什么区别？
8. 为什么 KV Cache 存在时，长上下文 Decode 仍会变慢？

### 12.2 参考答案

1. **为什么 token 数不能用字符数直接估算？**

   Tokenizer 按自己的词表和切分算法处理文本。一个 token 可能覆盖多个字符，一个字符也可能被拆成多个字节 token；空格、标点和特殊标记同样可能占用 token。因此必须使用目标模型配套的 Tokenizer 实际编码后计数。

2. **Tokenizer 输出 `[B,T]` 后，Embedding 为什么会得到 `[B,T,D]`？**

   `[B,T]` 中每个元素是一个 token ID。Embedding 矩阵为 $[V,D]$，每个 ID 索引出一行 $D$ 维向量，所以每个 token 位置新增隐藏维度 $D$，输出变为 `[B,T,D]`。

3. **一个 8192 token 的窗口已经使用 7000 token，是否一定还能生成 1192 token？**

   不一定。$8192-7000=1192$ 只是简化预算。服务可能还需要加入 EOS、聊天模板或其他特殊 token，并可能设置独立的最大输出长度和保留空间，应以最终编码结果与具体推理配置为准。

4. **RoPE 为什么旋转 Q、K 而通常不旋转 V？**

   Attention 的位置匹配发生在 $QK^\top$ 点积中。旋转 Q、K 可以使匹配分数包含相对位置信息；V 是根据已经得到的注意力权重被汇总的内容，通常不需要参与位置匹配旋转。

5. **为什么 RoPE 的点积能够表示相对位置？**

   因为旋转矩阵满足：

   $$
   (R(m)q)^\top(R(n)k)
   =q^\top R(n-m)k
   $$

   两个绝对位置的旋转在点积中合并为位置差 $n-m$。

6. **Temperature 是否会改变 token 的概率排名？**

   当 $\tau>0$ 且只执行温度缩放时，不会。所有 Logit 同除以同一个正数，大小顺序不变；改变的是 Softmax 后的概率差距。

7. **Top-k 和 Top-p 的候选集大小有什么区别？**

   Top-k 最多保留固定的 $k$ 个最高分 token。Top-p 保留累计概率达到阈值的最小集合，候选数量随当前概率分布动态变化。

8. **为什么 KV Cache 存在时，长上下文 Decode 仍会变慢？**

   KV Cache 避免重复计算历史 token 的 K、V，但每个新 Query 仍需读取并匹配全部历史 Key，再用权重汇总历史 Value。上下文长度 $T$ 增长时，单步 KV 读取量和 Attention 工作量仍近似线性增长。

## 13. 面试版总结

> Tokenizer 把文本切成 token ID，并直接决定序列长度和上下文消耗；Context Window 限制模型本次能够看到的 token 总量，长上下文会增加 Attention 计算和 KV Cache；RoPE 在每层 Attention 中按位置旋转 Q、K，使点积自然包含相对位置信息；模型输出 Logits 后，Temperature 调整概率分布的尖锐程度，Top-k 固定保留最高的 k 个候选，Top-p 动态保留累计概率达到阈值的最小候选集。前三者影响模型输入和内部计算，后三者只影响生成时如何选择下一个 token。

## 14. 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)：RoPE 的定义与相对位置性质。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer、Attention 与序列位置建模背景。
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词方法在神经机器翻译中的应用。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)：SentencePiece。
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)：Nucleus（Top-p）Sampling。
