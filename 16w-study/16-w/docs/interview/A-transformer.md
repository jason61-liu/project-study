# A. Transformer 与生成（1–15）

1. Attention 点积方差随 `d_k` 增长，除以 `sqrt(d_k)` 把尺度稳定在约常数量级，避免 softmax 饱和、梯度过小。
2. Q 表示当前 token 要找什么，K 表示可被匹配的特征，V 是被聚合的信息。输入通常从 `[B,T,d_model]` 投影并拆头为 `[B,H,T,d_head]`，注意力后合并回模型维度。
3. Causal Mask 禁止看到未来 token，保证自回归；Padding Mask 排除补齐位置，避免无效 token 参与注意力。二者可叠加但语义不同。
4. 多头有独立投影和 softmax，可在不同子空间形成不同注意力分布；一个宽单头只有一张分布，表达约束不同。
5. MHA 每头独立 K/V，质量上限高但 KV 最大；MQA 全头共享一组 K/V，缓存最省但可能损质量；GQA 让若干 Q 头共享 K/V，是常见折中。
6. RoPE 对 Q/K 按位置做复旋转，使点积自然携带相对位置。超过训练长度会发生相位分布外推、频率失配和注意力退化，需缩放策略且必须长上下文实测。
7. Pre-Norm 在子层前归一化，残差主路径更稳定，深层训练容易；Post-Norm 在残差相加后归一化，原论文采用但深层更难训，可能有不同最终质量。
8. Prefill 一次处理整段输入，矩阵大、并行度高，常偏算力受限；Decode 每步只生成一个 token，却反复读取权重和 KV，批量不足时偏带宽与调度受限。
9. KV Cache 保存每层历史 token 的 K/V。近似字节数为 `2 × layers × tokens × kv_heads × head_dim × bytes_per_element × batch`，还要计分页和碎片开销。
10. Temperature 缩放 logits；Top-k 只保留最高 k 个，Top-p 保留累计概率达到 p 的最小集合。通常先缩放再截断后归一化；多个约束叠加时更严格者主导候选集。
11. Tokenizer 决定同一文本的 token 数、词边界和稀有词拆分，因此直接影响上下文容量、计费、Prefill/Decode 时延与多语言质量。
12. 长上下文使普通 Attention 计算近似二次增长，并让 KV Cache 线性增长。MLA 压缩 KV 表示；稀疏注意力减少连接；线性注意力改变计算形式，但都带质量和适用分布条件。
13. EOS 是模型生成的结束 token，Stop Sequence 是运行时匹配文本后截断，业务终止条件则检查任务 Outcome、预算、审批或失败状态。前两者不能证明任务完成。
14. 模型输出 logits，经 temperature、mask、top-k/top-p 等处理后 softmax 得概率，再采样或取最大值选 token，追加上下文并循环到终止。
15. 速度还受架构、激活参数量、精度、内存带宽、KV、输入输出长度、批处理、并行策略、内核、硬件和调度影响；总参数量只是一个变量。

