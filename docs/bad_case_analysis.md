# SAM 阶段性 Bad Case 分析

本文档记录当前实验中暴露出的主要方法问题，以及已经完成或计划中的修正方向。这里不讨论环境配置问题，只讨论系统方法本身。

## 1. 图扩展带来收益，但也会引入噪声路径

现象：在 HotpotQA 300 条实验中，SAM-full 相比 Embedding Top-k 提升了证据召回率；但在部分样本中，图扩展会把有效向量候选挤出 top-k，导致答案命中没有同步提升。

已做修正：

- 增加 `sam_vector_anchor` 和 `sam_adaptive_anchor`，测试是否需要固定保留更多初始向量候选。
- 在 `cases.json` 中记录路径分数拆解，便于定位图路径是否过强。
- 在 `GraphBuilder` 中加入边质量约束，过滤低信息关键词导致的弱关系边。
- 在真实 embedding 300 条主实验中对比一跳与二跳扩展。二跳 run 中 SAM-full 证据召回率为 0.860，低于 Embedding Top-k 的 0.877；一跳 run 中 SAM-full 证据召回率提升到 0.890，高于 Embedding Top-k 和 SAM-no-graph 的 0.877。

当前判断：图扩展是有效模块，但当前阶段应以一跳局部联想作为稳定主设置。二跳路径不是不能用，而是必须依赖更强的 RelationJudge、边质量约束和路径重排，否则会把弱连通关系误当作推理链。

## 2. 反馈机制在单轮 HotpotQA 上不明显

现象：`sam_no_feedback` 与 SAM-full 在 300 条独立样本实验中的总体指标接近。这说明反馈事件虽然被写入，但没有在当前任务设置中形成足够强的后续影响。

原因分析：

- HotpotQA 每条问题的候选文档相对独立，同一记忆节点和边被跨问题复用的概率较低。
- 单轮问答更适合验证图扩展，不适合充分验证长期记忆演化。
- 当前反馈权重偏保守，主要用于记录和轻量调整，尚未形成强强化学习式更新。

后续修正：

- 构造同主题连续问答实验，让同一批记忆节点和边在多轮查询中反复出现。
- 比较 SAM-full、SAM-no-memory-state、SAM-no-feedback、SAM-static-graph。
- 输出反馈前后边权变化案例，证明记忆状态能影响后续排序。

## 3. RAPTOR 在部分设置中仍强于 SAM

现象：`fair_ablation_hotpotqa_300` 中 RAPTOR 的证据召回率高于 SAM-full，说明层次摘要结构在当前 HotpotQA 候选集上有效。

已做尝试：

- 增加 `sam_with_summary`，将 query summary memory node 接入动态图谱。
- 让摘要节点只能作为中间路径，不直接作为最终证据，避免摘要节点刷分。

当前问题：

- 简单把同题候选文档全部连接到一个摘要节点，会造成过宽跳转。
- 摘要边权和普通语义边尚未充分区分。
- 摘要粒度偏粗，容易把无关候选也纳入同一中心节点。

后续修正：

- 将摘要节点从“每题一个中心摘要”改为“主题簇摘要”或“证据链摘要”。
- 在摘要节点中加入覆盖范围、来源节点数和摘要置信度。
- 只允许摘要节点参与候选扩展，不让其直接扩大无关路径。

## 4. RelationJudge 已接入，但预算限制下覆盖不足

现象：低预算 GPT-5.4 关系判别 smoke 已经跑通，缓存也能生效；但在低预算设置下，RelationJudge 只覆盖少量高风险边，不能作为完整主实验结论。

已做修正：

- 增加 `cached_gpt54` 关系判别接口。
- 增加缓存，避免同一候选边重复消耗模型额度。
- 修正预算耗尽时错误写入 `budget_exhausted` 关系类型的问题，保留原始候选关系类型。

当前判断：RelationJudge 可以成为提升边质量的关键模块，但需要更完整的预算和更严格的采样评估。短期应优先用于高风险边，而不是所有边。

## 5. NovelQA 长文本任务仍是短板

现象：NovelQA 12 条 smoke 中，SAM 的指标明显弱于 HotpotQA。原因不是数据接入失败，而是长文本任务本身需要更强的章节切分、跨章节线索组织和答案生成评估。

当前问题：

- NovelQA 没有稳定 gold evidence，单纯证据召回指标不如 HotpotQA 直接。
- 小说文本 chunk 更长，实体和事件关系跨章节出现，简单局部边不够。
- 多选答案需要结合生成式判断，仅靠检索命中不足。

后续修正：

- 按章节、场景和人物共现建立更稳定的长文本记忆节点。
- 增加面向 NovelQA 的答案选项判别流程。
- 先做小规模 12 条或 30 条 smoke，再扩大到正式对照。

## 6. 在线 embedding 主实验已完成，但需要继续控制运行时成本

现象：公司 embedding endpoint 已经通过连通性测试，并完成 HotpotQA 30 条和 300 条真实 embedding 实验。扩大实验时确实会触发 qpm 限流，因此必须通过低并发、分批预热和 SQLite cache 控制调用成本。

已做修正：

- `.env.local` 模板改为低并发默认配置。
- `AzureOpenAISDKEmbeddingProvider` 增加 qpm/429 限流重试。
- README 和实验协议补充低并发、限流等待和 cache 使用方式。
- `scripts/plan_embedding_run.py` 和 `scripts/warm_embedding_cache.py` 已覆盖文档、query、query summary 和 RAPTOR runtime summary。
- `MemoryConsolidator` 已改为用证据节点向量合成长期记忆向量，避免反馈阶段每条样本再次调用在线 embedding。

当前结果：

- `outputs/runs/hotpotqa300_real_embedding_main_v4_hops1/` 中，SAM-full 证据召回率 0.890，Embedding Top-k 和 SAM-no-graph 均为 0.877。
- 该 run 的正式检索阶段复用本地 cache，没有继续请求线上 embedding。

## 总体判断

当前 SAM 的主要有效模块是按需建图和语义联想检索；当前主要短板是反馈演化没有在连续任务中充分显现，摘要结构还没有达到 RAPTOR 的稳定性，长文本 NovelQA 需要单独优化。下一阶段的关键不是继续堆方法名，而是围绕这些 bad case 做可验证的修正，并保证每个修正都有对应实验结果。
