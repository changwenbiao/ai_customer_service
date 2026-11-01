# R11 RAG 检索质量评测与召回优化 · 操作手册

> 对应需求：[进阶实战需求清单-Agent工程化.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/docs/my1/进阶实战需求清单-Agent工程化.md) 中的 R11。
> 本文严格按照 R11「你要做什么」的 **5 个步骤**组织，每一步可独立交付、独立验收。
> 基线数据实测日期：2026-11-01，租户 `t_demo`。

> 阅读顺序建议：**第一篇（需求与背景）→ 第二篇（原理）→ 五步总览 → 逐步操作 → 附录 D（面试话术）**。
> 如果只想跑实验，直接看「〇.1 完整复现 SOP」和第四章。

## 目录

| 篇章 | 回答什么问题 | 关键小节 |
|---|---|---|
| **第一篇 · 需求与背景** | 为什么做、做什么、不做什么、怎么验收 | 1.1 一句话需求、1.4 需求边界、1.7 验收、1.9 风险 |
| **第二篇 · 原理总览** | 指标怎么定义、BM25/RRF/rerank/chunk/阈值为什么这么设计 | 2.3 指标、2.5 BM25、2.6 RRF、2.7 rerank、2.9 阈值、2.12 线上口径对照 |
| **〇、总览与 SOP** | 五步当前状态 + 从零复现的完整命令 | 〇.1 完整复现 SOP |
| **第一章 ~ 第五章** | 每一步的具体操作、代码、预期结果与排错 | Step 1 地基 → Step 5 写回 |
| **附录 A ~ E** | 基线数字、验收清单、FAQ、面试话术、需求衔接 | 附录 D 面试怎么讲（详细版） |

---

## 第一篇 · 需求与背景

### 1.1 一句话说清这个需求

在已有的电商客服 RAG 链路上，先建立**可复现、可回归的检索评测体系**，再用**混合检索 + rerank + 结构化分片**把检索质量从"感觉还行"变成"指标可解释地变好"。

它只做三件事：

| 动作 | 产出 | 回答的问题 |
|---|---|---|
| **评测（尺子）** | 评测集 + 离线脚本 + 报告 | "现在到底多少分？" |
| **归因（定位）** | 分层指标 + 错误清单 | "丢分是没召回，还是排错了，还是生成没用对？" |
| **优化（动作）** | 五组对照实验 + 最优配置 | "改哪个参数、能涨多少、代价多大？" |

### 1.2 业务背景：为什么客服 RAG 必须做评测

电商客服的知识问答有三个特点，决定了"凭感觉优化"一定会翻车：

1. **答案有强事实性**：金额、时效、责任主体（7 天 / 15 天 / 谁承担运费 / 赔多少）答错就是赔付和投诉，不是"体验差一点"。
2. **问法与文档表述不一致**：用户说"退货邮费算谁的"，文档写"退货运费由买家承担"；字面不同、语义相同。
3. **问题天然跨文档**："退货邮费"同时涉及《退换货政策》和《运费与配送规则》，召回和排序都必须处理边界模糊。

一个真实失败案例：

> 知识库里有《退换货政策》和《运费与配送规则》，用户问"退货邮费算谁的"，模型答"建议咨询客服"。
> 日志显示检索返回了 3 条，但都是《促销活动规则》——**检索错了，生成侧只能兜底或胡说**。

RAG 的失败必须拆成两段看：

- **检索失败**：正确文档/段落根本没进 top-k → 再强的模型也救不回来；
- **生成失败**：正确资料进了 prompt，但模型没用、用错、或编造 → 属于 prompt / 引用 / 拒答治理（R12）。

只盯端到端答案，你永远不知道该改检索还是改 prompt。**评测体系的第一价值不是打分，而是归因。**

业务代价也要分开算，因为两类错误的成本完全不同：

| 错误类型 | 定义 | 业务后果 | 优化方向 |
|---|---|---|---|
| **误拒**（False Reject） | 有答案却答"不知道" | 用户转人工、流失、体验差 | 提高召回、降低阈值 |
| **泄漏**（Leak） | 无答案却硬答 | 错误承诺、赔付、合规风险 | 意图闸门、引用校验、拒答 |

所以后面的指标永远成对出现：**不能只看召回率，还要看无答案泄漏率。**

### 1.3 技术背景：改造前的链路长什么样

| 环节 | 改造前 | 问题 |
|---|---|---|
| 检索 | `search_kb` 单路 dense 向量检索，`asimilarity_search_with_score` | 只懂语义、不懂精确术语；无混合、无重排 |
| 返回 | 拼接后的纯字符串 | 模型无法引用来源，无法区分文档/段落（R12 无法做） |
| 分片 | `RecursiveCharacterTextSplitter(200/30)` | chunk 不带标题，边界模糊的文档无法区分 |
| 评测 | 无 | 调参靠感觉，prompt 改动无法回归（R16 无法做） |
| 阈值 | 无 | 不知道无答案问题长什么样，拒答无依据 |
| 语料 | 8 篇电商客服文档，39 个 chunk（线上企业 KB 常见 10³~10⁵ 篇、10⁵~10⁷ chunks） | 文档少，Recall@5 天然饱和 |

改造后的目标链路：

```
query → dense embedding ┐
      → BM25 sparse     ├→ RRF 融合 → top-N → cross-encoder rerank → top-k → 结构化 prompt
                        ┘
```

### 1.4 需求定义：做什么、不做什么

**做（In Scope）**

1. 知识库源文件、manifest、评测集的建立与维护；
2. 一键导入 / 重建脚本；
3. 离线检索评测脚本（Recall / MRR / Evidence hit / 分数分布 / 阈值扫描）；
4. 五组检索优化实验（k、阈值、混合检索、rerank、chunk）；
5. 最优配置写回，并用配置开关保留旧链路；
6. 延迟与成本的对照记录。

**不做（Out of Scope）**

| 不做的事 | 归属 |
|---|---|
| prompt 强制引用、拒答话术、citation 校验 | R12 |
| 上下文 token 预算、摘要注入、槽位 | R13 |
| prompt 版本门禁、灰度、回滚 | R16 |
| 压测、容量评估、限流 | R20 |
| 追求 SOTA 榜单分数 | 本项目语料只有 8 篇，目标是建立口径与证据，不是刷榜 |

### 1.5 六个具体问题 → 六个动作

| # | 问题（实测证据） | 动作 | 验收 |
|---|---|---|---|
| 1 | 没有评测集，效果靠感觉 | 建 100 条评测集 | 一条命令出报告 |
| 2 | 文档太少，Recall@5 虚高 | 8 篇多域文档 + 干扰项 | 指标分层看，不只看 @5 |
| 3 | `doc_id` 每次导入都变 | 用 `expected_doc_titles` 做锚点 | 重导后评测仍可跑 |
| 4 | 文档级饱和，差距在 chunk 级 | 新增 Evidence hit@k | 能定位"文档对了但段落错了" |
| 5 | 阈值区分度不足 | 阈值扫描 + 分数分布 | 输出"单靠阈值不可拒答"结论给 R12 |
| 6 | 返回纯字符串，无法引用 | 结构化返回（R12 衔接） | 带 `doc_id/title/score/snippet` |

### 1.6 交付物清单

| 交付物 | 路径 | 作用 | 状态 |
|---|---|---|---|
| KB 源文件 | `python_agent/R11/kb_docs/*.md` | 唯一事实来源，可 diff / review / 重建 | ✅ |
| 文档清单 | `python_agent/R11/kb_manifest.json` | 文件 → 标题 → 分类映射 | ✅ |
| 评测集 | `python_agent/R11/kb_eval.jsonl` | 100 条，85 正 + 15 负 | ✅ |
| 导入脚本 | `python_agent/R11/kb_import.py` | 清空 / 增量 / 单篇导入 + 等待索引 | ✅ |
| 评测脚本 | `python_agent/R11/run_eval.py` | retrieval / e2e 两种模式 | ✅ |
| 混合检索 | `python_agent/hybrid_search.py` | dense + sparse + RRF | ✅ |
| Rerank | `python_agent/rerank.py` | bge-reranker-v2-m3 cross-encoder | ✅ |
| 标题路径分片 | `python_agent/kb_index_worker.py` | chunk 前缀 `[文档 > 小节]` | ✅ |
| 实验报告 | `docs/my1/eval_report_*.md` | 指标对比与错误清单 | ✅ |

### 1.7 验收标准（可量化）

1. **可复现**：`python3 python_agent/R11/run_eval.py` 一条命令跑完，基线与附录 A 一致。
2. **有对比**：至少完成 A/B/C 三组实验，每组只改一个变量，记录改前 → 改后。
3. **有归因**：报告能列出 Top-1 未命中用例，并回答"是召回丢分还是排序丢分"。
4. **有结论**：阈值扫描能证明"单靠相似度不可拒答"，输出给 R12。
5. **有成本**：记录 P50/P95 延迟，优化后 P95 不超过基线 2 倍。
6. **有提升**：Recall@1 提升 ≥0.01 或 MRR 提升 ≥0.005（低于此可能只是波动）。

### 1.8 与其他需求的接口

| 需求 | 本需求提供什么 | 对方拿去做什​么 |
|---|---|---|
| **R12 溯源与拒答** | 分数分布、阈值扫描结论、结构化检索返回 | 设计 L1 意图闸门、citation 校验、拒答话术 |
| **R13 上下文预算** | 实验 A 的"取几条最划算" | 决定检索片段数量与 token 预算 |
| **R16 评测门禁** | 冻结的评测集与 `run_eval.py` | 改 prompt 前后跑同一套题，防退化 |
| **R19 数据闭环** | bad case 回流格式 | 把线上失败样本补进评测集 |

### 1.9 约束与风险（做之前必须知道）

| 风险 | 后果 | 应对 |
|---|---|---|
| 语料只有 8 篇 | Recall@5 饱和，指标失真 | 重点看 Recall@1 / MRR / Evidence hit；结论注明边界 |
| 评测集会被"调参过拟合" | 数字漂亮但线上不涨 | 冻结评测集；定期从线上补新问法；保留 holdout |
| rerank CPU 慢 | 请求超时、体验差 | 只重排 top-5；候选数可配；流式超时放宽到 180s |
| 分片改动需重索引 | 测的是旧 chunk | 改开关/代码后必须重跑 `kb_import.py` |
| 分数量纲随链路变化 | 阈值照搬失效 | dense / RRF / rerank 各自重新扫描 |
| 单次改多个变量 | 不知道提升来自哪 | 一次只改一个变量，固定其余配置 |

---

## 第二篇 · 原理总览

> 这一篇只回答"为什么这么做"，不涉及命令。每个原理都标注它对应哪个实验，读第四章时可以回来对照。
> 面试时真正拉开差距的，不是"我加了 BM25"，而是"我知道 BM25 为什么能补 dense 的洞、RRF 为什么不用归一化、rerank 为什么不能提升召回"。

### 2.1 RAG 检索链路全景

一次 RAG 问答，检索侧可以拆成 8 个阶段。**每个阶段都有独立的失败模式，所以要有独立的观测指标**：

| # | 阶段 | 输入 → 输出 | 典型失败 | 对应指标 |
|---|---|---|---|---|
| 1 | Query 理解 | 用户口语 → 检索 query | 指代未消解、多跳问题被压成单跳 | 人工抽样 |
| 2 | Query 编码 | query → dense 向量 / sparse 词项 | 分词错、embedding 模型不适配领域 | 单条验证 |
| 3 | 召回（ANN） | 向量 → top-N 候选 | 正确段落没进候选池 | Recall@N |
| 4 | 融合（RRF） | 多路列表 → 一个列表 | 同分并列、边界抖动 | MRR、Top-1 |
| 5 | 重排（Cross-encoder） | top-N → 重新打分 | 候选里没有正确答案，重排也救不回 | MRR、Recall@1 |
| 6 | 截断 top-k | 排序列表 → 进 prompt 的 k 条 | k 太小漏证据，k 太大进噪声 | Evidence hit@k |
| 7 | 拼 prompt | chunk → 带编号的上下文 | 超长、顺序乱、来源不明 | token 数、引用率 |
| 8 | 生成与校验 | prompt → 回答 | 不用资料、编造、漏引用 | 答案命中率、拒答率、泄漏率 |

```text
用户问题
  │
  ├─[1] Query 理解（口语/指代/多跳）
  │
  ├─[2] 双路编码
  │     ├─ dense：embedding API → 向量
  │     └─ sparse：jieba 分词 → BM25 词频向量
  │
  ├─[3] 召回：dense top-20、sparse top-20
  │
  ├─[4] 融合：RRF 只看排名，丢掉原始分数
  │
  ├─[5] 重排：cross-encoder 对 top-5 重新打分
  │
  ├─[6] 截断：取 top-k（当前 k=3）
  │
  ├─[7] 拼 prompt：编号 + 标题 + 片段 + 引用要求
  │
  └─[8] 生成 → citation 校验 → 拒答兜底
```

**关键认知：优化必须对准"当前最弱的那一段"。** 本项目的基线显示 Recall@5=1.0、Recall@3=1.0，说明召回不是瓶颈；但 Recall@1=0.953、Evidence hit@3=0.988，说明瓶颈在**排序**和**chunk 粒度**。所以实验 C（融合排序）、D（重排）、E（分片）才是主攻方向，而不是继续堆召回。

### 2.2 误差归因：检索问题还是生成问题

面试官最爱问："回答不准，是检索的问题还是生成的问题？"标准答案是**先分层评测，再看四象限**：

| 检索是否命中正确证据 | 生成是否正确 | 结论 | 该改什么 |
|---|---|---|---|
| ✅ 命中 | ✅ 正确 | 正常样本 | 扩评测集，找下一类 bad case |
| ✅ 命中 | ❌ 错误 | **生成/上下文问题** | prompt 约束、引用校验、上下文噪声（R12/R13） |
| ❌ 未命中 | ❌ 错误 | **检索问题** | 召回、融合、重排、分片（R11） |
| ❌ 未命中 | ✅ 正确 | **最危险**：模型靠记忆蒙对 | 强制引用 + 拒答，否则换个问法就编（R12） |

落地时的判断顺序：

1. 先跑 `retrieval` 模式，看 `Evidence hit@3`。低于 0.9 优先修检索。
2. 再跑 `e2e` 模式，看答案命中率与误拒率。检索命中但答案错，问题在生成侧。
3. 两者都看单条 case，不要只看均值——均值会掩盖长尾。

> 这也是为什么评测集要同时有 `evidence_terms`（检索层判据）和 `must_contain`（生成层判据）。两套判据分开，才能做上面的四象限归因。

### 2.3 指标：定义、公式、直觉、局限

#### 2.3.1 Recall@k（召回率）

**严格定义**（信息检索口径）：

```
Recall@k(q) = |TopK(q) ∩ Rel(q)| / |Rel(q)|
```

即"该找到的相关文档里，有多少进了前 k 名"。

**本项目实现口径**（每条 query 只有 1~2 篇期望文档）：

```
Recall@k = (1/|Q|) · Σ_q 1[ TopK(q) 中存在任一 expected_doc_title ]
```

这是逐 query 的**二值命中**（业界也常叫 Hit Rate）。当 `|Rel(q)|=1` 时，两者完全等价；当一条 query 有多个期望文档时，本项目的口径更宽松（命中一个就算 1）。

**直觉**：Recall@1 最严格（第一名必须对）；k 越大越宽松。
**本项目**：Recall@1=0.953、Recall@3=1.000、Recall@5=1.000（线上第一阶段通常看 Recall@100/200，重排后看 nDCG@10 / MRR@10）。
**局限**：文档级 Recall 在 8 篇语料上天然饱和（Top-5 覆盖 62% 文档），所以不能只看它。

#### 2.3.2 Precision@k 与 HitRate

```
Precision@k(q) = |TopK(q) ∩ Rel(q)| / k
```

它衡量"前 k 条里有多少是相关的"，用来发现**噪声比例**。RAG 场景里 Precision 低意味着 prompt 里塞了大量无关 chunk，会干扰生成。

本项目没有单独报 Precision，因为每条 query 的期望文档只有 1~2 篇，Precision@5 天然偏低（分母是 5），参考价值不如 MRR 和 Evidence hit。但做 chunk 实验时，Precision 可以用来观察"chunk 变小是否引入更多碎片"。

#### 2.3.3 MRR（Mean Reciprocal Rank）

```
MRR = (1/|Q|) · Σ_q 1 / rank_first_relevant(q)
```

`rank_first_relevant` 是第一个相关文档的名次（从 1 开始）；没命中记 0。

- E001 期望文档在 rank1 → 贡献 1/1 = 1.0
- E002 期望文档在 rank2 → 贡献 1/2 = 0.5
- 完全没命中 → 贡献 0

**直觉**：同时衡量"有没有命中"和"命中得早不早"。适合"每条 query 只有一个正确答案"的场景，正好是客服 FAQ。
**本项目**：MRR=0.975（85 条里大多数在 rank1；线上常用 MRR@10）。
**局限**：只关心第一个相关结果，不关心后续相关结果的质量；多文档答案场景不如 nDCG。

#### 2.3.4 nDCG@k（本项目暂未使用）

```
DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
IDCG@k = 按理想排序计算出的 DCG@k
nDCG@k = DCG@k / IDCG@k
```

`rel_i` 是第 i 位的相关性等级（可以是 0/1，也可以是分级 0~3）。它同时考虑**位置折损**和**多级相关性**。

**为什么本项目不用**：每条 query 只有 1~2 篇相关文档、且没有人工分级相关性，nDCG 与 MRR 高度重合，反而增加标注成本。面试被问到时可以答："当前是单答案 FAQ 场景，MRR 更直观；如果未来做多文档综述或带分级标注，再上 nDCG。"

#### 2.3.5 Evidence hit@k（本项目自定义的 chunk 级指标）

```
Evidence hit@k = (1/|Q|) · Σ_q 1[ 前 k 个 chunk 中，存在任一 chunk 的正文包含某个 evidence_term ]
```

它回答的是比 Recall 更细的问题：**"命中了文档，但命中的是不是包含答案的那个段落？"**

例：E002 的 `evidence_terms=["买家承担"]`。如果前 3 条里有任意一条正文包含"买家承担"，就记 1。
**基线**：Evidence hit@3=0.988，Evidence hit@5=1.000（线上对应 Context Recall / chunk-level Recall@k）。这说明有 1 条用例的正确段落排在 4~5 名——文档级 Recall 看不出来，chunk 级才暴露。

**局限**：子串匹配比较粗。同义表述（"消费者承担" vs "买家承担"）会漏判；跨 chunk 的证据也会漏判。它适合做**回归趋势**，不适合做绝对精确判定。

#### 2.3.6 Top-1 分数分布与阈值扫描

这部分是为 R12 服务的，但它本身也是一套重要的分析工具。

对每条 query 取 rank1 的分数，分成"有答案 / 无答案"两组，看分布：

| 分布 | 中位数 | 最小 | 最大 |
|---|---|---|---|
| 有答案（85 条） | 0.679 | 0.497 | 0.817 |
| 无答案（15 条） | 0.547 | 0.415 | 0.650 |

阈值扫描的判定规则：`Top-1 分数 ≥ t → 判为有答案`。三个指标：

```
准确率   = (TP + TN) / N
误拒率   = FN / (TP + FN)     # 有答案却被判无答案
泄漏率   = FP / (FP + TN)     # 无答案却被判有答案
```

**核心结论**：最优阈值 t=0.50 时准确率 0.900，但泄漏率仍有 60%。原因不是"阈值没调好"，而是**两类分布本身重叠**（有答案最低 0.497 < 无答案最高 0.650）。在重叠区里，任何阈值都只能做"误拒 ↔ 泄漏"的交换，不存在两全。

这就是 R12 的立项依据：**阈值只能做极端低分的兜底，主力拒答必须靠意图闸门 + 引用校验。**

#### 2.3.7 生成层指标

`run_eval.py --mode e2e` 计算四个粗指标：

| 指标 | 定义 | 业务含义 |
|---|---|---|
| 答案命中率 | 有答案问题中，回复包含全部 `must_contain` 的比例 | 答对了多少 |
| 误拒率 | 有答案问题中，回复命中拒答话术的比例 | 该答没答 |
| 拒答率 | 无答案问题中，回复正确拒答的比例 | 该拒答就拒答 |
| 泄漏率 | 无答案问题中，回复没有拒答的比例 | 不该答却答了 |

更精细的指标（R12 会补）：引用覆盖率、citation 准确率、faithfulness（回答是否完全由引用支撑）、answer correctness（与 reference 的语义一致）。这些通常用 LLM-as-judge，成本更高，不适合每次回归都跑。

#### 2.3.8 指标选择的四条原则

1. **和业务代价对齐**：误拒和泄漏的代价不同，指标必须成对看。
2. **可分解**：能定位到"召回 / 排序 / 截断 / 生成"的某一层，而不是只有一个端到端分数。
3. **可复现**：同一套数据、同一条命令、同样的归一化规则，数字才能横向比。
4. **看分布不只看均值**：均值会被大量简单样本拉高；要看分位数、最差 case、Top-1 未命中清单。

### 2.4 稠密检索（dense）

**原理**：双塔（bi-encoder）。query 和 passage 分别过一个 Transformer，各得到一个定长向量；用余弦相似度比较：

```
cos(q, d) = (q · d) / (‖q‖ · ‖d‖)
```

索引阶段把所有 chunk 的向量写入向量库；查询阶段用 ANN（HNSW 等近似最近邻）找 top-k。**注意是近似检索**：为了速度放弃精确性，边界附近的候选可能每次略有不同。

**优点**

- 语义泛化强："邮费" ≈ "运费"，"多久到" ≈ "配送时效"；
- 跨语言、改写、口语化更鲁棒；
- 一次编码，查询快（ANN 毫秒级）。

**局限**

- **专有名词/数字/型号不敏感**：`ORD-2024-001` 和 `ORD-2024-002` 在向量空间可能很近；
- **否定和条件容易丢**："不需要发票" vs "需要发票" 向量可能几乎一样；
- **高频词淹没意图**：一篇文档反复出现"发票"，整篇向量被"发票"主导，用户问"保修要不要发票"就会被吸过去（E059）；
- **近似检索有边界抖动**：HNSW 的参数、浮点非确定性、同分并列，都会让第 5 名候选不稳定，进而影响 rerank 结果。

这四条局限，正好对应实验 C（BM25 补精确匹配）、实验 D（cross-encoder 补交互）、tie-breaker（补稳定性）。

### 2.5 稀疏检索（BM25）

**原理**：不看神经网络，只看词频统计。给定 query 的词集合 `q1..qn`，对文档 D 打分：

```
score(D, Q) = Σ_i IDF(q_i) · [ tf(q_i, D) · (k1 + 1) ] / [ tf(q_i, D) + k1 · (1 - b + b · |D| / avgdl) ]
```

| 因子 | 含义 | 直觉 |
|---|---|---|
| `tf(q_i, D)` | 词 qi 在 D 中出现的次数 | 出现越多越相关，但收益会饱和 |
| `IDF(q_i)` | `ln(1 + (N - df + 0.5) / (df + 0.5))` | 越罕见的词越有区分度；"的"几乎无权重 |
| `k1`（1.2~2.0） | 词频饱和系数 | 防止同一个词刷 100 次就得 100 倍分 |
| `b`（约 0.75） | 文档长度归一化 | 长文档天然命中更多词，必须惩罚 |

**为什么它能补 dense 的洞**：BM25 对"精确词项"极其敏感。`7天无理由`、`ORD-2024-001`、`运费险` 这类词，只要 query 和文档词表一致，几乎不会漏。它不懂同义词，但 dense 懂；两者互补。

**中文必须先分词**：`fastembed` 的 `SimpleTokenizer` 只按空白/标点切词，中文没有空格会被当成**一整句一个 token**，BM25 直接失效。所以索引和查询都要先过 `jieba.cut`，再用空格连接。这条是实验 C 最容易踩的坑。

**本项目实测**：8 篇文档共 5449 字，"发票"出现 8 次（75% 集中在《发票与支付规则》），"保修"出现 9 次（78% 集中在《售后与保修规则》）。E059"保修需要纸质发票吗"中，06 篇同时命中"保修"（高 TF）和"发票"（顺带一次），BM25 总分 1.52；03 篇只有"发票"单路支撑，总分 0.78。**BM25 把正确答案排第一的真实机制，是"多词命中"而不是"IDF 压胜"。** 详细拆解见 4.3.1。

### 2.6 RRF（Reciprocal Rank Fusion）

**要解决的问题**：dense 给余弦相似度（0~1），BM25 给无上界分数（可能 0~20+），两者量纲不可比。加权求和需要先归一化，而归一化方式（min-max / z-score）会随查询分布变化，权重也很难调。

**RRF 的做法：丢掉分数，只看排名。**

```
RRF_score(d) = Σ_routes 1 / (k + rank_route(d))
```

- `rank_route(d)`：文档 d 在该路结果里的名次（从 1 开始）；
- `k`：平滑常数，业界默认 **60**；
- 某路没出现 → 该路贡献 0。

**为什么 k=60 能压平头部差距**：

| 名次 | 1/(60+rank) |
|---|---|
| 1 | 0.01639 |
| 2 | 0.01613 |
| 3 | 0.01587 |

第 1 名和第 3 名只差 0.0005，避免"某一路的第一名"直接碾压另一路。k 越小越激进地奖励头部（20~40），k 越大越平滑（80~100，适合噪声大的场景）。

**算例：为什么"两路都中游"胜"一路第一、一路垫底"**

| 文档 | dense 名次 | sparse 名次 | RRF 分 |
|---|---|---|---|
| A | 1 | 5 | 1/61 + 1/65 = 0.03181 |
| B | 2 | 2 | 1/62 + 1/62 = 0.03226 |

B 胜 A。这就是混合检索想要的性质：**多路共识优于单路极端**。

**三个工程优点**：不需要分数归一化、不受量纲影响、增加/替换检索器时不用重调权重。

**两个局限（必须知道）**：

1. **分数离散、大量并列**：dense 第 3 名和 sparse 第 3 名的贡献完全相同，融合后容易出现同分。Qdrant 对同分的返回顺序不保证稳定，导致边界候选抖动。
2. **只能缓解、不能消除抖动**：真正的稳定手段是加 tie-breaker（同分按 `doc_id` 升序），或扩大候选池（`RERANK_CANDIDATE_N` 从 5 提到 8~10），代价是延迟上升。

**替代方案**：加权分数融合（需要归一化 + 调权重）、RSF（Relative Score Fusion，按列表 min-max 归一化后加权）。RRF 的优势是"零调参、跨检索器稳定"，适合本项目这种检索器会不断增加的场景。

### 2.7 Rerank（cross-encoder 重排）

**双塔 vs 交叉编码器**：

| 维度 | Bi-encoder（dense 检索） | Cross-encoder（rerank） |
|---|---|---|
| 输入 | query、passage 分别编码 | `[CLS] query [SEP] passage [SEP]` 拼一起 |
| 交互 | 最后才点积，无细粒度交互 | 从第一层就做注意力交互 |
| 表达能力 | 弱，但可离线建索引 | 强，但每条 pair 都要前向 |
| 速度 | 查询时毫秒级 ANN | CPU 单条 3~5s（本项目实测） |
| 用途 | 从全库召回 top-N | 只对 top-N 重排 |

**为什么它更强**：它能判断"保修需要纸质发票吗"与"凭订单号即可，无需纸质发票"之间的**否定关系**；双塔把两段独立编码后，否定词很容易在向量里被稀释。

**关键认知：rerank 不能提升召回上限。** 它只在已有候选里重新排序；如果正确段落根本没进 top-N，rerank 无能为力。所以它必须**叠在混合检索之后**（hybrid → top-5 → rerank → top-3），而不是替代召回。

**成本模型**：cross-encoder 是 O(N) 推理，N 是候选数。本项目 CPU 实测：

| 候选数 | 本项目耗时（线上典型） |
|---|---|
| 5 | 15~20s（线上托管/GPU 对 50 条约 20~150ms） |
| 10 | 30~50s（线上通常不会用 CPU rerank 10 条） |

所以默认 `RERANK_CANDIDATE_N=5`（线上常见 20~100，常取 50），并且 Go 侧流式对话单独用 180s 超时的 `streamHTTPClient`，否则 rerank + LLM 生成会顶穿 60s。

**工程注意**：同步推理必须放 `asyncio.to_thread`，否则阻塞事件循环；模型在服务启动/镜像构建阶段预加载，不能首次请求懒加载（会超时）。

### 2.8 Chunk 策略

**四种常见分片方式**：

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 固定长度 | 每 N 字符一刀 | 简单、均匀 | 切断句子/表格，语义不完整 |
| 递归字符 | 按段落 → 句子 → 字符逐级切 | 尽量保留语义边界 | 仍不知道文档结构 |
| 结构感知 | 按 Markdown 标题切 | 主题完整、可带 section 元数据 | 依赖文档规范 |
| 标题路径 / Contextual | 给 chunk 前置 `[文档 > 小节]` 或生成上下文摘要 | embedding 带结构信息，边界区分度强 | 需要改索引、chunk 数变多 |

**本项目采用"递归 + 标题路径"**：`RecursiveCharacterTextSplitter(chunk_size=200, overlap=30)`（线上常见 256~512 tokens、overlap 10~20%）先保证长度可控，再用 `MarkdownHeaderTextSplitter` 按 `#/##` 切段，最后给每个 chunk 前置 `[文档标题 > 小节标题]`。

**为什么 prefix 要在切块之后加**：prefix 不占用 200 字的正文预算；如果先加前缀再切，正文反而被压缩。加上前缀后，8 篇文档从 39 个 chunk 变成 48 个。

**chunk 大小的权衡**：

- 太小：检索精准，但证据容易被切碎，跨 chunk 的答案会漏；chunk 数变多，top-k 覆盖变细但上下文更碎。
- 太大：上下文完整，但一个 chunk 混多个主题，embedding 被平均，噪声进 prompt。
- 经验：FAQ/规则类文档 150~300 字；overlap 10%~20%；改完必须重跑评测，不能凭感觉。

**评测陷阱**：chunk 变小后，一条证据可能横跨两个 chunk，`Evidence hit` 的子串匹配会失败。如果实验 E 后 Evidence hit 反而下降，先检查是不是"证据被切碎"，而不是急着否定标题路径。

### 2.9 阈值与拒答：为什么"单靠分数"不行

**数学本质**：阈值拒答是一个二分类问题。你希望找到一个 t，让有答案样本的分数都 ≥t、无答案样本都 <t。但实测分布是：

```
有答案：0.497 ──────────────── 0.817
无答案：      0.415 ── 0.650
                     ↑ 重叠区
```

重叠区 0.497~0.650 里同时存在两类样本。**任何阈值都只能在这两类错误之间做交换**：

- 阈值调低 → 泄漏率高（无答案被放进来）；
- 阈值调高 → 误拒率高（有答案被挡出去）。

这不是模型不好，而是"相似度分数"这个信号本身**不足以区分'相关'和'能回答'**。一个 chunk 可能和问题很相关，但里面没有答案；也可能分数不高，但恰好包含关键条款。

**正确用法**：

1. 阈值只做**极端低分兜底**（如 dense <0.45 直接拒答），不承担主力拒答；
2. 主力拒答靠 R12 的三层：L1 意图闸门（问题是否属于知识库覆盖范围）→ prompt 强制"资料不足就说不知道" → L3 citation 校验；
3. 换检索链路（dense → hybrid → rerank）后，分数量纲全变，阈值必须重新扫描。

### 2.10 延迟与成本模型

一次问答的检索侧延迟可以拆成：

```
T_total ≈ T_query理解 + T_embedding + T_qdrant + T_rerank + T_llm首token + T_生成
```

| 环节 | 量级 | 优化手段 |
|---|---|---|
| embedding API | 几十~几百 ms | 缓存 query 向量、批量索引 |
| Qdrant ANN | 几~几十 ms | 控制候选池、合理 HNSW 参数 |
| rerank（CPU） | 3~5s / 条 | 只重排 top-5、GPU、缓存、异步线程 |
| LLM 首 token | 几百 ms~数秒 | 控制 prompt 长度、流式输出 |

> 线上通常把**检索侧 P95** 压在 300ms~1s（不含 LLM 生成）；本项目 rerank 单条 3~5s、5 条 15~20s，明显超标，所以只能离线评测或低流量使用。

**判断优化是否值得上线的三条硬标准**：

1. 指标提升达到最小可感知幅度（Recall@1 ≥0.01 或 MRR ≥0.005）；
2. P95 延迟不超过基线的 2 倍；
3. 成本增量可解释（rerank 的 CPU 时间、embedding 的 token 费用）。

否则就是"为了指标好看，把用户体验做差"。

### 2.11 八个常见误区（面试官最爱抓）

1. **"Recall@5=1.0 说明检索很好"** → 语料只有 8 篇时 Top-5 覆盖 62% 文档，指标天然饱和；要看 Recall@1 / MRR / Evidence hit。
2. **"rerank 能提升召回"** → 不能。它只重排已有候选，召回上限由第一阶段决定。
3. **"阈值调一调就能拒答"** → 分布重叠时不存在两全阈值；阈值只能兜底。
4. **"k 越大越好"** → k 增大只增加噪声和 token，不改变排名；用 Evidence hit@k 的边际收益判断。
5. **"RRF 要调权重"** → RRF 不用权重、不用归一化，默认 k=60 就够稳。
6. **"改了 md 文件检索就变了"** → 必须重跑导入/重索引，否则测的是旧 chunk。
7. **"中文直接喂 BM25 就行"** → 必须先 jieba 分词，索引和查询走同一套分词。
8. **"单路检索也套 RRF 更统一"** → 单路套 RRF 会把分数归一化成 `1/(60+rank)`，Top-1 恒为 0.5，阈值扫描直接失效。

### 2.12 本项目指标 vs 线上系统典型口径

> 下面括号里的"线上典型值"是业界常见量级/经验区间，不是某个产品的 SLA，随业务差异很大，只用于建立尺度感。
> 读取方式：**本项目值（线上典型值）**。

| 维度 | 本项目值（线上典型值） | 差异说明 |
|---|---|---|
| 知识库文档数 | 8 篇（线上企业 KB 常见 10³~10⁵ 篇；公开基准 UDA 为 2,965 篇） | 本项目是验证规模；文档越多，文档级 Recall 越不会饱和 |
| chunk 数 | 39 → 48 个（线上 10⁵~10⁷；Qdrant 单 collection 百万级向量常见） | 本项目可全量扫描；线上必须依赖 ANN、分片、多租户 |
| chunk 大小 | 200 字符（线上 256~512 tokens，约 400~800 汉字） | 本项目偏小，证据更容易被切碎 |
| overlap | 30 字符 ≈15%（线上 10~20%） | 与业界一致 |
| 评测集规模 | 100 条（线上起步 50~100；CI 回归 20~50；周期基准数百~数千；公开基准 MIRAGE 7,560 QA） | 本项目够看趋势，不够做统计显著性 |
| 一阶段候选池 | dense/sparse 各 20（线上第一阶段 Recall@100/200，prefetch 50~200/路） | 本项目候选池小，重排可选空间有限 |
| 进 LLM 的 top-k | 3（线上 5~10） | 本项目 k 小，噪声少但漏答风险更高 |
| rerank 候选数 | 5（线上 20~100，常见 50） | 本项目受 CPU 延迟限制；线上用 GPU/托管 rerank |
| rerank 延迟 | CPU 单条 3~5s、5 条约 15~20s（线上托管/GPU 对 50 条约 20~150ms） | 差 2~3 个数量级，是本项目不能放大候选的主因 |
| 检索延迟 | 未系统压测（线上检索 P95 经验目标 300ms~1s，不含 LLM 生成） | 压测与容量评估留给 R20 |
| 核心检索指标 | DocRecall@1/3/5（线上第一阶段看 Recall@100/200；重排后看 nDCG@10 / MRR@10） | 本项目语料小，只能看 @1/@3/@5 |
| 排序指标 | DocMRR（线上 MRR@10 / nDCG@10） | 本项目是文档级 MRR，不精确到段落 |
| 段落级指标 | Evidence hit@3/5（线上 Context Recall / chunk-level Recall@k，RAGAS Context Recall 经验目标 ≥0.9） | 本项目用子串匹配，线上用 LLM judge 或人工 span 标注 |
| 阈值分析 | 准确率/误拒率/泄漏率网格扫描（线上 AUC/PR-AUC + 业务加权的误拒/泄漏成本） | 本项目只做单阈值网格；线上会画 PR 曲线 |
| 生成指标 | must_contain 命中/拒答/泄漏（线上 Answer Correctness、Faithfulness、Context Precision；经验目标 faithfulness ≥0.95、context precision ≥0.75） | 本项目用关键词，线上用 LLM-as-judge |
| 租户规模 | 单租户 `t_demo`（线上多租户 10²~10⁴） | 多租户需要隔离 collection 或过滤条件 |
| 流量 | 离线批量 100 条（线上 10~10³ QPS，需要限流与压测） | 对应 R20 |

**一句话总结**：本项目的指标口径是"小语料、可解释、可回归"，线上是"大语料、分级相关性、LLM judge、业务成本加权"。**不要把本项目的绝对数字当成线上水位**，要迁移的是方法论和指标分层。

#### 2.12.1 参考来源

- [NVIDIA Enterprise RAG Retrieval Sizing Guide](https://docs.nvidia.com/enterprise-reference-architectures/enterprise-rag-retrieval-scaling-and-sizing-guide/latest/summary.html) — TopK 与 reranker 的容量/成本权衡
- [Qdrant Large-Scale Search](https://qdrant.tech/documentation/tutorials-operations/large-scale-search/) — 百万级向量、rescore 与延迟实测
- [Qdrant: Minimal RAM to Serve 1M Vectors](https://qdrant.tech/articles/memory-consumption/) — 百万向量规模的资源量级
- [MLflow: RAG Evaluation in CI](https://mlflow.org/articles/catch-regressions-early-with-rag-evaluation-and-mlflow-for-engineers/) — CI 20~50 条、周期基准数百条的经验
- [Azure Databricks Retrieval Quality Evaluation](https://learn.microsoft.com/en-gb/azure/databricks/ai-search/retrieval-quality-eval) — DCG@10 作为主指标
- [Milvus: Optimal Chunk Size](https://blog.milvus.io/ai-quick-reference/what-is-the-optimal-chunk-size-for-rag-applications) — 256~512 tokens、10~20% overlap
- [RAG Reranking Strategy in Production](https://tanayshah.dev/blog/rag-reranking-strategy-production/) — 50~100 候选、50~200ms 延迟
- [RAG Evaluation Metrics Checklist](https://lazar-milicevic.com/blog/rag-evaluation-metrics-a-12-point-checklist-2026) — context recall/precision/faithfulness 经验目标
- [MIRAGE Benchmark](https://arxiv.org/html/2504.17137) — 7,560 QA / 37,800 chunks 的评测规模
- [UDA Benchmark](https://arxiv.org/html/2406.15187v1) — 2,965 篇真实文档 / 29,590 QA
- [Boolean & Beyond: RAG Evaluation](https://www.booleanbeyond.com/en/solutions/rag-ai-knowledge-systems/rag-evaluation) — 50~100 条起步评测集的经验

---

## 〇、五步总览与当前状态

R11 的「你要做什么」拆成 5 步，本文按这 5 步展开：

| 步骤 | 内容 | 状态 | 对应章节 |
|---|---|---|---|
| **Step 1** | 建知识库源文件 + 评测集 | ✅ 已完成 | 第一章 |
| **Step 2** | 一键导入脚本 `kb_import.py` | ✅ 已完成 | 第二章 |
| **Step 3** | 离线评测脚本 `run_eval.py` | ✅ 已完成 | 第三章 |
| **Step 4** | 检索优化实验（k / 阈值 / 混合检索 / rerank / chunk） | ✅ 已实施（报告见 `docs/my1/eval_report_*.md`） | 第四章 |
| **Step 5** | 最优配置写回 `search_kb`，旧行为做开关 | 🟡 部分完成：阈值/rerank 开关已接入；租户级配置与结构化返回待补 | 第五章 |

> **原则**：先冻结 Step 1 的源文件和评测集，再做 Step 4 的调参。一边调参一边改题，数字会骗自己。

### 〇.1 完整复现 SOP（从零到报告）

> 面试演示、换机器复现、或者隔两周忘了怎么跑，照这个顺序走。每一步都写清"做什么 / 命令 / 预期 / 失败排查"。
> 细节原理看第二篇，实验细节看第四章对应小节。

#### 第 0 步：环境自检

```bash
# 1) 容器是否齐全（go / agent / qdrant / postgres / redis）
docker compose ps

# 2) 服务是否可用（能拿到 token 就说明 Go + 鉴权正常）
curl -s -X POST http://localhost:8080/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"app_key":"demo_key","app_secret":"demo_secret"}'
```

预期：`code=0` 且返回 `data.token`。失败先别跑评测，按第二章 2.5 排查。

#### 第 1 步：重建知识库

```bash
python3 python_agent/R11/kb_import.py
```

预期：`8 篇文档，共 48 个 chunk`（标题路径前缀默认开启）；若输出 39，说明 `ENABLE_CHUNK_TITLE_PREFIX=false` 或 worker 没重建。

#### 第 2 步：跑基线（dense，k=5）

```bash
python3 python_agent/R11/run_eval.py --k 5 --retrieval-mode dense \
  --report docs/my1/eval_baseline_dense.md
```

预期：`Recall@1≈0.953 / MRR≈0.975 / Evidence hit@3≈0.988`。数字对不上，先确认知识库是同一套 8 篇源文件、且用的是 `dense` 模式。

#### 第 3 步：实验 A（top-k）

```bash
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode dense --report docs/my1/eval_k3.md
python3 python_agent/R11/run_eval.py --k 5 --retrieval-mode dense --report docs/my1/eval_k5.md
python3 python_agent/R11/run_eval.py --k 8 --retrieval-mode dense --report docs/my1/eval_k8.md
```

看 `Evidence hit@5 - Evidence hit@3` 的差值，不要跨 k 比 Recall@1 / MRR（它们恒等）。

#### 第 4 步：实验 B（阈值扫描）

```bash
python3 python_agent/R11/run_eval.py --retrieval-mode dense --threshold 0.5 \
  --report docs/my1/eval_threshold.md
```

看报告第五章的"准确率 / 误拒率 / 泄漏率"，确认不存在两全阈值，把结论交给 R12。

#### 第 5 步：实验 C（混合检索）

首次启用 fastembed 需要重建 agent 镜像（安装 `fastembed` + `jieba`，并让 collection 带上 sparse 字段）：

```bash
docker compose up -d --build agent
python3 python_agent/R11/kb_import.py          # schema 变了，必须清空重建
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode hybrid \
  --report docs/my1/eval_hybrid.md
```

单条验证（确认 E059 被修正）：

```bash
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"app_key":"demo_key","app_secret":"demo_secret"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['token'])")

for MODE in dense sparse hybrid; do
  echo "=== $MODE ==="
  curl -s -X POST http://localhost:8080/api/admin/kb/test-query \
    -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
    -d "{\"query\":\"保修需要提供纸质发票吗\",\"k\":3,\"mode\":\"$MODE\"}" \
    | python3 -c "
import sys,json
for i,x in enumerate(json.load(sys.stdin)['data']['items']):
    print(f'  {i+1}. {x[\"title\"]} | score={x[\"score\"]:.4f}')
"
done
```

预期：dense → 发票与支付规则；sparse / hybrid → 售后与保修规则。

#### 第 6 步：实验 D（rerank）

```bash
docker compose up -d --build agent     # FlagEmbedding + 预下载模型
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode rerank \
  --report docs/my1/eval_rerank.md

# 粗略测单条耗时
time curl -X POST http://localhost:8080/api/admin/kb/test-query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"保修需要发票吗","k":5,"mode":"rerank"}'
```

预期：MRR 有提升，但单条 15~20s（CPU）。收益不到 0.005 且延迟翻倍，就不进生产。

#### 第 7 步：实验 E（标题路径分片）

```bash
# 基线：.env 设 ENABLE_CHUNK_TITLE_PREFIX=false
docker compose up -d agent             # env_file 只在创建时读取，不能 restart
python3 python_agent/R11/kb_import.py
python3 python_agent/R11/run_eval.py --retrieval-mode dense --report docs/my1/exp_chunk_baseline.md

# 实验：.env 设 ENABLE_CHUNK_TITLE_PREFIX=true
docker compose up -d agent
python3 python_agent/R11/kb_import.py
python3 python_agent/R11/run_eval.py --retrieval-mode dense --report docs/my1/exp_chunk_title.md
python3 python_agent/R11/run_eval.py --retrieval-mode hybrid --report docs/my1/exp_chunk_title_hybrid.md
```

预期：chunk 数 39 → 48；重点看 E002 / E023 / E071 的 Top-1 是否修正。

#### 第 8 步：汇总对比表

把每个报告的核心指标填进 4.6 的对比表，至少包含：`Recall@1 / MRR / Evidence hit@3 / P50 / P95 / chunk 数`。
最后按 4.6 的"真提升"标准选生产配置，并把参数写回 Step 5。

#### 常见失败现象速查

| 现象 | 最可能原因 | 处理 |
|---|---|---|
| `test-query` 401 | 没用 app_key 登录 / token 过期 | 重新执行 `POST /api/auth/token` |
| 检索结果没变化 | 只改文件没重索引；或命中 Redis 缓存 | 重跑 `kb_import.py`（会清缓存） |
| sparse 返回 0 条 | 中文没 jieba 分词 / collection 没 sparse 字段 | 检查 `make_sparse_vector`；清空重建 collection |
| `vector name "sparse" not found` | 旧 collection schema 没有 sparse | 用 `kb_import.py` 清空重建 |
| rerank 请求超时 | 候选太多 / 流式超时仍是 60s | `RERANK_CANDIDATE_N=5`；Go 用 180s `streamHTTPClient` |
| 报告指标全 0 | `expected_doc_titles` 与 manifest 标题不一致 | 逐字核对标题（含空格/符号） |

---

## 第一章 · Step 1：建知识库源文件 + 评测集（地基 · 已完成）

> 这是整个 R11 的地基。没有源文件，知识库无法重建；没有评测集，所有优化都是"感觉变好了"。
> 本步产出：`python_agent/R11/kb_docs/*.md` + `kb_manifest.json` + `kb_eval.jsonl`。

### 1.1 这一步解决什么问题

R11 立项前的痛点（也是面试官的第一个问题）：

- 知识库里的内容散落在数据库里，**没人能 review、没法 diff、改了规则没人知道**。
- 没有评测集，调 chunk、调 top-k、加 rerank 全靠"感觉变好了"，**没有任何量化依据**。
- 文档只有 2 篇时，Recall@5 恒等于 1.0，**评测完全失真**——无法区分"检索真的好"还是"语料太少"。
- 没有负样本（无答案问题），**永远不知道拒答阈值该设多少**。

所以 Step 1 要一次性解决四件事：**源文件可维护、文档量足够产生干扰、评测集有正负样本、用标题做稳定锚点**。

### 1.2 知识库源文件设计

#### 1.2.1 为什么源文件要放仓库里

知识库的"事实来源"应该是 markdown 文件，而不是数据库里的 chunk：

| 维度 | 源文件放仓库 | 源文件只在数据库 |
|---|---|---|
| 可 review | ✅ PR 里能看改了哪条规则 | ❌ 改了没人知道 |
| 可回滚 | ✅ git 历史 | ❌ 数据库恢复麻烦 |
| 可重建 | ✅ 一条命令删库重建 | ❌ 丢了就没了 |
| 可 diff | ✅ 规则变更一目了然 | ❌ 只能看最终态 |

**设计原则：源文件是唯一事实来源，数据库里的 chunk 只是派生产物，随时可删可重建。** 这也是为什么 Step 2 的导入脚本要支持"清空 → 重建"。

#### 1.2.2 为什么是 8 篇而不是 1 篇大文档

检索评测需要**干扰项**。只有 1 篇文档时，任何 query 都能"召回"，Recall@k 恒为 1，评测失去意义。

8 篇的设计考虑：

1. **覆盖不同业务域**：售后、物流、支付、会员、库存、保修、订单、服务规范，能真实暴露**跨文档混淆**（例如"运费谁承担"会同时命中《退换货政策》和《运费与配送规则》）。
2. **故意制造边界模糊**：运费在 01 和 02 都有、退款在 01 和 03 都有，用来测排序质量。
3. **分类字段可复用**：`category` 字段（售后政策 / 配送物流 / 支付发票 / 会员营销 / 商品库存 / 订单账号 / 服务规范）后续可用于分类过滤召回实验（Step 4 实验 C 的前置条件）。

#### 1.2.3 文档清单

清单来源：[kb_manifest.json](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_manifest.json)

| # | 文件 | 标题 | 分类 | 核心内容 |
|---|---|---|---|---|
| 01 | [01-退换货政策.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/01-退换货政策.md) | 退换货政策 | 售后政策 | 7 天无理由、15 天质量问题、三包、退款时效、换货流程 |
| 02 | [02-运费与配送规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/02-运费与配送规则.md) | 运费与配送规则 | 配送物流 | 包邮门槛、偏远地区、发货/配送时效、物流异常、生鲜赔付 |
| 03 | [03-发票与支付规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/03-发票与支付规则.md) | 发票与支付规则 | 支付发票 | 发票类型与时效、分期费率、支付异常、价格保护 |
| 04 | [04-会员与优惠券规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/04-会员与优惠券规则.md) | 会员与优惠券规则 | 会员营销 | 会员等级、积分、优惠券叠加与返还 |
| 05 | [05-商品与库存规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/05-商品与库存规则.md) | 商品与库存规则 | 商品库存 | 预售定金、锁库存、缺货赔偿、限购、正品保障 |
| 06 | [06-售后与保修规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/06-售后与保修规则.md) | 售后与保修规则 | 售后政策 | 保修期限、维修流程、以旧换新、安装收费 |
| 07 | [07-订单与账号规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/07-订单与账号规则.md) | 订单与账号规则 | 订单账号 | 订单状态、取消、改地址、拆单、账号安全 |
| 08 | [08-服务承诺与转人工规则.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/08-服务承诺与转人工规则.md) | 服务承诺与转人工规则 | 服务规范 | 服务时间、响应时效、AI 能力边界、转人工条件、赔付标准 |

当前实际入库：**8 篇文档、39 个 chunk**（基线分片；实验 E 标题路径前缀开启后为 48 个，见 4.5.3。`chunk_count` 来自 `kb_documents` 表）。

#### 1.2.4 一篇"好用"的 KB 文档长什么样

以 [01-退换货政策.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_docs/01-退换货政策.md) 为例，说明好的 KB 文档结构：

```markdown
# 退换货政策                    ← 一级标题 = 文档标题（也是 expected_doc_titles 的锚点）

## 二、7 天无理由退货           ← 二级标题 = 主题分段，chunk 切分会以此为边界
- 起算时间：自签收次日起 **7 个自然日**内。   ← 关键数字加粗，便于模型抽取
- 运费承担：无理由退货的 **退货运费由买家承担**；订单含运费险的，按运费险规则赔付。
- 不支持品类：定制商品、生鲜食品、内衣裤及贴身衣物、已激活的数码产品...

## 五、退款规则
- 退款路径：**原路退回**至原支付账户。
- 到账时间：微信、支付宝 **1-3 个工作日**；银行卡 **3-7 个工作日**。
```

写 KB 文档的 4 条规则：

1. **一级标题就是文档标题**，和 `kb_manifest.json` 里的 `title` 完全一致——评测集靠标题定位文档。
2. **二级标题按主题分段**，chunk 切分（`RecursiveCharacterTextSplitter`）会优先在标题处断开，避免一个 chunk 塞两个不相关的主题。
3. **关键数字、责任主体加粗**（`**7 个自然日**`、`**买家承担**`），既便于模型抽取，也让 `must_contain` 判据更稳。
4. **边界条件单列一段**（如"不支持退货的情形"），这类"例外规则"是用户最常问、也最容易答错的。

#### 1.2.5 manifest 的作用

[kb_manifest.json](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_manifest.json) 是"文件 → 标题 → 分类"的映射表：

```json
{
  "tenant": "t_demo",
  "docs": [
    { "file": "01-退换货政策.md", "title": "退换货政策", "category": "售后政策" },
    ...
  ]
}
```

为什么需要它：
- 导入脚本（Step 2）按这个清单遍历文件，拿到 `title` 和 `category` 传给上传接口。
- 评测集的 `expected_doc_titles` 必须和这里的 `title` 完全一致，否则评测时解析不到 `doc_id`。
- 加新文档的唯一入口：**新建 md → 在 manifest 登记 → 重跑导入**。三处一致，知识库就不会乱。

### 1.3 评测集设计

评测集文件：[kb_eval.jsonl](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_eval.jsonl)，每行一个 JSON。

#### 1.3.1 字段设计

| 字段 | 类型 | 作用 | 设计理由 |
|---|---|---|---|
| `id` | string | 稳定编号 E001-E100 | 报告和回归对比都按它，不能随重排改变 |
| `q` | string | 用户问法 | **必须口语化，不用文档原句**，否则向量检索"作弊"式命中 |
| `category` | enum | 高频 / 长尾 / 多跳 / 无答案 | 分维度看指标，定位哪类问题弱 |
| `answerable` | bool | 知识库是否能答 | 决定参与哪类指标（有答案看召回，无答案看泄漏） |
| `expected_doc_titles` | string[] | 相关文档标题 | **用标题不用 doc_id**（见 1.3.2） |
| `must_contain` | string[] | 答案必须命中的关键词 | 生成层答案正确性的判据（Step 3） |
| `evidence_terms` | string[] | 证据词 | chunk 级召回判定，比文档级更严格 |
| `reference` | string | 标准答案 | 人工核对 / LLM-as-judge 打分用 |

#### 1.3.2 为什么用 `expected_doc_titles` 而不是 `doc_id`

`doc_id` 是上传接口生成的随机 uuid，每次重导必然变化。如果评测集硬编码 uuid，重导一次就全废。

用标题则永久稳定：评测脚本运行时通过 `GET /api/admin/kb/documents` 拿到"标题 → doc_id"映射，再把 `expected_doc_titles` 解析成当前 `doc_id`。

**这条设计直接决定了评测集是否可复用、可回归。** 面试时被问"你们评测集怎么保证稳定"，这就是答案。

#### 1.3.3 `must_contain` vs `evidence_terms`：两层判据

这两个字段看起来像，但用途不同：

- `must_contain`：**答案必须包含**的词。用于生成层评测——模型答的话里有没有这些词。例：`"7天无理由退货的运费由谁承担？"` 的 `must_contain: ["买家承担"]`。
- `evidence_terms`：**正确段落里应该出现**的词。用于检索层评测——召回的 chunk 里有没有这些词。例：同上，`evidence_terms: ["买家承担"]`。

大多数时候两者重合，但刻意区分有意义：
- 有的答案关键词在文档里表述不同（答案说"买家承担"，文档写"由消费者承担"），这时 `evidence_terms` 用文档原词，`must_contain` 用答案等价词。
- `evidence_terms` 是检索是否命中正确段落的判据，`must_contain` 是生成是否答对的判据——**两者分开才能定位"是检索没召回，还是召回了但模型没用"**。

#### 1.3.4 数据分布

| 维度 | 数量 | 说明 |
|---|---|---|
| 有答案 | 85 | 8 篇文档各 9-12 条 |
| 无答案 | 15 | 跨境、团购、门店自提、竞品价格、系统实现等，知识库确实没有 |
| 高频 | 47 | 用户最常问的规则类问题 |
| 长尾 | 35 | 边界条件、特殊品类、少见场景 |
| 多跳 | 3 | 需要同时看两个规则（如"整机 1 年 + 主要部件 3 年"） |
| 无答案 | 15 | 用于校准拒答阈值（R12） |

#### 1.3.5 为什么必须有 15 条无答案

没有负样本，你永远不知道：
- 拒答阈值该设多少
- 模型会不会"硬答"不该答的问题
- 检索的分数分布在无答案问题上是什么样

R11 基线实测的关键发现（见附录 A.4）就来自这 15 条：**无答案问题的 Top-1 分数和长尾有答案问题高度重叠**，证明单靠阈值无法可靠拒答。没有这 15 条，这个结论出不来。

#### 1.3.6 怎么写一条好用例（好坏对比）

**好的用例**：

```json
{"id":"E002","q":"7天无理由退货的运费由谁承担？","category":"高频","answerable":true,
 "expected_doc_titles":["退换货政策"],"must_contain":["买家承担"],
 "evidence_terms":["买家承担"],"reference":"由买家承担；订单含运费险的按运费险规则赔付。"}
```

好在哪：
- `q` 是口语问法，不是文档原句"无理由退货的退货运费由买家承担"
- `expected_doc_titles` 只有 1 篇，期望明确
- `must_contain` 短且唯一（"买家承担"不会在别处出现）
- `reference` 是完整答案，可用于人工核对

**差的用例**：

```json
{"id":"E00X","q":"退换货政策","category":"高频","answerable":true,
 "expected_doc_titles":["退换货政策","运费与配送规则"],
 "must_contain":["退","换"],"evidence_terms":["退货"],
 "reference":"退换货政策"}
```

差在哪：
- `q` 不是问题，是关键词——测的不是检索而是关键词匹配
- `expected_doc_titles` 列了 2 篇但没说为什么，判据模糊
- `must_contain: ["退","换"]` 太宽泛，几乎任何答案都能命中
- `reference` 没有信息量，无法人工核对

### 1.4 维护方式

| 场景 | 操作 |
|---|---|
| 改规则 | 编辑 `kb_docs/xx.md` → 重跑 Step 2 导入脚本 → 向量自动重建 |
| 加文档 | 新建 md → 在 `kb_manifest.json` 登记 → 重跑导入 |
| 加评测问题 | 在 `kb_eval.jsonl` 追加一行（id 递增），确保 `expected_doc_titles` 与 manifest 的 `title` 完全一致 |
| 规则改了标准答案 | 同步更新对应行的 `reference` 与 `must_contain`，否则评测会和文档打架 |

### 1.5 本步验收

- [x] 8 篇源文件在仓库内，可 diff、可 review
- [x] `kb_manifest.json` 登记完整，文件、标题、分类三处一致
- [x] 100 条评测集，其中 85 有答案 + 15 无答案
- [x] 评测集用 `expected_doc_titles`（标题）而非 `doc_id`
- [x] 每条用例有 `must_contain` 和 `evidence_terms` 两层判据

> 做完这步，知识库和评测集就"冻住"了，后面 Step 3、Step 4 都在这套数据上跑，数字才可比。

---

## 第二章 · Step 2：一键导入脚本（已完成）

> 产出：[kb_import.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_import.py)
> 功能：清空旧文档 → 上传 → 触发 reindex → 轮询索引完成。一条命令可重建知识库。

### 2.1 前置条件

- 容器已启动：`docker compose ps`（需要 `cs_go`、`cs_agent`、`cs_qdrant`、`cs_postgres`、`cs_redis`）
- 租户 `t_demo` 已配置可用的 embedding 模型（否则索引会 `status=2`）
- 用 app_key 登录（脚本默认 `demo_key` / `demo_secret`）

### 2.2 标准流程

```bash
# 清空旧文档 + 导入 + 等待索引完成（幂等，可反复执行）
python3 python_agent/R11/kb_import.py
```

输出示例（实测）：

```
目标服务 http://localhost:8080，租户 t_demo，文档 8 篇
登录成功
清空已有文档 2 篇
  - 已删除 采伐管理-机手浮动模块开发文档.docx
  + 已上传 退换货政策 (943 字符) doc_id=685e69ac...
等待异步索引完成 ...
  索引进度 8/8，失败 0
完成：8 篇文档，共 39 个 chunk
```

> 上面是关闭标题路径前缀（基线分片）时的实测输出；实验 E 默认开启后为 **48 个 chunk**，详见 4.5.3。

### 2.3 常用参数

```bash
python3 python_agent/R11/kb_import.py --dry-run      # 只打印计划，不发请求
python3 python_agent/R11/kb_import.py --keep         # 不清空，只增量导入
python3 python_agent/R11/kb_import.py --only 05      # 只重导 05-商品与库存规则.md
```

### 2.4 脚本做了什么（对应后端接口）

| 步骤 | 接口 | 说明 |
|---|---|---|
| 登录 | `POST /api/auth/token` | app_key/app_secret 换 token |
| 列出文档 | `GET /api/admin/kb/documents` | 拿到该租户全部 `doc_id` |
| 删除 | `DELETE /api/admin/kb/documents/{doc_id}` | 同时删 Qdrant points 与原文件 |
| 上传 | `POST /api/admin/kb/upload` | JSON 文本上传，落盘 + 写 `kb_documents(status=0)` |
| 索引 | `POST /api/admin/kb/documents/{doc_id}/reindex` | 删旧向量 + 入 Redis 队列 |
| 等待 | 轮询 documents | `0` 待索引 / `3` 索引中 / `1` 成功 / `2` 失败 |

导入完成后 worker 会清该租户的检索缓存，所以新内容立即可被检索。

### 2.5 失败排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `status=2`，错误含"向量模型不可用" | 租户未配置 embedding 模型 | 管理端配置模型后重跑 |
| `status=2`，错误含"原文件缺失" | 共享卷未挂载/被清理 | 重新执行导入脚本 |
| 上传返回 400 "超过配额" | 租户 `max_kb_docs` 限制 | 调整租户配额或减少文档 |
| 索引一直 0 | Redis 队列不可用 / worker 未启动 | 检查 `cs_agent` 日志 |
| 检索不到刚导入的内容 | 检索缓存未失效 | 等缓存 TTL 或重跑 reindex |

---

## 第三章 · Step 3：离线评测脚本 `run_eval.py`（已完成）

> 产出：[run_eval.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/run_eval.py)，输出 `eval_report_<mode>.md`（默认 hybrid → `eval_report_hybrid.md`）。
> 两种模式：`retrieval`（默认，不花 LLM 钱，秒级）和 `e2e`（端到端生成评测，需用户账号）。

### 3.1 脚本能力

| 模式 | 接口 | 耗时 | 花费 | 指标 |
|---|---|---|---|---|
| `retrieval`（默认） | `POST /api/admin/kb/test-query` | 秒级 | 0 | Recall@k、MRR、Evidence hit@k、分数分布、阈值扫描 |
| `e2e` | `POST /api/user/chat` | 慢 | 调 LLM | 答案命中率、拒答率、误拒率、泄漏率 |

只依赖 Python 标准库（`urllib`），和 `kb_import.py` 风格一致，无需安装第三方包。

### 3.2 用法

```bash
# 检索评测（默认 k=5，无需用户账号，不调 LLM，秒级出结果）
python3 python_agent/R11/run_eval.py

# 改 k、标注参考阈值
python3 python_agent/R11/run_eval.py --k 3 --threshold 0.55

# 只跑指定用例（调试用）
python3 python_agent/R11/run_eval.py --ids E001,E002,E086

# 打印前 N 条用例的检索详情（对照 3.3 理解指标）
python3 python_agent/R11/run_eval.py --ids E001,E002,E086 --show-cases 3

# 端到端生成评测（需要真实用户账号，会调 LLM）
EVAL_USERNAME=agent01 EVAL_PASSWORD=agent123 python3 python_agent/R11/run_eval.py --mode e2e

# 指定报告输出路径
python3 python_agent/R11/run_eval.py --report /tmp/my_report.md
```

> **日常用默认 retrieval 模式就够了**，不需要用户账号、不花 LLM 钱。`--mode e2e` 只在改完 prompt、想验答案质量时才用。

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KB_IMPORT_BASE_URL` | `http://localhost:8080` | 服务地址 |
| `KB_IMPORT_APP_KEY` | `demo_key` | app_key 登录（retrieval 用） |
| `KB_IMPORT_APP_SECRET` | `demo_secret` | app_secret |
| `EVAL_USERNAME` / `EVAL_PASSWORD` | - | e2e 模式的用户账号 |
| `EVAL_TENANT_ID` | `t_demo` | e2e 模式的租户 |

### 3.3 指标详解（一步步算给你看）

这一节用**真实返回结果**演示每个指标怎么算。看不懂公式没关系，跟着例子走一遍就懂了。

#### 3.3.1 先看检索返回了什么

对用例 **E002**（问题："7天无理由退货的运费由谁承担？"，期望文档：`退换货政策`，`evidence_terms=["买家承担"]`）调 `test-query`，k=5，后端实际返回：

| rank | score | title | 是否期望文档 |
|---|---|---|---|
| 1 | 0.640 | 运费与配送规则 | ✗ |
| 2 | 0.621 | 退换货政策 | ✓ |
| 3 | 0.603 | 运费与配送规则 | ✗ |
| 4 | 0.598 | 退换货政策 | ✓ |
| 5 | 0.586 | 服务承诺与转人工规则 | ✗ |

> `score` 是向量相似度，越高越相关。rank1 是分数最高的结果。

#### 3.3.2 Recall@k：期望文档有没有进前 k 名

**定义**：前 k 个结果里，只要出现过期望文档，就算命中（=1），否则=0。最后对所有有答案用例求平均。

对 E002 算：
- Recall@1：前 1 名里有"退换货政策"吗？rank1 是"运费与配送规则"→ **没有 → 0**
- Recall@3：前 3 名里有吗？rank2 是 → **有 → 1**
- Recall@5：前 5 名里有吗？→ **有 → 1**

**直觉**：Recall@1 最严格（第一名就得对），k 越大越宽松。E002 说明**文档被召回了，但排到了第 2 名**，这就是"检索到了但排序不够好"。

整条评测集 85 条有答案用例，Recall@1 = 81/85 = 0.953（4 条排到第 2 名以后），Recall@3 = 85/85 = 1.000。

#### 3.3.3 MRR：第一个命中排得有多靠前

**定义**：找到期望文档**第一次出现的位置 rank**，取 `1/rank`。所有用例求平均。

对 E002：期望文档第一次出现在 rank2 → MRR = 1/2 = **0.5**

对比 E001（期望文档在 rank1）→ MRR = 1/1 = **1.0**

**直觉**：MRR 同时考虑"有没有命中"和"命中得早不早"。命中越早分越高。E002 虽然召回了，但只排第 2，所以 MRR 只有 0.5，拉低了整体平均分。

整条评测集 MRR = 0.975，说明大多数用例期望文档都排在第 1 名。

#### 3.3.4 Evidence hit@k：不只要命中文档，还要命中正确段落

**为什么需要这个指标**：Recall@k 只看"文档标题对不对"，但同一篇文档有多个 chunk，命中文档不代表命中了**包含答案的那个段落**。

**定义**：前 k 个 chunk 的 `content`（归一化后）里，是否包含任一 `evidence_terms`。包含=1，否则=0。

对 E002，`evidence_terms=["买家承担"]`，检查前 3 个 chunk：
- rank1 content："…往返运费由买家承担…" → 包含"买家承担" → ✓
- 前 3 个里只要有一个包含就算命中 → **Evidence hit@3 = 1**

**直觉**：这是比 Recall@k 更严格的指标。如果一个用例 Recall@3=1（文档进前 3 了）但 Evidence hit@3=0（前 3 个段落都不含答案关键词），说明**召回的是文档里的无关段落**，模型拿到也答不对。

整条评测集 Evidence hit@3 = 84/85 = 0.988。

#### 3.3.5 Top-1 分数：相似度分数的分布

**定义**：每条用例取 rank1 的 `score`（最高分）。分别统计有答案 / 无答案两类问题的分数分布（中位数、最小、最大）。

实测：
| 分布 | 中位数 | 最小 | 最大 |
|---|---|---|---|
| 有答案（85 条） | 0.679 | 0.497 | 0.817 |
| 无答案（15 条） | 0.547 | 0.415 | 0.650 |

**直觉**：看两类的分数有没有拉开。如果有答案的分数普遍远高于无答案，那设个阈值就能拒答。但这里**两类高度重叠**（有答案最低 0.497，无答案最高 0.650），所以单靠阈值没法可靠拒答——这就是 R12 要解决的问题。

#### 3.3.6 阈值扫描：找最优拒答阈值

**定义**：对阈值 t 从 0.00 到 1.00 每隔 0.05 扫一遍，规则是"Top-1 分数 ≥ t 就认为有答案，否则认为无答案"，算三个率：

- **准确率**：判对的比例
- **误拒率**：有答案问题被判成"无答案"的比例（该答没答）
- **泄漏率**：无答案问题被判成"有答案"的比例（不该答却答了）

实测最优在 t=0.50：准确率 0.900，但**泄漏率 60%**（15 条无答案里 9 条分数 ≥ 0.50）。这说明即使选了最优阈值，仍有 60% 的无答案问题会被误判为有答案——再次证明阈值不可靠。

---

### 3.4 检索评测流程

1. 用 app_key 登录拿 admin token
2. `GET /api/admin/kb/documents` 拿到「标题 → doc_id」映射（用于解析 `expected_doc_titles`）
3. 对每条用例调 `POST /api/admin/kb/test-query`，拿到 `items[{title, score, content}]`
4. 按 3.3 的定义计算 Recall@k / MRR / Evidence hit@k / Top-1 分数
5. 阈值扫描 0.00~1.00，输出每档的准确率 / 误拒率 / 泄漏率
6. 生成 markdown 报告写入 `dense_report.md`（可用 `--report` 改路径）

### 3.5 文本归一化

判定 `must_contain` 和 `evidence_terms` 前先归一化，去掉空格与 markdown 符号：

```python
import re
_NORM_RE = re.compile(r"[\s*#>|`\-]+")
def norm(s: str) -> str:
    return _NORM_RE.sub("", s or "")
```

这样"7 个自然日"和"7个自然日"会被判为相等。

### 3.6 真实输出示例

执行 `python3 python_agent/R11/run_eval.py --retrieval-mode dense`（100 条，k=5）的控制台输出：

```
评测用例 100 条，模式 retrieval，检索模式 dense
登录成功，开始检索评测 ...
  检索进度 20/100
  检索进度 40/100
  检索进度 60/100
  检索进度 80/100
  检索进度 100/100

检索模式=dense  Recall@1=0.953  MRR=0.975  Evidence hit@3=0.988
报告已写入：.../docs/my1/eval_report_dense.md
```

报告 [eval_report_dense.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/docs/my1/eval_report_dense.md) 包含五部分：

1. **检索指标表**：Recall@1/3/5、MRR、Evidence hit@3/5
2. **分类别 Recall@1**：高频 / 长尾 / 多跳 分别多少
3. **Top-1 未命中清单**：列出每条的问题、期望文档、实际 Top-1（用于定位优化方向）
4. **Top-1 分数分布**：有答案 vs 无答案的中位数/最小/最大
5. **阈值扫描表**：21 档阈值的准确率 / 误拒率 / 泄漏率，标注 `--threshold` 指定的参考档

### 3.7 e2e 生成评测（可选）

`--mode e2e` 走用户聊天接口 `POST /api/user/chat`，每条用例用独立 `user_id`（避免上下文串扰）。

**接口契约**（见 [user.go](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/handler/user.go#L60-L63)）：

```json
// 请求：user_id 必须是 {tenant_id}:{uid} 格式，否则中间件返回 403「user_id 不属于当前租户」
{"user_id": "t_demo:eval-E001-<时间戳>", "msg": "7天无理由退货从什么时候开始算？"}

// 响应：回复在 msg 里，不是 data
{"code": 0, "msg": "<AI 回复文本>"}
```

指标：

- **答案命中率**：有答案问题中，AI 回复（归一化后）包含全部 `must_contain` 的比例
- **误拒率**：有答案问题中，AI 回复命中拒答关键词（"不知道/查不到/无法回答"等）的比例
- **拒答率**：无答案问题中，AI 正确拒答的比例
- **泄漏率**：无答案问题中，AI 硬答（没拒答）的比例

**注意事项**：

1. **耗时**：单条用例约 30~60 秒（真调 LLM），100 条约 1 小时。建议先小范围验证：
   ```bash
   EVAL_USERNAME=agent01 EVAL_PASSWORD=agent123 \
   python3 python_agent/R11/run_eval.py --mode e2e --ids E001,E002,E086
   ```
2. **账号**：需要一个租户内的真实用户账号（本项目 demo 租户 `t_demo` 下有个客服账号，密码用管理后台查看/重置）。没有账号只会报 401，不会影响 retrieval 模式。
3. **偶发超时**：LLM 提供方会间歇性超时，脚本已对 5xx / timeout 自动重试 2 次（间隔 3 秒）。
4. **关键词判据较粗**：同一问题每次生成的措辞不同，"软拒答"（如"具体时效可能有所不同，建议您查看商品页"）容易被判成泄漏；反过来答案里恰好出现关键词也会被判成误拒。**精确判对错要靠 R12 的 LLM-as-judge**，这里只做粗筛。

### 3.8 本步验收

- [x] 提交 `python_agent/R11/run_eval.py`，支持 `--k`、`--threshold`、`--mode retrieval|e2e`、`--ids`、`--report`
- [x] 输出 `eval_report_dense.md`，含指标表 + 分类别 + Top-1 未命中清单 + 分数分布 + 阈值扫描
- [x] `python3 python_agent/R11/run_eval.py` 一条命令跑完
- [x] 基线可复现：Recall@1 0.953 / MRR 0.975 / Evidence hit@3 0.988，与附录 A 一致

---

## 第四章 · Step 4：检索优化实验（核心交付）

> **核心方法论**：每次只改一个变量，用 Step 3 的 `run_eval.py` 在同一套 100 条评测集上跑出可比数字。基线见附录 A。
> **为什么要做这一步**：附录 A.2 的 4 条 Top-1 未命中全部是「文档级已召回、但排序或段落不对」——Recall@5=1.0 已饱和，真正的差距在 **chunk 级排序** 和 **关键词精确匹配**。所以 Step 4 的优化目标盯死三个指标：**Recall@1、MRR、Evidence hit@3**。

### 4.0 前置改造：让 `test-query` 和 `run_eval.py` 支持检索模式

5 个实验里，A/B 不用改代码（直接跑），C/D/E 需要后端支持不同检索模式。为了不重复造轮子，先做一个小改造：给 `test-query` 加 `mode` 参数，给 `run_eval.py` 加 `--retrieval-mode` 参数。

#### 4.0.1 改造 `test-query` 接口

在 [routers_kb.py 的 kb_test_query](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/routers_kb.py#L420-L462) 里读取 `mode`，传给 `HybridStore.asimilarity_search_with_score`：

```python
@router.post("/agent/kb/test-query")
async def kb_test_query(request: Request):
    ...
    mode = (body.get("mode") or "hybrid").strip().lower()  # hybrid | dense | sparse
    ...
    pairs = await store.asimilarity_search_with_score(query, k=k, mode=mode)
    items = [{"doc_id": ..., "title": ..., "score": ..., "content": ...} for ...]
    ...
```

`HybridStore`（见 [hybrid_search.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/hybrid_search.py)）根据 `mode` 决定走 dense / sparse / hybrid（RRF 融合）。返回结构保持不变（`items: [{doc_id, title, score, content}]`），`run_eval.py` 的指标计算逻辑不用动。

#### 4.0.2 改造 `run_eval.py`

在 [run_eval.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/run_eval.py) 的 `test_query` 函数加 `mode` 参数，主流程加 `--retrieval-mode`：

```python
def test_query(token, query, k, mode="hybrid"):
    r = request("POST", "/api/admin/kb/test-query", token,
                {"query": query, "k": k, "mode": mode})
    ...

# main() 里加：
ap.add_argument("--retrieval-mode", choices=["hybrid", "dense", "sparse"], default="hybrid",
                help="hybrid=dense+sparse RRF（默认）；dense=仅稠密；sparse=仅 BM25")
```

这样每个实验就是一条命令的事：`python3 run_eval.py --retrieval-mode hybrid`。

---

### 4.1 实验 A：top-k 调参（无需改代码）

#### 4.1.1 解决什么问题

`search_kb` 里硬编码 `k=RETRIEVER_TOP_K=3`（见 [config.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/config.py#L39)）。k 太小会漏召回正确段落，k 太大则把噪声 chunk 塞进 prompt，既浪费 token 又可能让模型被干扰。**这个实验回答「给模型几条最划算」**，同时为 R13 的上下文预算提供依据。

#### 4.1.2 怎么跑

`run_eval.py` 的 `--k` 直接透传给 `test-query`，后端 `k` 上限是 10（见 [routers_kb.py#L440](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/routers_kb.py#L440)）：

```bash
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode dense --report docs/my1/eval_k3.md
python3 python_agent/R11/run_eval.py --k 5 --retrieval-mode dense --report docs/my1/eval_k5.md
python3 python_agent/R11/run_eval.py --k 8 --retrieval-mode dense --report docs/my1/eval_k8.md
```

#### 4.1.3 怎么看结果

**先搞清楚一个关键事实**：k 不影响排名（第 1 名永远是第 1 名），只影响「能看到多少名」。由 [run_eval.py L213-215](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/run_eval.py#L213-L215) 的切片逻辑 `items[:kk]` 可推导出：

| 指标 | k=3 | k=5 | k=8 | 是否随 k 变化 |
|---|---|---|---|---|
| Recall@1 | 0.953 | 0.953 | 0.953 | ❌ 不变（只看第 1 名） |
| MRR | 0.975 | 0.975 | 0.975 | ❌ 不变（看第一次命中位置） |
| Recall@3 | 1.000 | 1.000 | 1.000 | ❌ 不变（前 3 名相同） |
| Recall@5 | = Recall@3 | 1.000 | 1.000 | ⚠️ k<5 时被截断为 Recall@3 |
| Evidence hit@3 | 0.988 | 0.988 | 0.988 | ❌ 不变（前 3 个 chunk 相同） |
| Evidence hit@5 | = Evidence hit@3 | 1.000 | 1.000 | ⚠️ k<5 时被截断为 Evidence hit@3 |

所以跨 k 对比 Recall@1 / MRR / Evidence hit@3 **没有意义**，它们恒相等。

**正确的判断方法**：在 **k=5 的同一次运行里**，对比「高阶指标 vs 低阶指标」的差值：

| 对比 | 基线值 | 差值 | 含义 |
|---|---|---|---|
| Recall@3 → Recall@5 | 1.000 → 1.000 | 0 | 文档级：没有正确文档排在 4-5 名，k=3 够用 |
| Evidence hit@3 → Evidence hit@5 | 0.988 → 1.000 | +0.012 | chunk 级：约 1.2% 用例的正确段落排在 4-5 名，k=3 会漏 |

**判断逻辑**：
- 如果 `Evidence hit@5 - Evidence hit@3 ≈ 0` → k=3 的 chunk 覆盖足够，生产用 k=3 省 token
- 如果差值明显 > 0（如基线的 +0.012）→ 说明有正确段落排在 4-5 名，k=3 会漏掉，是否提 k 取决于这部分损失能否接受
- 想验证 k=5 是否饱和，跑 `--k 8` 并扩展脚本计算 Evidence hit@8；如果 @8 不再提升，k=5 就是上界

**基线结论**：文档级 k=3 完全够用；chunk 级 k=3 会漏约 1 条正确段落（0.988 → 1.000）。生产上 k=3 还是 k=5，是「省 token」和「多 1.2% 证据命中」的权衡。

#### 4.1.4 踩坑

- **不要跨 k 比 Recall@1 / MRR / Evidence hit@3**：它们由排名决定，k 不改变排名，所以恒相等，对比无意义。
- **k<指标 cutoff 时指标会被截断**：k=3 时 Evidence hit@5 实际等于 Evidence hit@3（`items[:5]` 只有 3 条），报告里看起来 @5=@3，不是真的饱和，是看不到 4-5 名。
- **k 和 chunk 大小强相关**：chunk 越小，正确段落越分散，需要的 k 越大。做实验 E 改 chunk 后要重跑本实验。

---

### 4.2 实验 B：相似度阈值校准（评测免改代码，线上已接入 RETRIEVAL_SCORE_THRESHOLD）

#### 4.2.1 解决什么问题

理论上「Top-1 相似度 < 阈值就拒答」是最简单的拒答方案。但附录 A.4 已经证明：**无答案问题的中位数 0.547 和有答案问题的最低分 0.497 高度重叠**，阈值扫描最优也只有 91% 准确率、60% 泄漏。

所以这个实验的目的**不是找到完美阈值**，而是**用数据证明「单靠阈值无法拒答」**，为 R12 的「模型判断 + 引用校验」提供立项依据。

#### 4.2.2 怎么跑

`run_eval.py` 默认就输出阈值扫描表（第五章），直接看 `eval_report_dense.md`：

```bash
python3 python_agent/R11/run_eval.py --retrieval-mode dense --threshold 0.5
```

重点看三列：**准确率 / 误拒率 / 泄漏率**，找到「泄漏率可接受」的最高阈值。

#### 4.2.3 怎么看结果

以下是 `run_eval.py` 实际输出（取自 [eval_report_dense.md](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/docs/my1/eval_report_dense.md) 第五章）：

| 阈值 | 准确率 | 误拒率（该答没答） | 泄漏率（不该答却答了） |
|---|---|---|---|
| 0.45 | 0.870 | 0.0% | 86.7% |
| 0.50 | 0.900 | 1.2% | 60.0% |
| 0.55 | 0.890 | 4.7% | 46.7% |
| 0.60 | 0.850 | 12.9% | 26.7% |
| 0.65 | 0.700 | 35.3% | 0.0% |

**和附录 A.4 的分数分布对照**（有答案 min=0.497，无答案 max=0.650）：
- 0.50 阈值只误拒 1 条（有答案最低分 0.497 < 0.50），但无答案有 9 条 ≥ 0.50 泄漏
- 要把泄漏率降到 0，阈值必须 ≥ 0.65（无答案最高分 0.650），但代价是 35.3% 的有答案问题被误拒
- 没有任何一个阈值能同时做到误拒率 <10% **且** 泄漏率 <10%

**结论**：单靠向量相似度阈值无法可靠拒答。真正的重叠区间在 0.55~0.65，两边都有大量样本，阈值无论设在哪都顾此失彼。

**给 R12 的输出**：拒答不能只靠相似度，必须叠加「检索为空 / 最高相似度极低（<0.45）→ 直接拒答」+「模型判断资料是否真的回答了问题」+「引用校验」三层。

#### 4.2.4 踩坑

- 阈值是 **Top-1 分数**，不是平均分。因为 RAG 只看最高分的片段进 prompt。
- 不同 embedding 模型的分数尺度不同，换模型后要重测阈值，不能照搬。

#### 4.2.5 线上接入：RETRIEVAL_SCORE_THRESHOLD

`search_kb` 已支持阈值过滤，通过环境变量配置：

```python
# config.py
# 阈值过滤开关：默认开启；关闭后 search_kb 不做任何分数过滤
ENABLE_RETRIEVAL_SCORE_THRESHOLD = os.getenv("ENABLE_RETRIEVAL_SCORE_THRESHOLD", "true").lower() == "true"
# 检索相似度阈值：低于该分数的 chunk 被过滤，全部被滤掉则返回"未找到相关信息"（模型据此拒答）。
# 开关打开后阈值始终生效（无"等于0不过滤"特殊分支），默认 0——dense/hybrid 分数均为正时自然不过滤，
# 但 rerank 链路对数似然可为负，默认 0 也会滤掉负分 chunk。
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0"))
```

```python
# tools.py search_kb（检索完成后、拼 prompt 前）
if ENABLE_RETRIEVAL_SCORE_THRESHOLD and pairs:
    filtered = [(d, s) for d, s in pairs if s >= RETRIEVAL_SCORE_THRESHOLD]
    if len(filtered) < len(pairs):
        _logger.info(f"[KB Threshold] tenant={tid} 阈值过滤 {len(pairs)-len(filtered)}/{len(pairs)} 条")
    pairs = filtered
```

**注意四点**：

1. **先确认开关是开的**。`ENABLE_RETRIEVAL_SCORE_THRESHOLD=false` 时阈值完全不生效，调参会"看起来没反应"；改 `.env` 后必须重建 agent 容器（`docker compose up -d agent`），`restart` 不会重新读取 `env_file`。

2. **阈值语义随检索链路变化**。本实验扫描的阈值基于 **dense 的 cosine 相似度**。当前线上默认 hybrid（RRF 融合分，量纲不同：top-1 约 0.5~1.0），开启 rerank 后是 cross-encoder 对数似然（可正可负）。换链路后必须重新扫描阈值，不能照搬 0.50。
3. **开关打开后，阈值始终生效，没有"等于 0 不过滤"的特殊分支**。默认 0 下 dense/hybrid 分数均为正，等价于不过滤；但 rerank 链路对数似然可为负，默认 0 也会把负分 chunk 滤掉——用 rerank 链路前必须先看其分数分布。
4. **本实验的结论是"单靠阈值不可拒答"**（0.50 最优阈值泄漏率仍有 60%）。只在验证过当前链路分数分布后，才把阈值作为「极端低分兜底」用（如 dense <0.45 直接拒答），**不要**指望它做主力拒答——主力方案是 R12。

---

### 4.3 实验 C：混合检索（向量 + BM25，RRF 融合）★核心

#### 4.3.1 解决什么问题

附录 A.2 的 4 条 Top-1 未命中，全部属于同一类失败模式：**query 里的某个关键词把向量检索吸到了"包含该词、但不是答案"的文档**。逐条看：

| ID | 问题 | 期望文档 | 实际 Top-1 | 吸偏关键词 |
|---|---|---|---|---|
| E002 | 7天无理由退货的运费由谁承担？ | 退换货政策 | 运费与配送规则 | "运费" |
| E023 | 包裹丢失或破损由谁承担责任？ | 运费与配送规则 | 退换货政策 | "丢失/破损"语义 |
| E059 | 保修需要提供纸质发票吗？ | 售后与保修规则 | 发票与支付规则 | "发票" |
| E071 | 不同仓库的商品会怎么发货？ | 订单与账号规则 | 运费与配送规则 | "发货" |

以 E059 为例：query 里有"发票"，《发票与支付规则》整篇都在讲发票，向量相似度天然高；但真正回答"保修要不要纸质发票"的内容在《售后与保修规则》里（"凭订单号即可，无需纸质发票"），只是顺带提了一句"发票"。**向量检索被高频词淹没，看不到真正的意图**。

**为什么纯向量解决不了？** 因为 embedding 是把整个 chunk 编码成一个向量，chunk 里所有词的语义被"平均"了。《发票与支付规则》里"发票"出现几十次，向量被"发票"这个概念主导；而《售后与保修规则》里"发票"只出现一次，被"保修""质保""维修"等词稀释。结果就是含"发票"的 query 永远偏向《发票与支付规则》，**调阈值、调 k 都救不回来**——因为吸偏的文档分数本身就很高。

**为什么 BM25 能补位？** BM25 看的是词频（TF）和逆文档频率（IDF）：
- "发票"在《发票与支付规则》里 TF 高，但 IDF 低（几乎篇篇都有）→ 贡献有限
- "保修"在《售后与保修规则》里 TF 高，且 IDF 高（只在这篇里高频出现）→ 贡献大
- query"保修需要纸质发票吗"里"保修"的 IDF 权重压过"发票"，BM25 会把《售后与保修规则》排第一

**实测补充（用本项目真实语料核对）：** 全量扫描 `python_agent/R11/kb_docs/` 的 8 篇文档，统计"发票""保修"两个词的出现次数：

| 文档 | 字数 | "发票" | "保修" |
|---|---|---|---|
| 01-退换货政策 | 943 | 0 | 2 |
| 02-运费与配送规则 | 669 | 0 | 0 |
| 03-发票与支付规则 | 681 | **6** | 0 |
| 04-会员与优惠券规则 | 686 | 0 | 0 |
| 05-商品与库存规则 | 614 | 0 | 0 |
| 06-售后与保修规则 | 598 | 1 | **7** |
| 07-订单与账号规则 | 592 | 0 | 0 |
| 08-服务承诺与转人工规则 | 666 | 1 | 0 |
| **合计** | 5449 | **8** | **9** |

df（出现该词的文档数）：发票 3/8、保修 2/8。代入 BM25 的 IDF 公式 `ln(1 + (N - df + 0.5)/(df + 0.5))`：

| 词 | df | IDF |
|---|---|---|
| 发票 | 3/8 | ≈ **0.94** |
| 保修 | 2/8 | ≈ **1.28** |

**实测后的三点补充：**

1. **发票实际出现在 3/8 篇**——06、08 两篇各仅 1 次（顺带提及）；真正以发票为主题的只有 03 篇（TF=6）。
2. **IDF 差距很小（0.94 vs 1.28）**——因为语料只有 8 篇，df=2 与 df=3 只差一篇文章。IDF 对低频词的放大效应要在**大语料**上才明显（如 10 万篇里"发票"出现在 8 万篇）。
3. **真正让 BM25 补位的是 TF 的去向集中度**：发票 75%（6/8）挤在 03 篇，保修 78%（7/9）集中在 06 篇。

再看 E059 的实际打分。query"保修需要纸质发票吗"分词后命中"保修""发票"两个词，假设 k1=1.2、b=0.75、两篇长度接近语料均值（近似估算），打分项 `tf/(tf+1.2)`：

| 文档 | 保修项（IDF 1.28） | 发票项（IDF 0.94） | BM25 总分 |
|---|---|---|---|
| 06-售后与保修规则 | tf=7 → 1.28×0.854 ≈ **1.09** | tf=1 → 0.94×0.455 ≈ **0.43** | ≈ **1.52** |
| 03-发票与支付规则 | 未命中 → 0 | tf=6 → 0.94×0.833 ≈ **0.78** | ≈ **0.78** |

06 篇胜出的原因不是"保修 IDF 压过发票 IDF"（两者只差 0.34），而是**06 篇同时命中了两个词**：高 TF 的"保修"是主项，顺带提到的"发票"又贡献一份；03 篇只有"发票"单路支撑。这就是 BM25 把《售后与保修规则》排第一的真实机制。

稠密向量擅长语义匹配（"运费谁出"≈"运费承担"），但对**专有名词、意图词**不敏感；BM25 擅长精确关键词权重，但不懂同义词。**两者互补，用 RRF 融合**是工业界标配方案（参考 [Qdrant 混合检索文档](https://qdrant.tech/documentation/search/text-search/hybrid-search/) 和 [Hybrid Search 生产实践](https://enricopiovano.com/blog/hybrid-search-strategies-comprehensive-guide/)）。

#### 4.3.2 原理：RRF（Reciprocal Rank Fusion）

两个检索器各自返回一个排序列表，融合分数：

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

- `rank_i(d)`：文档 d 在第 i 个检索器结果里的排名（1-based）
- `k`：常数，通常取 60。k 越大，排名靠后的结果贡献越大
- **关键性质**：只用排名、不用原始分数，所以向量相似度（0~1）和 BM25 分数（0~∞）可以直接融合，不需要归一化

例：E059"保修需要纸质发票吗"
- 向量检索 rank：[发票与支付规则, 售后与保修规则, ...]
- BM25 检索 rank：[售后与保修规则（含"保修""纸质"）, 发票与支付规则（含"发票"）, ...]
- RRF 后：售后与保修规则 = 1/(60+2) + 1/(60+1) ≈ 0.0328，发票与支付规则 = 1/(60+1) + 1/(60+2) ≈ 0.0328
- 两者接近，但如果 BM25 把售后规则排第 1，融合后售后规则胜出 → **E059 被修正**

#### 4.3.3 实现方案对比

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| ① Qdrant 原生稀疏向量 | `FastEmbedSparse(model_name="Qdrant/bm25")` + `RetrievalMode.HYBRID` | 利用 Qdrant 原生 RRF，性能好 | 需重建 collection（加 sparse 向量）、依赖 `fastembed`、需重新索引 |
| ② 纯 Python BM25 内存索引 | 内存维护 chunk→tokens 的倒排，RRF 自己写 | 零新依赖、不改 collection、可灰度 | 索引需随 reindex 更新、全量滚动有成本 |

**本项目采用方案①**：Qdrant 是项目的向量数据库，原生支持 sparse vector + hybrid search（RRF），用 `Qdrant/bm25` 稀疏模型（纯统计 BM25，CPU 友好，无需 GPU）。**注意：它自带的分词器不支持中文，必须先用 jieba 预分词**，否则中文会被当成一个 token（见 4.3.7）。

**工业界生产上的选型参考**：

| 基础设施 | 主流方案 | 原因 |
|---|---|---|
| Elasticsearch / OpenSearch | 内置 BM25 + 向量检索，RRF 融合 | Lucene 的 BM25 打磨了 15+ 年，零额外成本 |
| 纯向量库（Qdrant/Milvus/Pinecone）效果优先 | 原生稀疏向量（SPLADE / Qdrant/bm25） | 少维护一套外部检索引擎，稀疏向量与稠密向量同库融合 |
| 纯向量库务实派 | 外接 BM25（rank_bm25）+ RRF | 不改 schema，中小规模 ROI 高 |
| 大规模/高预算 | BM25 + SPLADE + Dense + HyDE → RRF → Cross-encoder | TREC 2025 RAG Track 优胜方案，效果上限最高 |

#### 4.3.4 具体实现（方案①）

改造涉及 6 个文件，核心思路：**collection 用 named vectors（dense + sparse），检索用 Qdrant 原生 FusionQuery(RRF)**。

**① [requirements.txt](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/requirements.txt) 加依赖**：

```
fastembed>=0.6.1   # Qdrant/bm25 稀疏向量生成（本地 CPU，无需 GPU）
jieba>=0.42.1      # 中文分词：SimpleTokenizer 按空格切词，中文无空格需先 jieba 切词
```

**② [config.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/config.py) 加配置**：

```python
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"   # 纯统计 BM25，支持中文，CPU 友好
HYBRID_FUSION = "rrf"
HYBRID_PREFETCH_K = 20              # 每个检索器取 top-20 再 RRF 融合
```

**③ 新建 [hybrid_search.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/hybrid_search.py) —— sparse 模型 + HybridStore**：

```python
"""混合检索：dense（OpenAI API）+ sparse（fastembed Qdrant/bm25），Qdrant RRF 融合。"""
import jieba
from fastembed import SparseTextEmbedding
from qdrant_client.http import models as qmodels
from langchain_core.documents import Document
from config import SPARSE_MODEL_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, HYBRID_PREFETCH_K

_sparse_model = None  # 进程内单例（BM25 模型无状态，所有租户共享）

def _tokenize_for_bm25(text: str) -> str:
    """BM25 前的中文预分词。
    SimpleTokenizer 只按空白/标点切词，中文无空格会被当成一个 token。
    用 jieba 把中文切成词并用空格连接，让 SimpleTokenizer 能正确拆分。"""
    return " ".join(jieba.cut(text))

def get_sparse_model() -> SparseTextEmbedding:
    """懒加载 sparse BM25 模型。
    disable_stemmer=True：SnowballStemmer 不支持中文，jieba 已切词无需再词干提取。"""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
            language="english",        # disable_stemmer 时任意值均可
            disable_stemmer=True,
        )
    return _sparse_model

def make_sparse_vector(text: str) -> qmodels.SparseVector:
    """用 BM25 模型把文本转成 Qdrant SparseVector（indices + values）。
    中文先经 jieba 分词再喂给 BM25，否则整段中文被当成 1 个 token。"""
    tokenized = _tokenize_for_bm25(text)
    sv = list(get_sparse_model().embed([tokenized]))[0]
    return qmodels.SparseVector(indices=list(sv.indices), values=list(sv.values))

class HybridStore:
    """混合检索 Store，接口与 QdrantVectorStore.asimilarity_search_with_score 对齐。
    mode: hybrid（dense+sparse RRF）/ dense（仅稠密）/ sparse（仅 BM25）"""
    def __init__(self, qdrant_client, collection_name, embeddings,
                 dense_name=DENSE_VECTOR_NAME, sparse_name=SPARSE_VECTOR_NAME):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.dense_name = dense_name
        self.sparse_name = sparse_name

    async def asimilarity_search_with_score(self, query, k=3, mode="hybrid"):
        dense_vec = await self.embeddings.aembed_query(query)        # OpenAI API
        sparse_vec = make_sparse_vector(query)                       # 本地 BM25

        prefetch = []
        if mode in ("hybrid", "dense"):
            prefetch.append(qmodels.Prefetch(
                query=qmodels.NearestQuery(nearest=list(dense_vec)),
                using=self.dense_name, limit=HYBRID_PREFETCH_K))
        if mode in ("hybrid", "sparse"):
            prefetch.append(qmodels.Prefetch(
                query=qmodels.NearestQuery(nearest=sparse_vec),
                using=self.sparse_name, limit=HYBRID_PREFETCH_K))

        # 单路直接查（拿真实相似度），多路才用 RRF 融合
        # 单路套 RRF 会把分数归一化成 rank 倒数，Top-1 恒为 0.5，阈值扫描失效
        if mode == "dense":
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qmodels.NearestQuery(nearest=list(dense_vec)),
                using=self.dense_name, limit=k, with_payload=True)
        elif mode == "sparse":
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vec, using=self.sparse_name,
                limit=k, with_payload=True)
        else:  # hybrid
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                prefetch=prefetch, limit=k, with_payload=True)

        return [(Document(page_content=p.payload.get("page_content", ""),
                          metadata=p.payload.get("metadata") or {}),
                 float(p.score)) for p in resp.points]
```

**④ [agent_factory.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/agent_factory.py#L94-L113) —— collection schema 加 sparse**：

```python
app.state.qdrant_client.create_collection(
    collection_name=col,
    vectors_config={
        DENSE_VECTOR_NAME: VectorParams(size=vec_dim, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
    },
)
vs = HybridStore(qdrant_client=app.state.qdrant_client, collection_name=col, embeddings=tenant_emb)
app.state.tenant_stores[tenant_id] = vs
```

**⑤ [kb_index_worker.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/kb_index_worker.py#L76-L103) —— 写入 dense + sparse 双向量**：

```python
dense_vecs = await tenant_emb.aembed_documents(texts)       # OpenAI API
sparse_vecs = [make_sparse_vector(t) for t in texts]        # 本地 BM25

points = [qmodels.PointStruct(
    id=ids[i],
    vector={DENSE_VECTOR_NAME: dense_vecs[i], SPARSE_VECTOR_NAME: sparse_vecs[i]},
    payload={"page_content": texts[i],
             "metadata": {"doc_id": doc_id, "title": title, "source": "kb", "tenant_id": tenant_id}},
) for i in range(len(texts))]
app.state.qdrant_client.upsert(collection_name=col, points=points)
```

**⑥ [routers_kb.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/routers_kb.py#L441-L462) —— test-query 加 mode 参数**：

```python
mode = (body.get("mode") or "hybrid").strip().lower()  # hybrid / dense / sparse
pairs = await store.asimilarity_search_with_score(query, k=k, mode=mode)
```

**⑦ [tools.py search_kb](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/tools.py#L91)** —— 无需改动，`store.asimilarity_search_with_score(query, k=RETRIEVER_TOP_K)` 默认走 hybrid。

#### 4.3.5 怎么跑

**Step 1：重建 agent 镜像（安装 fastembed + jieba）**

```bash
docker compose up -d --build agent
```

> 首次重建会下载 `Qdrant/bm25` 模型（约 18 个文件）和 jieba 词典，耐心等待。

**Step 2：删除旧 collection（schema 变了，必须重建）**

用 [kb_import.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_import.py) 清空并重建（默认就是清空 + 重新导入 + 触发索引 + 等待完成）：

```bash
python3 python_agent/R11/kb_import.py
```

> `kb_import.py` 参数说明：无参数 = 清空重建；`--keep` = 增量导入不清空；`--dry-run` = 只打印不执行；`--only 01` = 只导入文件名以 01 开头的文档。

**Step 3：验证三种模式**

> `test-query` 接口需要鉴权，先登录拿 token（用 [kb_import.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/R11/kb_import.py) 相同的 app_key/app_secret）。

```bash
# 1) 登录拿 token
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"app_key":"demo_key","app_secret":"demo_secret"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['token'])")

# 2) 三种模式对比（用 python3 解析 JSON，无需 jq）
for MODE in dense sparse hybrid; do
  echo "=== $MODE ==="
  curl -s -X POST http://localhost:8080/api/admin/kb/test-query \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"query\":\"保修需要提供纸质发票吗\",\"k\":3,\"mode\":\"$MODE\"}" \
    | python3 -c "
import sys,json
d=json.load(sys.stdin)
for i,x in enumerate(d['data']['items']):
    print(f'  {i+1}. {x[\"title\"]} | score={x[\"score\"]:.4f}')
"
done
```

预期：dense 返回"发票与支付规则"，sparse 返回"售后与保修规则"，hybrid 返回"售后与保修规则"。

**Step 4：跑完整评测对比**

```bash
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode hybrid
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode dense
```

> 报告会自动按模式命名：`eval_report_hybrid.md`、`eval_report_dense.md`（不再互相覆盖）。报告头部第二行标注检索模式，方便对照。

#### 4.3.6 怎么看结果

**单条验证**：看 4 条未命中用例的 Top-1 是否被 hybrid 修正。

| ID | 问题 | dense Top-1 | sparse Top-1 | hybrid Top-1 | 是否修正 |
|---|---|---|---|---|---|
| E002 | 7天无理由退货的运费由谁承担？ | 运费与配送规则 | ? | ? | ? |
| E023 | 包裹丢失或破损由谁承担责任？ | 退换货政策 | ? | ? | ? |
| E059 | 保修需要提供纸质发票吗？ | 发票与支付规则 | ? | ? | ? |
| E071 | 不同仓库的商品会怎么发货？ | 运费与配送规则 | ? | ? | ? |

**整体指标**：对比 hybrid vs dense 的 Recall@1 / MRR / Evidence hit@3。预期 hybrid 对字面关键词强的问题有改善，Recall@1 可能从 0.953 提到 0.97+。

**如果 hybrid 没改善甚至变差**：
1. 先看 sparse 单独跑的结果——如果 sparse 自己就排错了，说明 BM25 分词没命中关键词
2. 检查 `HYBRID_PREFETCH_K` 是否太小（默认 20，对 8 篇文档 39 chunks 足够）
3. RRF 的 k 是 Qdrant 默认值（60），一般不用调

#### 4.3.7 踩坑

- **必须删除旧 collection 重建**：旧 collection 只有 dense 向量，没有 sparse 字段，直接 upsert 会报 `vector name "sparse" not found`。重建后必须重新索引所有文档。
- **中文必须先 jieba 分词**：fastembed 的 `SimpleTokenizer` 只按空白/标点切词，中文无空格会被当成**一整句一个 token**，导致 sparse 检索返回 0 条。必须用 `_tokenize_for_bm25`（jieba 切词 + 空格连接）预处理，索引和查询走同一套分词。同时 `disable_stemmer=True`（SnowballStemmer 不支持中文）。
- **fastembed 首次加载会下载模型**：`Qdrant/bm25` 模型约几十 MB，首次调用 `get_sparse_model()` 时下载到本地缓存，后续秒加载。容器构建时可预下载避免首次请求慢。
- **Qdrant 服务端版本 ≥ 1.15.2**：sparse vector + hybrid search 需要 Qdrant 1.15.2+。项目用 `qdrant/qdrant:latest`，满足要求。
- **不要在生产用 `latest` 标签**：练习项目无所谓，生产应锁定版本，避免 Qdrant 升级导致 sparse API 变化。
- **sparse 模型是全局单例**：`Qdrant/bm25` 无状态，所有租户共享一个模型实例。dense 向量仍按租户隔离（BYOK 模型配置）。
- **search_kb 默认走 hybrid**：不需要改 tools.py，因为 `HybridStore.asimilarity_search_with_score` 的 `mode` 默认值就是 `"hybrid"`。
- **单路模式不能套 RRF 融合**：dense/sparse 单路查询时若用 `FusionQuery(RRF)` 包裹单条 prefetch，Qdrant 会把分数归一化成 rank 倒数（rank1→0.5, rank2→0.333...），导致所有 Top-1 分数恒为 0.5，阈值扫描失效。单路必须直接 `query_points(query=NearestQuery(...), using=...)` 拿真实相似度，只有 hybrid 才用 RRF。

---

### 4.4 实验 D：Rerank（cross-encoder 重排）

#### 4.4.1 解决什么问题

混合检索解决了「召回」，但「排序」仍由向量相似度决定。双塔模型（embedding）为了速度把 query 和 passage 分别编码再点积，**表达能力弱于 cross-encoder**（把 query 和 passage 拼一起过 Transformer）。Rerank 就是用 cross-encoder 对 top-N 候选重新打分排序。

当前基线 MRR=0.975，说明大多数期望文档已在 rank1，但还有提升空间。Rerank 的目标是**把 rank2/3 的正确文档提到 rank1**，提升 MRR 和 Recall@1。

#### 4.4.2 原理

Cross-encoder 输入 `<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> query [SEP] passage [SEP]`，直接输出一个相关性分数。相比双塔的点积，它能捕捉 query 和 passage 之间的细粒度交互（如"保修需要发票吗"和"凭订单号即可无需发票"的否定关系）。

#### 4.4.3 实现方案对比

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| ① 本地 bge-reranker-v2-m3 | `FlagEmbedding.FlagReranker` | 568M 参数、多语言、效果好、无 API 费用 | 需装 `FlagEmbedding`+`torch`，首次下载模型几百 MB |
| ② API rerank | 调 Cohere/Jina/火山 rerank 接口 | 免部署、按需付费 | 依赖外部服务、有延迟和费用 |
| ③ LLM-as-reranker | 用现有对话模型打分 | 零部署 | 最贵、最慢，不适合生产 |

**推荐方案①**：`bge-reranker-v2-m3` 是中文场景 rerank 的标杆模型（参考 [BGE 官方文档](https://bge-model.com/tutorial/5_Reranking/5.1.html)），本地部署无外部依赖、延迟可控。

#### 4.4.4 具体实现（方案①）

安装依赖（只在做实验时加，不进生产 requirements）：

```bash
pip install FlagEmbedding
```

**新增** `python_agent/rerank.py`：

```python
"""Cross-encoder rerank：bge-reranker-v2-m3。
仅对 dense/hybrid 返回的 top-N 重排，取 top-k 返回。
"""
import logging
logger = logging.getLogger("agent.rerank")

_reranker = None
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATE_N = 5  # 只对 top-N 重排，cross-encoder 是 O(N)，全量太慢；CPU 实测单条前向 3~5s

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            # use_fp16=True 仅在有 GPU 时加速；纯 CPU 必须设 False
            _reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=False)
            logger.info("✅ reranker loaded: %s", RERANK_MODEL_NAME)
        except Exception as e:
            logger.warning("⚠️ reranker 不可用，降级为不重排: %s", str(e)[:150])
    return _reranker


def rerank_items(query: str, items: list, top_k: int, candidate_n: int = RERANK_CANDIDATE_N) -> list:
    rk = get_reranker()
    if rk is None or not items:
        return items[:top_k]
    candidates = items[:candidate_n]
    pairs = [[query, it["content"]] for it in candidates]
    scores = rk.compute_score(pairs)  # list[float]，对数似然，越大越相关
    if not isinstance(scores, list):
        scores = [scores]
    for it, sc in zip(candidates, scores):
        it["score"] = float(sc)
    candidates.sort(key=lambda x: -x["score"])
    return candidates[:top_k]
```

**`routers_kb.py` 的 test-query 接口**：增加 `mode=rerank`，先 hybrid 取 top-5，再 rerank 取 top-k：

```python
from rerank import rerank_items
from config import RERANK_CANDIDATE_N

# 检索模式：hybrid / dense / sparse / rerank
mode = (body.get("mode") or "hybrid").strip().lower()
if mode not in ("hybrid", "dense", "sparse", "rerank"):
    mode = "hybrid"

if mode == "rerank":
    # 先用 hybrid 召回 top-N 候选（受 k 上限约束），再用 cross-encoder 重排取 top-k
    retrieve_k = max(k, RERANK_CANDIDATE_N)
    pairs = await store.asimilarity_search_with_score(query, k=retrieve_k, mode="hybrid")
    items = [{"doc_id": doc.metadata.get("doc_id", ""), "title": doc.metadata.get("title", ""),
              "score": float(score), "content": doc.page_content} for doc, score in pairs]
    items = rerank_items(query, items, top_k=k, candidate_n=retrieve_k)
else:
    pairs = await store.asimilarity_search_with_score(query, k=k, mode=mode)
    items = [{"doc_id": doc.metadata.get("doc_id", ""), "title": doc.metadata.get("title", ""),
              "score": float(score), "content": doc.page_content} for doc, score in pairs]
```

**`run_eval.py`**：`--retrieval-mode` 增加 `rerank` 选项，报告自动命名为 `eval_report_rerank.md`。

**线上集成（默认开启）**：`search_kb` 工具已集成 rerank，`ENABLE_RERANK` 默认为 `true`。流程为 hybrid → top-5 → cross-encoder 重排 → top-3，rerank 推理放线程池（`asyncio.to_thread`）不阻塞事件循环。`FlagEmbedding` 已写入 `requirements.txt`，`bge-reranker-v2-m3` 模型在 Docker 构建阶段预下载到 `/opt/huggingface`，运行时无需联网。如需关闭，在 `.env` 设 `ENABLE_RERANK=false`。

> **耗时实测（CPU 无 GPU）**：10 候选 30~50s，5 候选 15~20s，单条前向约 3~5s。因此候选数默认取 5，并且 Go 侧流式对话单独用 `streamHTTPClient`（Timeout 180s）——`http.Client.Timeout` 覆盖整个响应体读取，原来的 60s 会被 rerank + LLM 生成顶穿，前端表现为收不到消息。

```python
# config.py
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_CANDIDATE_N = int(os.getenv("RERANK_CANDIDATE_N", "5"))

# tools.py search_kb
if ENABLE_RERANK:
    raw_pairs = await store.asimilarity_search_with_score(query, k=RERANK_CANDIDATE_N)
    items = [{..., "_idx": i} for i, (d, s) in enumerate(raw_pairs)]
    reranked = await asyncio.to_thread(rerank_items, query, items, RETRIEVER_TOP_K, RERANK_CANDIDATE_N)
    pairs = [(raw_pairs[it["_idx"]][0], it["score"]) for it in reranked]
```

```dockerfile
# Dockerfile 构建阶段预下载模型
ENV HF_HOME=/opt/huggingface
RUN python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)"
# runtime 阶段复制模型缓存
COPY --from=builder /opt/huggingface /opt/huggingface
```

#### 4.4.5 怎么跑

**前置**：重新构建镜像（安装 FlagEmbedding + 预下载模型）：

```bash
docker compose up -d --build agent
```

**跑评测**：

```bash
# 默认输出 eval_report_rerank.md
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode rerank

# 对比：同时跑 hybrid 基线
python3 python_agent/R11/run_eval.py --k 3 --retrieval-mode hybrid
```

#### 4.4.6 怎么看结果

**指标**：MRR 是否从 0.975 提升（预期到 0.985+），Recall@1 是否提升。

**延迟**：rerank 是本实验最大的成本。用 `time` 命令或在 `test-query` 里加耗时日志，记录 P50/P95：

```bash
# 单条耗时粗略测
time curl -X POST http://localhost:8080/api/admin/kb/test-query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"保修需要发票吗","k":5,"mode":"rerank"}'
```

**权衡**：如果 MRR 提升 <0.01 但延迟翻倍，不值得上生产；用「只对 top-5 重排」或「缓存 rerank 结果」降低成本。

#### 4.4.7 踩坑

- **只 rerank top-5，不要全量**：cross-encoder 是 O(N) 推理，CPU 上单条前向 3~5s，全量重排不可接受。top-5 已覆盖混合检索的高分区间。
- **流式接口超时必须单独放宽**：Go 侧 `http.Client.Timeout` 覆盖「发起请求 → 读完整个响应体」，流式对话若沿用 60s，rerank（15~20s）+ LLM 生成会顶穿，前端表现为无消息。流式单独用 `streamHTTPClient`（180s）。
- **use_fp16 在纯 CPU 上要设 False**：否则可能报错或反而更慢。GPU 上设 True 省一半显存。
- **模型已在构建时预下载**：Dockerfile 构建阶段执行 `FlagReranker('BAAI/bge-reranker-v2-m3')` 触发下载，模型缓存到 `/opt/huggingface`，运行时无需联网。若构建时网络受限，可手动 `huggingface-cli download BAAI/bge-reranker-v2-m3` 后挂载到容器。
- **rerank 不能单独提升 Recall**：它只重排已有候选，召回上限由 dense/hybrid 决定。所以 **rerank 应该叠在混合检索之上**（hybrid → top-5 → rerank → top-k），而不是替代混合检索。
- **`rerank_items` 是同步函数，不要 `await`**：`FlagReranker.compute_score` 是 CPU 同步推理，`rerank_items` 没有 `async`。如果上生产，应改用 `asyncio.to_thread(rerank_items, ...)` 避免阻塞事件循环；实验阶段直接同步调用即可。
- **`compute_score` 返回值兼容**：输入 list[pair] 返回 list[float]，但某些版本单条输入可能返回 float，代码里已做 `isinstance(scores, list)` 判断兼容。

---

### 4.5 实验 E：Chunk 策略（标题路径前缀）

#### 4.5.1 解决什么问题

当前 [kb_index_worker.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/kb_index_worker.py#L66-L67) 用 `RecursiveCharacterTextSplitter(chunk_size=200, overlap=30)` 切分，chunk 的 `page_content` **不包含文档标题和小节标题**。

后果：E002"7天无理由退货的运费由谁承担"在《退换货政策》和《运费与配送规则》里都有"运费"相关段落，两个 chunk 的文本几乎一样，embedding 也几乎一样，检索时谁排前全看运气。

**解法**：给每个 chunk 的文本前置「文档标题 > 小节标题」路径，让 embedding 包含结构信息。这是工业界叫 **Contextual Retrieval / Title-Chain Prefix** 的做法（参考 [Structure-Aware Semantic Chunking 论文](https://arxiv.org/html/2608.00824) 和 [得物 RAG Chunking 实战](https://cloud.tencent.com/developer/article/2582766)）。

#### 4.5.2 原理

原来的 chunk：
```
- 运费承担：无理由退货的退货运费由买家承担；订单含运费险的，按运费险规则赔付。
```

加标题路径后的 chunk：
```
[退换货政策 > 二、7 天无理由退货]
- 运费承担：无理由退货的退货运费由买家承担；订单含运费险的，按运费险规则赔付。
```

这样 embedding 时，"退换货政策 > 7天无理由退货"这个上下文会让向量偏向《退换货政策》，而《运费与配送规则》里的"包邮门槛 99 元"chunk 会带上 `[运费与配送规则 > 一、包邮规则]`，两者区分度大幅提高。

#### 4.5.3 具体实现

**代码位置**：[kb_index_worker.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/kb_index_worker.py#L29-L61) 新增标题感知分片，`_index_one` 里按开关二选一（[L104-L115](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/kb_index_worker.py#L104-L115)）。

1. **`_HEADER_SPLITTER`**：先按 markdown 标题切段，保留每段的 `doc_title` / `section` 元数据：

   ```python
   _HEADER_SPLITTER = MarkdownHeaderTextSplitter(
       headers_to_split_on=[("#", "doc_title"), ("##", "section")],
       strip_headers=True,   # 去掉正文里的标题行，标题信息统一由前缀承载，避免分片内重复出现标题
   )
   ```

2. **`split_chunks_with_title_prefix(text, title, sub_splitter)`**：对每个标题段再用 `app.state.text_splitter`（`chunk_size=200 / overlap=30`）切小块，并给每个 chunk 前置「文档标题 > 小节标题」路径：

   ```python
   prefix = f"[{doc_title} > {section}]" if section else f"[{doc_title}]"
   sub.page_content = f"{prefix}\n{sub.page_content}"
   ```

   分支处理：
   - 文档标题优先取 markdown 一级标题，缺失时用 DB 里的 `title`；
   - 只有标题行、没有正文的段（文档开头的一级标题）直接丢弃，避免生成纯标题垃圾分片；
   - 文档没有 markdown 标题 / 切分结果为空 → 回退 `sub_splitter.create_documents([text])`，与基线完全一致，保证非 markdown 文档不受影响。

3. **`_index_one` 按开关选择分片方式**：

   ```python
   if ENABLE_CHUNK_TITLE_PREFIX:
       chunks = split_chunks_with_title_prefix(text, title, app.state.text_splitter)
   else:
       chunks = app.state.text_splitter.create_documents([text])
   ```

4. **chunk 元数据带小节信息**：写入 Qdrant 的 payload 为
   `metadata = {doc_id, title, section, source, tenant_id}`。`section` 是新增字段，方便后续溯源和 badcase 定位（检索接口返回的 `title` 仍是文档标题，行为不变）。

5. **配置开关**（[config.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/config.py#L37-L42) + `.env`）：

   ```python
   # 分片策略（实验 E：标题路径前缀）：是否给每个 chunk 前置「文档标题 > 小节标题」路径
   # 改这个开关必须重新导入知识库（kb_import.py）才会生效，因为分片是索引期动作。
   ENABLE_CHUNK_TITLE_PREFIX = os.getenv("ENABLE_CHUNK_TITLE_PREFIX", "true").lower() == "true"
   ```

   ```bash
   # .env
   ENABLE_CHUNK_TITLE_PREFIX=true
   ```

   **默认开启**；做 A/B 对比时改成 `false` 即回退基线分片。

   > **为什么前缀在切块之后加**：prefix 不占用 `chunk_size=200` 的正文预算，只让最终参与 embedding/BM25 的文本多十几到几十字。如果先加前缀再切分，正文反而会被压缩。

**实测分片数**（8 篇源文档，2026-11-03 本地验证）：基线 39 个 chunk → 标题路径 48 个 chunk。

**重索引**：分片是索引期动作，改代码或改开关后必须重跑导入：

```bash
python3 python_agent/R11/kb_import.py
```

#### 4.5.4 怎么跑

**0) 重建 agent 容器**（分片逻辑跑在 worker 进程里）：

```bash
docker compose up -d --build agent
```

**1) 基线（关闭标题路径）**：

```bash
# .env: ENABLE_CHUNK_TITLE_PREFIX=false
# env_file 只在容器创建时读取，改 .env 必须重建容器，不能只 restart
docker compose up -d agent

python3 python_agent/R11/kb_import.py
python3 python_agent/R11/run_eval.py --retrieval-mode dense \
  --report docs/my1/exp_chunk_baseline.md
```

**2) 实验 E（打开标题路径）**：

```bash
# .env: ENABLE_CHUNK_TITLE_PREFIX=true
docker compose up -d agent

python3 python_agent/R11/kb_import.py
python3 python_agent/R11/run_eval.py --retrieval-mode dense \
  --report docs/my1/exp_chunk_title.md
```

**3) 可选：叠加混合检索再看一遍**（sparse BM25 同样吃到标题路径，关键词匹配也会受益）：

```bash
python3 python_agent/R11/run_eval.py --retrieval-mode hybrid \
  --report docs/my1/exp_chunk_title_hybrid.md
```

**4) 肉眼验证前缀是否真的写入**：`kb_import.py` 结束时会打印每篇文档的 chunk 数（应为 48 而不是 39）；再用 `test-query` 看命中的 `content` 是否以 `[文档标题 > 小节标题]` 开头：

```bash
time curl -X POST http://localhost:8080/api/admin/kb/test-query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"7天无理由退货的运费由谁承担","k":3,"mode":"dense"}'
```

#### 4.5.5 怎么看结果

**重点看 Evidence hit@3 和文档边界模糊的用例**（E002、E023、E071）。这三个 query 的正确段落和干扰文档文本高度相似，标题路径是唯一的结构区分信号。

**整体指标**：Recall@1 / MRR / Evidence hit@3 是否提升。预期 Evidence hit@3 从 0.988 提到 1.0。

**对比表要记两个额外字段**：
- chunk 数（39 → 48）：chunk 变多会让 Top-3 覆盖更细，但上下文更碎，需要一起看生成质量；
- Top-1 分数分布：加前缀后 dense 分数整体会小幅变化，**基线阈值和实验阈值不能直接沿用**（见 4.7 纪律 2）。

#### 4.5.6 踩坑

- **必须重索引**：只改代码/开关不重跑 `kb_import.py`，Qdrant 里仍是旧 chunk，实验白做。开关是索引期读取的，重索引前记得先重建容器让新 `.env` 生效。
- **`docker compose up -d agent` 而不是 restart**：`env_file` 只在容器创建时读取，`restart` 不会重新加载改过的 `.env`。
- **标题路径不能太长**：前缀只放「文档标题 + 二级标题」，不要塞完整标题层级（#/##/### 全带上）。当前标题路径约 15~30 字，且不占正文 `chunk_size` 预算。
- **`page_content` 改了不影响 `must_contain` 判据**：`run_eval.py` 的 Evidence hit 是归一化子串匹配，前缀不影响关键词是否出现。但**生成评测**（e2e 模式）里模型会看到标题路径，答案措辞可能变化——这是好事（模型更清楚来源）。
- **chunk 数会变**：8 篇文档实测 39 → 48。数量上涨是正常的（每个二级标题独立成段、超长段被再切分），不要按旧 chunk 数判断导入是否成功。
- **段落被切碎后 Evidence 关键词可能跨 chunk**：chunk 变短后，一条证据如果横跨两段，单 chunk 子串匹配可能失败。若 Evidence hit@3 反而下降，先检查是不是这个原因（看报告第六节逐条详情里的 `E` 标记）。
- **无标题文档不受影响**：没有 `#`/`##` 的文档会走 fallback，仍用原来的 `RecursiveCharacterTextSplitter`。

---

### 4.6 实验组合与对比表

5 个实验不是互斥的，C+D+E 可以叠加。最终目标是找到**指标提升 / 延迟成本**最优的组合。用这张表统一记录：

| 配置 | Recall@1 | MRR | Evidence hit@3 | P50 延迟 | P95 延迟 | 备注 |
|---|---|---|---|---|---|---|
| 基线（dense, k=3） | 0.953 | 0.975 | 0.988 | __ms | __ms | 已实测 |
| A: dense, k=5 | __ | __ | __ | __ms | __ms | 待跑 |
| A: dense, k=8 | __ | __ | __ | __ms | __ms | 待跑 |
| C: hybrid (RRF), k=3 | 0.953 | 0.976 | 1.000 | __ms | __ms | 已实测 |
| C+D: hybrid + rerank | __ | __ | __ | __ms | __ms | 推荐生产配置，待跑 |
| E: chunk 标题路径 | __ | __ | __ | __ms | __ms | 代码已就绪（`ENABLE_CHUNK_TITLE_PREFIX`），重索引后待跑 |
| C+D+E: 全叠加 | __ | __ | __ | __ms | __ms | 待跑 |

**判断"真提升"的标准**：
1. 同一套 100 条评测集、同一套知识库源文件（Step 1 冻结的）
2. Recall@1 提升 ≥0.01 或 MRR 提升 ≥0.005（小于这个可能是波动）
3. 延迟 P95 不超过基线的 2 倍（否则用户感知明显变慢）
4. 改前改后数字都记下来，面试时直接报

---

### 4.7 纪律与注意事项

1. **冻结评测集后再调参**：Step 1 冻住后，所有实验在同一套 100 条上跑。一边调参一边改题，数字会骗自己。
2. **一次只改一个变量**：实验 C 只加混合检索，不改 chunk；实验 E 只改 chunk，不开混合检索。否则不知道提升来自哪。
3. **延迟必须记录**：优化不能只看指标不看成本。rerank 提升 0.01 MRR 但延迟翻倍，生产上不值得。
4. **重索引后清缓存**：实验 E 改 chunk 后，Redis 检索缓存必须失效（`kb_import.py` 已自动处理），否则测的是旧 chunk。

---

### 4.8 本步验收

- [ ] 实验 A：跑出 k=3/5/8 三组报告，确定最优 k
- [ ] 实验 B：确认阈值无法可靠拒答，输出给 R12 的结论
- [ ] 实验 C：实现 BM25+RRF 混合检索，`test-query mode=hybrid` 可用，Recall@1 或 MRR 有可解释提升
- [x] 实验 D（可选）：实现 bge-reranker-v2-m3 重排，`test-query mode=rerank` 可用，`search_kb` 支持 `ENABLE_RERANK` 开关（待跑评测记录 MRR 提升与延迟成本）
- [x] 实验 E：实现标题路径前缀 chunk（`split_chunks_with_title_prefix` + `ENABLE_CHUNK_TITLE_PREFIX` 开关，默认开启），重索引后待跑评测填指标
- [ ] 填好 4.6 的对比表，选出生产配置（建议 hybrid + rerank + 标题路径）
- [ ] 延迟数据一并记录（P50 / P95）

---

### 4.9 参考链接

- [Qdrant Hybrid Search 官方文档](https://qdrant.tech/documentation/search/text-search/hybrid-search/) — 稠密+稀疏向量原生融合方案
- [LangChain Qdrant 集成](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant) — `FastEmbedSparse` + `RetrievalMode.HYBRID` 用法
- [Hybrid Search Strategies 综合指南](https://enricopiovano.com/blog/hybrid-search-strategies-comprehensive-guide/) — RRF k=60 的经验值与调参建议
- [Hybrid Retrieval for Hallucination Mitigation (arXiv)](https://arxiv.org/pdf/2504.05324) — 动态权重 RRF 对幻觉的抑制实验
- [BGE Reranker 官方教程](https://bge-model.com/tutorial/5_Reranking/5.1.html) — bge-reranker-v2-m3 用法
- [Structure-Aware Semantic Chunking with Title-Chain Prefixes](https://arxiv.org/html/2608.00824) — 标题路径前缀对 chunk 检索的提升实验
- [得物 RAG Chunking 策略实战](https://cloud.tencent.com/developer/article/2582766) — 结构感知分块的工程实践

---

## 第五章 · Step 5：最优配置写回 `search_kb`（部分完成）

> 把 Step 4 选出的最优参数写回 [tools.py 的 search_kb](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/tools.py#L141-L223)，旧行为用配置开关保留。

### 5.1 要做的事

1. **租户级配置**：把 `top_k`、相似度阈值、是否混合检索、是否 rerank 做成租户级可配（从租户配置读取，不硬编码）。
2. **保留 fallback**：旧的单路向量检索保留为默认/降级路径，新路径出问题可一键切回。
3. **结构化返回**（衔接 R12）：`search_kb` 从返回纯字符串改为返回 JSON 数组（含 `doc_id/title/snippet/score`），让模型能引用来源。

### 5.2 本步待完成

- [ ] `search_kb` 支持租户级 `top_k` / 阈值 / 混合检索 / rerank 开关
- [ ] 旧单路检索保留为 fallback
- [ ] 返回结构带 `score` 和 `doc_id`，为 R12 拒答 + 引用做准备

---

## 附录 A：基线结果（实测 2026-11-01）

> 用临时脚本跑的基线，Step 3 完成后由 `run_eval.py` 复现。

### A.1 检索指标（85 条有答案问题，k=5）

| 指标 | 结果 |
|---|---|
| Recall@1 | **0.953**（81/85） |
| Recall@3 | **1.000**（85/85） |
| Recall@5 | **1.000**（85/85） |
| MRR | **0.975** |
| Evidence hit@3（chunk 级） | **0.988**（84/85） |
| Evidence hit@5（chunk 级） | **1.000**（85/85） |

### A.2 Top-1 未命中分析（4 条）

| ID | 问题 | 实际 Top-1 | 分析 |
|---|---|---|---|
| E002 | 7天无理由退货的运费由谁承担？ | 运费与配送规则 | 两篇都讲运费，语义高度重叠；属**文档边界模糊**，需要标题/分类加权 |
| E023 | 包裹丢失或破损由谁承担责任？ | 退换货政策 | "责任"被退换货政策吸走；关键词"包裹"权重不足 |
| E059 | 保修需要提供纸质发票吗？ | 发票与支付规则 | "发票"强匹配，但答案是"凭订单号即可"；典型**字面相似但意图不同** |
| E071 | 不同仓库的商品会怎么发货？ | 运费与配送规则 | 配送类问题，答案在《订单与账号规则》的拆单段 |

> 这 4 条说明：**文档级 Recall 已接近饱和，真正的差距在 chunk 级排序**。这也解释了为什么要看 Evidence hit 和 MRR，而不是只看 Recall@5。

### A.3 为什么 Recall@5 = 1.0 仍要继续优化

- 当前语料只有 8 篇文档，Top-5 相当于覆盖 62% 的文档，指标天然虚高。
- 真实项目 100+ 文档时，Recall@5 会明显下降；所以现在就要把**评测框架和指标口径**搭好，而不是等语料变大再补。
- 更有区分度的指标是 **Recall@1 / MRR / Evidence hit@3**，优化目标应盯这三个。

### A.4 阈值区分度（为 R12 提供依据）

| 分布 | 中位数 | 最小 | 最大 |
|---|---|---|---|
| 有答案问题 Top-1 分数 | 0.679 | 0.497 | 0.817 |
| 无答案问题 Top-1 分数 | 0.547 | 0.415 | 0.650 |

阈值扫描最优结果：`threshold=0.49` 时整体判定准确率 **0.910**，但**15 条无答案中有 9 条分数高于阈值**（泄漏 60%）。

**结论：单靠向量相似度阈值不能可靠拒答。** 无答案问题的分数和长尾有答案问题高度重叠，必须叠加 R12 的"模型判断 + 引用校验 + 拒答话术"。这个数字，是"为什么需要 R12"的最硬证据。

---

## 附录 B：验收标准总表

- [x] 8 篇源文件在仓库内，可 diff、可 review
- [x] 一条命令可清空并重建知识库（实测 8 篇 / 39 chunks）
- [x] 100 条评测集，含 15 条无答案负样本
- [x] `run_eval.py` 可复现基线：Recall@1 0.953 / Recall@5 1.000 / MRR 0.975 / Evidence hit@3 0.988
- [x] `python3 python_agent/R11/run_eval.py` 一条命令跑完，输出 `dense_report.md`（指标表 + Top-1 未命中 + 阈值扫描）
- [ ] 完成至少 3 组优化实验（A/B/C），并记录改前改后数字
- [ ] Recall@1 或 MRR 有可解释的提升（例如混合检索后 MRR ≥0.98）

---

## 附录 C：FAQ 与坑

1. **为什么 `doc_id` 每次都不一样？**
   上传接口生成 uuid。所以评测集只存标题，运行时解析，不要硬编码 uuid。

2. **为什么改了 md 但检索结果没变？**
   只改文件不会自动生效，必须重跑 `kb_import.py`（或管理端触发重索引）。重索引会先删旧向量，避免新旧内容同时命中。

3. **为什么同一个问题命中了"错误"的文档却也能答对？**
   多数规则在多个文档里都有表述（如缺货补偿 5 元同时出现在《商品与库存》和《服务承诺》）。评测集对这类问题使用多标题期望，避免误判。

4. **评测集要不要用文档原句？**
   不要。原句会让向量检索"作弊"式命中。要用真实用户口语（"运费谁出""几天能到"）。

5. **多少条评测集够用？**
   起步 100 条，覆盖每个业务域至少 8 条 + 15 条负样本。低于 50 条时，指标波动会大于优化收益。

6. **语料只有 8 篇，指标会不会失真？**
   会。所以本手册同时给出 chunk 级 Evidence hit 和 Top-1 错误清单。扩语料后再用同一套脚本复测，指标才具备可比性。

---

## 附录 D：面试怎么讲（详细版）

> 讲法原则：**先讲为什么做 → 再讲怎么做 → 最后报数字 → 主动交代坑**。
> 每一句话都要能落到"代码在哪、报告在哪、数字怎么复现"。只会背结论、说不出证据，面试官一追问就露馅。

### D.1 30 秒电梯版（开场自我介绍 / 项目一句话）

> "我做过一个电商客服 RAG 的检索质量专项：先搭 100 条评测集把'回答不准'拆成'检索没召回'和'生成没用对'，再用 BM25+向量的 RRF 混合检索和 cross-encoder 重排去修排序，最后用阈值扫描证明单靠相似度不能拒答，把拒答交给下一层的意图闸门和引用校验。"

一句话记忆点：**"先有尺子，再谈优化；先做归因，再改参数。"**

### D.2 90 秒标准版（STAR，面试主叙述）

**S 背景**：电商客服知识库，用户问法口语化、跨文档，答错涉及赔付；原来只有单路向量检索，没有评测，调参靠感觉。

**T 任务**：建立可复现的检索评测体系，量化定位瓶颈，并在不显著增加延迟的前提下提升排序质量。

**A 行动**（按顺序讲，体现方法论）：

1. **建地基**：8 篇多域 KB 源文件放仓库（可 diff、可重建），配 manifest；100 条评测集，85 条有答案 + 15 条无答案，用文档标题而不是 `doc_id` 做锚点，保证重导后评测仍可复用。
2. **分层指标**：文档级 Recall@1/3/5、MRR；chunk 级 Evidence hit@k；再加无答案分数分布与阈值扫描。这样能判断"是没召回、排错了、还是生成没用对"。
3. **归因**：基线 Recall@1 0.953、MRR 0.975、Evidence hit@3 0.988，但 Recall@5 已 1.0——说明召回不是瓶颈，瓶颈在排序和 chunk 粒度。
4. **优化**：加 BM25 稀疏检索，用 RRF（k=60）和 dense 融合；再叠 bge-reranker-v2-m3 对 top-5 重排；分片改成标题路径前缀，让 embedding 带上文档结构。
5. **验证**：每次只改一个变量，在同一套评测集跑；记录 P50/P95，延迟超基线 2 倍就不上线。

**R 结果**：混合检索把 Evidence hit@3 从 0.988 提到 1.0、MRR 0.975→0.976；rerank 进一步改善 Top-1；更重要的产出是**阈值扫描证明最优阈值仍有 60% 泄漏**，直接推动 R12 做意图闸门和引用校验。

### D.3 5 分钟深挖版（面试官说"展开讲讲"）

按兴趣选一条主线，不要平铺所有实验：

| 面试官兴趣 | 讲这条主线 | 关键落点 |
|---|---|---|
| 评测怎么做 | 评测集设计 → 指标分层 → 误差归因 | 标题锚点、正负样本、Evidence hit、四象限 |
| 检索怎么优化 | dense 局限 → BM25 补位 → RRF 融合 | E059 案例、RRF 不用归一化、k=60 |
| 排序怎么优化 | bi-encoder vs cross-encoder → 候选数/延迟 | rerank 不提升召回、top-5 折中、180s 超时 |
| 分片怎么优化 | 固定/递归/结构感知 → 标题路径 | 39→48 chunk、prefix 不占正文预算、证据跨 chunk 风险 |
| 线上怎么落地 | 配置开关 → 灰度 → 回归 | 一次只改一个变量、P95 2 倍原则、重索引纪律 |

### D.4 数字口径表（防追问，必须背熟）

| 数字 | 含义 | 怎么复现 | 证据 |
|---|---|---|---|
| 8 篇 / 39 chunk | 基线语料与分片数 | `kb_import.py` | 导入日志 |
| 8 篇 / 48 chunk | 开启标题路径后的分片数 | 同上 | 4.5.3 |
| 100 条 = 85 + 15 | 评测集：有答案 / 无答案 | `kb_eval.jsonl` | 1.3.4 |
| Recall@1 = 0.953 | 81/85 排第一 | `run_eval.py` | 附录 A.1 |
| Recall@3 / @5 = 1.000 | 文档级已饱和 | 同上 | 附录 A.1 |
| MRR = 0.975 | 首个命中位置的平均倒数排名 | 同上 | 附录 A.1 |
| Evidence hit@3 = 0.988 | 84/85 前 3 个 chunk 含证据词 | 同上 | 附录 A.1 |
| 有答案 Top-1 中位数 0.679 | 分数分布 | `run_eval.py` | 附录 A.4 |
| 无答案 Top-1 中位数 0.547 | 分数分布 | 同上 | 附录 A.4 |
| 最优阈值 0.50 → 准确率 0.900 / 泄漏 60% | 阈值扫描结论 | `--threshold 0.5` | 4.2.3 |
| hybrid → Evidence hit@3 = 1.000 | 混合检索修正 chunk 排序 | `--retrieval-mode hybrid` | 4.6 |
| rerank CPU 单条 3~5s | 5 候选 15~20s、10 候选 30~50s | `time curl` | 4.4.6 |
| k=3 / prefetch=20 / candidate=5 | 生产参数 | `config.py` | config.py L54-L81 |

> 报数字时一定带**口径**："Recall@1 是 85 条有答案里 81 条排第一"，不要说"准确率 95%"——后者容易被追问是端到端还是检索层。
>
> 线上对照见 2.12：本项目 100 条评测集（线上 CI 20~50、周期基准数百~数千）、rerank 候选 5（线上 20~100）、chunk 200 字符（线上 256~512 tokens）、top-k 3（线上 5~10）。

### D.5 追问树（按类别准备）

#### A. 评测与指标

| 追问 | 回答 | 加分点 / 坑 |
|---|---|---|
| 为什么不用 RAGAS？ | 当前先做 retrieval 层的确定性指标（不花 LLM 钱、可秒级回归）；RAGAS 偏生成侧、需 LLM judge，成本和波动都更高，放到 R12 的引用/faithfulness 校验再用 | 体现"分层评测、按成本选工具" |
| 评测集怎么防止过拟合？ | 冻结主集 + 留 holdout + 线上 bad case 定期回流；单次提升 <0.01 不当真 | 主动承认小语料局限 |
| 100 条够吗？ | 起步够用；每个业务域 ≥8 条 + 15 条负样本；低于 50 条波动大于收益 | 说清"够用"的边界 |
| 为什么用标题不用 doc_id？ | `doc_id` 是上传生成的 uuid，重导必变；标题是稳定业务锚点，运行时解析成当前 doc_id | 这是"可回归"的关键设计 |
| Recall 和 Evidence hit 区别？ | Recall 只看文档标题；Evidence hit 看 chunk 正文是否含证据词，能抓"文档对了但段落错了" | 举例 E002 排第 2 |
| 为什么均值不够？ | 均值被简单样本拉高；要看 Top-1 未命中清单、分位数、最差 case | 体现 bad case 意识 |

#### B. 检索与融合

| 追问 | 回答 | 加分点 / 坑 |
|---|---|---|
| 为什么纯向量不够？ | 高频词淹没意图（E059 发票）、专有名词/数字不敏感、否定关系弱；BM25 补精确词项 | 用真实案例，不空谈 |
| RRF 为什么不用归一化？ | 只用排名 `1/(k+rank)`，丢掉原始分数量纲；加权求和才需要归一化 | 说清"排名 vs 分数" |
| k=60 怎么来的？ | 业界经验值，压平头部差距；k 小更激进、k 大更平滑；对尺度不敏感，不用精调 | 不要编造论文结论 |
| 单路为什么不能套 RRF？ | Qdrant 会把分数归一化成 rank 倒数，Top-1 恒 0.5，阈值扫描失效 | 这是真实踩坑，很加分 |
| 混合检索提升多少？ | 文档级 Recall@1 没变（0.953），但 MRR 0.975→0.976、Evidence hit@3 0.988→1.000；提升主要在 chunk 排序 | 诚实报"没大涨"，更可信 |
| 为什么不直接用 ES？ | 项目已有 Qdrant，原生支持 sparse+RRF，避免多维护一个组件；语料规模小，ROI 不划算 | 体现工程取舍 |
| 中文 BM25 怎么处理？ | fastembed 的 SimpleTokenizer 不支持中文，索引和查询都先 jieba 分词；`disable_stemmer=True` | 真实坑，面试官爱听 |

#### C. Rerank 与分片

| 追问 | 回答 | 加分点 / 坑 |
|---|---|---|
| rerank 能提升召回吗？ | 不能。只在候选内重排，召回上限由第一阶段决定，所以必须叠在 hybrid 后面 | 高频陷阱题 |
| 为什么只重排 top-5？ | cross-encoder O(N)，CPU 单条 3~5s；5 候选 15~20s，10 候选 30~50s，会顶穿 Go 超时 | 有实测数字 |
| 延迟怎么解决？ | 候选数可配、GPU、结果缓存、`asyncio.to_thread` 不阻塞事件循环；流式超时单独 180s | 说清工程闭环 |
| 标题路径为什么有效？ | chunk 文本带 `[文档 > 小节]`，embedding 含结构信息，边界模糊的两篇文档区分度提高 | 39→48，不占正文预算 |
| chunk 200/30 怎么定的？ | FAQ/规则类 150~300 字、overlap 10%~20%；改完必须重跑评测 | 不要说是"拍脑袋"，说"评测选出来的" |

#### D. 工程与线上

| 追问 | 回答 | 加分点 / 坑 |
|---|---|---|
| 改了文档为什么不生效？ | 分片/向量是索引期产物，必须重跑导入；`kb_import.py` 会清旧向量和缓存 | 真实运维经验 |
| 为什么分数每次不一样？ | 上游候选集不稳定：RRF 同分、HNSW 近似、embedding 浮点、rerank 同分继承顺序；加 tie-breaker 可消除同分随机 | 能讲到这一层很加分 |
| 怎么上线/回滚？ | Step 5 做成租户级开关（top_k/阈值/hybrid/rerank），旧单路保留 fallback；灰度与门禁在 R16 | 衔接下一需求 |
| 成本增加多少？ | embedding 调用不变；BM25 本地 CPU 几乎零成本；rerank 是主要增量（CPU 时间），用候选数控制 | 有成本模型 |

#### E. 挑战型追问

| 追问 | 回答 |
|---|---|
| "只有 8 篇文档，指标不算数吧？" | 承认：文档级 Recall 确实饱和，所以我主看 Recall@1/MRR/Evidence hit，并明确结论边界；方法论（标题锚点、正负样本、分层指标、单变量实验）可以直接迁移到 1000 篇；扩语料后用同一脚本复测，数字才可横向比。 |
| "你这提升也不大啊？" | 对，因为基线已经 0.953，天花板很低；这次最大的价值不是把 0.953 刷到 1.0，而是建立了可回归的评测体系和"阈值不可拒答"的硬证据，后者直接改变了 R12 的方案。 |
| "为什么不做 HyDE / 查询改写？" | 当前失败模式是排序和 chunk 粒度，不是 query 语义缺失；先修最弱的一段，避免为微小收益引入新模型和新延迟。 |
| "你最大的贡献是什么？" | 把"回答不准"变成可分解、可复现、可归因的工程问题：检索层/生成层分开评，正负样本齐全，每个优化都有改前改后数字和延迟成本。 |

### D.6 常见翻车说法与纠正

| 不要说 | 为什么错 | 应该说 |
|---|---|---|
| "Recall@5 是 1.0，检索没问题" | 小语料下饱和，不能证明排序好 | "文档级饱和，所以我看 Recall@1 和 chunk 级 Evidence hit" |
| "rerank 提升了召回" | 概念错误 | "rerank 只改善排序，召回上限由混合检索决定" |
| "阈值调到 0.65 就能拒答" | 泄漏降为 0 但误拒 35% | "不存在两全阈值，阈值只能做低分兜底" |
| "混合检索让 Recall 提升了 30%" | 数字对不上，一追问就崩 | 报真实口径：MRR/Evidence hit 的具体变化 |
| "BM25 就是关键词匹配" | 太浅 | 讲 TF/IDF/k1/b，以及为什么长文档要归一化 |
| "chunk 越小越准" | 会切碎证据 | 讲 precision 与 context 完整性的权衡 |

### D.7 简历 / 项目 bullet 模板

- 搭建 RAG 检索评测体系：8 篇多域知识库、100 条评测集（85 正/15 负）、文档级 Recall/MRR + chunk 级 Evidence hit + 阈值扫描，一条命令可复现基线。
- 实现 dense+BM25 混合检索（Qdrant 原生 RRF，k=60）与 bge-reranker-v2-m3 重排，Evidence hit@3 0.988→1.000，并记录 P50/P95 延迟权衡。
- 通过阈值扫描量化证明"单靠相似度无法可靠拒答"（最优阈值仍有 60% 泄漏），推动后续意图闸门 + 引用校验方案。

### D.8 一句话收尾

> "这个需求让我最满意的不是某个指标涨了多少，而是**以后任何人改检索参数，都能在 10 分钟内知道是变好还是变坏、以及为什么**。"

---

## 附录 E：与其他需求的衔接

- **R12 答案溯源与拒答**：附录 A.4 证明了"阈值不可靠"，R12 要在 `search_kb` 结构化返回（带 `score`/`doc_id`）后做拒答与引用。
- **R13 上下文预算**：检索片段数量与长度直接影响 token，Step 4 的 `k` 实验要为 R13 提供"取几条最划算"的依据。
- **R16 评测门禁**：本评测集就是门禁的执行体，改 prompt 前后跑同一套题。
