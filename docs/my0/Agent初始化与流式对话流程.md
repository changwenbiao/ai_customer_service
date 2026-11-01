# Agent 初始化与流式对话流程详解

> 以 `POST /agent/chat/stream` 的 `event_generator` 为主线，讲清楚一次流式对话从入参校验到 token 计费的 10 个步骤，其中重点展开第 5 步 `get_tenant_agent` 的懒构建逻辑，以及模型 / 工具 / 中间件 / Prompt / Checkpointer 各自的配置来源。

涉及文件：

| 文件 | 职责 |
| --- | --- |
| `../../python_agent/main.py` | FastAPI 入口、lifespan 初始化、`/agent/chat/stream`、`event_generator` |
| `../../python_agent/agent_factory.py` | Agent 构建、模型 BYOK、Qdrant/retriever、中间件、配置签名校验 |
| `../../python_agent/tools.py` | 三个工具 + app 注入模式 |
| `../../python_agent/db_ops.py` | 会话元数据、Prompt、配额、token 用量 |
| `../../python_agent/checkpoint_safe.py` | 并发安全 + 断线重连的 Checkpointer 包装 |
| `../../python_agent/config.py` | 环境变量与检索参数 |

---

## 一、整体时序

```
Go 侧 POST /agent/chat/stream  (SSE)
        │
        ▼
agent_chat_stream → StreamingResponse(event_generator())
        │
        ├─1 校验 thread_id / user_message
        ├─2 tenant_from_thread(thread_id)         租户 = thread_id 前缀
        ├─3 写 ContextVar（tenant/thread/pending_ticket）
        ├─4 并发会话配额校验（调 Go /api/agent/tenant-concurrency）
        ├─5 get_tenant_agent(app, tenant_id)      ★ 懒构建 / 缓存命中 / 签名校验
        ├─6 config = {"configurable": {"thread_id": ...}}
        ├─7 touch_session_retry                  会话心跳 +1
        ├─8 agent.astream(stream_mode="messages") 逐 token yield
        ├─9 aget_state 检查 interrupts
        └─10 _record_usage_after_chat             token 计费
```

`event_generator` 是异步生成器：调用时不执行函数体，由 `StreamingResponse` 通过 `await anext()` 反复拉取，每 `yield` 一次就向客户端写一段 `data: {...}\n\n`。

---

## 二、10 个步骤逐一说明

### 步骤 1：参数校验

**这一步干什么**：在真正消耗模型额度、查库、建向量库之前，先把明显不合法的请求挡掉。`thread_id` 为空会导致无法定位会话历史，`user_message` 为空会让模型空转并照样计费。

```python
if not thread_id or not user_message:
    yield f"data: {_json.dumps({'type':'error','msg':'thread_id 和 user_message 不能为空'}, ensure_ascii=False)}\n\n"
    return
```

- 校验放在生成器内部，因为此时 HTTP 200 + `text/event-stream` 响应头已经发出，无法再改状态码。
- `return` 在异步生成器里等价于 `StopAsyncIteration`，是正常结束而非异常。

### 步骤 2：解析租户

**这一步干什么**：后续所有"按租户隔离"的资源——模型（BYOK）、系统 Prompt、Qdrant 集合、配额、计费——都以这个 `tenant_id` 为 key，所以必须最先确定。它决定了"这次对话用谁的钱、谁的知识库、谁的提示词"。

```python
tenant_id = tenant_from_thread(thread_id)
```

`db_ops.py` 实现：`thread_id` 形如 `{tenant_id}:{uid}`，按第一个 `:` 切分。没有前缀则为空串（走 `kb_default` 集合与全局 Prompt）。

> 注意：这里只做解析，不做归属校验。租户边界由 Go 侧鉴权保证，Python 内部接口依赖 `X-Internal-Token` 信任。

### 步骤 3：写入 ContextVar

**这一步干什么**：把"这次请求属于谁、哪个会话、有没有排队工单"这三个信息挂到当前协程上下文里。因为工具函数是全局单例、被所有租户共用，只能靠上下文区分调用方，否则并发时 A 租户的问题会去查 B 租户的知识库。

```python
tenant_var.set(tenant_id)
thread_var.set(thread_id)
pending_ticket_var.set(req.has_pending_ticket)
```

三个工具是**全局共享**的单例函数，靠 ContextVar 才能在并发请求中区分租户、会话、是否有排队工单。这些值在工具被调用的同一协程链中可见。

### 步骤 4：并发会话配额

**这一步干什么**：按租户限制"同时正在进行多少路对话"，防止某个客户端的连接数把模型额度、PG 连接池、向量库额度打满，也防止单租户故障影响其他租户。它检查的是**在线连接数**，不是消息数，所以是"并发"限制而非频率限制。

```python
async with _pool_conn(request.app) as conn:
    quota_err = await check_concurrent_sessions(conn, request.app, tenant_id)
```

- 先从 `platform_settings` 读租户的 `max_concurrent_sessions`；`<= 0` 表示不限。
- 再实时向 Go 请求 `/api/agent/tenant-concurrency`（WS Hub 统计的在线连接数）。
- 达到上限返回 `429` 语义的 msg，直接 `yield error` 结束。
- 请求 Go 失败只记 warning、放行（配额是保护性限制，不阻断业务）。

### 步骤 5：获取 Agent（重点，见下一章）

**这一步干什么**：拿到"这个租户专属的、已经装配好模型/工具/Prompt/记忆"的 LangGraph 图。它同时承担四件事——①发现配置是否变更、②命中缓存直接复用、③首次使用时懒构建（向量库、retriever、Prompt、模型）、④构建结果缓存起来。之所以做成懒构建而不是启动时为所有租户建好，是为了避免"一万个租户各建一份图 + 各建一个 Qdrant 集合"的启动开销。

```python
agent = await get_tenant_agent(request.app, tenant_id)
```

### 步骤 6：构造 LangGraph 运行配置

**这一步干什么**：告诉 LangGraph "这次执行属于哪个会话线程"。图本身是共享的，但状态（历史消息、工具往返、中断点）是按 `thread_id` 存放的——同一个图对象被多个 thread 复用，靠这个 key 区分。如果这里传错，用户会看到别人的历史或丢失上下文。

```python
config = {"configurable": {"thread_id": thread_id}}
```

`thread_id` 是 Checkpointer 的主键，决定：
- 历史消息从哪里恢复（`AsyncPostgresSaver` 按 thread 存 checkpoint）
- `interrupt` 状态挂在哪个 thread 上
- `/agent/resume` 恢复的是哪个会话

### 步骤 7：会话心跳

**这一步干什么**：把这次对话记为"刚刚活跃过"，供三个后台机制使用——会话回收任务据此判断哪些 thread 已长时间无人使用可以删；配额/统计能看出活跃度；运维排查时能确认会话确实在流转。

```python
await touch_session_retry(request.app, thread_id)
```

`touch_session` 是 `INSERT ... ON CONFLICT DO UPDATE`（幂等）：`last_active_at = now()`，`message_count + 1`。外层 `_retry` 对瞬时段连错误（PG 重启残留死连接）重取连接重试一次。

这一步同时为「会话回收」和「配额/统计」提供数据。

### 步骤 8：流式调用 LangGraph

**这一步干什么**：真正执行图。模型逐 token 生成、工具按需调用、每产出一段文本就立刻推给前端，让用户不用等到全部生成完才看到回复。同时把整轮的消息收集起来，供最后的计费和结果归档。

```python
async for msg, metadata in agent.astream(
    {"messages": [HumanMessage(content=user_message)]},
    config=config,
    stream_mode="messages",
):
    stream_msgs.append(msg)
    if hasattr(msg, "content") and msg.content and not isinstance(msg, HumanMessage):
        chunk = msg.content if isinstance(msg.content, str) else str(msg.content)
        if chunk:
            full_content += chunk
            yield f"data: {_json.dumps({'type':'chunk','content':chunk}, ensure_ascii=False)}\n\n"
```

要点：

- `stream_mode="messages"` 是 LangGraph 的消息级流式，逐 token 产出 `(message_chunk, metadata)`。
- 过滤器只推非空的 AI 内容，因此 **tool_calls 的分片（content 为空）不会显示**，工具执行过程对用户不可见。
- `metadata` 拿到了但未使用；本来可以用 `metadata["langgraph_node"]` 区分 `model` / `tools` 节点。
- `full_content` 跨 `yield` 保留在栈帧里，用于最后 `end` 事件；`stream_msgs` 收集全部消息用于计费。
- 未传 `recursion_limit`，使用 LangGraph 默认值 25（约 12 轮工具往返）。

### 步骤 9：中断检测

**这一步干什么**：判断这一轮是否需要"停下来等人"。比如模型调用了 `transfer_to_human`，图会在工具节点挂起等待人工确认；这里读图状态就能知道有没有挂起、挂起的载荷是什么（转接原因、工单号等）。之所以要单独检查，是因为中断时流已经结束，前端需要的是 `interrupt` 事件而不是普通的 `end` 事件。

```python
state = await agent.aget_state(config)
if state.tasks and state.tasks[0].interrupts:
    is_interrupt = True
    interrupt_data = state.tasks[0].interrupts[0].value
```

中断不是从流里识别，而是流结束后**读图状态**判断。`transfer_to_human` 里调用 `interrupt()` 时，图在 tools 节点挂起，状态被 Checkpointer 落库。

- 有中断 → `yield {"type":"interrupt","data":...}`
- 无中断 → `yield {"type":"end","content":full_content}`

Go 侧收到 `interrupt` 会建工单，等人工处理完再调 `/agent/resume`，用 `Command(resume={confirmed, wait_time, ticket_id})` 继续执行。

### 步骤 10：token 计费

**这一步干什么**：把这一轮消耗的输入/输出 token 按当时的单价落库，用于出账、成本核算和租户用量监控。之所以用"单价快照"，是因为调价频繁，历史账单必须按当时价格计算才可追溯；之所以区分缓存命中/未命中，是因为不同计费档价格不同。

```python
async with _pool_conn(request.app) as conn:
    await _record_usage_after_chat(request, conn, tenant_id, thread_id, stream_msgs)
```

- `extract_usage_metadata` 从**最后一条带 `usage_metadata` 的 AI 消息**取用量。
- 同时解析 `response_metadata.usage` 里的 `prompt_cache_hit_tokens / prompt_cache_miss_tokens`（DeepSeek/通义等支持，缺省按 0 计）。
- 单价来自 `current_chat_prices`（模型配置里的三类单价）并按当时价格快照落库，后续调价不影响历史账单。

> 已知副作用：这段代码在 `async for` 之后。若客户端中途断开，Starlette 会在某个 `yield` 处取消任务，循环之后的计费逻辑不会执行，这一轮用量不入库。

---

## 三、第 5 步深挖：`get_tenant_agent`

```python
async def get_tenant_agent(app, tenant_id):
    sig_version = await _check_config_version(app, tenant_id)   # ① 配置签名懒检查
    if tenant_id in app.state.tenant_agents:                    # ② 缓存命中
        return app.state.tenant_agents[tenant_id]
    await ensure_tenant_vector(app, tenant_id)                  # ③ 向量库
    get_tenant_retriever(app, tenant_id)                        # ④ retriever
    async with app.state.db_pool.connection() as conn:
        prompt = await load_active_prompt(conn, tenant_id)      # ⑤ Prompt
    model = await get_tenant_model(app, tenant_id)              # ⑥ 模型（BYOK）
    agent = create_agent(...)                                   # ⑦ 建图
    app.state.tenant_agents[tenant_id] = agent                  # ⑧ 缓存
    _agent_sigs[tenant_id] = sig_version                        # ⑨ 记录签名
    return agent
```

调用顺序不可调换：③④ 决定工具能否检索，⑤ 决定 system_prompt，⑥ 决定模型，⑦ 才把四者装进图。

### ① 配置签名懒检查（广播丢失兜底）

**这一步干什么**：确认"租户的模型/Prompt 配置有没有在别处被改过"。后台改配置时 Go 会通过 Redis 广播通知各 Agent 实例清缓存，但广播可能丢（实例重启、网络抖动、订阅断开），所以这里用一个 `config_version` 签名做最终一致兜底：拉到的版本与构建 Agent 时记录的不一致，就清缓存让下一步走重建路径。

```python
_sig_check_ts[tenant_id] = now       # 10 秒 TTL
remote = await _fetch_config_version(app, tenant_id)   # GET Go /api/agent/model-config?tenant_id=...
if remote != _agent_sigs.get(tenant_id):
    clear_model_cache(tenant_id, "all")
    app.state.tenant_agents.pop(tenant_id, None)
    app.state.tenant_retrievers.pop(tenant_id, None)
```

- 主路径是 Redis 广播即时失效；这里是**最终一致兜底**，最多延迟 10 秒发现配置变化。
- 拉取失败（Go 不可用/超时 5s）只记 warning 并跳过本轮，不阻断聊天。
- TTL 内直接返回 `None`，不请求 Go，保证每轮对话最多只多一次外部调用。

### ② 缓存命中

**这一步干什么**：绝大多数请求都从这里直接返回，省掉建图、查库、建 embedding 的开销。图构建一次成本不低（要读 PG Prompt、可能创建 Qdrant 集合、创建模型客户端），而它本身无状态（状态都在 checkpointer 里），所以可以安全复用。

`app.state.tenant_agents` 是 `tenant_id -> CompiledGraph` 的字典。命中就直接返回**同一个图对象**。

因为 `create_agent` 把模型实例、工具列表、prompt、checkpointer 都固化进图里，所以这些配置的任何变更都必须清缓存重建——只改 DB 而不清缓存是不生效的。

### ③ `ensure_tenant_vector`：向量库与集合

**这一步干什么**：为这个租户准备好"语义检索的底座"——用它的 embedding 模型建好 Qdrant 集合，并封装成 LangChain 的 VectorStore。没有它，`search_kb` 工具就无库可查；有了它，知识库上传/检索都走同一个集合，天然实现租户隔离。

| 动作 | 细节 |
| --- | --- |
| 取租户 embedding | `get_tenant_embeddings`：自定义配置优先，否则平台默认（`t_admin` 的配置） |
| 集合命名 | `kb_{tenant_id}`，无租户用 `kb_default` |
| 集合不存在则创建 | `VectorParams(size=vec_dim, distance=Distance.COSINE)` |
| 维度来源 | 启动时探测一次 `app.state.vec_dim`；若为 0（启动时 embedding 不可用）则用租户 embedding **实时探测**并回写 |
| 缓存 | `app.state.tenant_stores[tenant_id] = QdrantVectorStore(...)` |
| 失败降级 | 记 `_vec_fail_ts`，60 秒冷却期内直接返回 `None`，**不抛异常**，对话退化为纯 LLM |

`OpenAIEmbeddings` 的两个非默认参数：
- `check_embedding_ctx_length=False`：不按 OpenAI 分词器校验上下文长度（第三方 embedding 常见超 3072 token 的误判）
- `chunk_size=10`：批量请求条数

### ④ `get_tenant_retriever`

**这一步干什么**：把 VectorStore 包装成"检索器"，并固定检索策略与条数。工具只需要"给一个 query，拿回几条文档"，不必关心向量库怎么连、用哪个 embedding、取几条。

```python
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_TOP_K})
```

- `RETRIEVER_TOP_K = 3`（`config.py`），即每轮检索返回 3 条。
- store 为 `None`（降级）时返回 `None`，此时 `search_kb` 会返回"知识库暂未就绪"，不会抛异常。
- retriever 缓存到 `app.state.tenant_retrievers`，`search_kb` 通过 `ContextVar` 拿到 `tenant_id` 后从这里取。

### ⑤ `load_active_prompt`：Prompt 三级回落

**这一步干什么**：取这次对话要用的系统提示词（角色设定、话术规范、业务约束）。三级回落保证"永远不会没有 Prompt"：租户自定义 → 平台全局 → 代码内置默认。这样新租户开箱即用，又有能力覆盖平台策略。

```sql
-- 1) 租户自有激活 Prompt
SELECT content FROM prompts WHERE tenant_id = %s AND is_active = true ORDER BY updated_at DESC LIMIT 1
-- 2) 全局激活 Prompt（tenant_id = ''）
SELECT content FROM prompts WHERE tenant_id = '' AND is_active = true ORDER BY updated_at DESC LIMIT 1
-- 3) 都没有 → DEFAULT_SYSTEM_PROMPT（代码常量）
```

- 启动时 `seed_default_prompt` 保证全局至少有一条激活记录。
- 多版本由 `key` 区分，同一租户同时只有一条 `is_active = true`（`upsert_prompt` 会把其他条置 false）。

### ⑥ `get_tenant_model`：BYOK 决策

**这一步干什么**：决定"这次对话用哪个模型、谁的 key、哪个网关"。这是多租户 SaaS 的核心商业逻辑——客户可以自带模型（BYOK）省钱或用自有额度，没配的走平台默认模型由平台承担成本。

```
get_tenant_model_config(tenant_id)      ← 缓存 300s（_MODEL_CONFIG_TTL）
        │  HTTP GET Go /api/agent/model-config
        ▼
  use_custom == false ──► get_platform_model(app)   ← 即 t_admin 的配置（平台默认）
        │
        └─ true ──► init_chat_model(model, model_provider="openai", api_key, base_url, ...)
```

关键参数（平台默认与租户自定义**完全一致**）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `model_provider` | `"openai"` | 统一走 OpenAI 兼容协议，靠自定义 `base_url` 接第三方/自建网关 |
| `temperature` | `0` | 客服场景要求确定性，避免发散 |
| `max_tokens` | `1024` | 单轮输出上限，也是成本上限 |
| `timeout` | `30` | 单次模型调用超时（非整轮对话） |
| `max_retries` | `0` | 429/5xx 不重试，直接抛，由 `_friendly_llm_error` 映射为"模型服务当前繁忙" |

缓存设计：

- 实例缓存 key 形如 `_model_{tenant}:{model}:{base_url}:{api_key[-4:]}`，换模型/换网关/换 key 都会命中新实例。
- 配置缓存 key 直接用 `tenant_id`，300 秒过期；配错 key 不会回退到平台模型，而是直接抛 `RuntimeError`（避免"悄悄用错模型"）。
- 平台默认模型在 lifespan 里预热到 `app.state.model`，失败则降级启动，请求时再懒构建重试。

### ⑦ `create_agent`：把 5 样东西装进图

**这一步干什么**：把前面准备好的四样东西（模型、工具、Prompt、checkpointer）加上内容守卫，装配成一个可执行、可中断、可恢复的 Agent 图。这一步之后得到的是一个 `CompiledGraph`，后续 `astream` / `aget_state` / `Command(resume=...)` 都在它上面调用。

```python
create_agent(
    model=model,                          # 租户级模型实例（BYOK）
    tools=app.state.tools,                # 全局唯一的 3 个工具
    middleware=[content_guard],           # before_model 内容守卫
    checkpointer=app.state.checkpointer,  # 全局单例 PG Checkpointer
    system_prompt=prompt,                 # 租户 Prompt（DB）
)
```

生成的图（由 `langchain.agents.factory` 内部用 `StateGraph` + `ToolNode` 组装）：

```
START → before_model(content_guard) → model 节点
             ↑                          │ tool_calls
             │                          ▼
           tools 节点 ◄──────────────────┘
             │ 无 tool_calls
             ▼
            END
```

- `interrupt()` 在 tools 节点内触发挂起，checkpoint 写入 PostgreSQL。
- `Command(resume=...)` 从断点续跑，这就是 `/agent/resume`。

### ⑧⑨ 缓存与签名记录

**这一步干什么**：把刚建好的图存起来供后续请求复用，同时记下"这张图是按哪个配置版本构建的"。前者是性能，后者是可观测性——没有它，签名检查就无法判断"配置变了"（会误判或漏判，导致重建风暴或不更新）。

- `app.state.tenant_agents[tenant_id] = agent`
- `_agent_sigs[tenant_id] = sig_version`：若签名检查因 TTL 跳过返回 `None`，则从模型配置缓存里的 `config_version` 补记，避免 10 秒后误判为"变化"而多重建一次。

---

## 四、四类配置的最终来源总表

| 配置项 | 来源 | 变更后如何生效 |
| --- | --- | --- |
| 模型 / key / base_url | Go `/api/agent/model-config`（DB） | Redis 广播立即清缓存；10s 签名检查兜底 |
| Embedding | 同上（`embedding` 子配置） | 同上；集合维度变化需重建 collection |
| Prompt | PG `prompts` 表 | `/prompt` 接口调 `rebuild_tenant_agent`；平台级变更只影响无自有 Prompt 的租户 |
| 工具 | 代码固定 `[search_kb, query_order, transfer_to_human]` | 需重新发布 |
| 中间件 | 代码固定 `[content_guard]` | 需重新发布 |
| Checkpointer | lifespan 单例 `SafeAsyncPostgresSaver` | 无需变更 |
| 检索参数 | `config.py`：`k=3`、chunk 200/30 | 需重启 |

---

## 五、工具与 ContextVar 的关系（多租户并发的关键）

`tools.py` 采用 **app 注入模式**：

```python
# main.py
tools.set_app(app, logger, tenant_var, pending_ticket_var, thread_var)

# tools.py
@tool
async def search_kb(query: str) -> str:
    tid = _tenant_var.get() or ""
    retriever = _app.state.tenant_retrievers.get(tid)
```

推论：

1. 工具函数只有一份，工具 schema 里**没有租户参数**，LLM 无从得知租户。
2. 租户隔离完全依赖 ContextVar 在协程链中的传递。
3. 若在工具内部再用 `asyncio.create_task` / 线程池，ContextVar 可能丢失或复制，`get()` 会拿到空串——目前三个工具都没这么做，属于必须遵守的约束。

三个工具的行为：

| 工具 | 行为 | 失败策略 |
| --- | --- | --- |
| `search_kb` | Redis 缓存 → `retriever.ainvoke(query)` → 写 `kb_hits`（命中明细/未命中都记） | 异常→"知识库暂未就绪"字符串返回，不抛 |
| `query_order` | httpx GET Go `/api/business/order`，5s 超时 | 异常→字符串返回 |
| `transfer_to_human` | `interrupt({action, reason, message})`；已有 pending 工单则只提示等待 | — |

---

## 六、Checkpointer：为什么需要包装

`SafeAsyncPostgresSaver`（`checkpoint_safe.py`）解决两个真实问题：

1. **并发写**：`AsyncPostgresSaver` 只有一条常驻 psycopg 连接，多租户并发 `aput` 会报 `another command is already in progress` → 内部加锁串行化。
2. **断线**：PG 重启后常驻连接变 closed → 检测到 `conn.closed` 时按 `dsn` 重建连接并重跑 `setup()`；瞬时错误自动重试一次。

所有租户共用这个实例，隔离维度仍是 `thread_id`。

会话清理：`session_cleanup_loop` 按 `SESSION_RETENTION_DAYS` 找到过期 thread，调 `checkpointer.adelete_thread(tid)` 删 checkpoint，再删 `session_meta`。

---

## 七、常见坑与风险点

1. **`build_agent()` 是死代码**：`agent_factory.py` 里的 `build_agent` 只写 `app.state.agent`，聊天路径完全不读它；实际使用的是 `get_tenant_agent`。
2. **改配置不清缓存 = 不生效**：模型/prompt 都被 `create_agent` 固化，必须清 `tenant_agents`（签名检查里是 `clear_model_cache` + `pop` 一起做）。
3. **客户端断开会吞掉计费**：`_record_usage_after_chat` 在循环之后，取消任务时不会执行；如需精确计费应挪进 `finally` 或流中周期记录。
4. **429 与连接失败未区分**：`max_retries=0` 让限流立即暴露，但错误文案统一成"模型服务当前繁忙"，排查时需看上游原始响应。
5. **`tenant_id` 未校验归属**：信任边界在 Go 侧，Python 内部接口只认 `X-Internal-Token`。
6. **工具调用对用户不可见**：流里过滤了空 content，前端看不到"正在检索知识库"，需要额外事件才能做进度提示。
7. **恢复轮次非流式**：`/agent/resume` 用 `ainvoke` 同步返回，人工确认后的回复是一次性吐出。

---

## 八、关键文件与行号索引

| 位置 | 内容 |
| --- | --- |
| `../../python_agent/main.py` | `agent_chat_stream` 与 `event_generator`（约 366-447 行） |
| `../../python_agent/main.py` | lifespan 初始化（约 112-225 行）：连接池、checkpointer、qdrant、模型、工具、后台任务 |
| `../../python_agent/agent_factory.py` | `content_guard`（27-40）、`get_tenant_agent`（429-458）、`get_tenant_model`（344-388）、`ensure_tenant_vector`（69-108）、`get_tenant_retriever`（111-123）、`_check_config_version`（487-509） |
| `../../python_agent/tools.py` | `set_app`（21-28）、`search_kb`（57-94）、`query_order`（97-111）、`transfer_to_human`（114-132） |
| `../../python_agent/db_ops.py` | `tenant_from_thread`（18-22）、`touch_session`（60-70）、`check_concurrent_sessions`（232-252）、`load_active_prompt`（255-267） |
| `../../python_agent/checkpoint_safe.py` | `SafeAsyncPostgresSaver`（23 起） |
