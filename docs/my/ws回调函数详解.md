# ws.go 189-247 行四个回调函数详解

> 文件：[ws.go](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/handler/ws.go)
> 行号：189-247
> 所处上下文：`HandleUserWS` 的 `"chat"` 分支内，一个 `go func(){...}()` 子协程里
> 调用方：[CallAgentChatStream](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/pkg/agent/client.go#L128-L130)

## 一、总体定位

这四个不是普通函数，而是**匿名闭包（closure）**，作为回调参数传给 `agent.CallAgentChatStream`。它们捕获了外层作用域的 `userID`、`fullContent`、`traceID` 等变量，在 `client.go` 的 SSE 解析循环里被逐个调用。

```
client.go SSE 读取循环                 ws.go 回调闭包
─────────────────────                  ─────────────────
收到 "chunk" 事件 ──────────────────►  onChunk(chunk)
收到 "end"   事件 ──────────────────►  onEnd(content)
收到 "error" 事件 ──────────────────►  onError(err)
收到 "interrupt" 事件 ──────────────►  onInterrupt(data)
```

调用链：

```
HandleUserWS (WS主循环)
  └─ go func() { defer UnlockThread() }   ← 子协程，不阻塞主循环
      └─ agent.CallAgentChatStream(ctx, ..., onChunk, onEnd, onError, onInterrupt)
          └─ httpClient.Do(req)            ← 阻塞直到响应头
              └─ reader.Read 循环          ← 逐块阻塞读 SSE
                  ├─ "chunk" → onChunk(chunk.Content)
                  ├─ "end"   → onEnd(chunk.Content); return
                  ├─ "error" → onError(...); return
                  └─ "interrupt" → onInterrupt(chunk.Data); return
```

**关键机制**：四个回调里，`onEnd` 和 `onError`、`onInterrupt` 是**互斥**的——`client.go` 在收到 `end`/`error`/`interrupt` 任意一个后立刻 `return`，所以一次流式对话只会走到其中一个收尾回调。这意味着：触发中断时，`onEnd` 不会被调用，`onChunk` 累积的 `fullContent` 不会落库。

---

## 二、onChunk（189-194 行）：逐 token 推送

```go
func(chunk string) {
    fullContent += chunk
    ws.WSHub.SendToUser(userID, map[string]interface{}{
        "type": "chat_chunk", "chunk": chunk,
    })
},
```

### 干了什么

1. **累加全文**：`fullContent += chunk`
   - `fullContent` 是子协程闭包变量（在 `go func()` 外层声明），每次回调在同一个协程内被顺序调用，不存在并发写问题。
   - 用途：给 `onEnd` 兜底——如果 Python 侧没传完整 `content`，Go 侧用累加结果。
2. **实时推送**：`SendToUser(userID, {type:"chat_chunk", chunk})`
   - 只投递到该用户 WS 连接的 `send` 通道，由 `WriteLoop` 串行写 socket。
   - 前端收到后把 `chunk` 追加到气泡末尾，实现逐字输出。

### 触发条件

SSE 事件 `type == "chunk"` 且 `content` 非空（`client.go` 里只有 `msg.content` 非空才产出 chunk）。注意：模型输出 `tool_calls` 时 `content` 为空，不会触发 `onChunk`，所以工具调用过程对用户不可见。

### 副作用

- 修改闭包变量 `fullContent`
- 向用户端 WS 推一条 `chat_chunk` 消息

---

## 三、onEnd（196-205 行）：流结束，落库 + 推送完整内容

```go
func(content string) {
    if htk, _ := repository.GetHandlingTicket(userID); htk != nil {
        _ = repository.SaveTicketMessage(userID, "ai", content, "text", htk.ID)
    } else {
        _ = repository.SaveMessage(userID, "ai", content, "text")
    }
    ws.WSHub.SendToUser(userID, map[string]interface{}{
        "type": "chat_end", "msg": content,
    })
},
```

### 干了什么

1. **判断是否在人工接管中**：`GetHandlingTicket(userID)`（查 `tickets` 表 `status = handling`）
   - 该函数只查 `status = handling` 的工单（[store.go](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/repository/store.go#L76-L87)）。
   - 有 `handling` 工单 → 这条 AI 回复**归属到工单**：调用 `SaveTicketMessage` → INSERT 进 **`messages` 表**，`ticket_id` 字段填工单 ID，`role=ai`。这样客服在工单历史里能看到 AI 最后说了什么。
   - 没有 → 普通线程级落库：`SaveMessage` → INSERT 进 **`messages` 表**，`ticket_id` 为 NULL。
2. **推送完整内容**：`SendToUser(userID, {type:"chat_end", msg: content})`
   - 前端收到 `chat_end` 后，用 `msg` 替换掉之前逐字累加的内容（权威全文，避免分片丢字），并标记本轮回复结束。

### 参数 `content` 从哪来

Python 侧 SSE 的 `end` 事件带了完整 `content`；`client.go` 把它透传过来。`fullContent` 此时没有被使用——如果上游 `content` 为空，这里会存空字符串（潜在 bug：应该 `if content == "" { content = fullContent }` 兜底）。

### 触发条件

SSE 事件 `type == "end"`。**注意：触发 `interrupt` 时不会走到这里**，因为 `client.go` 在 `interrupt` 分支直接 `return` 了。

### 副作用

- 数据库：INSERT 一条 `role=ai` 的消息进 **`messages` 表**（`ticket_id` 为工单 ID 或 NULL）
- 向用户端 WS 推一条 `chat_end` 消息

---

## 四、onError（207-211 行）：业务错误推送

```go
func(err error) {
    ws.WSHub.SendToUser(userID, map[string]interface{}{
        "type": "error", "msg": "调用agent失败:" + err.Error(),
    })
},
```

### 干了什么

1. 把 Python 侧 SSE `error` 事件里的 `msg` 包装成 `调用agent失败: xxx` 推给前端。

### 触发条件

SSE 事件 `type == "error"`。**这是唯一的业务错误出口**——`CallAgentChatStream` 自身返回的网络错误（连接失败、超时）走的是外层 `streamErr` 判断（在 `onInterrupt` 之后的代码），而不是这个回调。

### 注意点

- 错误**不落库**，只推前端。如果需要审计模型调用失败，要在这里加日志或写表。
- 429 限流、模型余额不足等上游错误目前混在同一个文案里，前端无法区分。

---

## 五、onInterrupt（213-247 行）：转人工 + 建工单 + 恢复图

这是四个回调里最复杂的，分四步。

### 第 1 步：提取转人工原因（214-219 行）

```go
reason := ""
if interruptData != nil {
    if r, ok := interruptData["reason"].(string); ok {
        reason = r
    }
}
```

`interruptData` 来自 Python 的 `interrupt({action, reason, message})`。这里用 type assertion 安全取 `reason` 字段，取不到就置空——空 reason 不阻塞建单。

### 第 2 步：建单或复用（220-226 行）

```go
t, reused, err := service.EnsureTicket(userID, userID, reason)
if err != nil {
    ws.WSHub.SendToUser(userID, map[string]interface{}{
        "type": "error", "msg": "创建工单失败:" + err.Error(),
    })
    return
}
```

`EnsureTicket(threadID, userID, reason)` 内部（[ticket_service.go](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/service/ticket_service.go#L127-L189)）：

1. **行锁查现有工单**：`SELECT ... FROM tickets WHERE thread_id=? AND status IN (pending, handling) FOR UPDATE`
   - 有 → 复用，只 UPDATE `reason` 字段，返回 `reused=true`
2. **新建**：INSERT 进 **`tickets` 表**，`ticket_no = "TK-" + uuid[:8]`，`status=pending`；同时 INSERT 一条 `action=create` 进 **`ticket_logs` 表**
3. **自动分配**：`PickIdleAgent(tenantID)` 查 `agent_status` 表找同租户在线且处理量最少的子账号客服
   - 找到 → UPDATE `tickets` 状态为 `handling`、绑 `handler_uid/handler_name`、`agent_status.handling_count +1`；INSERT 一条 `action=assign` 进 **`ticket_logs` 表**
   - 找不到 → 保持 `pending`，等客服手动接管

**事务 + 行锁**保证 WS 通道和 HTTP 通道同时触发转人工时不会建出两张单。

失败时推 `error` 并 `return`（不会继续往下推 ticket、也不会 resume）。

### 第 3 步：新建工单的后续动作（227-239 行，仅 `!reused` 时执行）

```go
if !reused {
    // 系统消息
    _ = repository.SaveTicketMessage(userID, "system", "已提交人工客服，工单号："+t.TicketNo, "text", t.ID)
    // 把用户那条"转人工"诉求消息归属到新工单
    _ = repository.AssignLastUserMessage(userID, t.ID)
    // 广播给同租户所有客服端
    ws.WSHub.SendToAdminsByTenant(auth.TenantFromUserID(userID), map[string]interface{}{
        "type": "new_ticket",
        "ticket": map[string]interface{}{...},  // id, ticket_no, thread_id, user_id, reason, status, created_at
    })
}
```

- **系统消息**：`SaveTicketMessage` → INSERT 进 **`messages` 表**，`role=system`、`ticket_id=新工单ID`，内容"已提交人工客服，工单号：TK-xxx"。让用户和客服在会话历史里看到这条提示。
- **AssignLastUserMessage**：UPDATE **`messages` 表**中该线程最新一条 `role=user` 且 `ticket_id IS NULL` 的消息，把 `ticket_id` 设为新工单 ID（[store.go](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/go_service/internal/repository/store.go#L68-L73)）。这样客服打开工单能看到用户的转人工诉求原文。
- **new_ticket 广播**：推给该租户所有在线客服端，客服列表实时出现新工单。

### 第 4 步：通知用户 + 恢复 Agent 图（240-246 行）

```go
ws.WSHub.SendToUser(userID, map[string]interface{}{
    "type": "ticket", "ticket_no": t.TicketNo,
    "reused": reused, "msg": "已提交人工客服，请等待处理",
})
_, _ = agent.CallAgentResume(agent.ResumeReq{
    ThreadID: userID, Confirmed: true, WaitTime: 3, TicketID: t.TicketNo,
}, traceID)
```

- **推 `ticket` 事件**：前端切换到"等待人工"状态，显示工单号。`reused=true` 时前端可提示"您已提交过工单"。
- **CallAgentResume**：调 Python `/agent/resume`，传 `confirmed=true, wait_time=3, ticket_id=TK-xxx`。
  - Python 侧用 `Command(resume={...})` 从挂起的 checkpoint 继续执行（[main.py](file:///Users/changwenbiao/PycharmProjects/ai_customer_service/python_agent/main.py#L470-L477)），让 `interrupt()` 返回该值，工具函数据此生成"已为您转接人工客服，预计等待 3 分钟"的文本，模型收尾，图跑完。
  - **返回值被丢弃**（`_, _ =`）：模型说的"预计等待 3 分钟"不会推给用户，用户实际看到的是上一步 `ticket` 事件的"已提交人工客服，请等待处理"。

### 触发条件

SSE 事件 `type == "interrupt"`，即模型调用了 `transfer_to_human` 工具且 Python 侧 `aget_state` 检测到中断。

### 副作用汇总

- **`tickets` 表**：新建或复用工单（INSERT 或 UPDATE reason/status/handler）
- **`ticket_logs` 表**：写 `create` / `assign` 状态流转日志
- **`messages` 表**：新建时 INSERT 一条 `role=system` 系统消息；UPDATE 最新用户消息的 `ticket_id`
- WS：推 `error`（失败）/ `new_ticket`（客服端）/ `ticket`（用户端）
- 外部：调 Python `/agent/resume` 恢复 LangGraph 图

---

## 六、四个回调的协作关系

```
正常对话:
  onChunk(×N) ──► onEnd(full)    ← 落库 + chat_end

转人工:
  onChunk(×N) ──► onInterrupt(reason)
                    ├─ EnsureTicket (建单/复用 + 自动分配)
                    ├─ [新建时] 系统消息 + AssignLastUserMessage + new_ticket广播
                    ├─ 推 ticket 给用户
                    └─ CallAgentResume (图继续跑完，回复丢弃)
  注意: onEnd 不会被调用，fullContent 不落库

出错:
  onChunk(×N?) ──► onError(err)   ← 只推 error，不落库
```

## 七、几处值得注意的设计点

1. **`fullContent` 在中断路径被丢弃**：`onInterrupt` 不调用 `onEnd`，所以模型在调 `transfer_to_human` 之前说的"好的，我帮您转接"不会进 `messages` 表。人工客服看到的 AI 最后一句话是上一轮的。
2. **`onEnd` 没有用 `fullContent` 兜底**：如果 Python 的 `end` 事件 `content` 为空（理论上不会），会存空字符串。建议 `if content == "" { content = fullContent }`。
3. **`onError` 不区分错误类型**：429、超时、模型配置缺失都走同一个 `调用agent失败: xxx`，前端无法差异化提示。
4. **`reused=true` 时不推 `new_ticket`**：复用已有工单意味着客服端已经在列表里了，无需重复通知；但 `ticket` 事件仍推给用户，让用户确认"已受理"。
5. **`EnsureTicket` 行锁只锁查询行**：若该 thread 从未有过工单，`FOR UPDATE` 无行可锁，并发场景下可能两个请求都查不到然后各建一单——靠 `thread_id + status IN (pending,handling)` 的唯一约束（若有）兜底，否则需应用层 `threadLocks` 补充（目前转人工路径没加 `TryLockThread`，存在极小概率双建单）。
6. **`CallAgentResume` 失败被忽略**：图恢复失败只在 Python 日志里体现，Go 侧不推错误；此时用户已看到"已提交"，但图实际挂起，下次用户发消息时 LangGraph 会因 checkpoint 仍在中断态而行为异常（需手动关单或再次 resume）。
