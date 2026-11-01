package service

import (
	"os"
	"strconv"
	"time"

	"customer_service/internal/model"
	"customer_service/internal/pkg/logger"
	"customer_service/internal/pkg/ws"
	"customer_service/internal/repository"
)

// StartSlaMonitor 启动工单 SLA 超时提醒扫描。
// 每分钟扫描一次 pending 且超过阈值未接管的工单，给用户推系统消息 + 广播客服。
// 幂等：MarkSlaReminded 的条件更新保证同一张工单只会被提醒一次。
//
// 环境变量：
//   - TICKET_SLA_MINUTES：超时阈值（分钟），默认 2
//   - SLA_SCAN_INTERVAL：扫描间隔（秒），默认 60
func StartSlaMonitor() {
	timeout := 2 * time.Minute
	if v := os.Getenv("TICKET_SLA_MINUTES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			timeout = time.Duration(n) * time.Minute
		}
	}
	interval := 60 * time.Second
	if v := os.Getenv("SLA_SCAN_INTERVAL"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			interval = time.Duration(n) * time.Second
		}
	}

	logger.Info("SLA 超时提醒扫描已启动", logger.Fields{
		"timeout_minutes": int(timeout.Minutes()),
		"scan_interval_s": int(interval.Seconds()),
	})

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			scanSlaTimeouts(timeout)
		}
	}()
}

// scanSlaTimeouts 执行一次 SLA 扫描
func scanSlaTimeouts(timeout time.Duration) {
	tickets, err := repository.ListSlaTimeoutTickets(timeout)
	if err != nil {
		logger.Error("SLA 扫描失败", logger.Fields{"error": err.Error()})
		return
	}
	if len(tickets) == 0 {
		return
	}
	logger.Info("SLA 扫描执行", logger.Fields{"found": len(tickets)})

	reminded := 0
	for _, t := range tickets {
		// 条件更新防并发：只有标记成功的才发消息
		ok, err := repository.MarkSlaReminded(t.ID)
		if err != nil {
			logger.Error("SLA 标记失败", logger.Fields{"ticket_id": t.ID, "error": err.Error()})
			continue
		}
		if !ok {
			continue // 已被其他扫描者标记
		}

		waitMin := int(time.Since(t.CreatedAt).Minutes())
		notice := "您的咨询已等待较久，客服即将接入，请稍候"

		// 1) 落库为工单内的系统消息：用户离线时重连拉 history 也能看到
		_ = repository.SaveTicketMessage(t.UserID, "system", notice, "text", t.ID)

		// 2) 用户在线则实时推送（前端 chat 页只处理 notice 类型）
		if t.UserID != "" {
			ws.WSHub.SendToUser(t.UserID, map[string]interface{}{
				"type": "notice",
				"msg":  notice,
			})
		}

		// 3) 广播给该租户的在线客服（管理端提示）
		ws.WSHub.SendToAdminsByTenant(t.TenantID, map[string]interface{}{
			"type":      "sla_alert",
			"ticket_no": t.TicketNo,
			"ticket_id": t.ID,
			"wait_min":  waitMin,
		})

		// 4) 写工单操作日志，留痕
		_ = repository.AddTicketLog(t.ID, t.ThreadID, t.TenantID, "sla_remind",
			model.TicketStatusPending, model.TicketStatusPending,
			t.Priority, t.Priority,
			"system", "工单等待超时，已提醒用户和客服")

		reminded++
		logger.Info("SLA 超时提醒已发送", logger.Fields{
			"ticket_id": t.ID, "ticket_no": t.TicketNo,
			"tenant": t.TenantID, "wait_min": waitMin,
		})
	}

	if reminded > 0 {
		logger.Info("SLA 扫描完成", logger.Fields{"timeout_count": len(tickets), "reminded": reminded})
	}
}
