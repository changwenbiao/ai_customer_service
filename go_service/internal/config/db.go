package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"customer_service/internal/model"

	"gopkg.in/yaml.v3"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

// Config 全局配置（P4-4：统一配置管理）
// 支持 YAML 配置文件 + 环境变量覆盖（环境变量优先）
type Config struct {
	Mysql struct {
		Dsn string `yaml:"dsn"`
	} `yaml:"mysql"`
	AgentURL string `yaml:"agent_url"`
	Server   struct {
		Port string `yaml:"port"`
	} `yaml:"server"`
	Auth AuthConfig `yaml:"auth"`
	// P4-4：数据库连接池配置
	DBPool struct {
		MaxOpenConns    int `yaml:"max_open_conns"`
		MaxIdleConns    int `yaml:"max_idle_conns"`
		ConnMaxLifetime int `yaml:"conn_max_lifetime_minutes"`
		ConnMaxIdleTime int `yaml:"conn_max_idle_minutes"`
	} `yaml:"db_pool"`
	// P4-4：内部接口鉴权 Token（Go <-> Agent 内部通信）
	InternalAPIToken string `yaml:"internal_api_token"`
	// P1-4：消息归档保留期（天，默认 90）
	MsgRetentionDays int `yaml:"msg_retention_days"`
}

// AuthConfig 鉴权与租户配置
type AuthConfig struct {
	Enabled  bool           `yaml:"enabled"`
	TokenTTL string         `yaml:"token_ttl"`
	Tenants  []model.Tenant `yaml:"tenants"`
}

var GlobalConfig Config
var DB *gorm.DB

// LoadConfig 加载配置：YAML 文件 + 环境变量覆盖
func LoadConfig(path string) error {
	// 本地开发：存在 go_service/.env.local 时先加载（容器内无此文件，不影响线上）
	loadLocalEnv(".env.local")

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := yaml.Unmarshal(data, &GlobalConfig); err != nil {
		return err
	}

	// P4-4：统一环境变量覆盖（容器化部署优先使用环境变量）
	if v := os.Getenv("CS_MYSQL_DSN"); v != "" {
		GlobalConfig.Mysql.Dsn = v
	}
	if v := os.Getenv("CS_AGENT_URL"); v != "" {
		GlobalConfig.AgentURL = v
	}
	if v := os.Getenv("INTERNAL_API_TOKEN"); v != "" {
		GlobalConfig.InternalAPIToken = v
	}
	// 连接池配置环境变量覆盖
	if v := os.Getenv("CS_DB_MAX_OPEN_CONNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			GlobalConfig.DBPool.MaxOpenConns = n
		}
	}
	if v := os.Getenv("CS_DB_MAX_IDLE_CONNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			GlobalConfig.DBPool.MaxIdleConns = n
		}
	}
	// P1-4：消息保留期环境变量覆盖
	if v := os.Getenv("CS_MSG_RETENTION_DAYS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			GlobalConfig.MsgRetentionDays = n
		}
	}

	// P4-4：配置默认值
	if GlobalConfig.DBPool.MaxOpenConns == 0 {
		GlobalConfig.DBPool.MaxOpenConns = 25
	}
	if GlobalConfig.DBPool.MaxIdleConns == 0 {
		GlobalConfig.DBPool.MaxIdleConns = 10
	}
	if GlobalConfig.DBPool.ConnMaxLifetime == 0 {
		GlobalConfig.DBPool.ConnMaxLifetime = 60 // 60分钟
	}
	if GlobalConfig.DBPool.ConnMaxIdleTime == 0 {
		GlobalConfig.DBPool.ConnMaxIdleTime = 30 // 30分钟
	}
	if GlobalConfig.Server.Port == "" {
		GlobalConfig.Server.Port = ":8080"
	}
	// 本地多服务并存时端口可能冲突（nginx 占用 8080），允许环境变量改监听端口
	if v := os.Getenv("CS_SERVER_PORT"); v != "" {
		GlobalConfig.Server.Port = v
	}

	return nil
}

// loadLocalEnv 读取本地开发环境文件（KEY=VALUE 形式）
// 仅文件存在时生效；已存在的进程环境变量优先，避免覆盖容器注入的配置
func loadLocalEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key != "" && os.Getenv(key) == "" {
			_ = os.Setenv(key, value)
		}
	}
}

// ValidateConfig P4-4：配置验证，启动时检查关键配置
func ValidateConfig() error {
	if GlobalConfig.Mysql.Dsn == "" {
		return fmt.Errorf("MySQL DSN 未配置（CS_MYSQL_DSN 或 config.yaml mysql.dsn）")
	}
	if GlobalConfig.AgentURL == "" {
		return fmt.Errorf("Agent URL 未配置（CS_AGENT_URL 或 config.yaml agent_url）")
	}
	return nil
}

func InitDB() error {
	db, err := gorm.Open(mysql.Open(GlobalConfig.Mysql.Dsn), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("mysql connect err: %w", err)
	}

	// 获取底层 *sql.DB 对象，设置连接池参数（P4-4：使用配置值）
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("get sql db err: %w", err)
	}

	sqlDB.SetMaxOpenConns(GlobalConfig.DBPool.MaxOpenConns)
	sqlDB.SetMaxIdleConns(GlobalConfig.DBPool.MaxIdleConns)
	sqlDB.SetConnMaxLifetime(time.Duration(GlobalConfig.DBPool.ConnMaxLifetime) * time.Minute)
	sqlDB.SetConnMaxIdleTime(time.Duration(GlobalConfig.DBPool.ConnMaxIdleTime) * time.Minute)

	DB = db

	// 自动建表：会话消息/工单/租户/审计/平台配置/模型提供商/归档消息/工单日志/订单
	// 新项目结构（与 deploy/production/init/mysql/01_schema.sql 对齐）：
	// 模型已含 tenant_id、租户配额、provider_id 等全部列与索引，AutoMigrate 幂等建全。
	if err := DB.AutoMigrate(&model.Message{}, &model.Ticket{}, &model.TenantRow{},
		&model.AuditLog{}, &model.PlatformConfig{}, &model.ModelProvider{},
		&model.ArchivedMessage{}, &model.TicketLog{}, &model.Order{},
		&model.CustomerProfile{}, &model.User{}, &model.Role{}, &model.UserRole{},
		&model.QuickReply{}, &model.AgentStatus{}, &model.SatisfactionRating{}); err != nil {
		return fmt.Errorf("auto migrate err: %w", err)
	}
	// 种子租户：tenants 表为空时从 config.yaml 写入（app_secret 哈希，支持 env 覆盖）
	if err := seedTenants(); err != nil {
		return fmt.Errorf("seed tenants err: %w", err)
	}
	return nil
}
