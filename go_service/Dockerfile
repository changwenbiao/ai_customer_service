# syntax=docker/dockerfile:1.4
FROM golang:1.25-alpine AS builder
WORKDIR /build
ENV GOPROXY=https://goproxy.cn,direct
# go 模块缓存：只在 go.mod/go.sum 变化时才重新下载依赖
COPY go_service/go.mod go_service/go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download
COPY go_service/ .
RUN go mod tidy
# go 构建缓存：改源码不重拉依赖，增量编译
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o go_service .

FROM alpine:3.24
RUN apk add --no-cache tzdata
ENV TZ=Asia/Shanghai

WORKDIR /app
COPY --from=builder /build/go_service ./
COPY go_service/config.yaml ./
EXPOSE 8080
CMD ["./go_service"]
