# 基于uvloop httptools构建的快速ASGI（异步服务器网关接口）服务器 
import uvicorn

from reworkd_platform.settings import settings

# 程序的入口点
def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        #应用的路径
        "reworkd_platform.web.application:get_app",
        workers=settings.workers_count,
        #指定服务器绑定的主机地址
        host=settings.host,
        #服务器绑定的端口号
        port=settings.port,
        #settings.reload如果设置为True uvicorn以热重载模式运行（源代码改变 会自动重启服务器）
        reload=settings.reload,
        #设置日志级别 常见值info debug warning error critical 有助于监控应用并调试
        log_level=settings.log_level.lower(),
        #get_app是工厂函数 确保每个worker有一个独立的实例 适用于多进程
        factory=True,
    )


if __name__ == "__main__":
    main()
