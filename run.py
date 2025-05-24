import uvicorn
from app.config import settings

# from app.main import app # 如果 uvicorn.run 使用的是字符串 "app.main:app"，则不需要导入 app

if __name__ == "__main__":
    print(
        f"启动 Uvicorn 服务器，地址: http://{settings.server_host}:{settings.server_port}"
    )
    # print(f"允许的 CORS 源: {settings.allowed_origins}") # <--- 注释掉或删除此行
    print("CORS 配置: 允许所有源 ('*')")  # 可以用这条日志替代
    print(f"模型映射: 已移除，直接使用前端指定的 Gemini 模型名")  # 更新日志
    if not settings.gemini_api_key or "YOUR_GEMINI_API_KEY" in settings.gemini_api_key:
        print("\n警告: .env 文件中的 GEMINI_API_KEY 未设置或为占位符！")
        print("应用将启动，但对 Gemini 的 API 调用将会失败。\n")

    # 推荐使用字符串形式指定 app，以便 reload 等功能更好工作
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
        reload=True,  # 如果您希望在开发时自动重载
    )
