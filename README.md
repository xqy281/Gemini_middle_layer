# Gemini API 到 OpenAI 兼容本地代理

此服务充当本地代理，将 OpenAI API 请求（特别是聊天完成）转换为 Google Gemini API 请求。它允许为 OpenAI API 设计的应用程序通过 `localhost` 端点使用 Gemini 模型。

## 功能特性

-   **OpenAI API 兼容性**: 模拟 `POST /v1/chat/completions`。
-   **请求/响应转换**: 将 OpenAI 请求/响应结构与 Gemini 的格式相互转换。
-   **系统提示处理**: 将 OpenAI `system` 消息转换为 Gemini 的 `system_instruction`。
-   **参数映射**: 映射常用参数，如 `temperature`, `max_tokens`, `top_p`, `stop_sequences`。
-   **CORS 支持**: 可配置的跨源资源共享，用于前端集成。
-   **错误处理**: 尝试返回 OpenAI 风格的错误消息。
-   **通过 `.env` 配置**: 安全地管理 API 密钥、服务器设置和模型映射。
-   **日志记录**: 对请求、响应和错误进行基本日志记录。

## 先决条件

-   Python 3.8+
-   一个 Google Gemini API 密钥。

## 安装设置

1.  **克隆仓库（或按上述说明创建文件）。**

2.  **导航到项目目录：**
    ```bash
    cd gemini_openai_proxy
    ```

3.  **创建 Python 虚拟环境：**
    ```bash
    python -m venv venv
    ```

4.  **激活虚拟环境：**
    -   在 macOS 和 Linux 上：
        ```bash
        source venv/bin/activate
        ```
    -   在 Windows 上：
        ```bash
        .\venv\Scripts\activate
        ```

5.  **安装依赖项：**
    ```bash
    pip install -r requirements.txt
    ```

6.  **配置服务：**
    -   将 `.env.example` 重命名或复制为 `.env`（如果您创建了示例文件），或在项目根目录中创建一个新的 `.env` 文件。
    -   编辑 `.env` 文件，填入您的实际 `GEMINI_API_KEY` 并根据需要调整其他设置：

        ```env
        # Gemini API 配置
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

        # 服务器配置
        SERVER_HOST="127.0.0.1"
        SERVER_PORT="3000"

        ```
    -   **重要提示**：
        -   将 `YOUR_GEMINI_API_KEY_HERE` 替换为您的实际 Gemini API 密钥。

## 运行服务

您可以通过几种方式运行该服务：

**选项 1：直接使用 Uvicorn（推荐用于开发）**

```bash
# 确保您的虚拟环境已激活
# 使用 run.py (它会从 .env 读取配置)
python run.py