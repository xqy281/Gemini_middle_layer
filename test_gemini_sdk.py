# test_gemini_sdk.py
import os
import asyncio
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import google.api_core.exceptions # <--- 添加这一行

# --- 在导入 google.generativeai 或进行任何网络调用之前设置代理 ---
# 确保将端口号替换为您 v2rayN 的实际 HTTP 代理端口
PROXY_URL = "http://127.0.0.1:10808" # 例如 v2rayN 的 HTTP 代理端口
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

# 配置基本日志，方便查看 SDK 内部信息（如果 SDK 使用 logging）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "YOUR_GEMINI_API_KEY_HERE" in api_key:
        logger.error("错误: GEMINI_API_KEY 未在 .env 文件中找到或仍为占位符。")
        return

    try:
        logger.info(f"正在使用 API Key: ...{api_key[-4:]} 配置 Gemini API") # 只打印最后4位以保护密钥
        genai.configure(api_key=api_key)
        logger.info("Gemini API 配置成功。")

        # 选项 1: 列出可用模型 (一个简单的 API 调用检查)
        logger.info("尝试列出可用模型...")
        models_listed = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"可用模型 (支持 generateContent): {m.name}")
                models_listed = True
        if not models_listed:
            logger.warning("未能列出任何支持 generateContent 的模型。请检查 API Key 权限或 API 是否已启用。")
        
        # 选项 2: 尝试一个简单的聊天补全
        # 请确保这里的模型名称是您 API Key 有权限访问的真实 Gemini 模型
        # 例如 'gemini-1.5-flash-latest' 或 'gemini-pro'
        model_name = "gemini-1.5-flash-latest" 
        logger.info(f"\n尝试使用模型 '{model_name}' 进行聊天补全...")
        model = genai.GenerativeModel(model_name)
        
        logger.info("发送内容 'Hello, Gemini!' 到模型...")
        # 注意：对于简单的文本输入，可以直接传递字符串
        response = await model.generate_content_async("Hello, Gemini!") 
        
        logger.info("\n收到 Gemini 响应:")
        if response.candidates:
            # .text 是一个方便的属性，用于获取第一个候选者的文本内容
            logger.info(f"响应文本: {response.text}")
        else:
            logger.warning("响应中没有候选内容。")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logger.warning(f"提示被阻止，原因: {response.prompt_feedback.block_reason.name}")
            if response.candidates and not response.candidates[0].content:
                logger.warning(f"候选内容为空，完成原因: {response.candidates[0].finish_reason.name if response.candidates[0].finish_reason else 'N/A'}")


    except google.api_core.exceptions.PermissionDenied as e:
        logger.error(f"Gemini API 权限被拒绝: {e}")
        logger.error("请检查您的 API 密钥是否有效、是否具有 Gemini API 的使用权限，以及 API 是否已在您的 Google Cloud 项目中启用。")
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"发生 Google API 错误: {e}")
    except Exception as e:
        logger.error(f"发生意外错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())