import os
import json  # json 仍然可能被 allowed_origins 使用
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List  # List 仍然被 allowed_origins 使用

# 从项目根目录加载 .env 文件
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)


class Settings(BaseSettings):
    gemini_api_key: str = "YOUR_GEMINI_API_KEY_HERE"
    server_host: str = "127.0.0.1"
    server_port: int = 3000  # 请确保这个端口与您前端配置的端口一致

    # MODEL_MAPPING 相关的配置已移除

    # 如果您在 app/main.py 中硬编码了 allow_origins=["*"]，则此配置不再严格需要
    # 但保留它以备将来可能需要更细致的 CORS 控制
    _allowed_origins_str: str = os.getenv("ALLOWED_ORIGINS", '["*"]')  # 默认允许所有源

    model_config = SettingsConfigDict(
        env_file=dotenv_path, env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def allowed_origins(self) -> List[str]:
        # 如果 _allowed_origins_str 是 '["*"]'，则直接返回 ["*"]
        if self._allowed_origins_str == '["*"]':
            return ["*"]
        try:
            parsed_origins = json.loads(self._allowed_origins_str)
            if isinstance(parsed_origins, list) and all(
                isinstance(item, str) for item in parsed_origins
            ):
                return parsed_origins
            else:
                print(
                    f"警告: ALLOWED_ORIGINS 格式不正确: {self._allowed_origins_str}。将使用默认值 ['*']。"
                )
                return ["*"]
        except json.JSONDecodeError:
            print(
                f"警告: 无法解析 ALLOWED_ORIGINS: {self._allowed_origins_str}。将使用默认值 ['*']。"
            )
            return ["*"]


settings = Settings()

if settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE" or not settings.gemini_api_key:
    print(
        "警告: .env 文件中的 GEMINI_API_KEY 未设置或仍为占位符。应用可能无法正常工作。"
    )
