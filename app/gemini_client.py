import logging
import asyncio
import google.generativeai as genai
import google.api_core.exceptions
from app.config import settings
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from google.generativeai.types import (
    GenerateContentResponse,
    AsyncGenerateContentResponse,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    # ... (__init__ and list_available_models methods) ...
    def __init__(self, api_key: str):
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error(
                "Gemini API 密钥未配置或为占位符。请在 .env 文件中设置 GEMINI_API_KEY。"
            )
            self.configured_correctly = False
        else:
            try:
                genai.configure(api_key=api_key)
                self.configured_correctly = True
                logger.info("Gemini API 配置成功。")
            except Exception as e:
                logger.error(f"使用提供的密钥配置 Gemini API 失败: {e}")
                self.configured_correctly = False

    async def generate_chat_completion(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]] = None,
        generation_config: Optional[genai.types.GenerationConfigDict] = None,
        stream: bool = False,
    ) -> Union[
        AsyncGenerateContentResponse,
        GenerateContentResponse,
        AsyncGenerator[AsyncGenerateContentResponse, None],
    ]:
        if not self.configured_correctly:
            raise ConnectionError("Gemini API 客户端未正确配置 (可能是 API 密钥问题)。")

        try:
            logger.debug(
                f"Initializing Gemini model: {model_name} with system instruction: {bool(system_instruction)}"
            )
            model_kwargs = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            model = genai.GenerativeModel(model_name, **model_kwargs)

            logger.debug(f"Sending to Gemini - Model: {model_name}, Stream: {stream}")

            # This call returns Union[AsyncGenerateContentResponse, AsyncIterable[AsyncGenerateContentResponse]]
            response_data = await model.generate_content_async(
                contents=contents, generation_config=generation_config, stream=stream
            )

            # response_data is now either an AsyncGenerator or a single AsyncGenerateContentResponse
            # The type hints for google-generativeai might be AsyncIterable, which is compatible with AsyncGenerator.
            if stream:
                if isinstance(response_data, AsyncGenerator):
                    logger.info("Gemini client returned an async generator for stream.")
                    return response_data  # type: ignore
                elif isinstance(
                    response_data,
                    (AsyncGenerateContentResponse, GenerateContentResponse),
                ):
                    # If stream=True but SDK gives a single object, it means the response was immediate.
                    # We return this single object. The caller (main.py) will handle it.
                    logger.warning(
                        f"Gemini client returned a single response object for stream=True. Type: {type(response_data)}. Returning it directly."
                    )
                    return response_data  # type: ignore
                else:
                    logger.error(
                        f"Unexpected response type from Gemini for stream=True: {type(response_data)}"
                    )

                    async def empty_stream_error_gen():
                        if False:
                            yield None

                    return empty_stream_error_gen()  # type: ignore
            else:  # Non-stream
                if not isinstance(
                    response_data,
                    (AsyncGenerateContentResponse, GenerateContentResponse),
                ):
                    logger.error(
                        f"Unexpected response type from Gemini for stream=False: {type(response_data)}"
                    )
                    raise TypeError(
                        f"Expected AsyncGenerateContentResponse or GenerateContentResponse, got {type(response_data)}"
                    )
                logger.debug(
                    f"Raw Gemini Non-Stream Response. Candidates: {len(response_data.candidates) if response_data.candidates else 'None'}"
                )
                return response_data

        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API Error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Gemini client: {e}", exc_info=True)
            raise

    async def list_available_models(self) -> List[Dict[str, Any]]:
        # ... (此方法不变)
        if not self.configured_correctly:
            raise ConnectionError("Gemini API 客户端未正确配置 (可能是 API 密钥问题)。")
        models_data = []
        try:
            logger.info("Fetching available models from Gemini API...")

            def sync_list_models_op():
                return list(genai.list_models())

            all_gemini_models = await asyncio.to_thread(sync_list_models_op)
            for model_info in all_gemini_models:
                if "generateContent" in model_info.supported_generation_methods:
                    models_data.append(
                        {"id": model_info.name, "object": "model", "owned_by": "google"}
                    )
            logger.info(
                f"Found {len(models_data)} models supporting 'generateContent'."
            )
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Error fetching models from Gemini API: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error while listing models: {e}", exc_info=True)
            raise
        return models_data


gemini_client_instance = GeminiClient(api_key=settings.gemini_api_key)
