import logging
import json
import time
import uuid
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator, Union  #
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import google.api_core.exceptions
import google.generativeai as genai

from app.config import settings
from app.schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIErrorDetail,
    OpenAIModelObject,
    OpenAIModelList,
)
from app.conversion import (
    convert_openai_to_gemini_params,
    convert_gemini_to_openai_response,
    convert_gemini_chunk_to_openai_stream_response_str,
)
from app.gemini_client import gemini_client_instance

# --- 代理配置 ---
import os

PROXY_URL = "http://127.0.0.1:10808"  # 替换为您的代理端口
if PROXY_URL and "YOUR_HTTP_PROXY_PORT" not in PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    logging.info(f"HTTP/HTTPS proxy configured: {PROXY_URL}")
else:
    logging.info("No proxy explicitly configured in main.py.")
# --- 代理配置结束 ---

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemini to OpenAI Compatible Proxy",
    version="1.2.2",  # 版本更新
    description="Local proxy for Gemini API, OpenAI compatible, with streaming and model listing.",
)

# 使用 settings.allowed_origins，它现在可以处理 ["*"]
logger.info(f"Configuring CORS with allowed origins: {settings.allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # 从 config 读取，可以是 ["*"]
    allow_credentials=(
        True if settings.allowed_origins != ["*"] else False
    ),  # 凭证与 "*" 不兼容
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    logger.info(f"Server Host: {settings.server_host}, Port: {settings.server_port}")
    if not settings.gemini_api_key or "YOUR_GEMINI_API_KEY" in settings.gemini_api_key:
        logger.error("CRITICAL: GEMINI_API_KEY not configured or is placeholder.")
    else:
        logger.info("GEMINI_API_KEY loaded.")
    if not gemini_client_instance.configured_correctly:
        logger.warning("Gemini client not configured correctly at startup.")


def create_openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400,
) -> JSONResponse:
    error_detail = OpenAIErrorDetail(
        message=message, type=err_type, param=param, code=code
    )
    return JSONResponse(
        status_code=status_code,
        content=OpenAIErrorResponse(error=error_detail).model_dump(),
    )


@app.get("/v1/models", response_model=OpenAIModelList)
async def list_models_endpoint():
    try:
        logger.info("Request for /v1/models")
        models_data = await gemini_client_instance.list_available_models()
        response = OpenAIModelList(data=[OpenAIModelObject(**md) for md in models_data])
        logger.info(f"Returning {len(response.data)} models.")
        return response
    except ConnectionError as e:
        logger.error(f"/v1/models: ConnectionError: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"/v1/models: GoogleAPIError: {e}", exc_info=True)
        status = (
            401 if isinstance(e, google.api_core.exceptions.PermissionDenied) else 500
        )
        raise HTTPException(status_code=status, detail=f"Gemini API Error: {e.message}")
    except Exception as e:
        logger.error(f"/v1/models: Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_gemini_response_as_openai_sse(
    gemini_response_input: Union[
        AsyncGenerator[genai.types.AsyncGenerateContentResponse, None],
        genai.types.AsyncGenerateContentResponse,
        genai.types.GenerateContentResponse,
    ],
    request_model_name: str,
):
    response_id, created_time = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    is_first_content_chunk_map: Dict[int, bool] = {}

    async def process_and_yield_chunk(
        chunk_to_process: genai.types.AsyncGenerateContentResponse,
    ):  # Type hint to AsyncGCR
        nonlocal is_first_content_chunk_map

        resolved_chunk = chunk_to_process
        if hasattr(chunk_to_process, "resolve") and callable(chunk_to_process.resolve):
            logger.debug("Stream: Attempting to await resolve() on chunk.")
            try:
                # --- MODIFICATION HERE: Directly await resolve() ---
                await chunk_to_process.resolve()
                # After awaiting resolve, the chunk_to_process object itself should be updated/resolved.
                resolved_chunk = chunk_to_process
                logger.debug("Stream: Chunk resolve() awaited successfully.")
            except Exception as e_resolve:
                logger.error(
                    f"Stream: Error awaiting resolve() on chunk: {e_resolve}",
                    exc_info=True,
                )
                # Proceed with the original chunk if resolve fails, though it might still error out

        sse_event_str = await convert_gemini_chunk_to_openai_stream_response_str(
            resolved_chunk,
            request_model_name,
            response_id,
            created_time,
            is_first_content_chunk_map,
        )
        if sse_event_str:
            if "'finish_reason':" in sse_event_str:
                logger.debug(
                    f"Stream: Sent chunk with finish_reason: {sse_event_str.strip()}"
                )
            return sse_event_str
        return None

    try:
        if isinstance(gemini_response_input, AsyncGenerator):
            logger.debug("Stream: Processing async generator from Gemini.")
            async for gemini_chunk_from_generator in gemini_response_input:
                if not isinstance(
                    gemini_chunk_from_generator,
                    (genai.types.AsyncGenerateContentResponse),
                ):  # Expecting AsyncGCR from stream
                    logger.warning(
                        f"Stream: Skipping unexpected chunk type in generator: {type(gemini_chunk_from_generator)}."
                    )
                    continue

                sse_event = await process_and_yield_chunk(gemini_chunk_from_generator)
                if sse_event:
                    yield sse_event

        elif isinstance(
            gemini_response_input,
            (
                genai.types.AsyncGenerateContentResponse,
                genai.types.GenerateContentResponse,
            ),
        ):
            logger.debug(
                "Stream: Processing single response object as a one-chunk stream."
            )
            # This input is already an AsyncGenerateContentResponse or GenerateContentResponse
            # If it's GenerateContentResponse, resolve() is sync. If AsyncGCR, resolve() is async.
            # The process_and_yield_chunk now handles awaiting resolve() if it's an async method.
            sse_event = await process_and_yield_chunk(gemini_response_input)  # type: ignore
            if sse_event:
                yield sse_event
        else:
            logger.error(
                f"Stream: Unexpected input type for streaming: {type(gemini_response_input)}"
            )

        logger.info("Stream: Processing finished. Sending [DONE].")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream: Error during SSE generation: {e}", exc_info=True)
        err_code = "gemini_stream_error"
        err_msg = f"Error during stream: {str(e)}"
        if isinstance(e, google.api_core.exceptions.GoogleAPIError):
            err_code = "gemini_api_error"
            err_msg = f"Gemini API Error during stream: {e.message}"
        elif isinstance(e, AttributeError):
            err_msg = f"Streaming error (AttributeError): {str(e)}."

        err_detail = OpenAIErrorDetail(message=err_msg, type="api_error", code=err_code)
        err_json = OpenAIErrorResponse(error=err_detail).model_dump_json()
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"


# ... (chat_completions endpoint and other parts of main.py remain the same as the last complete version)
# ...
@app.post("/v1/chat/completions")
async def chat_completions(
    request_body: OpenAIChatCompletionRequest, http_request: Request
):
    logger.info(
        f"Request /v1/chat/completions from {http_request.client.host if http_request.client else 'N/A'} for model: {request_body.model}, stream: {request_body.stream}"
    )
    try:
        gemini_model_name, gemini_contents, system_instruction, generation_config = (
            convert_openai_to_gemini_params(request_body)
        )

        gemini_output = await gemini_client_instance.generate_chat_completion(
            model_name=gemini_model_name,
            contents=gemini_contents,
            system_instruction=system_instruction,
            generation_config=generation_config,
            stream=request_body.stream,
        )

        if request_body.stream:
            logger.info(f"Processing stream request for model: {gemini_model_name}")
            if not (
                isinstance(gemini_output, AsyncGenerator)
                or isinstance(
                    gemini_output,
                    (
                        genai.types.AsyncGenerateContentResponse,
                        genai.types.GenerateContentResponse,
                    ),
                )
            ):
                logger.error(
                    f"Stream Error: Gemini client returned unexpected type for stream=True: {type(gemini_output)}"
                )
                return create_openai_error(
                    "Internal server error: Failed to initiate stream with Gemini due to unexpected response type.",
                    status_code=500,
                )

            return StreamingResponse(
                stream_gemini_response_as_openai_sse(gemini_output, request_body.model),  # type: ignore
                media_type="text/event-stream",
            )
        else:
            logger.info(f"Processing non-stream request for model: {gemini_model_name}")
            if not isinstance(
                gemini_output,
                (
                    genai.types.AsyncGenerateContentResponse,
                    genai.types.GenerateContentResponse,
                ),
            ):
                logger.error(
                    f"Non-Stream Error: Gemini client returned unexpected type: {type(gemini_output)}"
                )
                return create_openai_error(
                    "Internal server error: Unexpected response type from Gemini (non-stream).",
                    status_code=500,
                )

            gemini_response_obj = gemini_output

            # For a non-streamed, complete response, .candidates should be safe.
            # If it's an AsyncGenerateContentResponse, it should represent the fully resolved response.
            # However, to be absolutely safe after seeing IncompleteIterationError,
            # we might consider awaiting resolve() here too if it's an AsyncGCR.
            if isinstance(
                gemini_response_obj, genai.types.AsyncGenerateContentResponse
            ) and hasattr(gemini_response_obj, "resolve"):
                logger.debug(
                    "Non-stream: Awaiting resolve() on AsyncGenerateContentResponse."
                )
                await gemini_response_obj.resolve()  # Await if it's async

            try:
                if not gemini_response_obj.candidates:
                    block_msg = (
                        "Response generation failed/blocked by Gemini (no candidates)."
                    )
                    if (
                        hasattr(gemini_response_obj, "prompt_feedback")
                        and gemini_response_obj.prompt_feedback.block_reason
                    ):
                        block_msg = f"Prompt blocked by Gemini: {genai.types.BlockReason(gemini_response_obj.prompt_feedback.block_reason).name}."
                    return create_openai_error(
                        message=block_msg,
                        err_type="invalid_request_error",
                        code="content_policy_violation",
                        status_code=400,
                    )
            except genai.types.generation_types.IncompleteIterationError:
                logger.error(
                    "Non-stream: IncompleteIterationError accessing .candidates on a supposedly complete response. This is unexpected."
                )
                return create_openai_error(
                    "Internal server error: Failed to process Gemini response.",
                    status_code=500,
                )

            openai_response = convert_gemini_to_openai_response(
                gemini_response_obj, request_body.model
            )
            logger.info(
                f"Successfully processed non-stream response for model {request_body.model}."
            )
            return openai_response

    except ValueError as e:
        logger.error(f"ValueError during request processing: {e}", exc_info=True)
        return create_openai_error(
            str(e),
            err_type="invalid_request_error",
            code="parameter_error",
            status_code=400,
        )
    except ConnectionError as e:
        logger.error(
            f"ConnectionError (Gemini client not configured): {e}", exc_info=True
        )
        return create_openai_error(
            str(e),
            err_type="api_connection_error",
            code="gemini_config_error",
            status_code=500,
        )
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"A Google API error occurred: {e}", exc_info=True)
        err_type, status, code, msg = (
            "api_error",
            500,
            "gemini_api_error",
            f"An error occurred with the Gemini API: {e.message}",
        )
        if isinstance(e, google.api_core.exceptions.PermissionDenied):
            status, code, err_type = 401, "invalid_api_key", "authentication_error"
        elif isinstance(e, google.api_core.exceptions.InvalidArgument):
            status, code, err_type = (
                400,
                "gemini_invalid_argument",
                "invalid_request_error",
            )
        elif isinstance(e, google.api_core.exceptions.ResourceExhausted):
            status, code, err_type = 429, "rate_limit_exceeded", "rate_limit_error"
        elif isinstance(e, google.api_core.exceptions.FailedPrecondition):
            status, code, err_type = (
                400,
                "gemini_precondition_failed",
                "invalid_request_error",
            )
        return create_openai_error(
            message=msg, err_type=err_type, code=code, status_code=status
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return create_openai_error(
            f"An unexpected internal server error occurred: {str(e)}",
            err_type="internal_server_error",
            code="unexpected_error",
            status_code=500,
        )


# ... (root endpoint and __main__ block) ...
@app.get("/", summary="Health Check")
async def root():
    return {
        "message": "Gemini to OpenAI Proxy is running.",
        "status": "healthy",
        "version": app.version,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(
        f"Starting server: http://{settings.server_host}:{settings.server_port}"
    )
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
