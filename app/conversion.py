import logging
import json
from typing import Tuple, Dict, Any, List, Optional, AsyncGenerator
import google.generativeai as genai

from app.schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatResponseMessage,
    OpenAIChatChoice,
    OpenAIUsage,
    OpenAIChatCompletionResponse,
    OpenAIChatResponseStreamDelta,
    OpenAIChatStreamChoice,
    OpenAIChatCompletionStreamResponse,
)

logger = logging.getLogger(__name__)


def convert_openai_to_gemini_params(
    request: OpenAIChatCompletionRequest,
) -> Tuple[
    str,
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[genai.types.GenerationConfigDict],
]:
    gemini_model_name = request.model
    logger.info(f"Using Gemini model name directly from request: {gemini_model_name}")
    gemini_contents = []
    system_instruction_parts = []
    for msg in request.messages:
        role, content = msg.role, msg.content
        if role == "system":
            system_instruction_parts.append(content)
        elif role == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": content}]})

    system_instruction_for_gemini = None
    if system_instruction_parts:
        system_instruction_for_gemini = {
            "parts": [{"text": "\n".join(system_instruction_parts)}]
        }

    generation_config: genai.types.GenerationConfigDict = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["top_p"] = float(request.top_p) if request.top_p > 0 else None
    if request.max_tokens is not None:
        generation_config["max_output_tokens"] = request.max_tokens
    if request.stop:
        generation_config["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else request.stop
        )
    if request.n is not None and request.n > 0:
        if request.n > 1 and request.stream:
            logger.warning("Streamed n>1 may not fully map to OpenAI.")
        generation_config["candidate_count"] = request.n
    return (
        gemini_model_name,
        gemini_contents,
        system_instruction_for_gemini,
        generation_config if generation_config else None,
    )


def convert_gemini_to_openai_response(
    gemini_response: genai.types.GenerateContentResponse,
    requested_openai_model: str,
    num_prompt_tokens_fallback: int = 0,
) -> OpenAIChatCompletionResponse:
    choices = []
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

    # For non-streamed (complete) responses, .candidates and .usage_metadata should be safe to access.
    if gemini_response.usage_metadata:
        prompt_tokens = gemini_response.usage_metadata.prompt_token_count
        completion_tokens = gemini_response.usage_metadata.candidates_token_count
        total_tokens = gemini_response.usage_metadata.total_token_count
    else:
        prompt_tokens = num_prompt_tokens_fallback
        logger.warning(
            "Non-stream: Gemini response missing usage_metadata. Tokens inaccurate."
        )

    if gemini_response.candidates:  # Check if candidates exist
        for i, candidate in enumerate(gemini_response.candidates):
            content_parts = (
                [part.text for part in candidate.content.parts if hasattr(part, "text")]
                if candidate.content and candidate.content.parts
                else []
            )
            full_content = "".join(content_parts) if content_parts else ""

            openai_finish_reason = "stop"
            if candidate.finish_reason:
                try:
                    reason_name = genai.types.FinishReason(candidate.finish_reason).name
                except ValueError:
                    reason_name = "OTHER"
                if reason_name == "STOP":
                    openai_finish_reason = "stop"
                elif reason_name == "MAX_TOKENS":
                    openai_finish_reason = "length"
                elif reason_name in ["SAFETY", "RECITATION"]:
                    openai_finish_reason = "content_filter"
                else:
                    logger.warning(
                        f"Non-stream: Unknown Gemini finish_reason '{reason_name}'. Mapping to 'stop'."
                    )

            choices.append(
                OpenAIChatChoice(
                    index=i,
                    message=OpenAIChatResponseMessage(
                        role="assistant", content=full_content
                    ),
                    finish_reason=openai_finish_reason,
                )
            )
            if not gemini_response.usage_metadata and full_content:
                completion_tokens += len(full_content.split())
    else:  # No candidates in a non-streamed response (e.g., prompt blocked)
        logger.warning("Non-stream: No candidates found in Gemini response.")
        # We might still have prompt_feedback
        if (
            hasattr(gemini_response, "prompt_feedback")
            and gemini_response.prompt_feedback.block_reason
        ):
            try:
                block_name = genai.types.BlockReason(
                    gemini_response.prompt_feedback.block_reason
                ).name
            except ValueError:
                block_name = "UNKNOWN"
            logger.warning(f"Non-stream: Prompt feedback indicates block: {block_name}")
            # Create a choice with empty content and finish_reason 'content_filter'
            choices.append(
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatResponseMessage(role="assistant", content=""),
                    finish_reason="content_filter",
                )
            )

    if not gemini_response.usage_metadata:
        total_tokens = prompt_tokens + completion_tokens
    usage = OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return OpenAIChatCompletionResponse(
        model=requested_openai_model, choices=choices, usage=usage
    )


async def convert_gemini_chunk_to_openai_stream_response_str(
    gemini_chunk: genai.types.GenerateContentResponse,
    request_model_name: str,
    response_id: str,
    created_time: int,
    is_first_content_chunk_map: Dict[int, bool],
) -> Optional[str]:
    sse_choices = []
    has_meaningful_change = False
    choice_index = 0

    delta_content: Optional[str] = None
    current_chunk_has_content = False
    openai_finish_reason = None

    # --- Safely access text and finish_reason ---
    chunk_text: Optional[str] = None
    chunk_finish_reason_value: Optional[int] = None  # Store the enum value
    chunk_prompt_feedback = None

    try:
        # .text should be the safest way to get current chunk's text
        if hasattr(gemini_chunk, "text"):  # Check before access
            chunk_text = gemini_chunk.text  # This might be None or empty string

        # .finish_reason should also be directly on the chunk object
        if hasattr(gemini_chunk, "finish_reason"):  # Check before access
            chunk_finish_reason_value = gemini_chunk.finish_reason

        if hasattr(gemini_chunk, "prompt_feedback"):
            chunk_prompt_feedback = gemini_chunk.prompt_feedback

    except genai.types.generation_types.IncompleteIterationError as e:
        logger.warning(
            f"Stream: IncompleteIterationError accessing basic chunk attributes (text/finish_reason): {e}. This chunk might be problematic."
        )
        # If even basic access fails, this chunk is likely unusable for content/finish_reason
        # We might still check prompt_feedback if it was accessible before this error.
        if chunk_prompt_feedback and chunk_prompt_feedback.block_reason:
            try:
                block_name = genai.types.BlockReason(
                    chunk_prompt_feedback.block_reason
                ).name
            except ValueError:
                block_name = "UNKNOWN_BLOCK_REASON"
            logger.warning(
                f"Stream chunk indicates prompt block (from fallback): {block_name}"
            )
        return None  # Cannot process this chunk further

    except AttributeError as e:  # Catch other AttributeErrors
        logger.warning(
            f"Stream: AttributeError accessing chunk attributes: {e}. Chunk type: {type(gemini_chunk)}"
        )
        return None

    if chunk_text is not None:  # chunk_text can be "" (empty string)
        delta_content = chunk_text
        current_chunk_has_content = True
        has_meaningful_change = True

    if chunk_finish_reason_value is not None:
        has_meaningful_change = True
        try:
            reason_name = genai.types.FinishReason(chunk_finish_reason_value).name
        except ValueError:
            reason_name = "OTHER"

        if reason_name == "STOP":
            openai_finish_reason = "stop"
        elif reason_name == "MAX_TOKENS":
            openai_finish_reason = "length"
        elif reason_name in ["SAFETY", "RECITATION"]:
            openai_finish_reason = "content_filter"
        else:
            logger.warning(
                f"Stream: Unknown Gemini finish_reason '{reason_name}'. Mapping to 'stop'."
            )
            openai_finish_reason = "stop"

    if delta_content is not None or openai_finish_reason:
        delta_obj_params = {}
        if delta_content is not None:
            delta_obj_params["content"] = delta_content

        if current_chunk_has_content and is_first_content_chunk_map.get(
            choice_index, True
        ):
            delta_obj_params["role"] = "assistant"
            is_first_content_chunk_map[choice_index] = False
        elif (
            not current_chunk_has_content
            and openai_finish_reason
            and is_first_content_chunk_map.get(choice_index, True)
        ):
            delta_obj_params["role"] = "assistant"
            is_first_content_chunk_map[choice_index] = False

        delta_obj = OpenAIChatResponseStreamDelta(**delta_obj_params)
        sse_choices.append(
            OpenAIChatStreamChoice(
                index=choice_index, delta=delta_obj, finish_reason=openai_finish_reason
            )
        )
        # has_meaningful_change is already true if we are here

    if not has_meaningful_change:
        if chunk_prompt_feedback and chunk_prompt_feedback.block_reason:
            try:
                block_name = genai.types.BlockReason(
                    chunk_prompt_feedback.block_reason
                ).name
            except ValueError:
                block_name = "UNKNOWN_BLOCK_REASON"
            logger.warning(
                f"Stream chunk (no content/finish) indicates prompt block: {block_name}"
            )
        else:
            logger.debug(
                f"Stream: Chunk has no meaningful change (no text, no finish_reason, no block): {gemini_chunk}"
            )
        return None

    stream_response_obj = OpenAIChatCompletionStreamResponse(
        id=response_id,
        created=created_time,
        model=request_model_name,
        choices=sse_choices,
    )
    return f"data: {stream_response_obj.model_dump_json()}\n\n"


# ... (convert_openai_to_gemini_params and convert_gemini_to_openai_response remain the same)
def convert_openai_to_gemini_params(
    request: OpenAIChatCompletionRequest,
) -> Tuple[
    str,
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[genai.types.GenerationConfigDict],
]:
    gemini_model_name = request.model
    logger.info(f"Using Gemini model name directly from request: {gemini_model_name}")
    gemini_contents, system_instruction_parts = [], []
    for msg in request.messages:
        role, content = msg.role, msg.content
        if role == "system":
            system_instruction_parts.append(content)
        elif role == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": content}]})
    system_instruction_for_gemini = (
        {"parts": [{"text": "\n".join(system_instruction_parts)}]}
        if system_instruction_parts
        else None
    )
    generation_config: genai.types.GenerationConfigDict = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["top_p"] = float(request.top_p) if request.top_p > 0 else None
    if request.max_tokens is not None:
        generation_config["max_output_tokens"] = request.max_tokens
    if request.stop:
        generation_config["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else request.stop
        )
    if request.n is not None and request.n > 0:
        if request.n > 1 and request.stream:
            logger.warning("Streamed n>1 may not fully map to OpenAI.")
        generation_config["candidate_count"] = request.n
    return (
        gemini_model_name,
        gemini_contents,
        system_instruction_for_gemini,
        generation_config if generation_config else None,
    )


def convert_gemini_to_openai_response(
    gemini_response: genai.types.GenerateContentResponse,
    requested_openai_model: str,
    num_prompt_tokens_fallback: int = 0,
) -> OpenAIChatCompletionResponse:
    choices = []
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
    if gemini_response.usage_metadata:
        prompt_tokens = gemini_response.usage_metadata.prompt_token_count
        completion_tokens = gemini_response.usage_metadata.candidates_token_count
        total_tokens = gemini_response.usage_metadata.total_token_count
    else:
        prompt_tokens = num_prompt_tokens_fallback
        logger.warning(
            "Non-stream: Gemini response missing usage_metadata. Tokens inaccurate."
        )

    if gemini_response.candidates:
        for i, candidate in enumerate(gemini_response.candidates):
            content_parts = (
                [part.text for part in candidate.content.parts if hasattr(part, "text")]
                if candidate.content and candidate.content.parts
                else []
            )
            full_content = "".join(content_parts) if content_parts else ""
            openai_finish_reason = "stop"
            if candidate.finish_reason:
                try:
                    reason_name = genai.types.FinishReason(candidate.finish_reason).name
                except ValueError:
                    reason_name = "OTHER"
                if reason_name == "STOP":
                    openai_finish_reason = "stop"
                elif reason_name == "MAX_TOKENS":
                    openai_finish_reason = "length"
                elif reason_name in ["SAFETY", "RECITATION"]:
                    openai_finish_reason = "content_filter"
                else:
                    logger.warning(
                        f"Non-stream: Unknown Gemini finish_reason '{reason_name}'. Mapping to 'stop'."
                    )
            choices.append(
                OpenAIChatChoice(
                    index=i,
                    message=OpenAIChatResponseMessage(
                        role="assistant", content=full_content
                    ),
                    finish_reason=openai_finish_reason,
                )
            )
            if not gemini_response.usage_metadata and full_content:
                completion_tokens += len(full_content.split())
    else:
        logger.warning("Non-stream: No candidates found in Gemini response.")
        if (
            hasattr(gemini_response, "prompt_feedback")
            and gemini_response.prompt_feedback.block_reason
        ):
            try:
                block_name = genai.types.BlockReason(
                    gemini_response.prompt_feedback.block_reason
                ).name
            except ValueError:
                block_name = "UNKNOWN"
            logger.warning(f"Non-stream: Prompt feedback indicates block: {block_name}")
            choices.append(
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatResponseMessage(role="assistant", content=""),
                    finish_reason="content_filter",
                )
            )
    if not gemini_response.usage_metadata:
        total_tokens = prompt_tokens + completion_tokens
    usage = OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return OpenAIChatCompletionResponse(
        model=requested_openai_model, choices=choices, usage=usage
    )
