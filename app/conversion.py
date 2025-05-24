# app/conversion.py
import logging
import json
import re  # 导入正则表达式模块
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

# ... (其他导入) ...
logger = logging.getLogger(__name__)


def strip_markdown_code_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    # --- 修改正则表达式，将 (?s) 移到开头 ---
    # Pattern to match ```json ... ``` or ```JSON ... ``` or ``` ... ```
    # (?s) at the beginning makes . match newlines for the entire pattern.
    # We also make the language specifier (json|JSON) optional and non-capturing.
    # And handle optional leading/trailing whitespace around the entire block.

    # Pattern 1: With optional language specifier (json or JSON)
    # The (?s) flag must be at the very start of the pattern string.
    pattern_with_lang = r"(?s)^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$"
    match = re.match(pattern_with_lang, text.strip())
    if match:
        return match.group(1).strip()  # group(1) is the content inside ```...```

    # Pattern 2: Simple ``` ... ``` without language specifier (if the first didn't match)
    # This is actually covered by the first pattern if (json|JSON)? is truly optional.
    # Let's refine the first pattern to be more robust or keep two distinct patterns if necessary.
    # A single, more robust pattern:
    # (?s) - DOTALL flag
    # ^\s*``` - Start of line, optional whitespace, then ```
    # (?:[a-zA-Z0-9]*)? - Optional language specifier (non-capturing group)
    # \s*\n? - Optional whitespace then optional newline
    # (.*?) - The content (non-greedy match)
    # \n?```\s*$ - Optional newline, then ```, optional whitespace, end of line.

    # Refined single pattern:
    # (?s) must be at the start of the string passed to re.match, not inside the r"" string if it's not first.
    # So, we either use re.compile with flags, or ensure (?s) is truly first.

    # Let's use re.compile for clarity with flags
    # No, re.match takes the pattern string directly. (?s) must be at the start of THIS string.

    # Corrected pattern string with (?s) at the beginning:
    # This pattern tries to capture content between ``` optionally preceded by a language.
    # r"(?s)^\s*```(?:[a-zA-Z0-9]+)?\s*\n?(.*?)\n?```\s*$"
    # The previous one was: r"^\s*```(?:json|JSON)?\s*\n?(?P<content>(?s).*)\n?```\s*$"
    # The issue was (?s) inside the named group.

    # Let's try with (?s) at the very beginning of the raw string for re.match
    # This makes . match any character, including newline, for the whole expression.

    # Attempt 1: More specific for json/JSON then generic
    # (?s) makes . match newlines
    json_pattern_str = r"^\s*```(?:json|JSON)\s*\n?(.*?)\n?```\s*$"
    # For the above, if we want DOTALL, we'd pass re.DOTALL to re.match if it accepted flags,
    # or use re.compile. Since re.match doesn't take flags directly in its call,
    # the inline flag (?s) must be at the start of the pattern string itself.

    # So, if we want DOTALL for the content part:
    # r"^\s*```(?:json|JSON)?\s*\n?(?P<content>.*?)\n?```\s*$" with re.DOTALL flag
    # Or, using inline flag for the content part only (less common for re.match):
    # This was the problematic approach.

    # Correct way with re.match and inline flag for the whole pattern:
    # Pattern 1: ```json ... ``` or ```JSON ... ```
    # We need to make the dotall apply to the content part.
    # The (?s) flag applies to the rest of the regex from where it appears.
    # So, if we put it inside the group, it only applies there.
    # If we put it at the start of the whole regex string, it applies to all dots.

    # Let's use re.DOTALL flag with re.compile for clarity and correctness.
    # This is generally the most robust way.

    # Compiled pattern for ```json ... ``` or ```JSON ... ```
    lang_code_block_pattern = re.compile(
        r"^\s*```(?:json|JSON)\s*\n?(.*?)\n?```\s*$", re.DOTALL
    )
    match = lang_code_block_pattern.match(text.strip())
    if match:
        return match.group(1).strip()

    # Compiled pattern for generic ``` ... ```
    generic_code_block_pattern = re.compile(r"^\s*```\s*\n?(.*?)\n?```\s*$", re.DOTALL)
    match = generic_code_block_pattern.match(text.strip())
    if match:
        return match.group(1).strip()

    return text  # If no code block markers are found


# ... (convert_openai_to_gemini_params, convert_gemini_to_openai_response,
#      and convert_gemini_chunk_to_openai_stream_response_str functions remain the same
#      as in the previous "complete code" response, as they call this updated strip_markdown_code_block)
# ...
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
            full_raw_content = "".join(content_parts) if content_parts else ""
            full_content = strip_markdown_code_block(full_raw_content)
            if full_content is None:
                full_content = ""

            openai_finish_reason = "stop"
            if candidate.finish_reason is not None:
                try:
                    reason_name = candidate.finish_reason.name
                except AttributeError:
                    logger.warning(
                        f"Non-stream: candidate.finish_reason (value: {candidate.finish_reason}) no .name. Treating as OTHER."
                    )
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
            and gemini_response.prompt_feedback.block_reason is not None
        ):
            try:
                block_name = gemini_response.prompt_feedback.block_reason.name
            except AttributeError:
                logger.warning(
                    f"Non-stream: prompt_feedback.block_reason (value: {gemini_response.prompt_feedback.block_reason}) no .name. Treating as UNKNOWN."
                )
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
    delta_content_final: Optional[str] = None  # Renamed to avoid confusion
    current_chunk_has_content = False
    openai_finish_reason = None
    chunk_text_raw: Optional[str] = None
    chunk_finish_reason_enum_member = None
    chunk_prompt_feedback = None

    try:
        if hasattr(gemini_chunk, "text"):
            chunk_text_raw = gemini_chunk.text
        if hasattr(gemini_chunk, "finish_reason"):
            chunk_finish_reason_enum_member = gemini_chunk.finish_reason
        if hasattr(gemini_chunk, "prompt_feedback"):
            chunk_prompt_feedback = gemini_chunk.prompt_feedback
    except genai.types.generation_types.IncompleteIterationError as e:
        logger.warning(f"Stream: IncompleteIterationError: {e}.")
    except AttributeError as e:
        logger.warning(f"Stream: AttributeError: {e}.")

    if chunk_text_raw is not None:
        delta_content_final = strip_markdown_code_block(chunk_text_raw)
        current_chunk_has_content = True
        has_meaningful_change = True

    if chunk_finish_reason_enum_member is not None:
        has_meaningful_change = True
        try:
            reason_name = chunk_finish_reason_enum_member.name
        except AttributeError:
            logger.warning(
                f"Stream: chunk.finish_reason (value: {chunk_finish_reason_enum_member}) no .name. Treating as OTHER."
            )
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

    if delta_content_final is not None or openai_finish_reason:
        delta_obj_params = {}
        if delta_content_final is not None:
            delta_obj_params["content"] = delta_content_final
        is_first = is_first_content_chunk_map.get(choice_index, True)
        if (
            current_chunk_has_content and delta_content_final is not None and is_first
        ) or (not current_chunk_has_content and openai_finish_reason and is_first):
            delta_obj_params["role"] = "assistant"
            is_first_content_chunk_map[choice_index] = False
        delta_obj = OpenAIChatResponseStreamDelta(**delta_obj_params)
        sse_choices.append(
            OpenAIChatStreamChoice(
                index=choice_index, delta=delta_obj, finish_reason=openai_finish_reason
            )
        )

    if not has_meaningful_change:
        if chunk_prompt_feedback and chunk_prompt_feedback.block_reason is not None:
            try:
                block_name = chunk_prompt_feedback.block_reason.name
            except AttributeError:
                logger.warning(
                    f"Stream: chunk.prompt_feedback.block_reason (value: {chunk_prompt_feedback.block_reason}) no .name. Treating as UNKNOWN."
                )
                block_name = "UNKNOWN_BLOCK_REASON"
            logger.warning(
                f"Stream chunk (no content/finish) indicates prompt block: {block_name}"
            )
        else:
            logger.debug(f"Stream: Chunk has no meaningful change: {gemini_chunk}")
        return None
    if not sse_choices:
        logger.debug("Stream: No SSE choices generated, skipping.")
        return None
    stream_response_obj = OpenAIChatCompletionStreamResponse(
        id=response_id,
        created=created_time,
        model=request_model_name,
        choices=sse_choices,
    )
    return f"data: {stream_response_obj.model_dump_json()}\n\n"
