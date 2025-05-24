from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import time
import uuid


# --- OpenAI 请求模式 ---
class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


# --- OpenAI 非流式响应模式 ---
class OpenAIChatResponseMessage(BaseModel):
    role: str
    content: Optional[str] = None


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatResponseMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage
    system_fingerprint: Optional[str] = None


# --- OpenAI 流式响应增量 (Delta) ---
class OpenAIChatResponseStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class OpenAIChatStreamChoice(BaseModel):
    index: int
    delta: OpenAIChatResponseStreamDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChatStreamChoice]
    system_fingerprint: Optional[str] = None


# --- 错误模式 (OpenAI 风格) ---
class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


# --- OpenAI 模型对象和列表模式 (用于 /v1/models 端点) ---
class OpenAIModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "google"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModelObject]
