import logging
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Type, Union

from dbgpt.core import MessageConverter, ModelMetadata, ModelOutput, ModelRequest
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    ResourceCategory,
    auto_register_resource,
)
from dbgpt.model.proxy.base import (
    AsyncGenerateStreamFunction,
    GenerateStreamFunction,
    ProxyLLMClient,
    register_proxy_model_adapter,
)
from dbgpt.model.proxy.llms.chatgpt import OpenAICompatibleDeployModelParameters
from dbgpt.model.proxy.llms.proxy_model import ProxyModel, parse_model_request
from dbgpt.util.i18n_utils import _

if TYPE_CHECKING:
    from httpx._types import ProxiesTypes
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@auto_register_resource(
    label=_("Burncloud Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("Burncloud Proxy LLM"),
    documentation_url="https://ai.burncloud.com/docs",
    show_in_ui=False,
)
@dataclass
class BurncloudDeployModelParameters(OpenAICompatibleDeployModelParameters):
    """Deploy model parameters for Burncloud."""

    provider: str = "proxy/burncloud"

    api_base: Optional[str] = field(
        default="${env:BURNCLOUD_API_BASE:-https://ai.burncloud.com/v1}",
        metadata={
            "help": _("The base url of the Burncloud API."),
        },
    )

    api_key: Optional[str] = field(
        default="${env:BURNCLOUD_API_KEY}",
        metadata={
            "help": _("The API key of the Burncloud API."),
            "tags": "privacy",
        },
    )


async def burncloud_generate_stream(
    model: ProxyModel,
    tokenizer: Any,
    params: Dict[str, Any],
    device: str,
    context_len=2048,
) -> AsyncIterator[ModelOutput]:
    client: BurncloudLLMClient = model.proxy_llm_client
    request = parse_model_request(params, client.default_model, stream=True)
    async for r in client.generate_stream(request):
        yield r


class BurncloudLLMClient(ProxyLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        proxies: Optional["ProxiesTypes"] = None,
        timeout: Optional[int] = 240,
        model_alias: Optional[str] = "claude-3-5-sonnet-20241022",
        context_length: Optional[int] = 8192,
        client: Optional["AsyncOpenAI"] = None,
        burncloud_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Could not import python package: openai "
                "Please install openai by command `pip install openai"
            ) from exc

        if not model:
            model = "claude-3-5-sonnet-20241022"
        self._client = client
        self._model = model
        self._api_key = self._resolve_env_vars(api_key)
        self._api_base = self._resolve_env_vars(api_base) or "https://ai.burncloud.com/v1"
        self._proxies = proxies
        self._timeout = timeout
        self._burncloud_kwargs = burncloud_kwargs or {}
        self._model_alias = model_alias

        super().__init__(
            model_names=[model_alias],
            context_length=context_length,
        )

    @classmethod
    def new_client(
        cls,
        model_params: BurncloudDeployModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "BurncloudLLMClient":
        return cls(
            api_key=model_params.api_key,
            api_base=model_params.api_base,
            model=model_params.real_provider_model_name,
            proxies=model_params.http_proxy,
            model_alias=model_params.real_provider_model_name,
            context_length=max(model_params.context_length or 8192, 8192),
        )

    @classmethod
    def param_class(cls) -> Type[BurncloudDeployModelParameters]:
        """Get the model parameters class."""
        return BurncloudDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        """Get generate stream function.

        Returns:
            Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
                generate stream function
        """
        return burncloud_generate_stream

    @property
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            )
        return self._client

    @property
    def default_model(self) -> str:
        """Default model name"""
        model = self._model
        if not model:
            model = "claude-3-5-sonnet-20241022"
        return model

    def _build_request(
        self, request: ModelRequest, stream: Optional[bool] = False
    ) -> Dict[str, Any]:
        payload = {"stream": stream}
        model = request.model or self.default_model
        payload["model"] = model
        # Apply burncloud kwargs
        for k, v in self._burncloud_kwargs.items():
            payload[k] = v
        if request.temperature:
            payload["temperature"] = request.temperature
        if request.max_new_tokens:
            payload["max_tokens"] = request.max_new_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.top_p:
            payload["top_p"] = request.top_p
        return payload

    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        request = self.local_covert_message(request, message_converter)
        messages = request.to_common_messages()
        payload = self._build_request(request)
        logger.info(
            f"Send request to burncloud, payload: {payload}\n\n messages:\n{messages}"
        )
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages, **payload
            )
            reasoning_content = ""
            message_obj = chat_completion.choices[0].message
            if hasattr(message_obj, "reasoning_content"):
                reasoning_content = message_obj.reasoning_content
            text = chat_completion.choices[0].message.content
            usage = chat_completion.usage.dict() if chat_completion.usage else None
            return ModelOutput.build(text, reasoning_content, usage=usage)
        except Exception as e:
            return ModelOutput(
                text=f"**Burncloud Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )

    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        request = self.local_covert_message(request, message_converter)
        messages = request.to_common_messages()
        payload = self._build_request(request, stream=True)
        logger.info(
            f"Send request to burncloud, payload: {payload}\n\n messages:\n{messages}"
        )
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages, **payload
            )
            text = ""
            reasoning_content = ""
            usage = None
            async for r in chat_completion:
                if len(r.choices) == 0:
                    continue
                # Check for empty 'choices' issue
                if r.choices[0] is not None and r.choices[0].delta is None:
                    continue
                delta_obj = r.choices[0].delta
                if hasattr(delta_obj, "reasoning_content"):
                    reasoning_content += delta_obj.reasoning_content or ""
                if r.choices[0].delta.content is not None:
                    text += r.choices[0].delta.content
                if text or reasoning_content:
                    if hasattr(r, "usage") and r.usage is not None:
                        usage = r.usage.dict()
                    yield ModelOutput.build(text, reasoning_content, usage=usage)
        except Exception as e:
            yield ModelOutput(
                text=f"**Burncloud Generate Stream Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )

    async def models(self) -> List[ModelMetadata]:
        model_metadata = ModelMetadata(
            model=self._model_alias,
            context_length=await self.get_context_length(),
        )
        return [model_metadata]

    async def get_context_length(self) -> int:
        """Get the context length of the model.

        Returns:
            int: The context length.
        """
        return self.context_length


register_proxy_model_adapter(
    BurncloudLLMClient,
    supported_models=[
        ModelMetadata(
            model=[
                "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
            ],
            context_length=200 * 1024,
            max_output_length=8 * 1024,
            description="Claude models provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-4o-2024-08-06",
            ],
            context_length=128000,
            max_output_length=16384,
            description="GPT-4 models provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "o1",
                "o1-mini",
                "o1-preview",
            ],
            context_length=200000,
            max_output_length=100000,
            description="OpenAI reasoning models provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "gpt-image-1",
            ],
            context_length=128000,
            max_output_length=16384,
            description="GPT image model provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "gemini-2.5-pro-preview-05-06",
            ],
            context_length=1000000,
            max_output_length=8192,
            description="Gemini 2.5 Pro model provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "deepseek-r1",
                "deepseek-v3",
            ],
            context_length=128000,
            max_output_length=16384,
            description="DeepSeek models provided by Burncloud",
            link="https://ai.burncloud.com/docs",
            function_calling=True,
        ),
    ],
)