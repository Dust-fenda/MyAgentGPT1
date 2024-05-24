from typing import Any, Callable, Dict, TypeVar

from langchain import BasePromptTemplate, LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser, OutputParserException
from openai.error import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)

from reworkd_platform.schemas.agent import ModelSettings
from reworkd_platform.web.api.errors import OpenAIError

T = TypeVar("T")


def parse_with_handling(parser: BaseOutputParser[T], completion: str) -> T:
    try:
        return parser.parse(completion)
    except OutputParserException as e:
        raise OpenAIError(
            e, "There was an issue parsing the response from the AI model."
        )


async def openai_error_handler(
    func: Callable[..., Any], *args: Any, settings: ModelSettings, **kwargs: Any
) -> Any:
    try:
        return await func(*args, **kwargs)
    except ServiceUnavailableError as e:
        raise OpenAIError(
            e,
            "OpenAI is experiencing issues. Visit "
            "https://status.openai.com/ for more info.",
            should_log=not settings.custom_api_key,
        )
    except InvalidRequestError as e:
        if e.user_message.startswith("The model:"):
            raise OpenAIError(
                e,
                f"Your API key does not have access to your current model. Please use a different model.",
                should_log=not settings.custom_api_key,
            )
        raise OpenAIError(e, e.user_message)
    except AuthenticationError as e:
        raise OpenAIError(
            e,
            "Authentication error: Ensure a valid API key is being used.",
            should_log=not settings.custom_api_key,
        )
    except RateLimitError as e:
        if e.user_message.startswith("You exceeded your current quota"):
            raise OpenAIError(
                e,
                f"Your API key exceeded your current quota, please check your plan and billing details.",
                should_log=not settings.custom_api_key,
            )
        raise OpenAIError(e, e.user_message)
    except Exception as e:
        raise OpenAIError(
            e, "There was an unexpected issue getting a response from the AI model."
        )

#调用语言模型并处理可能的错误
async def call_model_with_handling(
    #函数签名
    model: BaseChatModel,
    prompt: BasePromptTemplate,
    args: Dict[str, str],
    settings: ModelSettings,
    #其他可选参数
    **kwargs: Any,
) -> str:
    #传递语言模型和提示模板实例来创建 LLMChain实例
    chain = LLMChain(llm=model, prompt=prompt)
    #调用 openai_error_handler 函数来执行链的运行并处理可能的错误
    return await openai_error_handler(chain.arun, args, settings=settings, **kwargs)
