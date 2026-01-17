from functools import cache
import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.cache import SQLiteCache


def configure_llm_cache(enable: bool = True):
    os.makedirs(".cache", exist_ok=True)
    if enable:
        set_llm_cache(SQLiteCache(database_path="./.cache/langchain.db"))
    else:
        set_llm_cache(None)


@cache
def chat_model(model: str, **kwargs) -> BaseChatModel:
    kwargs, provider = {**kwargs}, "openai"

    if model == "claude-haiku":
        model = "claude-haiku-4-5-20251001"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "claude-sonnet":
        model = "claude-sonnet-4-5-20250929"
        provider = "anthropic"
        kwargs["temperature"] = 0.0
    if model == "minimax-m2.1":
        model = "accounts/fireworks/models/minimax-m2p1"
        provider = "fireworks"
        kwargs["temperature"] = 0.0
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=0.1,
        )

    return init_chat_model(model, model_provider=provider, **kwargs)
