"""The LiteLLM gateway in front of every model call (Day 4, 5.1).

Our code only ever asks for the alias "paper-explainer". The gateway maps it to a real model and owns the
fallback, the retries and the timeout. Library mode: a litellm.Router in this process. Proxy mode: the same
settings live in gateway/litellm_config.yaml, and each user calls the proxy with their own virtual key.

Each deployment has a model_info id ("primary", "fallback"): it comes back on every reply as
AIMessage.response_metadata["model_id"], which is how the service reports `answered_by`.
"""
import contextvars
import os
from typing import Optional

from paper_agent.service import config
from paper_agent.service.config import ALIAS, EMBED_ALIAS, FALLBACK_ALIAS, MAX_TOKENS, NUM_RETRIES, TIMEOUT_S

GEMINI_HOST = "generativelanguage.googleapis.com"
# The calling user's virtual key (proxy mode), for calls made below the agent: query embeddings in search_paper.
CURRENT_KEY: contextvars.ContextVar = contextvars.ContextVar("gateway_key", default=None)


def litellm_params(model: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    """LiteLLM's name for a model behind LLM_BASE_URL. Gemini gets LiteLLM's native provider (it round-trips
    Gemini's tool-call thought signatures); any other endpoint is called as an OpenAI-compatible API."""
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = api_key or os.environ.get("LLM_API_KEY")
    if GEMINI_HOST in base_url:
        return {"model": f"gemini/{model}", "api_key": api_key}
    return {"model": f"openai/{model}", "api_base": base_url, "api_key": api_key}


def quiet_litellm() -> None:
    """No "Give Feedback / Get Help" banners or debug lines in notebook and service output."""
    import logging

    import litellm
    litellm.suppress_debug_info = True
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


def make_router(primary: Optional[str] = None, fallback: Optional[str] = None, num_retries: int = NUM_RETRIES,
                timeout: float = TIMEOUT_S):
    """The gateway as a library: two deployments behind two aliases, a fallback from one to the other, and
    LiteLLM's retries and timeout. Models default to LLM_MODEL (first entry) and LLM_FALLBACK_MODEL (or the
    primary again, if you have only one model). fallback=False: no fallback at all."""
    from litellm import Router
    quiet_litellm()
    primary = primary or os.environ["LLM_MODEL"].split(",")[0].strip()
    if fallback is None:
        fallback = (os.environ.get("LLM_FALLBACK_MODEL") or "").split(",")[0].strip() or primary
    model_list = [{"model_name": ALIAS, "litellm_params": litellm_params(primary), "model_info": {"id": "primary"}},
                  {"model_name": EMBED_ALIAS, "litellm_params": litellm_params(os.environ.get("EMBED_MODEL", primary)),
                   "model_info": {"id": "embed", "mode": "embedding"}}]
    if fallback:
        model_list.append({"model_name": FALLBACK_ALIAS, "litellm_params": litellm_params(fallback),
                           "model_info": {"id": "fallback"}})
    return Router(model_list=model_list, fallbacks=[{ALIAS: [FALLBACK_ALIAS]}] if fallback else [],
                  num_retries=num_retries, timeout=timeout, disable_cooldowns=True)


def make_chat_model(router=None, user_key: Optional[str] = None):
    """The agent's chat model. Library mode: LangChain's wrapper around the Router. Proxy mode (or a user_key):
    Day 3's ChatOpenAI pointed at the proxy with the calling user's virtual key.
    One retry layer (the Day 1 rule): LiteLLM retries; LangChain makes ONE attempt (max_retries=1 is
    tenacity's attempt count in langchain-litellm, 0 is the OpenAI client's retry count). Retrying in both
    would stack: 3 x 3 = 9 calls for one failure."""
    if user_key or config.GATEWAY_MODE == "proxy":
        from paper_agent.agent.model import ToolCallSignatures  # Day 3's ChatOpenAI (Gemini-safe)
        # No key, no call: the api swaps in each caller's key (pipeline.user_key_model) before every call.
        return ToolCallSignatures(base_url=f"{config.GATEWAY_URL}/v1", api_key=user_key or "no-key-set",
                                  model=ALIAS, temperature=0, timeout=TIMEOUT_S, max_tokens=MAX_TOKENS,
                                  max_retries=0, include_response_headers=True)
    from langchain_litellm import ChatLiteLLMRouter
    return ChatLiteLLMRouter(router=router or make_router(), model_name=ALIAS, max_tokens=MAX_TOKENS,
                             temperature=0, max_retries=1)


def deployment_of(message) -> Optional[str]:
    """Which deployment answered: the Router's model_info id, or the proxy's x-litellm-model-id header."""
    meta = getattr(message, "response_metadata", None) or {}
    headers = {k.lower(): v for k, v in (meta.get("headers") or {}).items()}
    return meta.get("model_id") or headers.get("x-litellm-model-id")


class GatewayEmbedder:
    """Day 2's embedding calls (day2.llm.embed) through the gateway's "paper-embed" alias."""

    def __init__(self, router=None):
        self.router = router

    def embed(self, texts: list, model: Optional[str] = None) -> list:
        if self.router is not None:
            response = self.router.embedding(model=EMBED_ALIAS, input=list(texts))
        else:
            import litellm
            quiet_litellm()
            response = litellm.embedding(model=f"openai/{EMBED_ALIAS}", input=list(texts),
                                         api_base=f"{config.GATEWAY_URL}/v1",
                                         api_key=CURRENT_KEY.get() or os.environ.get("LLM_API_KEY"))
        return [item["embedding"] for item in response.data]
