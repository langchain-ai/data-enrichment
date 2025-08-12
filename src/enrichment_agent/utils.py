"""Utility functions used in our graph."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def init_model(model_name: str) -> BaseChatModel:
    """Initialize the configured chat model."""
    if "/" in model_name:
        provider, model = model_name.split("/", maxsplit=1)
    else:
        provider = None
        model = model_name
    return init_chat_model(model, model_provider=provider)
