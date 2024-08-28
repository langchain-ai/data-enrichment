from langchain_core.messages import AnyMessage
from typing import Optional

from enrichment_agent.configuration import Configuration
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel


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


def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the configured chat model."""
    configuration = Configuration.from_runnable_config(config)
    return init_chat_model(configuration.model_name)
