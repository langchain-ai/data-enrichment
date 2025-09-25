import os

from enrichment_agent.context import Context


def test_context_init() -> None:
    context = Context(model="test-model")
    assert context.model == "test-model"


def test_context_init_with_env_vars() -> None:
    os.environ["MODEL"] = "env-model"
    context = Context()
    assert context.model == "env-model"
    # Clean up
    del os.environ["MODEL"]


def test_context_init_with_env_vars_and_passed_values() -> None:
    os.environ["MODEL"] = "env-model"
    context = Context(model="passed-model")
    assert context.model == "passed-model"
    # Clean up
    del os.environ["MODEL"]
