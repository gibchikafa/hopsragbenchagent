"""That every agent module loads and wires the same tools.

Written after an extraction moved the store out of support_agent.py and left it
referencing seven names it no longer had. That failed at import, in a deployment,
as "predictor terminated unsuccessfully" — the slowest possible place to find a
NameError.

The dependencies are stubbed, so this checks the wiring rather than the store:
no cluster, no model, no network.

    python -m pytest chinook/test_wiring.py
"""

from __future__ import annotations

import sys
import types
from inspect import Signature, signature
from unittest import mock

import pytest

TOOLS = (
    "lookup_track", "lookup_album", "lookup_artist", "purchase_history",
    "place_order", "remember_interest", "recall_interests",
)

AGENT_MODULES = (
    "support_agent",
    "support_agent_llamaindex",
    "support_agent_openai",
    "support_agent_claude",
)

AGENT_FILES = tuple(f"{name}.py" for name in AGENT_MODULES)

STUBBED = (
    "hopsworks_agent_protocol",
    "hopsworks", "pandas", "sentence_transformers", "tabulate", "anthropic",
    "langchain_anthropic", "langchain_core", "langchain_core.tools",
    "langgraph", "langgraph.graph", "langgraph.graph.message",
    "langgraph.prebuilt", "langgraph.types",
    "llama_index", "llama_index.core", "llama_index.core.agent",
    "llama_index.core.agent.workflow", "llama_index.core.tools",
    "llama_index.core.llms", "llama_index.llms", "llama_index.llms.anthropic",
    "agents", "claude_agent_sdk",
)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    # `tool` returns the function, so wrapped names stay callable and the
    # assertions below are about this repo rather than about LangChain.
    sys.modules["langchain_core.tools"].tool = lambda fn: fn
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def model_dump(self):
            return dict(getattr(self, "__dict__", {}))

    pydantic.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
    sys.modules["hopsworks"].login = mock.MagicMock()
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("store", *AGENT_MODULES):
        sys.modules.pop(name, None)
    yield


def test_the_store_imports(stubbed):
    import store  # noqa: PLC0415

    assert store is not None


@pytest.mark.parametrize("name", TOOLS)
def test_every_tool_is_a_callable_with_a_docstring(stubbed, name):
    # the frameworks read the docstring as the description the model chooses on,
    # so an undocumented tool is one the model cannot use properly
    import store  # noqa: PLC0415

    tool = getattr(store, name)
    assert callable(tool)
    assert tool.__doc__, f"{name} has no description for the model"


def test_the_langgraph_agent_imports(stubbed):
    # the failure this file exists for: a NameError at import, found in a
    # deployment rather than here
    import support_agent  # noqa: PLC0415

    assert support_agent.agent_app is not None


def test_the_llamaindex_agent_imports(stubbed):
    import support_agent_llamaindex  # noqa: PLC0415

    assert support_agent_llamaindex.agent_app is not None


def test_the_openai_agent_imports(stubbed):
    import support_agent_openai  # noqa: PLC0415

    assert support_agent_openai.agent_app is not None


def test_the_claude_agent_imports(stubbed):
    import support_agent_claude  # noqa: PLC0415

    assert support_agent_claude.agent_app is not None


def test_all_agents_use_the_same_system_prompt(stubbed):
    # a suite written against one should hold against the other, so a difference
    # in results is a difference in the framework and not in the instructions
    import support_agent  # noqa: PLC0415
    import support_agent_claude  # noqa: PLC0415
    import support_agent_llamaindex  # noqa: PLC0415
    import support_agent_openai  # noqa: PLC0415

    for module in (
        support_agent_llamaindex,
        support_agent_openai,
        support_agent_claude,
    ):
        assert support_agent.SYSTEM_PROMPT == module.SYSTEM_PROMPT


def test_all_agents_use_the_same_identity_gate(stubbed):
    import support_agent  # noqa: PLC0415
    import support_agent_claude  # noqa: PLC0415
    import support_agent_llamaindex  # noqa: PLC0415
    import support_agent_openai  # noqa: PLC0415

    for module in (
        support_agent_llamaindex,
        support_agent_openai,
        support_agent_claude,
    ):
        assert support_agent.IDENTIFY_INSTRUCTIONS == module.IDENTIFY_INSTRUCTIONS
        assert support_agent.ASK_FOR_IDENTITY == module.ASK_FOR_IDENTITY


def test_all_agents_wire_every_store_tool(stubbed):
    import store  # noqa: PLC0415
    import support_agent  # noqa: PLC0415
    import support_agent_claude  # noqa: PLC0415
    import support_agent_llamaindex  # noqa: PLC0415
    import support_agent_openai  # noqa: PLC0415

    for name in TOOLS:
        assert getattr(support_agent, name, None) is not None, name
        assert getattr(support_agent_llamaindex, name, None) is not None, name
        assert getattr(support_agent_openai, name, None) is not None, name
        assert getattr(support_agent_claude, name, None) is not None, name
        assert getattr(store, name) is not None, name
    assert len(support_agent_llamaindex.TOOLS) >= len(TOOLS)
    assert len(support_agent_openai.TOOLS) >= len(TOOLS)
    assert len(support_agent_claude.TOOLS) >= len(TOOLS)


def test_store_tool_annotations_are_concrete_for_llamaindex(stubbed):
    """LlamaIndex builds Pydantic schemas from inspect.signature directly."""
    import store  # noqa: PLC0415

    for name in TOOLS:
        for param in signature(getattr(store, name)).parameters.values():
            if param.annotation is Signature.empty:
                continue
            assert not isinstance(param.annotation, str), (
                f"{name}.{param.name} annotation must be a real type, not a "
                "postponed string"
            )


@pytest.mark.parametrize("name", AGENT_FILES)
def test_agent_files_start_uvicorn_when_executed_as_scripts(name):
    """The serving platform runs the selected agent file with Python directly."""
    from pathlib import Path

    source = (Path(__file__).parent / name).read_text()
    assert 'if __name__ == "__main__":' in source
    assert "uvicorn.run(agent_app" in source


def test_no_undefined_names_in_any_agent_file(stubbed):
    """Every name each file uses is one it has.

    Importing catches only what runs at import — the ChatAnthropic call that
    killed the deployment was module level, but a name used inside a tool would
    have waited until a customer asked for it. This reads the whole file.
    """
    import io
    from pathlib import Path

    pytest.importorskip("pyflakes", reason="pyflakes is a dev tool")
    from pyflakes import api as pyflakes_api  # noqa: PLC0415
    from pyflakes.reporter import Reporter  # noqa: PLC0415

    problems = io.StringIO()
    here = Path(__file__).parent
    for name in ("store.py", *AGENT_FILES):
        # Reporter(warnings, errors) — undefined names are warnings, so passing
        # the collector second discarded exactly what this test looks for.
        pyflakes_api.checkPath(
            str(here / name), Reporter(problems, problems)
        )
    undefined = [
        line for line in problems.getvalue().splitlines() if "undefined name" in line
    ]
    assert undefined == [], "\n".join(undefined)
