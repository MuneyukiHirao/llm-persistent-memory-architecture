import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from click.testing import CliRunner

from src.agents.agent_registry import AgentDefinition
from src.cli.main import agent as cli_agent


class DummyAgentRegistry:
    def __init__(self, agents=None):
        self._agents = agents or []

    def get_active_agents(self):
        return [a for a in self._agents if a.status == "active"]

    def get_all(self):
        return list(self._agents)

    def get_by_id(self, agent_id: str):
        for agent in self._agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def register(self, agent):
        self._agents.append(agent)

    def update(self, agent):
        return agent


class DummyMemoryRepository:
    def search_by_agent(self, agent_id: str, limit: int = 10000):
        return []


class DummyOrchestrator:
    def process_request(self, task_summary, items=None, session_id=None):
        routing = SimpleNamespace(
            selected_agent_id="dummy_agent",
            confidence=0.9,
            selection_reason="dummy",
        )
        return SimpleNamespace(
            routing_decision=routing,
            agent_result={"output": "ok"},
            session_id=session_id or uuid4(),
        )


class DummyTaskExecutor:
    def execute_task(self, agent_id, task_description, perspective=None):
        return {"output": f"{agent_id}:{task_description}"}


def _patch_context(monkeypatch, agents=None):
    def _init(self):
        self.db = None
        self.config = None
        self.agent_registry = DummyAgentRegistry(agents or [])
        self.memory_repository = DummyMemoryRepository()
        self.embedding_client = SimpleNamespace(get_embedding=lambda _: [0.0])
        self.vector_search = None
        self._task_executor = DummyTaskExecutor()
        self._orchestrator = DummyOrchestrator()
        self._initialized = True

    monkeypatch.setattr("src.cli.main.CLIContext.initialize", _init, raising=False)


def test_list_json_output(monkeypatch):
    agents = [
        AgentDefinition(
            agent_id="agent_1",
            name="Agent One",
            role="Role",
            perspectives=["A", "B", "C"],
            system_prompt="",
            capabilities=[],
            status="active",
        )
    ]
    _patch_context(monkeypatch, agents)

    runner = CliRunner()
    result = runner.invoke(cli_agent, ["list", "--format", "json", "--status", "all"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert payload[0]["agent_id"] == "agent_1"


def test_task_async(monkeypatch):
    _patch_context(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli_agent, ["task", "hello", "--async"])
    assert result.exit_code == 0
    assert "非同期" in result.output


def test_task_invalid_session(monkeypatch):
    _patch_context(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli_agent, ["task", "hello", "--session", "invalid-uuid"])
    assert result.exit_code == 2


def test_register_direct(monkeypatch):
    _patch_context(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli_agent,
        [
            "register",
            "--id",
            "agent_x",
            "--name",
            "Agent X",
            "--role",
            "Role",
            "--perspectives",
            "A,B,C",
        ],
    )
    assert result.exit_code == 0
    assert "エージェントを登録しました" in result.output


def test_educate_dry_run(monkeypatch, tmp_path):
    agents = [
        AgentDefinition(
            agent_id="agent_x",
            name="Agent X",
            role="Role",
            perspectives=["A", "B", "C"],
            system_prompt="",
            capabilities=[],
            status="active",
        )
    ]
    _patch_context(monkeypatch, agents)

    yaml_content = """
    title: サンプル教科書
    scope_level: project
    chapters:
      - title: 章1
        content: 内容
        quiz:
          - question: Q1
            answer: A1
    """
    file_path = tmp_path / "textbook.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_agent,
        [
            "educate",
            "agent_x",
            "-f",
            str(file_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
