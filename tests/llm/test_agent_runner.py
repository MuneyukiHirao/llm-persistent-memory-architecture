"""
AgentRunner のテスト

モックを使った単体テストで、LLM呼び出しを行わずに
AgentRunner の機能を検証します。
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from src.llm.agent_runner import (
    AgentRunner,
    AgentRunnerError,
    AgentResult,
    Escalation,
)
from src.llm.tool_executor import (
    ToolExecutor,
    Tool,
    ToolExecutionResult,
)
from src.llm.claude_client import (
    ClaudeClient,
    ClaudeResponse,
    ToolUse,
)


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def project_root(tmp_path):
    """テスト用プロジェクトルート"""
    # prompts/agents ディレクトリを作成
    agents_dir = tmp_path / "prompts" / "agents"
    agents_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_agent_yaml():
    """サンプルエージェント定義"""
    return """
agent_id: test_agent
name: テストエージェント
description: テスト用のエージェントです

role: |
  テスト用のエージェントとして動作します。
  与えられたタスクを実行します。

perspectives:
  - name: テスト観点1
    description: テスト観点1の説明
  - name: テスト観点2
    description: テスト観点2の説明

universal_principles:
  - 原則1
  - 原則2

memory_system:
  type: postgresql
  identifier: agent_id

report_types:
  - progress
  - completed
  - error
"""


@pytest.fixture
def mock_tool_executor():
    """モックToolExecutor"""
    executor = MagicMock(spec=ToolExecutor)
    executor.get_tools_for_llm.return_value = [
        {
            "name": "file_read",
            "description": "ファイルを読み込む",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    executor.list_tools.return_value = ["file_read"]
    return executor


@pytest.fixture
def mock_claude_client():
    """モックClaudeClient"""
    client = MagicMock(spec=ClaudeClient)
    return client


# =============================================================================
# AgentRunner 初期化テスト
# =============================================================================

class TestAgentRunnerInit:
    """初期化テスト"""

    def test_init_success(self, project_root, mock_tool_executor, mock_claude_client):
        """正常な初期化"""
        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )
        assert runner.project_root == project_root
        assert runner.tool_executor == mock_tool_executor
        assert runner.claude_client == mock_claude_client

    def test_init_invalid_project_root(self, mock_tool_executor, mock_claude_client):
        """存在しないプロジェクトルート"""
        with pytest.raises(AgentRunnerError) as excinfo:
            AgentRunner(
                project_root="/nonexistent/path",
                tool_executor=mock_tool_executor,
                claude_client=mock_claude_client,
            )
        assert "プロジェクトルートが存在しません" in str(excinfo.value)


# =============================================================================
# load_agent テスト
# =============================================================================

class TestLoadAgent:
    """エージェント読み込みテスト"""

    def test_load_agent_success(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """エージェント読み込み成功"""
        # YAMLファイルを作成
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        prompt = runner.load_agent("test_agent")

        assert "テストエージェント" in prompt
        assert "テスト用のエージェントとして動作します" in prompt
        assert "テスト観点1" in prompt
        assert "原則1" in prompt

    def test_load_agent_not_found(
        self, project_root, mock_tool_executor, mock_claude_client
    ):
        """エージェント定義ファイルが見つからない"""
        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        with pytest.raises(AgentRunnerError) as excinfo:
            runner.load_agent("nonexistent_agent")
        assert "エージェント定義ファイルが見つかりません" in str(excinfo.value)

    def test_load_agent_cached(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """キャッシュされたプロンプトを返す"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        # 1回目の読み込み
        prompt1 = runner.load_agent("test_agent")
        # 2回目の読み込み（キャッシュから）
        prompt2 = runner.load_agent("test_agent")

        assert prompt1 == prompt2


# =============================================================================
# run_task テスト
# =============================================================================

class TestRunTask:
    """タスク実行テスト"""

    def test_run_task_simple(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """シンプルなタスク実行（ツール使用なし）"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        # LLMレスポンスを設定
        mock_claude_client.complete.return_value = ClaudeResponse(
            content="タスクが完了しました。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        result = runner.run_task(
            agent_id="test_agent",
            task="Hello",
        )

        assert result.success is True
        assert result.response == "タスクが完了しました。"
        assert len(result.tool_calls) == 0
        assert result.tokens_used["input_tokens"] == 100
        assert result.tokens_used["output_tokens"] == 50

    def test_run_task_with_tool_use(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """ツール使用を含むタスク実行"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        # ツール使用 -> 完了の2回のLLM呼び出し
        mock_claude_client.complete.side_effect = [
            ClaudeResponse(
                content="ファイルを読み込みます。",
                stop_reason="tool_use",
                tool_uses=[
                    ToolUse(
                        id="tool_001",
                        name="file_read",
                        input={"path": "README.md"},
                    )
                ],
                usage={"input_tokens": 100, "output_tokens": 30},
            ),
            ClaudeResponse(
                content="ファイルの内容を確認しました。",
                stop_reason="end_turn",
                tool_uses=[],
                usage={"input_tokens": 150, "output_tokens": 40},
            ),
        ]

        # ツール実行結果
        mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="file_read",
            success=True,
            result=json.dumps({"success": True, "data": {"content": "Hello World"}}),
        )

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        result = runner.run_task(
            agent_id="test_agent",
            task="README.mdを読んでください",
        )

        assert result.success is True
        assert result.response == "ファイルの内容を確認しました。"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "file_read"
        assert result.tool_calls[0]["success"] is True
        assert result.tokens_used["input_tokens"] == 250
        assert result.tokens_used["output_tokens"] == 70

    def test_run_task_with_escalation(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """エスカレーション発生時"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        # ツール使用 -> 完了
        mock_claude_client.complete.side_effect = [
            ClaudeResponse(
                content="コマンドを実行します。",
                stop_reason="tool_use",
                tool_uses=[
                    ToolUse(
                        id="tool_001",
                        name="bash_execute",
                        input={"command": "npm install"},
                    )
                ],
                usage={"input_tokens": 100, "output_tokens": 30},
            ),
            ClaudeResponse(
                content="エスカレーションが必要です。",
                stop_reason="end_turn",
                tool_uses=[],
                usage={"input_tokens": 150, "output_tokens": 40},
            ),
        ]

        # エスカレーションを含むツール実行結果
        mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="bash_execute",
            success=False,
            result=json.dumps({
                "success": False,
                "escalation_required": True,
                "escalation_reason": "コマンド 'npm' は許可リストに含まれていません",
                "requested_action": "npm install",
                "risk_level": "medium",
            }),
            error="エスカレーションが必要です",
        )

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        result = runner.run_task(
            agent_id="test_agent",
            task="npm install を実行してください",
        )

        assert result.success is True
        assert len(result.escalations) == 1
        assert result.escalations[0]["tool_name"] == "bash_execute"
        assert "許可リスト" in result.escalations[0]["reason"]
        assert result.escalations[0]["risk_level"] == "medium"

    def test_run_task_max_iterations(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """最大イテレーション超過"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        # 常にツール使用を返す
        mock_claude_client.complete.return_value = ClaudeResponse(
            content="ツールを使用します。",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="tool_001", name="file_read", input={"path": "test.txt"})
            ],
            usage={"input_tokens": 100, "output_tokens": 30},
        )

        mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="file_read",
            success=True,
            result=json.dumps({"success": True}),
        )

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        result = runner.run_task(
            agent_id="test_agent",
            task="test",
            max_tool_iterations=3,
        )

        # 最大イテレーションに達しても success=True（処理は継続）
        assert result.success is True
        assert len(result.tool_calls) == 3

    def test_run_task_agent_not_found(
        self, project_root, mock_tool_executor, mock_claude_client
    ):
        """エージェントが見つからない場合"""
        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        result = runner.run_task(
            agent_id="nonexistent_agent",
            task="test",
        )

        assert result.success is False
        assert "エージェント定義ファイルが見つかりません" in result.error


# =============================================================================
# list_agents テスト
# =============================================================================

class TestListAgents:
    """エージェント一覧テスト"""

    def test_list_agents(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """エージェント一覧取得"""
        # 複数のエージェントを作成
        agents_dir = project_root / "prompts" / "agents"
        (agents_dir / "agent_a.yaml").write_text(sample_agent_yaml)
        (agents_dir / "agent_b.yaml").write_text(sample_agent_yaml)
        (agents_dir / "agent_c.yaml").write_text(sample_agent_yaml)

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        agents = runner.list_agents()

        assert len(agents) == 3
        assert "agent_a" in agents
        assert "agent_b" in agents
        assert "agent_c" in agents

    def test_list_agents_empty(
        self, project_root, mock_tool_executor, mock_claude_client
    ):
        """エージェントがない場合"""
        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        agents = runner.list_agents()
        assert agents == []


# =============================================================================
# get_agent_info テスト
# =============================================================================

class TestGetAgentInfo:
    """エージェント情報取得テスト"""

    def test_get_agent_info_success(
        self, project_root, sample_agent_yaml, mock_tool_executor, mock_claude_client
    ):
        """エージェント情報取得成功"""
        yaml_path = project_root / "prompts" / "agents" / "test_agent.yaml"
        yaml_path.write_text(sample_agent_yaml)

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        info = runner.get_agent_info("test_agent")

        assert info is not None
        assert info["agent_id"] == "test_agent"
        assert info["name"] == "テストエージェント"
        assert "テスト観点1" in str(info["perspectives"])

    def test_get_agent_info_not_found(
        self, project_root, mock_tool_executor, mock_claude_client
    ):
        """エージェントが見つからない場合"""
        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        info = runner.get_agent_info("nonexistent_agent")
        assert info is None


# =============================================================================
# AgentResult テスト
# =============================================================================

class TestAgentResult:
    """AgentResult データクラステスト"""

    def test_to_dict(self):
        """辞書変換"""
        result = AgentResult(
            success=True,
            response="完了しました",
            tool_calls=[{"tool_name": "test"}],
            escalations=[{"reason": "test"}],
            tokens_used={"input_tokens": 100, "output_tokens": 50},
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["response"] == "完了しました"
        assert len(data["tool_calls"]) == 1
        assert len(data["escalations"]) == 1
        assert data["tokens_used"]["input_tokens"] == 100

    def test_to_dict_with_error(self):
        """エラー付き辞書変換"""
        result = AgentResult(
            success=False,
            response="",
            error="エラーが発生しました",
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["error"] == "エラーが発生しました"


# =============================================================================
# Escalation テスト
# =============================================================================

class TestEscalation:
    """Escalation データクラステスト"""

    def test_to_dict(self):
        """辞書変換"""
        esc = Escalation(
            tool_name="bash_execute",
            reason="許可リストに含まれていません",
            requested_action="npm install",
            risk_level="medium",
        )

        data = esc.to_dict()

        assert data["tool_name"] == "bash_execute"
        assert data["reason"] == "許可リストに含まれていません"
        assert data["requested_action"] == "npm install"
        assert data["risk_level"] == "medium"

    def test_to_dict_minimal(self):
        """最小限の辞書変換"""
        esc = Escalation(
            tool_name="test_tool",
            reason="テスト理由",
        )

        data = esc.to_dict()

        assert data["tool_name"] == "test_tool"
        assert data["reason"] == "テスト理由"
        assert "requested_action" not in data
        assert "risk_level" not in data


# =============================================================================
# システムプロンプト生成テスト
# =============================================================================

class TestGenerateSystemPrompt:
    """システムプロンプト生成テスト"""

    def test_generate_with_all_fields(
        self, project_root, mock_tool_executor, mock_claude_client
    ):
        """全フィールドを含むプロンプト生成"""
        yaml_content = """
agent_id: full_agent
name: フルエージェント
description: すべてのフィールドを持つエージェント

role: |
  完全な役割定義
  複数行で記述

perspectives:
  - name: 観点A
    description: 観点Aの説明
  - name: 観点B
    description: 観点Bの説明

universal_principles:
  - 原則1
  - 原則2
  - 原則3

task_execution_flow:
  step1_init:
    description: 初期化
    actions:
      - アクション1
      - アクション2
  step2_execute:
    description: 実行
    actions:
      - アクション3

learning_rules: |
  学びのルールを記載

reference_docs:
  - docs/spec.md
  - docs/design.md

memory_system:
  type: postgresql
  identifier: agent_id

report_types:
  - progress
  - completed
"""
        yaml_path = project_root / "prompts" / "agents" / "full_agent.yaml"
        yaml_path.write_text(yaml_content)

        runner = AgentRunner(
            project_root=str(project_root),
            tool_executor=mock_tool_executor,
            claude_client=mock_claude_client,
        )

        prompt = runner.load_agent("full_agent")

        # 各セクションが含まれていることを確認
        assert "# フルエージェント" in prompt
        assert "## 役割" in prompt
        assert "完全な役割定義" in prompt
        assert "## 判断の観点" in prompt
        assert "観点A" in prompt
        assert "## 基本原則" in prompt
        assert "原則1" in prompt
        assert "## タスク実行フロー" in prompt
        assert "初期化" in prompt
        assert "## 学びの記録ルール" in prompt
        assert "## 参照ドキュメント" in prompt
        assert "docs/spec.md" in prompt
        assert "## 外部メモリシステム" in prompt
        assert "postgresql" in prompt
        assert "search_memories" in prompt
        assert "## 報告タイプ" in prompt
