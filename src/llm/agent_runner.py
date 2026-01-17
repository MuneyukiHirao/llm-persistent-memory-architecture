# エージェント実行フレームワーク
# 実装仕様: docs/phase1-implementation-spec.ja.md
# エージェント設計: docs/development-agents.ja.md
"""
エージェント実行モジュール

YAMLプロンプト + Claude API + 実行ツールを統合し、
エージェントにタスクを依頼できる基盤を提供します。

設計方針（タスク実行フローエージェント観点）:
- フロー整合性: YAML読み込み→ツール初期化→LLM呼び出しの順序を保証
- エラー処理: ツール実行失敗時もフローを継続、エスカレーションを適切に処理
- ログ可視性: 各ステップでのログ出力
- 拡張性: 新規エージェントの追加が容易

使用例:
    runner = AgentRunner(project_root="/path/to/project")

    # エージェントでタスクを実行
    result = runner.run_task(
        agent_id="task_execution_agent",
        task="docs/architecture.ja.md を読んで概要を教えてください"
    )

    print(result.response)
    print(f"ツール呼び出し回数: {len(result.tool_calls)}")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.llm.claude_client import ClaudeClient, ClaudeClientError
from src.llm.tool_executor import (
    ToolExecutor,
    create_tool_executor_with_all_tools,
)
from src.llm.task_planner import (
    TaskPlanner,
    TaskPlan,
    TaskPlannerError,
    get_planning_prompt,
)
from src.config.llm_config import LLMConfig

# TaskExecutorはオプションのため、型ヒント用に条件付きインポート
try:
    from src.core.task_executor import TaskExecutor as TaskExecutorType
except ImportError:
    TaskExecutorType = None  # type: ignore

logger = logging.getLogger(__name__)


class AgentRunnerError(Exception):
    """AgentRunner のエラー

    エージェント読み込み失敗、タスク実行失敗などで発生。

    Attributes:
        message: エラーメッセージ
        agent_id: 対象のエージェントID（存在する場合）
        original_error: 元の例外（あれば）
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [self.message]
        if self.agent_id:
            parts.append(f"agent_id={self.agent_id}")
        if self.original_error:
            parts.append(f"原因: {self.original_error}")
        return " ".join(parts)


@dataclass
class Escalation:
    """エスカレーション情報

    ツール実行がエスカレーションを必要とした場合の情報。

    Attributes:
        tool_name: エスカレーションを発生させたツール名
        reason: エスカレーション理由
        requested_action: 要求されたアクション
        risk_level: リスクレベル（low/medium/high）
    """

    tool_name: str
    reason: str
    requested_action: Optional[str] = None
    risk_level: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = {
            "tool_name": self.tool_name,
            "reason": self.reason,
        }
        if self.requested_action:
            data["requested_action"] = self.requested_action
        if self.risk_level:
            data["risk_level"] = self.risk_level
        return data


@dataclass
class AgentResult:
    """エージェント実行結果

    AgentRunner.run_task() の戻り値として、タスク実行の全結果を格納。

    Attributes:
        success: タスクが正常に完了したかどうか
        response: LLMからの最終応答テキスト
        tool_calls: 呼び出したツールの記録リスト
        escalations: エスカレーション情報のリスト
        tokens_used: トークン使用量（input_tokens, output_tokens）
        error: エラーメッセージ（失敗時）
    """

    success: bool
    response: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    escalations: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = {
            "success": self.success,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "escalations": self.escalations,
            "tokens_used": self.tokens_used,
        }
        if self.error:
            data["error"] = self.error
        return data


class AgentRunner:
    """エージェント実行クラス

    YAMLプロンプト、Claude API、実行ツールを統合し、
    エージェントにタスクを依頼できる実行環境を提供します。

    処理フロー:
        1. 初期化: プロジェクトルート設定、ツール初期化
        2. エージェント読み込み: YAML からシステムプロンプト生成
        3. タスク実行: LLM呼び出し、ツール使用ループ
        4. 結果返却: レスポンス、ツール呼び出し記録、エスカレーション

    Attributes:
        project_root: プロジェクトルートディレクトリ
        tool_executor: ToolExecutor インスタンス
        claude_client: ClaudeClient インスタンス
    """

    DEFAULT_MAX_TOOL_ITERATIONS = 10

    def __init__(
        self,
        project_root: str,
        llm_config: Optional[LLMConfig] = None,
        tool_executor: Optional[ToolExecutor] = None,
        claude_client: Optional[ClaudeClient] = None,
        task_executor: Optional[Any] = None,
    ):
        """AgentRunner を初期化

        Args:
            project_root: プロジェクトルートディレクトリのパス
            llm_config: LLM設定（省略時はデフォルト設定を使用）
            tool_executor: ToolExecutor インスタンス（省略時は自動生成）
            claude_client: ClaudeClient インスタンス（省略時は自動生成）
            task_executor: TaskExecutor インスタンス（省略時はメモリツール無効）
                          メモリシステムとの連携を有効にするには、
                          Phase 1-4 で実装した TaskExecutor を渡す

        Raises:
            AgentRunnerError: 初期化に失敗した場合
        """
        self.project_root = Path(project_root).resolve()

        if not self.project_root.exists():
            raise AgentRunnerError(
                f"プロジェクトルートが存在しません: {self.project_root}"
            )

        # ツール初期化
        # task_executor が渡された場合、メモリツール（search_memories, record_learning）が有効化される
        if tool_executor is not None:
            self.tool_executor = tool_executor
        else:
            self.tool_executor = create_tool_executor_with_all_tools(
                task_executor=task_executor,
                project_root=str(self.project_root),
                working_dir=str(self.project_root),
            )

        # TaskExecutor を保持（後続処理で使用する場合）
        self.task_executor = task_executor

        # LLMクライアント初期化
        if claude_client is not None:
            self.claude_client = claude_client
        else:
            config = llm_config or LLMConfig()
            try:
                self.claude_client = ClaudeClient(config)
            except ClaudeClientError as e:
                raise AgentRunnerError(
                    "ClaudeClient の初期化に失敗しました",
                    original_error=e,
                ) from e

        # エージェントプロンプトのキャッシュ
        self._agent_prompts: Dict[str, str] = {}

        logger.info(
            f"AgentRunner 初期化完了: project_root={self.project_root}, "
            f"tools={self.tool_executor.list_tools()}"
        )

    def load_agent(self, agent_id: str) -> str:
        """エージェント定義からシステムプロンプトを生成

        prompts/agents/{agent_id}.yaml を読み込んでシステムプロンプトを生成します。

        Args:
            agent_id: エージェントID（YAMLファイル名から拡張子を除いたもの）

        Returns:
            システムプロンプト文字列

        Raises:
            AgentRunnerError: エージェント定義の読み込みに失敗した場合
        """
        # キャッシュ確認
        if agent_id in self._agent_prompts:
            return self._agent_prompts[agent_id]

        # YAMLファイルパス
        yaml_path = self.project_root / "prompts" / "agents" / f"{agent_id}.yaml"

        if not yaml_path.exists():
            raise AgentRunnerError(
                f"エージェント定義ファイルが見つかりません: {yaml_path}",
                agent_id=agent_id,
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                agent_def = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise AgentRunnerError(
                f"エージェント定義の解析に失敗しました: {yaml_path}",
                agent_id=agent_id,
                original_error=e,
            ) from e

        # システムプロンプト生成
        system_prompt = self._generate_system_prompt(agent_id, agent_def)

        # キャッシュに保存
        self._agent_prompts[agent_id] = system_prompt

        logger.info(f"エージェント読み込み完了: agent_id={agent_id}")

        return system_prompt

    def _generate_system_prompt(
        self,
        agent_id: str,
        agent_def: Dict[str, Any],
    ) -> str:
        """エージェント定義からシステムプロンプトを生成

        Args:
            agent_id: エージェントID
            agent_def: YAML から読み込んだエージェント定義

        Returns:
            システムプロンプト文字列
        """
        lines = []

        # ヘッダー
        name = agent_def.get("name", agent_id)
        lines.append(f"# {name}")
        lines.append("")

        # 役割
        if "role" in agent_def:
            lines.append("## 役割")
            lines.append(agent_def["role"].strip())
            lines.append("")

        # 観点
        if "perspectives" in agent_def:
            lines.append("## 判断の観点")
            for perspective in agent_def["perspectives"]:
                pname = perspective.get("name", "")
                pdesc = perspective.get("description", "")
                lines.append(f"- **{pname}**: {pdesc}")
            lines.append("")

        # 基本原則
        if "universal_principles" in agent_def:
            lines.append("## 基本原則")
            for principle in agent_def["universal_principles"]:
                lines.append(f"- {principle}")
            lines.append("")

        # タスク実行フロー
        if "task_execution_flow" in agent_def:
            lines.append("## タスク実行フロー")
            for step_name, step_def in agent_def["task_execution_flow"].items():
                desc = step_def.get("description", step_name)
                lines.append(f"### {desc}")
                if "actions" in step_def:
                    for action in step_def["actions"]:
                        lines.append(f"- {action}")
            lines.append("")

        # 学びのルール
        if "learning_rules" in agent_def:
            lines.append("## 学びの記録ルール")
            lines.append(agent_def["learning_rules"].strip())
            lines.append("")

        # 参照ドキュメント
        if "reference_docs" in agent_def:
            lines.append("## 参照ドキュメント")
            for doc in agent_def["reference_docs"]:
                lines.append(f"- {doc}")
            lines.append("")

        # 外部メモリシステム
        if "memory_system" in agent_def:
            lines.append("## 外部メモリシステム")
            memory_sys = agent_def["memory_system"]
            mem_type = memory_sys.get("type", "postgresql")
            mem_id = memory_sys.get("identifier", "agent_id")
            lines.append(f"- **タイプ**: {mem_type}")
            lines.append(f"- **識別子**: {mem_id}")
            lines.append("- **利用可能なツール**: search_memories, record_learning")
            lines.append(f"- **備考**: {agent_id} をキーとしてメモリを検索・保存できます")
            lines.append("")

        # レポートタイプ
        if "report_types" in agent_def:
            lines.append("## 報告タイプ")
            lines.append(", ".join(agent_def["report_types"]))
            lines.append("")

        return "\n".join(lines)

    def run_task(
        self,
        agent_id: str,
        task: str,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> AgentResult:
        """指定エージェントでタスクを実行

        YAMLからシステムプロンプトを読み込み、LLMを呼び出してタスクを実行します。
        LLMがツールを使用する場合は、ツール実行→結果追加→再呼び出しのループを行います。

        Args:
            agent_id: エージェントID
            task: 実行するタスクの説明
            max_tool_iterations: ツール使用ループの最大回数

        Returns:
            AgentResult: タスク実行結果

        処理フロー:
            1. エージェント定義を読み込み
            2. ツール一覧を取得
            3. LLM呼び出しループ
            4. エスカレーション検出と記録
            5. 結果を返す
        """
        logger.info(
            f"タスク実行開始: agent_id={agent_id}, task={task[:50]!r}..."
        )

        # システムプロンプト読み込み
        try:
            system_prompt = self.load_agent(agent_id)
        except AgentRunnerError as e:
            return AgentResult(
                success=False,
                response="",
                error=str(e),
            )

        # 結果記録用
        tool_calls: List[Dict[str, Any]] = []
        escalations: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0

        # メッセージ初期化
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": task}
        ]

        # ツール一覧
        tools = self.tool_executor.get_tools_for_llm()

        # LLM呼び出しループ
        while True:
            iterations += 1

            # 最大イテレーション回数チェック
            if iterations > max_tool_iterations:
                logger.warning(
                    f"最大イテレーション回数を超過: {max_tool_iterations}"
                )
                return AgentResult(
                    success=True,
                    response=self._extract_last_content(messages),
                    tool_calls=tool_calls,
                    escalations=escalations,
                    tokens_used={
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                    },
                )

            logger.info(f"LLM呼び出し: iteration={iterations}")

            try:
                response = self.claude_client.complete(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools if tools else None,
                )
            except ClaudeClientError as e:
                logger.error(f"LLM呼び出しエラー: {e}")
                return AgentResult(
                    success=False,
                    response="",
                    tool_calls=tool_calls,
                    escalations=escalations,
                    tokens_used={
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                    },
                    error=str(e),
                )

            # トークン使用量を累積
            if response.usage:
                total_input_tokens += response.usage.get("input_tokens", 0)
                total_output_tokens += response.usage.get("output_tokens", 0)

            logger.debug(
                f"LLM応答: stop_reason={response.stop_reason}, "
                f"content_length={len(response.content)}, "
                f"tool_uses={len(response.tool_uses)}"
            )

            # 停止理由による分岐
            if response.stop_reason == "tool_use":
                # アシスタントのレスポンスをメッセージに追加
                assistant_content = self._build_assistant_content(response)
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # 各ツールを実行
                tool_results: List[Dict[str, Any]] = []
                for tool_use in response.tool_uses:
                    logger.info(
                        f"ツール実行: name={tool_use.name}, id={tool_use.id}"
                    )

                    # ツール実行
                    exec_result = self.tool_executor.execute_tool_safe(
                        name=tool_use.name,
                        input_data=tool_use.input,
                    )

                    # 記録を追加
                    tool_record = {
                        "tool_use_id": tool_use.id,
                        "tool_name": tool_use.name,
                        "tool_input": tool_use.input,
                        "result": exec_result.result if exec_result.success else "",
                        "success": exec_result.success,
                    }
                    if exec_result.error:
                        tool_record["error"] = exec_result.error
                    tool_calls.append(tool_record)

                    # エスカレーション検出
                    import json
                    try:
                        result_data = json.loads(exec_result.result) if exec_result.result else {}
                    except json.JSONDecodeError:
                        result_data = {}

                    if result_data.get("escalation_required"):
                        escalation = Escalation(
                            tool_name=tool_use.name,
                            reason=result_data.get("escalation_reason", "不明"),
                            requested_action=result_data.get("requested_action"),
                            risk_level=result_data.get("risk_level"),
                        )
                        escalations.append(escalation.to_dict())
                        logger.warning(
                            f"エスカレーション発生: tool={tool_use.name}, "
                            f"reason={escalation.reason}"
                        )

                    # tool_result を構築
                    if exec_result.success:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": exec_result.result,
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: {exec_result.error}",
                            "is_error": True,
                        })

                    logger.info(
                        f"ツール実行完了: name={tool_use.name}, "
                        f"success={exec_result.success}"
                    )

                # ツール結果をユーザーメッセージとして追加
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                # 次のイテレーションへ
                continue

            else:
                # end_turn または max_tokens の場合は終了
                logger.info(
                    f"タスク実行完了: stop_reason={response.stop_reason}, "
                    f"iterations={iterations}, tool_calls={len(tool_calls)}"
                )

                return AgentResult(
                    success=True,
                    response=response.content,
                    tool_calls=tool_calls,
                    escalations=escalations,
                    tokens_used={
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                    },
                )

    def _build_assistant_content(
        self,
        response,
    ) -> List[Dict[str, Any]]:
        """ClaudeResponse からアシスタントメッセージのコンテンツを構築

        Args:
            response: ClaudeResponse インスタンス

        Returns:
            アシスタントメッセージのコンテンツ配列
        """
        content: List[Dict[str, Any]] = []

        # テキストコンテンツがあれば追加
        if response.content:
            content.append({
                "type": "text",
                "text": response.content,
            })

        # ツール使用を追加
        for tool_use in response.tool_uses:
            content.append({
                "type": "tool_use",
                "id": tool_use.id,
                "name": tool_use.name,
                "input": tool_use.input,
            })

        return content

    def _extract_last_content(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """メッセージ履歴から最後のアシスタント応答を抽出

        Args:
            messages: メッセージ履歴

        Returns:
            最後のアシスタント応答のテキスト
        """
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    texts = [
                        c.get("text", "")
                        for c in content
                        if c.get("type") == "text"
                    ]
                    return " ".join(texts)
                elif isinstance(content, str):
                    return content
        return ""

    def list_agents(self) -> List[str]:
        """利用可能なエージェント一覧を取得

        prompts/agents/ ディレクトリ内の YAML ファイルからエージェントIDを取得します。

        Returns:
            エージェントIDのリスト
        """
        agents_dir = self.project_root / "prompts" / "agents"
        if not agents_dir.exists():
            return []

        agent_ids = []
        for yaml_file in agents_dir.glob("*.yaml"):
            agent_ids.append(yaml_file.stem)

        return sorted(agent_ids)

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """エージェント情報を取得

        Args:
            agent_id: エージェントID

        Returns:
            エージェント情報の辞書（見つからない場合は None）
        """
        yaml_path = self.project_root / "prompts" / "agents" / f"{agent_id}.yaml"

        if not yaml_path.exists():
            return None

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                agent_def = yaml.safe_load(f)
            return {
                "agent_id": agent_id,
                "name": agent_def.get("name", agent_id),
                "description": agent_def.get("description", ""),
                "perspectives": agent_def.get("perspectives", []),
                "memory_system": agent_def.get("memory_system"),
            }
        except Exception:
            return None

    # ==========================================================================
    # 計画ベースのタスク実行（トークン効率改善）
    # ==========================================================================

    def execute_with_planning(
        self,
        agent_id: str,
        spec_file: str,
        task_description: str,
        plan_name: Optional[str] = None,
        auto_execute: bool = True,
    ) -> Dict[str, Any]:
        """計画ベースでタスクを実行（トークン効率改善）

        大きな仕様書を参照するタスクを効率的に実行します。

        処理フロー:
            1. 仕様書を読んで計画を作成（Phase 1）
            2. 計画をファイルに保存
            3. 計画に基づいてサブタスクを順次実行（Phase 2）

        Args:
            agent_id: 実行するエージェントID
            spec_file: 仕様書ファイルパス
            task_description: タスク説明
            plan_name: 計画ファイル名（省略時は自動生成）
            auto_execute: True の場合、計画作成後に自動的に実行

        Returns:
            実行結果の辞書:
            {
                "plan_file": 計画ファイルパス,
                "plan": 計画オブジェクト,
                "results": サブタスク実行結果リスト（auto_execute時）,
                "success": 全体の成功/失敗,
            }
        """
        logger.info(
            f"計画ベース実行開始: agent_id={agent_id}, spec_file={spec_file}"
        )

        # TaskPlanner 初期化
        planner = TaskPlanner(project_root=str(self.project_root))

        # Phase 1: 計画作成
        plan = self._create_plan_from_spec(
            agent_id=agent_id,
            spec_file=spec_file,
            task_description=task_description,
        )

        # 計画を保存
        plan_path = planner.save_plan(plan, plan_name)

        result = {
            "plan_file": str(plan_path),
            "plan": plan,
            "results": [],
            "success": True,
        }

        if not auto_execute:
            logger.info(f"計画作成完了（実行なし）: {plan_path}")
            return result

        # Phase 2: サブタスクを順次実行
        results = self._execute_plan(agent_id, plan, planner, plan_path)
        result["results"] = results
        result["success"] = all(r.get("success", False) for r in results)

        logger.info(
            f"計画ベース実行完了: success={result['success']}, "
            f"subtasks={len(results)}"
        )

        return result

    def execute_from_plan(
        self,
        agent_id: str,
        plan_path: str,
    ) -> Dict[str, Any]:
        """既存の計画ファイルからタスクを実行

        Args:
            agent_id: 実行するエージェントID
            plan_path: 計画ファイルパス

        Returns:
            実行結果の辞書
        """
        logger.info(f"計画ファイルから実行: plan_path={plan_path}")

        planner = TaskPlanner(project_root=str(self.project_root))
        plan = planner.load_plan(plan_path)

        results = self._execute_plan(agent_id, plan, planner, plan_path)

        return {
            "plan_file": plan_path,
            "plan": plan,
            "results": results,
            "success": all(r.get("success", False) for r in results),
        }

    def _create_plan_from_spec(
        self,
        agent_id: str,
        spec_file: str,
        task_description: str,
    ) -> TaskPlan:
        """仕様書を読んで計画を作成

        Args:
            agent_id: エージェントID
            spec_file: 仕様書ファイルパス
            task_description: タスク説明

        Returns:
            TaskPlan インスタンス
        """
        # 仕様書を読み込み
        spec_path = self.project_root / spec_file
        if not spec_path.exists():
            raise AgentRunnerError(f"仕様書が見つかりません: {spec_path}")

        with open(spec_path, "r", encoding="utf-8") as f:
            spec_content = f.read()

        # 計画作成プロンプトを生成
        planning_prompt = get_planning_prompt(task_description, spec_content)

        # LLM に計画作成を依頼
        logger.info("LLM に計画作成を依頼中...")

        result = self.run_task(
            agent_id=agent_id,
            task=planning_prompt,
            max_tool_iterations=3,  # 計画作成は少ないイテレーションで十分
        )

        if not result.success:
            raise AgentRunnerError(
                f"計画作成に失敗しました: {result.error}",
                agent_id=agent_id,
            )

        # レスポンスから YAML を抽出
        plan_data = self._extract_yaml_from_response(result.response)

        # TaskPlanner で計画オブジェクトを作成
        planner = TaskPlanner(project_root=str(self.project_root))
        plan = planner.create_plan_from_spec(
            spec_file=spec_file,
            task_description=task_description,
            subtasks=plan_data.get("subtasks", []),
            context_summary=plan_data.get("context_summary", ""),
        )

        logger.info(
            f"計画作成完了: subtasks={len(plan.subtasks)}, "
            f"estimated_tokens={plan.total_estimated_tokens}"
        )

        return plan

    def _extract_yaml_from_response(self, response: str) -> Dict[str, Any]:
        """LLM レスポンスから YAML を抽出

        Args:
            response: LLM からのレスポンステキスト

        Returns:
            パースされた YAML データ
        """
        import re

        # ```yaml ... ``` ブロックを探す
        yaml_match = re.search(r"```yaml\s*\n(.*?)```", response, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
        else:
            # バッククォートなしの場合、レスポンス全体を試す
            yaml_content = response

        try:
            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                data = {}
            return data
        except yaml.YAMLError as e:
            logger.warning(f"YAML パースエラー: {e}")
            return {}

    def _execute_plan(
        self,
        agent_id: str,
        plan: TaskPlan,
        planner: TaskPlanner,
        plan_path: str,
    ) -> List[Dict[str, Any]]:
        """計画のサブタスクを順次実行

        Args:
            agent_id: エージェントID
            plan: 実行する計画
            planner: TaskPlanner インスタンス
            plan_path: 計画ファイルパス

        Returns:
            各サブタスクの実行結果リスト
        """
        results = []

        while not plan.is_complete():
            subtask = plan.get_next_subtask()
            if subtask is None:
                # 実行可能なタスクがない（依存関係エラー）
                logger.warning("実行可能なサブタスクがありません（依存関係エラー）")
                break

            logger.info(
                f"サブタスク実行: id={subtask.id}, "
                f"description={subtask.description[:50]}..."
            )

            # サブタスクを実行
            subtask.status = "in_progress"
            planner.update_plan(plan, plan_path)

            # タスク説明に context_summary を追加
            task_with_context = self._build_subtask_prompt(plan, subtask)

            try:
                agent_result = self.run_task(
                    agent_id=agent_id,
                    task=task_with_context,
                )

                if agent_result.success:
                    plan.mark_subtask_completed(
                        subtask.id,
                        result=agent_result.response[:500],  # 長すぎる場合は切り詰め
                    )
                    results.append({
                        "subtask_id": subtask.id,
                        "success": True,
                        "response": agent_result.response,
                        "tool_calls": len(agent_result.tool_calls),
                    })
                else:
                    plan.mark_subtask_failed(
                        subtask.id,
                        error=agent_result.error or "Unknown error",
                    )
                    results.append({
                        "subtask_id": subtask.id,
                        "success": False,
                        "error": agent_result.error,
                    })

            except Exception as e:
                logger.error(f"サブタスク実行エラー: {e}")
                plan.mark_subtask_failed(subtask.id, error=str(e))
                results.append({
                    "subtask_id": subtask.id,
                    "success": False,
                    "error": str(e),
                })

            # 計画を更新して保存
            planner.update_plan(plan, plan_path)

        return results

    def _build_subtask_prompt(self, plan: TaskPlan, subtask) -> str:
        """サブタスク用のプロンプトを構築

        context_summary を含めて、仕様書を読まなくても
        タスクが実行できるようにします。

        Args:
            plan: 計画
            subtask: 実行するサブタスク

        Returns:
            プロンプト文字列
        """
        lines = []

        # コンテキスト要約（仕様書の代わり）
        if plan.context_summary:
            lines.append("## 背景情報")
            lines.append(plan.context_summary.strip())
            lines.append("")

        # タスク指示
        lines.append("## タスク")
        lines.append(subtask.task_description.strip())
        lines.append("")

        # 対象ファイル
        if subtask.files:
            lines.append("## 対象ファイル")
            for f in subtask.files:
                lines.append(f"- {f}")
            lines.append("")

        return "\n".join(lines)
