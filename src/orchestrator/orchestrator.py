# オーケストレーター本体
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.2
"""
オーケストレーターモジュール

専門エージェントへのタスクルーティング、結果評価、進捗管理を行う中心コンポーネント。

Phase 2 MVP:
- Router: タスクに最適なエージェントを選択
- Evaluator: ユーザーフィードバックを評価
- TaskExecutor: Phase 1 の外部メモリ機能を再利用

統一設計原則（architecture.ja.md より）:
オーケストレーターは専門エージェントと全く同じ仕組みで動く。違いは「役割」と「観点」だけ。
- 外部メモリ（強度管理、減衰、定着レベル）
- 観点（役割に応じた5つ程度）
- タスクごとに睡眠
- 容量制限と強制剪定
- アーカイブと再活性化

処理フロー:
1. タスク分析（外部メモリ検索）
2. ルーティング判断（Router）
3. タスク委譲（エージェントに依頼）
4. 結果をユーザーに返す
5. フィードバック受信（Evaluator）
6. 学びを記録

設計方針（タスク実行フローエージェント観点）:
- API設計: process_request/receive_feedback の2メソッドで主要フローを提供
- フロー整合性: 外部メモリ検索→ルーティング→委譲→評価の順序を保証
- エラー処理: タイムアウト、エージェント不在時の適切なハンドリング
- 拡張性: Phase 2 MVP はモック委譲、Phase 3 で実際のClaude API呼び出し
- テスト容易性: 各コンポーネントを依存性注入で差し替え可能
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from src.orchestrator.router import Router, RoutingDecision
from src.orchestrator.evaluator import Evaluator, FeedbackResult
from src.agents.agent_registry import AgentRegistry
from src.agents.meta_agent import MetaAgent
from src.core.task_executor import TaskExecutor
from src.config.phase2_config import Phase2Config, ORCHESTRATOR_PERSPECTIVES
from src.orchestrator.progress_manager import ProgressManager, SessionStateRepository
from src.db.connection import DatabaseConnection

# LLM統合（オプション）
try:
    from src.llm import LLMTaskExecutor, LLMTaskResult
    LLM_AVAILABLE = True
except ImportError:
    LLMTaskExecutor = None  # type: ignore
    LLMTaskResult = None  # type: ignore
    LLM_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """オーケストレーター実行結果

    process_request() の戻り値として、オーケストレーション結果を格納。

    Attributes:
        session_id: セッション識別子（UUID）
        routing_decision: ルーティング判断結果
        agent_result: エージェントからの結果（dict形式）
        status: 処理ステータス
            - success: 正常完了
            - partial_success: 一部成功
            - failure: 失敗
            - timeout: タイムアウト
            - no_agent: 適切なエージェントが見つからない
        error_message: エラーメッセージ（失敗時のみ）
        executed_at: 実行日時
    """

    session_id: UUID
    routing_decision: RoutingDecision
    agent_result: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換

        Returns:
            辞書形式の結果（JSON API 対応）
        """
        return {
            "session_id": str(self.session_id),
            "routing_decision": self.routing_decision.to_dict(),
            "agent_result": self.agent_result,
            "status": self.status,
            "error_message": self.error_message,
            "executed_at": self.executed_at.isoformat(),
        }

    @property
    def is_success(self) -> bool:
        """成功したか"""
        return self.status in ("success", "partial_success")

    @property
    def is_failure(self) -> bool:
        """失敗したか"""
        return self.status in ("failure", "timeout", "no_agent")


@dataclass
class SessionContext:
    """セッションコンテキスト

    オーケストレーターのセッション状態を保持。
    中間睡眠からの復帰に使用。

    Attributes:
        session_id: セッション識別子
        task_summary: タスクの概要
        items: 論点リスト
        routing_decision: ルーティング判断結果
        subtask_count: 完了したサブタスク数
        conversation_history: 会話履歴のリスト（user_input, agent_output のペア）
        created_at: セッション作成日時
        last_activity_at: 最後のアクティビティ日時
    """

    session_id: UUID
    task_summary: str
    items: List[str] = field(default_factory=list)
    routing_decision: Optional[RoutingDecision] = None
    subtask_count: int = 0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)


class Orchestrator:
    """オーケストレーター

    専門エージェントへのタスクルーティング、結果評価、進捗管理を行う。

    使用例:
        # コンポーネントの初期化
        db = DatabaseConnection()
        agent_registry = AgentRegistry(db)
        router = Router(agent_registry)
        evaluator = Evaluator()
        task_executor = TaskExecutor(...)  # Phase 1 のコンポーネント

        # オーケストレーターの初期化
        orchestrator = Orchestrator(
            agent_id="orchestrator_01",
            router=router,
            evaluator=evaluator,
            task_executor=task_executor,
        )

        # リクエスト処理
        result = orchestrator.process_request(
            task_summary="ユーザー認証機能を実装",
            items=["認証フロー", "セッション管理"],
        )

        # フィードバック受信
        feedback = orchestrator.receive_feedback(
            session_id=result.session_id,
            user_response="ありがとう、良さそうです",
        )

    Attributes:
        agent_id: オーケストレーター自身のエージェントID
        router: Router インスタンス
        evaluator: Evaluator インスタンス
        task_executor: TaskExecutor インスタンス（Phase 1）
        config: Phase2Config インスタンス
    """

    def __init__(
        self,
        agent_id: str,
        router: Router,
        evaluator: Evaluator,
        task_executor: TaskExecutor,
        config: Optional[Phase2Config] = None,
        llm_task_executor: Optional["LLMTaskExecutor"] = None,
        meta_agent: Optional[MetaAgent] = None,
        db: Optional[DatabaseConnection] = None,
    ):
        """Orchestrator を初期化

        Args:
            agent_id: オーケストレーター自身のエージェントID
            router: Router インスタンス
            evaluator: Evaluator インスタンス
            task_executor: TaskExecutor インスタンス（Phase 1 の仕組みを再利用）
            config: Phase2Config インスタンス（省略時はデフォルト設定）
            llm_task_executor: LLMTaskExecutor インスタンス（オプション、省略時はモック動作）
            meta_agent: MetaAgent インスタンス（オプション、自動エージェント生成機能）
            db: DatabaseConnection インスタンス（セッション永続化用、オプション）
        """
        self.agent_id = agent_id
        self.router = router
        self.evaluator = evaluator
        self.task_executor = task_executor
        self.config = config or Phase2Config()
        self.llm_task_executor = llm_task_executor
        self.meta_agent = meta_agent

        # セッション管理（Phase 2 MVP: インメモリ管理）
        # Phase 3 で DB テーブル（session_state）に移行予定
        self._sessions: Dict[UUID, SessionContext] = {}

        # 進捗管理（セッション永続化用）
        self.progress_manager = None
        if db:
            try:
                session_repository = SessionStateRepository(db)
                self.progress_manager = ProgressManager(
                    session_repository=session_repository,
                    config=config or Phase2Config(),
                )
            except Exception as e:
                logger.warning(f"ProgressManager の初期化に失敗: {e}")

        # 統計情報（睡眠判定に使用）
        self._subtask_completed_count: int = 0
        self._last_activity_time: datetime = datetime.now()

        logger.info(
            f"Orchestrator 初期化完了: agent_id={agent_id}, "
            f"meta_agent_enabled={meta_agent is not None}, "
            f"progress_manager_enabled={self.progress_manager is not None}"
        )

    def process_request(
        self,
        task_summary: str,
        items: Optional[List[str]] = None,
        session_id: Optional[UUID] = None,
    ) -> OrchestratorResult:
        """リクエストを処理

        処理フロー:
        1. セッション管理（新規 or 既存）
        2. 外部メモリで類似タスクを検索（task_executor.search_memories）
        3. ルーティング判断（router.decide）
        4. タスク委譲（_delegate_task）
        5. 結果を返す

        Args:
            task_summary: タスクの概要
            items: 論点リスト（オプション）
            session_id: 既存セッションのID（継続時に指定）

        Returns:
            OrchestratorResult インスタンス

        Note:
            - Phase 2 MVP では同期処理
            - Phase 3 で非同期処理（async/await）に移行予定
        """
        logger.info(
            f"リクエスト処理開始: task_summary={task_summary[:50]!r}..., "
            f"items={len(items or [])}件, session_id={session_id}"
        )

        items = items or []
        self._last_activity_time = datetime.now()

        # 1. セッション管理（新規 or 既存）
        session = self._get_or_create_session(session_id, task_summary, items)

        # 1.1. 既存セッションの場合、会話履歴をタスク概要に追加
        enhanced_task_summary = task_summary
        if session.conversation_history:
            # 直近の会話履歴をコンテキストとして追加（最大3件）
            recent_history = session.conversation_history[-3:]
            context_lines = []
            for idx, entry in enumerate(recent_history, 1):
                context_lines.append(f"[前回の会話 {idx}]")
                context_lines.append(f"ユーザー: {entry.get('user_input', '')}")
                context_lines.append(f"応答: {entry.get('agent_output', '')[:100]}...")

            context = "\n".join(context_lines)
            enhanced_task_summary = f"{context}\n\n[今回の入力]\n{task_summary}"
            logger.info(f"会話履歴を追加: {len(recent_history)}件")

        try:
            # 2. 外部メモリで類似タスクを検索
            past_experiences = self._search_past_experiences(enhanced_task_summary)

            # 3. ルーティング判断
            routing_decision = self.router.decide(
                task_summary=task_summary,
                items=items,
                past_experiences=past_experiences,
            )

            # セッションにルーティング判断を保存
            session.routing_decision = routing_decision

            # 3.1. メタエージェントによる新規エージェント作成判断
            # ルーティング確信度が低い場合、新規エージェント作成を検討
            if (
                self.meta_agent is not None
                and self.config.meta_agent_enabled
                and routing_decision.confidence < self.config.min_routing_confidence
            ):
                logger.info(
                    f"ルーティング確信度が低いため、メタエージェントで新規作成を検討: "
                    f"confidence={routing_decision.confidence}, "
                    f"threshold={self.config.min_routing_confidence}"
                )

                # 既存エージェントを取得
                existing_agents = self.router.agent_registry.get_active_agents()

                # 新規エージェント作成が必要か判断
                if self.meta_agent.should_create_new_agent(
                    task_summary=task_summary,
                    existing_agents=existing_agents,
                    routing_confidence=routing_decision.confidence,
                ):
                    try:
                        # エージェント要件を分析
                        requirements = self.meta_agent.analyze_task_requirements(
                            task_summary=task_summary
                        )

                        # AgentDefinition を生成
                        new_agent = self.meta_agent.generate_agent_definition(
                            requirements=requirements
                        )

                        # AgentRegistry に登録
                        self.router.agent_registry.register(new_agent)

                        logger.info(
                            f"新規エージェントを自動生成して登録: "
                            f"agent_id={new_agent.agent_id}, "
                            f"name={new_agent.name}"
                        )

                        # 自動教育プロセスを実行
                        self._educate_new_agent(new_agent.agent_id)

                        # 新規エージェントを選択
                        routing_decision = RoutingDecision(
                            selected_agent_id=new_agent.agent_id,
                            confidence=0.7,
                            selection_reason=(
                                f"タスクに最適な新規エージェント「{new_agent.name}」を"
                                f"自動生成しました。専門性: {requirements.specialization}"
                            ),
                            candidates=[{
                                "agent_id": new_agent.agent_id,
                                "name": new_agent.name,
                                "score": 0.7,
                                "reason": "新規作成エージェント",
                            }],
                        )

                        # セッションのルーティング判断を更新
                        session.routing_decision = routing_decision

                    except Exception as e:
                        logger.warning(f"新規エージェント作成でエラー: {e}、既存判断を継続")

            # エージェントが見つからない場合
            if not routing_decision.selected_agent_id:
                logger.warning(
                    f"適切なエージェントが見つかりません: "
                    f"task_summary={task_summary[:50]!r}"
                )
                return OrchestratorResult(
                    session_id=session.session_id,
                    routing_decision=routing_decision,
                    agent_result={},
                    status="no_agent",
                    error_message="適切なエージェントが見つかりません",
                )

            # 4. タスク委譲
            agent_result = self._delegate_task(routing_decision, enhanced_task_summary)

            # サブタスク完了をカウント
            self._subtask_completed_count += 1
            session.subtask_count += 1

            # 4.1. 会話履歴を更新
            session.conversation_history.append({
                "user_input": task_summary,
                "agent_output": agent_result.get("output", ""),
                "timestamp": datetime.now().isoformat(),
            })

            # 5. 結果を返す
            result = OrchestratorResult(
                session_id=session.session_id,
                routing_decision=routing_decision,
                agent_result=agent_result,
                status="success",
            )

            logger.info(
                f"リクエスト処理完了: session_id={session.session_id}, "
                f"selected_agent={routing_decision.selected_agent_id}, "
                f"status={result.status}"
            )

            return result

        except Exception as e:
            logger.error(f"リクエスト処理でエラー: {e}")
            return OrchestratorResult(
                session_id=session.session_id,
                routing_decision=session.routing_decision or RoutingDecision(
                    selected_agent_id="",
                    selection_reason="エラーにより判断不能",
                ),
                agent_result={},
                status="failure",
                error_message=str(e),
            )

        finally:
            # オーケストレーター自身の自動睡眠フェーズ
            # _should_sleep() の内部ロジックを使用（サブタスク完了数またはアイドル時間）
            if self.config.auto_sleep_after_task and self._should_sleep():
                logger.info("オーケストレーター睡眠トリガー条件を満たしました")
                self._run_sleep_phase()

    def receive_feedback(
        self,
        session_id: UUID,
        user_response: str,
    ) -> FeedbackResult:
        """ユーザーフィードバックを受信して評価

        処理フロー:
        1. evaluator.evaluate で判定
        2. 学びを記録（必要な場合）
        3. 結果を返す

        Args:
            session_id: セッション識別子
            user_response: ユーザーの応答テキスト

        Returns:
            FeedbackResult インスタンス

        Note:
            - セッションが見つからない場合は neutral を返す
            - positive/negative/redo_requested の場合に学びを記録
        """
        logger.info(
            f"フィードバック受信: session_id={session_id}, "
            f"user_response={user_response[:30]!r}..."
        )

        self._last_activity_time = datetime.now()

        # 1. フィードバックを評価
        feedback_result = self.evaluator.evaluate(user_response)

        # セッション情報を取得（学び記録に使用）
        session = self._sessions.get(session_id)

        # 2. 学びを記録（必要な場合）
        if session and feedback_result.feedback_type in (
            "positive", "negative", "redo_requested"
        ):
            try:
                self._record_routing_learning(
                    session_id=session_id,
                    routing_decision=session.routing_decision,
                    feedback_type=feedback_result.feedback_type,
                )
            except Exception as e:
                logger.warning(f"学び記録でエラー: {e}")

        # 実行履歴を記録（負荷分散用）
        if session and session.routing_decision:
            success = feedback_result.feedback_type in ("positive", "neutral")
            self.router.record_execution(
                agent_id=session.routing_decision.selected_agent_id,
                success=success,
            )

        logger.info(
            f"フィードバック処理完了: feedback_type={feedback_result.feedback_type}, "
            f"confidence={feedback_result.confidence}"
        )

        return feedback_result

    def _get_or_create_session(
        self,
        session_id: Optional[UUID],
        task_summary: str,
        items: List[str],
    ) -> SessionContext:
        """セッションを取得または作成

        Args:
            session_id: 既存セッションのID（継続時）
            task_summary: タスクの概要
            items: 論点リスト

        Returns:
            SessionContext インスタンス
        """
        # メモリ内に存在する場合
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity_at = datetime.now()
            logger.debug(f"既存セッションを継続（メモリ内）: session_id={session_id}")
            return session

        # DBから復元を試みる
        if session_id and self.progress_manager:
            try:
                session_state = self.progress_manager.restore_state(session_id)
                if session_state:
                    # SessionStateからSessionContextを復元
                    session = SessionContext(
                        session_id=session_id,
                        task_summary=session_state.user_request.get("original", task_summary),
                        items=items,
                        conversation_history=session_state.task_tree.get("conversation_history", []),
                        created_at=session_state.created_at,
                        last_activity_at=datetime.now(),
                    )
                    self._sessions[session_id] = session
                    logger.info(f"セッションをDBから復元: session_id={session_id}")
                    return session
            except Exception as e:
                logger.warning(f"セッション復元に失敗: {e}")

        # 新規セッション作成
        new_session_id = session_id or uuid4()
        session = SessionContext(
            session_id=new_session_id,
            task_summary=task_summary,
            items=items,
        )
        self._sessions[new_session_id] = session

        logger.debug(f"新規セッション作成: session_id={new_session_id}")
        return session

    def _search_past_experiences(
        self,
        task_summary: str,
    ) -> List[Dict[str, Any]]:
        """過去の経験（類似タスク）を検索

        TaskExecutor.search_memories() を使用して、
        オーケストレーター自身の外部メモリから類似タスクを検索。

        Args:
            task_summary: タスクの概要

        Returns:
            過去の経験リスト（dict形式）
        """
        try:
            # オーケストレーター自身の外部メモリを検索
            scored_memories = self.task_executor.search_memories(
                query=task_summary,
                agent_id=self.agent_id,
                perspective="エージェント適性",  # ORCHESTRATOR_PERSPECTIVES[1]
            )

            # ScoredMemory を dict 形式に変換
            experiences = []
            for sm in scored_memories:
                memory = sm.memory
                # routing_context があれば抽出
                routing_context = {}
                if hasattr(memory, "routing_context") and memory.routing_context:
                    routing_context = memory.routing_context

                experiences.append({
                    "memory_id": str(memory.id),
                    "content": memory.content,
                    "agent_id": routing_context.get("agent_selected"),
                    "selected_agent_id": routing_context.get("agent_selected"),
                    "success": routing_context.get("success"),
                    "result_status": "success" if routing_context.get("success") else "failure",
                    "user_feedback": routing_context.get("feedback_type"),
                    "score": sm.final_score,
                })

            logger.debug(
                f"過去の経験検索完了: query={task_summary[:30]!r}..., "
                f"found={len(experiences)}件"
            )

            return experiences

        except Exception as e:
            logger.warning(f"過去の経験検索でエラー: {e}")
            return []

    def _delegate_task(
        self,
        routing_decision: RoutingDecision,
        task_summary: str,
    ) -> Dict[str, Any]:
        """エージェントにタスクを委譲

        LLMTaskExecutor が利用可能な場合は実際の Claude API を呼び出し、
        そうでない場合はモック結果を返す（テスト互換性のため）。

        Args:
            routing_decision: ルーティング判断結果
            task_summary: タスクの概要

        Returns:
            エージェントからの結果（dict形式）

        Note:
            - llm_task_executor が設定されている場合は実際のLLM呼び出し
            - llm_task_executor が None の場合はモック動作（後方互換性）
        """
        agent_id = routing_decision.selected_agent_id

        logger.info(
            f"タスク委譲: agent_id={agent_id}, "
            f"task_summary={task_summary[:50]!r}..."
        )

        # LLMTaskExecutor が利用可能な場合は実際のLLM呼び出し
        if self.llm_task_executor is not None:
            return self._delegate_task_with_llm(
                routing_decision=routing_decision,
                task_summary=task_summary,
            )

        # フォールバック: モック結果を返す（テスト互換性）
        mock_result = {
            "agent_id": agent_id,
            "task_summary": task_summary,
            "status": "completed",
            "output": f"[Phase 2 MVP] {agent_id} によるタスク実行結果（モック）",
            "executed_at": datetime.now().isoformat(),
            "metadata": {
                "routing_confidence": routing_decision.confidence,
                "selection_reason": routing_decision.selection_reason,
            },
        }

        logger.debug(f"タスク委譲完了（モック）: agent_id={agent_id}")

        return mock_result

    def _delegate_task_with_llm(
        self,
        routing_decision: RoutingDecision,
        task_summary: str,
    ) -> Dict[str, Any]:
        """LLMを使用してタスクを委譲

        エージェント定義からsystem_promptを取得し、
        LLMTaskExecutor.execute_task_with_tools() を呼び出す。

        Args:
            routing_decision: ルーティング判断結果
            task_summary: タスクの概要

        Returns:
            エージェントからの結果（dict形式）
        """
        agent_id = routing_decision.selected_agent_id

        # エージェント定義を取得
        agent_definition = self.router.agent_registry.get_by_id(agent_id)
        if agent_definition is None:
            logger.error(f"エージェント定義が見つかりません: agent_id={agent_id}")
            return {
                "agent_id": agent_id,
                "task_summary": task_summary,
                "status": "error",
                "output": f"エージェント定義が見つかりません: {agent_id}",
                "executed_at": datetime.now().isoformat(),
                "metadata": {
                    "routing_confidence": routing_decision.confidence,
                    "selection_reason": routing_decision.selection_reason,
                    "error": "agent_not_found",
                },
            }

        # system_prompt を取得
        system_prompt = agent_definition.system_prompt

        # 観点を取得（エージェント定義の最初の観点を使用）
        perspective = None
        if agent_definition.perspectives:
            perspective = agent_definition.perspectives[0]

        logger.info(
            f"LLMタスク実行: agent_id={agent_id}, "
            f"perspective={perspective}"
        )

        try:
            # LLMTaskExecutor でタスク実行
            llm_result = self.llm_task_executor.execute_task_with_tools(
                agent_id=agent_id,
                system_prompt=system_prompt,
                task_description=task_summary,
                perspective=perspective,
            )

            # LLMTaskResult を dict 形式に変換
            result = {
                "agent_id": agent_id,
                "task_summary": task_summary,
                "status": "completed" if llm_result.stop_reason in ("end_turn", "max_tokens") else "partial",
                "output": llm_result.content,
                "executed_at": datetime.now().isoformat(),
                "metadata": {
                    "routing_confidence": routing_decision.confidence,
                    "selection_reason": routing_decision.selection_reason,
                    "llm_result": llm_result.to_dict(),
                },
            }

            logger.info(
                f"LLMタスク実行完了: agent_id={agent_id}, "
                f"stop_reason={llm_result.stop_reason}, "
                f"tool_calls={len(llm_result.tool_calls)}"
            )

            return result

        except Exception as e:
            logger.error(f"LLMタスク実行でエラー: {e}")
            return {
                "agent_id": agent_id,
                "task_summary": task_summary,
                "status": "error",
                "output": f"LLMタスク実行でエラー: {str(e)}",
                "executed_at": datetime.now().isoformat(),
                "metadata": {
                    "routing_confidence": routing_decision.confidence,
                    "selection_reason": routing_decision.selection_reason,
                    "error": str(e),
                },
            }

    def _record_routing_learning(
        self,
        session_id: UUID,
        routing_decision: Optional[RoutingDecision],
        feedback_type: str,
    ) -> None:
        """ルーティング結果の学びを記録

        TaskExecutor.record_learning() を使用して、
        オーケストレーター自身の外部メモリに学びを記録。

        Args:
            session_id: セッション識別子
            routing_decision: ルーティング判断結果
            feedback_type: フィードバックタイプ（positive/negative/redo_requested）
        """
        if not routing_decision:
            return

        session = self._sessions.get(session_id)
        if not session:
            return

        # 学びの内容を構築
        agent_id = routing_decision.selected_agent_id
        success = feedback_type == "positive"
        task_summary = session.task_summary[:100]

        if success:
            content = (
                f"タスク「{task_summary}」を{agent_id}に委譲して成功。"
                f"選択理由: {routing_decision.selection_reason}"
            )
            learning = (
                f"{agent_id}はこのタイプのタスクに適している。"
                f"確信度: {routing_decision.confidence:.2f}"
            )
        elif feedback_type == "redo_requested":
            content = (
                f"タスク「{task_summary}」を{agent_id}に委譲したが、"
                f"やり直しが要求された。選択理由: {routing_decision.selection_reason}"
            )
            learning = (
                f"{agent_id}はこのタイプのタスクには不適切だった可能性。"
                f"別のエージェントを検討すべき。"
            )
        else:  # negative
            content = (
                f"タスク「{task_summary}」を{agent_id}に委譲したが、"
                f"否定的なフィードバックを受けた。"
            )
            learning = (
                f"{agent_id}のこのタイプのタスクへの対応を改善する必要あり。"
            )

        try:
            memory_id = self.task_executor.record_learning(
                agent_id=self.agent_id,
                content=content,
                learning=learning,
                perspective="エージェント適性",
            )

            logger.info(
                f"ルーティング学び記録完了: memory_id={memory_id}, "
                f"feedback_type={feedback_type}"
            )

        except Exception as e:
            logger.warning(f"ルーティング学び記録でエラー: {e}")

    def _should_sleep(self) -> bool:
        """睡眠すべきか判定

        睡眠トリガー条件（仕様書より）:
        1. サブタスク完了数が orchestrator_subtask_batch_size 以上
        2. アイドル時間が orchestrator_idle_timeout_minutes 以上

        Returns:
            睡眠すべき場合 True

        Note:
            - コンテキスト使用率（orchestrator_context_threshold）は
              Phase 3 でトークンカウント実装後に対応
        """
        # 条件1: サブタスク完了数
        if self._subtask_completed_count >= self.config.orchestrator_subtask_batch_size:
            logger.debug(
                f"睡眠条件1: サブタスク完了数 "
                f"{self._subtask_completed_count} >= "
                f"{self.config.orchestrator_subtask_batch_size}"
            )
            return True

        # 条件2: アイドル時間
        idle_threshold = timedelta(
            minutes=self.config.orchestrator_idle_timeout_minutes
        )
        idle_time = datetime.now() - self._last_activity_time

        if idle_time >= idle_threshold:
            logger.debug(
                f"睡眠条件2: アイドル時間 "
                f"{idle_time.total_seconds() / 60:.1f}分 >= "
                f"{self.config.orchestrator_idle_timeout_minutes}分"
            )
            return True

        return False

    def _run_sleep_phase(self) -> None:
        """睡眠フェーズを実行

        TaskExecutor.run_sleep_phase() を使用して、
        オーケストレーター自身の外部メモリの睡眠処理を実行。
        """
        logger.info(f"睡眠フェーズ開始: agent_id={self.agent_id}")

        try:
            result = self.task_executor.run_sleep_phase(self.agent_id)

            # サブタスク完了カウントをリセット
            self._subtask_completed_count = 0

            logger.info(
                f"睡眠フェーズ完了: "
                f"decayed={result.decayed_count}, "
                f"archived={result.archived_count}, "
                f"consolidated={result.consolidated_count}"
            )

        except Exception as e:
            logger.warning(f"睡眠フェーズでエラー: {e}")

    def _educate_new_agent(self, agent_id: str) -> None:
        """新規エージェントに基礎知識を自動注入

        デフォルト教科書をロードし、教育プロセスを実行して
        新規エージェントに基礎知識を注入する。

        Args:
            agent_id: 教育対象のエージェントID

        Note:
            - 教育失敗してもエージェント作成は成功扱い
            - config.auto_educate_new_agents が False の場合は何もしない
        """
        if not self.config.auto_educate_new_agents:
            logger.debug(f"自動教育は無効です: agent_id={agent_id}")
            return

        try:
            # デフォルト教科書をロード
            from src.education.textbook import TextbookLoader
            from src.education.education_process import EducationProcess

            loader = TextbookLoader()
            textbook = loader.load(self.config.default_textbook_path)

            logger.info(
                f"新規エージェント {agent_id} への教育開始: "
                f"textbook={textbook.title}"
            )

            # 教育プロセス実行
            education = EducationProcess(
                agent_id=agent_id,
                textbook=textbook,
                repository=self.task_executor.repository,
                embedding_client=self.task_executor.vector_search.embedding_client,
                config=self.config,
            )

            result = education.run()

            # ログ出力
            logger.info(
                f"新規エージェント {agent_id} への教育完了: "
                f"章={result.chapters_completed}, "
                f"記憶={result.memories_created}, "
                f"テスト合格率={result.pass_rate:.1%}"
            )

        except FileNotFoundError as e:
            # 教科書ファイルが見つからない場合は警告
            logger.warning(
                f"デフォルト教科書が見つかりません: {self.config.default_textbook_path}. "
                f"エージェント {agent_id} への自動教育をスキップします。"
            )

        except Exception as e:
            # 教育失敗してもエージェント作成は成功扱い
            logger.warning(f"エージェント {agent_id} への自動教育に失敗: {e}")

    def get_session(self, session_id: UUID) -> Optional[SessionContext]:
        """セッションを取得

        Args:
            session_id: セッション識別子

        Returns:
            SessionContext インスタンス、見つからない場合は None
        """
        return self._sessions.get(session_id)

    def clear_sessions(self) -> None:
        """全セッションをクリア（テスト用）"""
        self._sessions.clear()
        logger.debug("全セッションをクリアしました")

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報（dict形式）
        """
        return {
            "agent_id": self.agent_id,
            "active_sessions": len(self._sessions),
            "subtask_completed_count": self._subtask_completed_count,
            "last_activity_time": self._last_activity_time.isoformat(),
            "idle_minutes": (
                datetime.now() - self._last_activity_time
            ).total_seconds() / 60,
        }
