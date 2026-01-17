# タスク実行フロー統合クラス
# メモリ検索・2段階強化・学び記録のフローを統合
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション3 (タスク実行フロー)
# アーキテクチャ: docs/architecture.ja.md セクション4.3 (タスク実行フロー)
"""
タスク実行フロー統合モジュール

タスク実行時のメモリ検索・2段階強化・学び記録フローを統合管理。
各コンポーネント（VectorSearch, MemoryRanker, StrengthManager等）を
コンポジションパターンで保持し、一貫したフローを提供。

設計方針（タスク実行フローエージェント観点）:
- API設計: 呼び出し側の負担を最小化。複数モジュールの連携を隠蔽
- フロー整合性: 2段階強化のタイミング（候補時 vs 使用時）を厳密に分離
- エラー処理: 部分的な失敗時も処理継続可能な設計
- 拡張性: 依存性注入でコンポーネント差し替え可能
- テスト容易性: 各コンポーネントをモック差し替え可能

参照メモリ:
- mem_task_001: コンポジションパターンの有効性
- mem_task_002: 2段階強化の分離設計
- mem_task_003: keyword方式の使用判定
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from src.config.phase1_config import Phase1Config
from src.core.memory_repository import MemoryRepository
from src.core.strength_manager import StrengthManager
from src.core.sleep_processor import SleepPhaseProcessor, SleepPhaseResult
from src.models.memory import AgentMemory
from src.search.vector_search import VectorSearch
from src.search.ranking import MemoryRanker, ScoredMemory


logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """タスク実行結果

    execute_task() の戻り値として、タスク実行の全結果を格納。
    検索されたメモリ、実際に使用されたメモリ、タスク結果などを含む。

    Attributes:
        task_result: タスク関数の実行結果（Any型）
        searched_memories: 検索でヒットしたメモリのリスト（ScoredMemory）
        used_memory_ids: 実際に使用されたと判定されたメモリのIDリスト
        recorded_memory_id: 新たに記録された学びのメモリID（Optional）
        executed_at: 実行日時
        errors: 処理中に発生したエラーのリスト
    """
    task_result: Any
    searched_memories: List[ScoredMemory] = field(default_factory=list)
    used_memory_ids: List[UUID] = field(default_factory=list)
    recorded_memory_id: Optional[UUID] = None
    executed_at: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """辞書形式に変換

        REST API レスポンスや JSON シリアライズ用。

        Returns:
            辞書形式の結果
        """
        return {
            "task_result": self.task_result,
            "searched_memories": [
                {
                    "memory_id": str(sm.memory.id),
                    "content": sm.memory.content[:100],  # 先頭100文字
                    "final_score": sm.final_score,
                }
                for sm in self.searched_memories
            ],
            "used_memory_ids": [str(mid) for mid in self.used_memory_ids],
            "recorded_memory_id": str(self.recorded_memory_id) if self.recorded_memory_id else None,
            "executed_at": self.executed_at.isoformat(),
            "errors": self.errors,
            "success": len(self.errors) == 0,
        }


class TaskExecutor:
    """タスク実行フロー統合クラス

    メモリ検索・2段階強化・学び記録のフローを統合管理。

    コンポジションパターン（mem_task_001参照）:
        VectorSearch, MemoryRanker, StrengthManager, SleepPhaseProcessor,
        MemoryRepository を内部で保持し、外部には統合されたメソッドを公開。

    2段階強化の分離（mem_task_002参照）:
        - search_memories(): 候補強化（candidate_count++）のみ
        - reinforce_used_memories(): 使用強化（access_count++, strength強化）

    使用例:
        # 依存コンポーネントの初期化
        db = DatabaseConnection()
        config = Phase1Config()
        embedding_client = AzureEmbeddingClient()

        vector_search = VectorSearch(db, embedding_client, config)
        ranker = MemoryRanker(config)
        repository = MemoryRepository(db, config)
        strength_manager = StrengthManager(repository, config)
        sleep_processor = SleepPhaseProcessor(db, config)

        # TaskExecutor の初期化
        executor = TaskExecutor(
            vector_search=vector_search,
            ranker=ranker,
            strength_manager=strength_manager,
            sleep_processor=sleep_processor,
            repository=repository,
            config=config,
        )

        # メモリ検索（候補強化のみ）
        memories = executor.search_memories("緊急調達のコスト", "agent_01")

        # タスク実行（検索→タスク→使用判定→強化→学び記録の統合フロー）
        result = executor.execute_task(
            query="緊急調達のコスト",
            agent_id="agent_01",
            task_func=my_task_function,
        )

    Attributes:
        vector_search: VectorSearch インスタンス（Stage 1 検索）
        ranker: MemoryRanker インスタンス（Stage 2 ランキング）
        strength_manager: StrengthManager インスタンス（2段階強化）
        sleep_processor: SleepPhaseProcessor インスタンス（睡眠フェーズ）
        repository: MemoryRepository インスタンス（CRUD操作）
        config: Phase1Config インスタンス
    """

    def __init__(
        self,
        vector_search: VectorSearch,
        ranker: MemoryRanker,
        strength_manager: StrengthManager,
        sleep_processor: SleepPhaseProcessor,
        repository: MemoryRepository,
        config: Optional[Phase1Config] = None,
    ):
        """TaskExecutor を初期化

        依存性注入パターンで各コンポーネントを受け取る。
        テスト時にはモックを注入可能。

        Args:
            vector_search: VectorSearch インスタンス
            ranker: MemoryRanker インスタンス
            strength_manager: StrengthManager インスタンス
            sleep_processor: SleepPhaseProcessor インスタンス
            repository: MemoryRepository インスタンス
            config: Phase1Config インスタンス（省略時はデフォルト設定）
        """
        self.vector_search = vector_search
        self.ranker = ranker
        self.strength_manager = strength_manager
        self.sleep_processor = sleep_processor
        self.repository = repository
        self.config = config or Phase1Config()
        self._task_count = 0  # タスク実行カウンター（自動睡眠フェーズ用）

        logger.info("TaskExecutor 初期化完了")

    # === 検索フロー ===

    def search_memories(
        self,
        query: str,
        agent_id: str,
        perspective: Optional[str] = None,
    ) -> List[ScoredMemory]:
        """メモリ検索を実行（2段階強化の Stage 1: 候補強化）

        クエリに基づいてメモリを検索し、検索候補として強化する。

        処理フロー:
            1. VectorSearch で候補取得（Stage 1: 関連性フィルタ）
            2. MemoryRanker でスコア合成・ランキング（Stage 2: 優先度ランキング）
            3. StrengthManager.mark_as_candidate() で候補強化（candidate_count++）
            4. 結果返却

        Args:
            query: 検索クエリ（テキスト）
            agent_id: 検索対象のエージェントID
            perspective: 観点（指定時は観点別強度を考慮）

        Returns:
            ScoredMemory のリスト（スコア降順）

        Note:
            - この時点では candidate_count++ のみ（strength は変更しない）
            - 実際に使用された場合は reinforce_used_memories() を別途呼び出す
            - 空のクエリの場合は空リストを返す

        Raises:
            VectorSearchError: ベクトル検索に失敗した場合
        """
        # 空クエリチェック
        if not query or not query.strip():
            logger.warning("空のクエリが渡されました")
            return []

        logger.info(
            f"メモリ検索開始: query={query[:30]!r}..., "
            f"agent_id={agent_id}, perspective={perspective}"
        )

        # Stage 1: ベクトル検索で候補を取得
        # VectorSearch.search_candidates() → List[Tuple[AgentMemory, float]]
        candidates = self.vector_search.search_candidates(
            query=query,
            agent_id=agent_id,
            perspective=perspective,
        )

        if not candidates:
            logger.info("検索候補が見つかりませんでした")
            return []

        # Stage 2: スコア合成・ランキング
        # MemoryRanker.rank() → List[ScoredMemory]
        ranked_memories = self.ranker.rank(
            candidates=candidates,
            perspective=perspective,
        )

        if not ranked_memories:
            logger.info("ランキング後の結果が空です")
            return []

        # 2段階強化 Stage 1: 候補強化（candidate_count++）
        # 検索候補として選ばれた時点で candidate_count をインクリメント
        memory_ids = [sm.memory.id for sm in ranked_memories]
        updated_count = self.strength_manager.mark_as_candidate(memory_ids)

        logger.info(
            f"メモリ検索完了: "
            f"candidates={len(candidates)}, "
            f"ranked={len(ranked_memories)}, "
            f"reinforced={updated_count}"
        )

        return ranked_memories

    # === 使用判定 ===

    def identify_used_memories(
        self,
        task_result: Any,
        candidates: List[ScoredMemory],
    ) -> List[UUID]:
        """タスク結果からメモリ使用を判定（keyword方式）

        タスク結果テキストにメモリの内容（キーワード）が含まれるかを判定。
        Phase 1 では keyword 方式を採用（mem_task_003参照）。

        処理フロー:
            1. タスク結果を文字列に変換
            2. 各候補メモリの content からキーワードを抽出
            3. タスク結果にキーワードが含まれるかを判定
            4. 使用されたと判定されたメモリのIDリストを返却

        Args:
            task_result: タスク関数の実行結果（文字列に変換可能な型）
            candidates: 検索候補の ScoredMemory リスト

        Returns:
            使用されたと判定されたメモリの UUID リスト

        Note:
            - Phase 1 では keyword 方式（シンプルなキーワードマッチング）
            - Phase 2 以降で similarity 方式や llm 方式への移行を検討
            - 判定失敗時は空リストを返す（例外をスローしない）
            - config.use_detection_method で方式を切り替え可能（将来対応）
        """
        # 空チェック（防御的実装：例外をスローせず空リストを返す）
        if not candidates:
            logger.debug("identify_used_memories: candidates が空です")
            return []

        # タスク結果を文字列に変換
        try:
            task_result_str = str(task_result) if task_result is not None else ""
        except Exception as e:
            logger.warning(
                f"identify_used_memories: task_result の文字列変換に失敗: {e}"
            )
            return []

        # 空の task_result には空リストを返す
        if not task_result_str or not task_result_str.strip():
            logger.debug("identify_used_memories: task_result が空です")
            return []

        logger.info(
            f"使用判定開始: candidates={len(candidates)}, "
            f"task_result={task_result_str[:50]!r}..."
        )

        used_memory_ids: List[UUID] = []

        for scored_memory in candidates:
            memory = scored_memory.memory

            # メモリの content からキーワードを抽出
            keywords = self._extract_keywords(memory.content)

            if not keywords:
                # キーワードが抽出できない場合はスキップ
                logger.debug(
                    f"メモリ {memory.id}: キーワード抽出結果が空、スキップ"
                )
                continue

            # タスク結果にキーワードが含まれるか判定
            if self._matches_any_keyword(task_result_str, keywords):
                used_memory_ids.append(memory.id)
                logger.debug(
                    f"メモリ {memory.id}: 使用されたと判定 "
                    f"(keywords={keywords[:5]}...)"
                )
            else:
                logger.debug(
                    f"メモリ {memory.id}: 使用されていないと判定"
                )

        logger.info(
            f"使用判定完了: {len(used_memory_ids)}/{len(candidates)} が使用された"
        )

        return used_memory_ids

    # === 使用強化 ===

    def reinforce_used_memories(
        self,
        memory_ids: List[UUID],
        agent_id: str,
        perspective: Optional[str] = None,
    ) -> int:
        """使用されたメモリを強化（2段階強化の Stage 2: 使用強化）

        実際に使用されたメモリの access_count と strength を強化する。

        処理フロー:
            1. 各メモリに対して StrengthManager.mark_as_used() を呼び出し
            2. 観点指定時は観点別強度も強化
            3. 強化されたメモリ数を返却

        Args:
            memory_ids: 使用されたメモリの UUID リスト
            agent_id: エージェントID（ログ用）
            perspective: 観点（指定時は strength_by_perspective も強化）

        Returns:
            強化されたメモリ数

        Note:
            - search_memories() で候補強化済みのメモリに対して呼び出す
            - 強化量は config.strength_increment_on_use (0.1) を使用
            - 観点指定時は config.perspective_strength_increment (0.15) も加算
            - 部分的な失敗時もログ記録して継続
        """
        # 空リストの場合は早期リターン
        if not memory_ids:
            logger.debug(
                f"reinforce_used_memories: memory_ids が空です "
                f"(agent_id={agent_id})"
            )
            return 0

        logger.info(
            f"使用強化開始: "
            f"count={len(memory_ids)}, "
            f"agent_id={agent_id}, "
            f"perspective={perspective}"
        )

        success_count = 0
        failed_ids: List[UUID] = []

        # 各メモリに対して mark_as_used を呼び出す
        for memory_id in memory_ids:
            try:
                result = self.strength_manager.mark_as_used(
                    memory_id=memory_id,
                    perspective=perspective,
                )
                if result is not None:
                    success_count += 1
                    logger.debug(
                        f"メモリ強化成功: memory_id={memory_id}, "
                        f"new_strength={result.strength}, "
                        f"new_access_count={result.access_count}"
                    )
                else:
                    # mark_as_used が None を返した場合（メモリが存在しない）
                    failed_ids.append(memory_id)
                    logger.warning(
                        f"メモリ強化失敗（メモリが存在しない）: memory_id={memory_id}"
                    )
            except Exception as e:
                # 例外が発生した場合もログを記録して継続
                failed_ids.append(memory_id)
                logger.warning(
                    f"メモリ強化失敗（例外）: memory_id={memory_id}, error={e}"
                )

        # 最終結果をログに記録
        if failed_ids:
            logger.warning(
                f"使用強化完了（一部失敗）: "
                f"success={success_count}/{len(memory_ids)}, "
                f"failed_ids={[str(fid) for fid in failed_ids]}"
            )
        else:
            logger.info(
                f"使用強化完了: "
                f"success={success_count}/{len(memory_ids)}, "
                f"agent_id={agent_id}"
            )

        return success_count

    # === 学び記録 ===

    def record_learning(
        self,
        agent_id: str,
        content: str,
        learning: str,
        perspective: Optional[str] = None,
    ) -> UUID:
        """例外的なイベントを学びとして記録

        エラー解決、予想外の挙動発見、効率的な方法発見などの
        例外的なイベントのみを新規メモリとして保存する。

        注意: 「テスト36件成功」「0.1秒で完了」のような報告や、
        汎用的な自動生成メッセージは記録しない。
        呼び出し側が明示的に学びを指定した場合のみ記録する。

        処理フロー:
            1. content の空チェック
            2. AzureEmbeddingClient でエンベディングを生成
            3. strength_by_perspective の初期化（perspective 指定時）
            4. AgentMemory.create() で新しいメモリを作成（source = "task"）
            5. MemoryRepository.create() でDBに保存
            6. 作成されたメモリのUUIDを返却

        Args:
            agent_id: エージェントID
            content: 学びの内容（メモリのメインコンテンツ）
            learning: 学びの詳細テキスト（単純なTEXT）
            perspective: 観点名（強度管理で使用、指定時は
                        strength_by_perspective[perspective] = 1.0）

        Returns:
            作成されたメモリの UUID

        Note:
            - 初期 strength は 1.0（config.initial_strength）
            - source フィールドは "task" を設定
            - perspective 指定時は strength_by_perspective[perspective] = 1.0

        Raises:
            ValueError: content が空の場合
            AzureEmbeddingError: エンベディング生成に失敗した場合（呼び出し側でハンドリング）
        """
        # content の空チェック
        if not content or not content.strip():
            raise ValueError("content を空にすることはできません")

        logger.info(
            f"学び記録開始: agent_id={agent_id}, "
            f"content={content[:30]!r}..., perspective={perspective}"
        )

        # エンベディングを生成
        # VectorSearch が保持している embedding_client を使用
        embedding = self.vector_search.embedding_client.get_embedding(content)

        # strength_by_perspective の初期化
        # perspective 指定時はその観点を 1.0 で初期化
        strength_by_perspective: Dict[str, float] = {}
        if perspective:
            strength_by_perspective[perspective] = 1.0

        # AgentMemory インスタンスを生成
        memory = AgentMemory.create(
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            strength=1.0,  # 初期強度
            strength_by_perspective=strength_by_perspective,
            learning=learning,  # 単純なTEXT
            source="task",  # タスク実行で得た学び
        )

        # MemoryRepository.create() でDBに保存
        created_memory = self.repository.create(memory)

        logger.info(
            f"学び記録完了: memory_id={created_memory.id}, "
            f"agent_id={agent_id}"
        )

        return created_memory.id

    # === 統合フロー ===

    def execute_task(
        self,
        query: str,
        agent_id: str,
        task_func: Callable[[List[ScoredMemory]], Any],
        perspective: Optional[str] = None,
        extract_learning: bool = False,
        learning_content: Optional[str] = None,
        learning_text: Optional[str] = None,
        auto_sleep: bool = True,
    ) -> TaskExecutionResult:
        """タスク実行の統合フロー

        メモリ検索→タスク実行→使用判定→使用強化→学び記録の
        一連のフローを統合実行する。

        処理フロー:
            1. search_memories() でメモリ検索（候補強化）
            2. task_func() を実行（検索結果を引数として渡す）
            3. identify_used_memories() で使用判定
            4. reinforce_used_memories() で使用強化
            5. learning_content と learning_text が明示的に指定された場合のみ学び記録
            6. TaskExecutionResult を返却

        学び記録について:
            学びは「例外的なイベント」のみ記録する。
            - エラー解決、予想外の挙動発見、効率的な方法発見など
            - 「テスト成功」「処理完了」のような報告は記録しない
            - 汎用的な自動生成メッセージは記録しない
            呼び出し側が learning_content と learning_text を明示的に指定した場合のみ記録。

        Args:
            query: 検索クエリ
            agent_id: エージェントID
            task_func: 実行するタスク関数
                      シグネチャ: (memories: List[ScoredMemory]) -> Any
            perspective: 観点（検索・強化に影響）
            extract_learning: [deprecated] 自動抽出は行わない、明示的指定のみ有効
            learning_content: 記録する学びの content（明示的に指定する場合）
            learning_text: 記録する学びの learning テキスト（明示的に指定する場合）
            auto_sleep: 自動睡眠フェーズを実行するか（デフォルト: True）

        Returns:
            TaskExecutionResult インスタンス

        Note:
            - 各ステップでエラーが発生しても後続処理は継続（fail-soft）
            - 重大なエラー（検索失敗等）は例外を伝播
            - 全エラーは result.errors に集約
            - task_func の例外は捕捉せず、そのまま伝播

        Example:
            def my_task(memories: List[ScoredMemory]) -> str:
                # メモリを参照してタスクを実行
                return "タスク結果"

            result = executor.execute_task(
                query="緊急調達のコスト",
                agent_id="agent_01",
                task_func=my_task,
                perspective="コスト",
                learning_content="緊急調達は1.5倍のコストがかかる",
                learning_text="納期短縮のトレードオフを学んだ",
            )
        """
        logger.info(
            f"タスク実行開始: query={query[:30]!r}..., "
            f"agent_id={agent_id}, perspective={perspective}"
        )

        errors: List[str] = []
        searched_memories: List[ScoredMemory] = []
        used_memory_ids: List[UUID] = []
        recorded_memory_id: Optional[UUID] = None
        task_result: Any = None

        # ========================================
        # Step 1: メモリ検索（候補強化を含む）
        # ========================================
        # 検索失敗は重大エラーとして例外を伝播
        try:
            searched_memories = self.search_memories(
                query=query,
                agent_id=agent_id,
                perspective=perspective,
            )
            logger.info(f"Step 1 完了: 検索結果 {len(searched_memories)} 件")
        except Exception as e:
            # 検索失敗は重大エラー - 例外を伝播
            logger.error(f"Step 1 失敗（検索エラー）: {e}")
            raise

        # ========================================
        # Step 2: タスク実行
        # ========================================
        # task_func の例外はそのまま伝播（呼び出し側でハンドリング）
        task_result = task_func(searched_memories)
        logger.info(f"Step 2 完了: タスク実行結果取得")

        # ========================================
        # Step 3: 使用判定
        # ========================================
        try:
            used_memory_ids = self.identify_used_memories(
                task_result=task_result,
                candidates=searched_memories,
            )
            logger.info(
                f"Step 3 完了: 使用メモリ {len(used_memory_ids)}/{len(searched_memories)} 件"
            )
        except Exception as e:
            # 使用判定失敗は記録して継続
            error_msg = f"使用判定でエラー: {e}"
            logger.warning(f"Step 3 失敗: {error_msg}")
            errors.append(error_msg)

        # ========================================
        # Step 4: 使用強化
        # ========================================
        if used_memory_ids:
            try:
                reinforced_count = self.reinforce_used_memories(
                    memory_ids=used_memory_ids,
                    agent_id=agent_id,
                    perspective=perspective,
                )
                logger.info(
                    f"Step 4 完了: 強化成功 {reinforced_count}/{len(used_memory_ids)} 件"
                )
                # 部分的な失敗を検出（reinforce_used_memories は内部で例外をキャッチ）
                if reinforced_count < len(used_memory_ids):
                    failed_count = len(used_memory_ids) - reinforced_count
                    error_msg = f"使用強化で部分的失敗: {failed_count}/{len(used_memory_ids)} 件が失敗"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            except Exception as e:
                # 想定外の例外は記録して継続
                error_msg = f"使用強化でエラー: {e}"
                logger.warning(f"Step 4 失敗: {error_msg}")
                errors.append(error_msg)
        else:
            logger.info("Step 4 スキップ: 使用メモリなし")

        # ========================================
        # Step 5: 学び記録（例外的なイベントのみ）
        # ========================================
        # 学びは「例外的なイベント」のみ記録
        # - エラー解決、予想外の挙動発見、効率的な方法発見など
        # - 「テスト成功」「処理完了」のような報告は記録しない
        # - learning_content と learning_text が明示的に指定された場合のみ記録
        # - extract_learning は deprecated（自動抽出は行わない）
        should_record_learning = learning_content and learning_text

        if should_record_learning:
            try:
                recorded_memory_id = self.record_learning(
                    agent_id=agent_id,
                    content=learning_content,
                    learning=learning_text,
                    perspective=perspective,
                )
                logger.info(
                    f"Step 5 完了: 学び記録 memory_id={recorded_memory_id}"
                )
            except Exception as e:
                # 学び記録失敗は記録して継続
                error_msg = f"学び記録でエラー: {e}"
                logger.warning(f"Step 5 失敗: {error_msg}")
                errors.append(error_msg)
        else:
            logger.info("Step 5 スキップ: 学び記録なし（明示的な指定なし）")

        # ========================================
        # Step 6: 結果返却
        # ========================================
        result = TaskExecutionResult(
            task_result=task_result,
            searched_memories=searched_memories,
            used_memory_ids=used_memory_ids,
            recorded_memory_id=recorded_memory_id,
            executed_at=datetime.now(),
            errors=errors,
        )

        if errors:
            logger.warning(
                f"タスク実行完了（エラーあり）: {len(errors)} 件のエラー"
            )
        else:
            logger.info("タスク実行完了（成功）")

        # ========================================
        # Step 7: 自動睡眠フェーズ（新規追加）
        # ========================================
        # タスクカウンターを更新
        self._task_count += 1

        # 自動睡眠判定
        if auto_sleep and self.config.auto_sleep_after_task:
            if self._task_count % self.config.auto_sleep_threshold_tasks == 0:
                try:
                    sleep_result = self.run_sleep_phase(agent_id)
                    logger.info(
                        f"自動睡眠フェーズ完了: "
                        f"減衰={sleep_result.decayed_count}, "
                        f"アーカイブ={sleep_result.archived_count}"
                    )
                except Exception as e:
                    # 睡眠フェーズ失敗してもタスク実行は成功扱い
                    logger.warning(f"自動睡眠フェーズに失敗: {e}")

        return result

    def _extract_learning_content(
        self,
        task_result: Any,
        query: str,
    ) -> Optional[str]:
        """タスク結果から学びの content を抽出

        Phase 1 ではシンプルな抽出（task_result の文字列化）を行う。
        Phase 2 以降で LLM による高度な抽出に移行予定。

        Args:
            task_result: タスク関数の実行結果
            query: 元のクエリ（コンテキスト情報として使用）

        Returns:
            学びの content（抽出失敗時は None）

        Note:
            - Phase 1 では task_result を文字列化して返すシンプルな実装
            - 空の結果や変換失敗時は None を返す
        """
        try:
            if task_result is None:
                return None

            result_str = str(task_result)
            if not result_str or not result_str.strip():
                return None

            # 結果が長すぎる場合は先頭500文字に制限
            max_length = 500
            if len(result_str) > max_length:
                result_str = result_str[:max_length] + "..."

            return result_str

        except Exception as e:
            logger.warning(f"学び content 抽出に失敗: {e}")
            return None

    # === 睡眠フェーズ ===

    def run_sleep_phase(self, agent_id: str) -> SleepPhaseResult:
        """睡眠フェーズを実行

        SleepPhaseProcessor に処理を委譲する。
        減衰・アーカイブ・統合処理を一括実行。

        Args:
            agent_id: 処理対象のエージェントID

        Returns:
            SleepPhaseResult インスタンス

        Note:
            - タスク完了後に呼び出すことを想定
            - 処理中のエラーは result.errors に集約
            - SleepPhaseProcessor.process_all() は内部で各処理の
              try-except を持つため、障害耐性が確保されている
        """
        logger.info(f"睡眠フェーズ開始（TaskExecutor経由）: agent_id={agent_id}")

        result = self.sleep_processor.process_all(agent_id)

        logger.info(
            f"睡眠フェーズ完了（TaskExecutor経由）: agent_id={agent_id}, "
            f"decayed={result.decayed_count}, archived={result.archived_count}, "
            f"consolidated={result.consolidated_count}, errors={len(result.errors)}"
        )

        return result

    # === 内部ヘルパーメソッド ===

    # ストップワード（Phase 1: 英語のみ、日本語は Phase 2 以降）
    _STOPWORDS = frozenset({
        # 冠詞
        "a", "an", "the",
        # be動詞
        "is", "am", "are", "was", "were", "be", "been", "being",
        # 助動詞
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could",
        # 代名詞
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        # 前置詞・接続詞
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
        "about", "into", "over", "after", "and", "or", "but", "if", "then",
        # その他
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "how", "when", "where", "why", "not", "no", "yes", "so", "as",
    })

    def _extract_keywords(self, content: str) -> List[str]:
        """コンテンツからキーワードを抽出

        identify_used_memories() で使用する内部メソッド。
        Phase 1 ではシンプルな単語分割でキーワードを抽出する。

        処理フロー:
            1. 小文字に変換
            2. 単語分割（英数字・アンダースコア以外を区切り文字として扱う）
            3. ストップワード除外
            4. 短すぎる単語（3文字未満）除外
            5. 重複除去（順序保持）

        Args:
            content: キーワード抽出対象のテキスト

        Returns:
            抽出されたキーワードのリスト（小文字、重複なし）

        Note:
            - Phase 1 では単純な単語分割（日本語の形態素解析は Phase 2 以降）
            - ストップワードは英語のみ（日本語は Phase 2 以降）
            - 空のコンテンツには空リストを返す
        """
        if not content or not content.strip():
            return []

        # 小文字変換
        text = content.lower()

        # 単語分割（英数字・アンダースコア以外を区切りとして扱う）
        # 日本語文字はそのまま残る（分割されない）
        import re
        words = re.split(r'[^\w]+', text, flags=re.UNICODE)

        # キーワード抽出（重複除去、順序保持）
        seen = set()
        keywords = []
        for word in words:
            # 空文字スキップ
            if not word:
                continue
            # 短すぎる単語は除外（3文字未満）
            if len(word) < 3:
                continue
            # ストップワード除外
            if word in self._STOPWORDS:
                continue
            # 重複除去（既出の単語はスキップ）
            if word in seen:
                continue
            seen.add(word)
            keywords.append(word)

        return keywords

    def _matches_any_keyword(
        self,
        text: str,
        keywords: List[str],
    ) -> bool:
        """テキストにキーワードが含まれるかを判定

        identify_used_memories() で使用する内部メソッド。
        タスク結果テキストにメモリのキーワードが含まれているかを判定する。

        Args:
            text: 検索対象のテキスト（タスク結果など）
            keywords: マッチングするキーワードのリスト（メモリから抽出したもの）

        Returns:
            いずれかのキーワードが含まれる場合 True、そうでなければ False

        Note:
            - 大文字小文字を区別しない（両方を小文字に変換して比較）
            - 部分一致で判定（キーワードがテキスト内に含まれるか）
            - 空のテキストまたは空のキーワードリストには False を返す
        """
        # 空チェック
        if not text or not keywords:
            return False

        # 小文字に変換
        text_lower = text.lower()

        # いずれかのキーワードが含まれるかチェック
        for keyword in keywords:
            # キーワードも小文字に変換（呼び出し側が大文字を渡した場合の対応）
            if keyword.lower() in text_lower:
                return True

        return False
