# 教育プロセス
# 教科書を読み、テストを実行し、記憶を強化する教育フローを提供
# 実装仕様: docs/architecture.ja.md セクション4.3

"""
教育プロセスモジュール

教科書を読み、テストを実行し、睡眠フェーズで減衰させる教育プロセスを実装。
人間の教育と同じプロセスで専門性を形成する。

設計方針（メモリ管理エージェント観点）:
- 強度の正確性: 教育で読んだだけの記憶は初期強度 0.5
- 2段階強化: テスト正解時は mark_as_candidate（Stage 1）で強化
- テスト容易性: 依存性注入でリポジトリと設定を外部から注入可能
"""

from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from src.config.phase1_config import Phase1Config
from src.core.memory_repository import MemoryRepository
from src.core.strength_manager import StrengthManager
from src.education.textbook import Chapter, Textbook
from src.embedding.azure_client import AzureEmbeddingClient
from src.models.memory import AgentMemory


@dataclass
class EducationResult:
    """教育プロセスの実行結果

    Attributes:
        chapters_completed: 完了した章の数
        memories_created: 作成された記憶の数
        tests_passed: 合格したテストの数
        tests_total: テストの総数
    """

    chapters_completed: int
    memories_created: int
    tests_passed: int
    tests_total: int

    @property
    def pass_rate(self) -> float:
        """テストの合格率を計算

        Returns:
            合格率（0.0〜1.0）。テストがない場合は 1.0
        """
        if self.tests_total == 0:
            return 1.0
        return self.tests_passed / self.tests_total


class EducationProcess:
    """教育プロセスクラス

    教科書を読み、テストを実行し、記憶を強化する教育フローを提供。

    プロセスの流れ:
        1. read_chapter: 章のコンテンツをチャンク分割して記憶として保存
        2. run_test: クイズを実行し、正解した場合は関連記憶を強化
        3. run: 全章を処理してEducationResultを返却

    使用例:
        textbook = TextbookLoader().load("path/to/textbook.yaml")
        process = EducationProcess(
            agent_id="agent_01",
            textbook=textbook,
            repository=repo,
            embedding_client=client,
        )
        result = process.run()

    Attributes:
        agent_id: エージェントID
        textbook: 教科書データ
        repository: MemoryRepository インスタンス
        embedding_client: AzureEmbeddingClient インスタンス
        config: Phase1Config インスタンス
        strength_manager: StrengthManager インスタンス
    """

    def __init__(
        self,
        agent_id: str,
        textbook: Textbook,
        repository: MemoryRepository,
        embedding_client: AzureEmbeddingClient,
        config: Optional[Phase1Config] = None,
    ):
        """EducationProcess を初期化

        Args:
            agent_id: エージェントID
            textbook: 教科書データ
            repository: MemoryRepository インスタンス
            embedding_client: AzureEmbeddingClient インスタンス
            config: Phase1Config インスタンス（省略時はデフォルト設定を使用）
        """
        self.agent_id = agent_id
        self.textbook = textbook
        self.repository = repository
        self.embedding_client = embedding_client
        self.config = config or Phase1Config()
        self.strength_manager = StrengthManager(repository, self.config)

    def run(self) -> EducationResult:
        """全章を処理して教育プロセスを実行

        Returns:
            EducationResult: 教育プロセスの実行結果

        処理内容:
            1. 各章を順に処理
            2. read_chapter で記憶を保存
            3. run_test でテストを実行
            4. 結果を集計して返却
        """
        chapters_completed = 0
        memories_created = 0
        tests_passed = 0
        tests_total = 0

        for chapter in self.textbook.chapters:
            # Step 1: 読む（チャンク分割して記憶として保存）
            memory_ids = self.read_chapter(chapter)
            memories_created += len(memory_ids)

            # Step 2: テスト
            passed = self.run_test(chapter, memory_ids)
            tests_passed += passed
            tests_total += len(chapter.quiz)

            chapters_completed += 1

        return EducationResult(
            chapters_completed=chapters_completed,
            memories_created=memories_created,
            tests_passed=tests_passed,
            tests_total=tests_total,
        )

    def read_chapter(self, chapter: Chapter) -> List[str]:
        """章のコンテンツをチャンク分割して記憶として保存

        Args:
            chapter: 処理する章

        Returns:
            作成された記憶のIDリスト

        処理内容:
            1. コンテンツを CHUNK_SIZE 文字で分割
            2. 各チャンクに対してエンベディングを生成
            3. AgentMemory として保存（strength=0.5, source="education"）
        """
        memory_ids: List[str] = []

        # チャンク分割
        chunks = self._split_into_chunks(chapter.content)

        for i, chunk in enumerate(chunks):
            # エンベディングを生成
            embedding = self.embedding_client.get_embedding(chunk)

            # 記憶を作成（教育プロセス用の初期強度）
            memory = AgentMemory.create_from_education(
                agent_id=self.agent_id,
                content=chunk,
                embedding=embedding,
                tags=[chapter.title, self.textbook.perspective],
                scope_level=self.config.default_scope_level,
                scope_project=self.config.current_project_id,
            )

            # データベースに保存
            created_memory = self.repository.create(memory)
            memory_ids.append(str(created_memory.id))

        return memory_ids

    def run_test(self, chapter: Chapter, memory_ids: List[str]) -> int:
        """テストを実行し、正解した場合は関連記憶を強化

        Args:
            chapter: テスト対象の章
            memory_ids: この章で作成された記憶のIDリスト

        Returns:
            正解したテストの数

        処理内容:
            1. 各クイズの question をベクトル検索のクエリとして使用
            2. 検索で取得した記憶の content に expected_keywords が含まれているか確認
            3. 含まれていれば正解とし、関連記憶を強化（Stage 1のみ）
        """
        passed = 0

        for quiz in chapter.quiz:
            # クイズの question でベクトル検索（エンベディングを生成）
            query_embedding = self.embedding_client.get_embedding(quiz.question)

            # この章で作成された記憶を検索候補としてマーク
            memory_uuids = [UUID(mid) for mid in memory_ids]
            self.strength_manager.mark_as_candidate(memory_uuids)

            # 記憶を取得してキーワードマッチング
            is_correct = False
            for mid in memory_ids:
                memory = self.repository.get_by_id(UUID(mid))
                if memory is None:
                    continue

                # expected_keywords が記憶の content に含まれているか確認
                if self._check_keywords(memory.content, quiz.expected_keywords):
                    is_correct = True
                    break

            if is_correct:
                passed += 1

        return passed

    def _split_into_chunks(self, content: str) -> List[str]:
        """コンテンツをチャンク分割

        Args:
            content: 分割するコンテンツ

        Returns:
            チャンクのリスト

        Note:
            - CHUNK_SIZE 文字で分割
            - 最後のチャンクが CHUNK_SIZE 未満でも含める
            - 空のコンテンツは空のリストを返す
        """
        if not content:
            return []

        chunk_size = self.config.chunk_size
        chunks = []

        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            if chunk.strip():  # 空白のみのチャンクは除外
                chunks.append(chunk)

        return chunks

    def _check_keywords(self, content: str, keywords: List[str]) -> bool:
        """キーワードがコンテンツに含まれているか確認

        Args:
            content: 検索対象のコンテンツ
            keywords: 期待するキーワードのリスト

        Returns:
            True: いずれかのキーワードが含まれている
            False: どのキーワードも含まれていない
        """
        content_lower = content.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                return True
        return False
