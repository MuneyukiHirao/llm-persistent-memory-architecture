#!/usr/bin/env python3
"""
LLM永続メモリ Phase 1 - 基本操作サンプル

このスクリプトは、LLM永続メモリシステムの基本操作を網羅的に学べるサンプルです。
各機能を関数に分離し、実行順序がわかるように構成しています。

quickstart.py との違い:
    - quickstart.py: 最小構成（動作確認用）
    - basic_usage.py: 基本操作の網羅（学習用）

含まれる機能:
    1. メモリの保存（複数件）
    2. メモリの検索（ベクトル検索 + ランキング）
    3. 2段階強化（候補強化 + 使用強化）
    4. 睡眠フェーズ（減衰・アーカイブ）
    5. タスク実行フロー（execute_task）

実行方法:
    cd /path/to/llm-persistent-memory
    source venv/bin/activate
    python examples/basic_usage.py

前提条件:
    1. Docker で PostgreSQL + pgvector が起動していること
       docker compose -f docker/docker-compose.yml up -d
    2. .env ファイルに環境変数が設定されていること
    3. データベーススキーマが初期化されていること
"""

import os
import sys
from typing import List, Optional
from uuid import UUID

# ============================================
# プロジェクトルートをPythonパスに追加
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================
# グローバル変数（サンプル実行用）
# ============================================
# 各 Example 間でコンポーネントを共有するための変数
executor = None
repository = None
strength_manager = None
db = None
config = None

# 作成したメモリのIDを保持（クリーンアップ用）
created_memory_ids: List[UUID] = []

# サンプル用のエージェントID
AGENT_ID = "basic_usage_agent"


# ============================================
# 環境変数の確認
# ============================================
def check_environment():
    """必要な環境変数が設定されているか確認"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[OK] .env ファイルを読み込みました")
    except ImportError:
        print("[INFO] python-dotenv がインストールされていません")

    errors = []

    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not postgres_password:
        errors.append("POSTGRES_PASSWORD が設定されていません")

    if not os.getenv("DATABASE_URL") and postgres_password:
        os.environ["DATABASE_URL"] = (
            f"postgresql://agent:{postgres_password}@localhost:5432/agent_memory"
        )
        print("[OK] DATABASE_URL を設定しました")

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OpenAIEmbeddingURI")
    if not azure_endpoint:
        errors.append("Azure OpenAI エンドポイントが設定されていません")

    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OpenAIEmbeddingKey")
    if not azure_key:
        errors.append("Azure OpenAI APIキーが設定されていません")

    if errors:
        print("\n" + "=" * 50)
        print("環境変数エラー")
        print("=" * 50)
        for error in errors:
            print(f"[ERROR] {error}")
        sys.exit(1)

    print("[OK] 環境変数の確認が完了しました\n")


# ============================================
# コンポーネント初期化
# ============================================
def init_components():
    """全コンポーネントを初期化"""
    global executor, repository, strength_manager, db, config

    print("=" * 60)
    print("コンポーネント初期化")
    print("=" * 60)

    from src.config.phase1_config import Phase1Config
    from src.db.connection import DatabaseConnection
    from src.embedding.azure_client import AzureEmbeddingClient
    from src.core.memory_repository import MemoryRepository
    from src.core.strength_manager import StrengthManager
    from src.core.sleep_processor import SleepPhaseProcessor
    from src.search.vector_search import VectorSearch
    from src.search.ranking import MemoryRanker
    from src.core.task_executor import TaskExecutor

    config = Phase1Config()
    db = DatabaseConnection()

    if not db.health_check():
        print("[ERROR] データベースに接続できません")
        sys.exit(1)

    embedding_client = AzureEmbeddingClient()
    repository = MemoryRepository(db, config)
    strength_manager = StrengthManager(repository, config)
    sleep_processor = SleepPhaseProcessor(db, config)
    vector_search = VectorSearch(db, embedding_client, config)
    ranker = MemoryRanker(config)

    executor = TaskExecutor(
        vector_search=vector_search,
        ranker=ranker,
        strength_manager=strength_manager,
        sleep_processor=sleep_processor,
        repository=repository,
        config=config,
    )

    print("[OK] 全コンポーネントの初期化完了\n")


# ============================================
# Example 1: メモリの保存（複数件）
# ============================================
def run_example_1_save_memories():
    """
    Example 1: メモリの保存

    record_learning() を使って複数の学びを保存します。
    観点（perspective）を指定することで、観点別の強度管理が可能になります。
    """
    global created_memory_ids

    print("\n" + "=" * 60)
    print("Example 1: メモリの保存（複数件）")
    print("=" * 60)

    # 保存する学びのデータ
    learnings = [
        {
            "content": "PostgreSQLのpgvectorは1万件未満ならインデックスなしで十分高速",
            "learning": "ベクトル検索のパフォーマンスに関する知見",
            "perspective": "performance",
        },
        {
            "content": "コネクションプールのサイズは同時接続数の2倍程度が目安",
            "learning": "データベース接続に関するベストプラクティス",
            "perspective": "performance",
        },
        {
            "content": "エンベディングのバッチ処理は100件程度で区切ると効率的",
            "learning": "Azure OpenAI APIの使い方に関する知見",
            "perspective": "api_usage",
        },
        {
            "content": "メモリの強度は使用頻度と経過時間で自動調整される",
            "learning": "メモリ管理システムの動作原理",
            "perspective": "system_design",
        },
        {
            "content": "2段階強化により、検索と使用を区別して重要度を判定できる",
            "learning": "2段階強化の設計意図",
            "perspective": "system_design",
        },
    ]

    print(f"\n{len(learnings)} 件のメモリを保存します...\n")

    for i, data in enumerate(learnings, 1):
        memory_id = executor.record_learning(
            agent_id=AGENT_ID,
            content=data["content"],
            learning=data["learning"],
            perspective=data["perspective"],
        )
        created_memory_ids.append(memory_id)

        print(f"  [{i}] 保存完了")
        print(f"      memory_id: {memory_id}")
        print(f"      content: {data['content'][:40]}...")
        print(f"      perspective: {data['perspective']}")
        print()

    print(f"[OK] Example 1 完了: {len(learnings)} 件のメモリを保存しました")


# ============================================
# Example 2: メモリの検索
# ============================================
def run_example_2_search():
    """
    Example 2: メモリの検索

    search_memories() を使ってメモリを検索します。
    検索はベクトル検索（Stage 1）とランキング（Stage 2）の2段階で行われます。
    """
    print("\n" + "=" * 60)
    print("Example 2: メモリの検索")
    print("=" * 60)

    queries = [
        "ベクトル検索のパフォーマンス",
        "データベース接続の最適化",
        "メモリ管理の仕組み",
    ]

    for query in queries:
        print(f"\n検索クエリ: \"{query}\"")
        print("-" * 50)

        results = executor.search_memories(
            query=query,
            agent_id=AGENT_ID,
        )

        if results:
            print(f"  検索結果: {len(results)} 件")
            for i, scored_memory in enumerate(results, 1):
                memory = scored_memory.memory
                print(f"\n  [{i}] memory_id: {memory.id}")
                print(f"      content: {memory.content[:50]}...")
                print(f"      final_score: {scored_memory.final_score:.4f}")
                print(f"      strength: {memory.strength:.2f}")
                print(f"      candidate_count: {memory.candidate_count}")
        else:
            print("  検索結果がありません")

    print(f"\n[OK] Example 2 完了: 検索が正常に動作しました")


# ============================================
# Example 3: 2段階強化
# ============================================
def run_example_3_two_stage_reinforcement():
    """
    Example 3: 2段階強化の動作確認

    2段階強化の仕組み:
    1. 検索候補として参照 → candidate_count がインクリメント
    2. 実際に使用された → access_count と strength が増加

    このExampleでは、検索後にメモリを「使用した」とマークし、
    強度がどのように変化するかを確認します。
    """
    print("\n" + "=" * 60)
    print("Example 3: 2段階強化")
    print("=" * 60)

    if not created_memory_ids:
        print("  [SKIP] 保存済みメモリがありません")
        return

    # 最初のメモリを取得
    memory_id = created_memory_ids[0]
    memory_before = repository.get_by_id(memory_id)

    print(f"\n対象メモリ: {memory_id}")
    print(f"  content: {memory_before.content[:50]}...")
    print()
    print("【強化前の状態】")
    print(f"  strength: {memory_before.strength:.4f}")
    print(f"  access_count: {memory_before.access_count}")
    print(f"  candidate_count: {memory_before.candidate_count}")

    # Stage 1: 検索（候補強化）
    print("\n--- Stage 1: 検索による候補強化 ---")
    print("search_memories() を呼び出すと candidate_count が増加します")

    executor.search_memories(
        query="パフォーマンス最適化",
        agent_id=AGENT_ID,
    )

    memory_after_search = repository.get_by_id(memory_id)
    print(f"\n【検索後の状態】")
    print(f"  strength: {memory_after_search.strength:.4f} (変化なし)")
    print(f"  access_count: {memory_after_search.access_count} (変化なし)")
    print(f"  candidate_count: {memory_after_search.candidate_count} (+1)")

    # Stage 2: 使用強化
    print("\n--- Stage 2: 使用による強化 ---")
    print("mark_as_used() を呼び出すと access_count と strength が増加します")

    strength_manager.mark_as_used(
        memory_id=memory_id,
        perspective="performance",
    )

    memory_after_use = repository.get_by_id(memory_id)
    print(f"\n【使用後の状態】")
    print(f"  strength: {memory_after_use.strength:.4f} (+0.1)")
    print(f"  access_count: {memory_after_use.access_count} (+1)")
    print(f"  candidate_count: {memory_after_use.candidate_count} (変化なし)")

    # 強度変化のサマリー
    print("\n【強度変化のサマリー】")
    print(f"  strength: {memory_before.strength:.4f} → {memory_after_use.strength:.4f}")
    print(f"  access_count: {memory_before.access_count} → {memory_after_use.access_count}")
    print(f"  candidate_count: {memory_before.candidate_count} → {memory_after_use.candidate_count}")

    print(f"\n[OK] Example 3 完了: 2段階強化の動作を確認しました")


# ============================================
# Example 4: 睡眠フェーズ
# ============================================
def run_example_4_sleep_phase():
    """
    Example 4: 睡眠フェーズ

    run_sleep_phase() を使って睡眠フェーズを実行します。
    睡眠フェーズでは以下の処理が行われます:
    1. 強度減衰: 定着レベルに応じた減衰率で strength を減少
    2. アーカイブ: 閾値(0.1)以下のメモリを archived に変更
    3. 統合処理: 類似メモリの consolidation_level を管理（Phase 1は簡易版）
    """
    print("\n" + "=" * 60)
    print("Example 4: 睡眠フェーズ")
    print("=" * 60)

    if not created_memory_ids:
        print("  [SKIP] 保存済みメモリがありません")
        return

    # 睡眠フェーズ前の状態を取得
    print("\n【睡眠フェーズ前の状態】")
    print("-" * 50)

    memories_before = []
    for memory_id in created_memory_ids[:3]:  # 最初の3件のみ表示
        memory = repository.get_by_id(memory_id)
        if memory:
            memories_before.append(memory)
            print(f"  memory_id: {memory.id}")
            print(f"    content: {memory.content[:40]}...")
            print(f"    strength: {memory.strength:.4f}")
            print(f"    status: {memory.status}")
            print()

    # 睡眠フェーズを実行
    print("睡眠フェーズを実行中...")
    result = executor.run_sleep_phase(AGENT_ID)

    print(f"\n【睡眠フェーズの処理結果】")
    print(f"  減衰処理: {result.decayed_count} 件")
    print(f"  アーカイブ: {result.archived_count} 件")
    print(f"  統合処理: {result.consolidated_count} 件")
    print(f"  エラー: {len(result.errors)} 件")

    # 睡眠フェーズ後の状態を取得
    print(f"\n【睡眠フェーズ後の状態】")
    print("-" * 50)

    for i, memory_before in enumerate(memories_before):
        memory_after = repository.get_by_id(memory_before.id)
        if memory_after:
            strength_diff = memory_after.strength - memory_before.strength
            print(f"  memory_id: {memory_after.id}")
            print(f"    strength: {memory_before.strength:.4f} → {memory_after.strength:.4f} ({strength_diff:+.4f})")
            print(f"    status: {memory_after.status}")
            print()

    print(f"[OK] Example 4 完了: 睡眠フェーズを実行しました")


# ============================================
# Example 5: タスク実行フロー統合
# ============================================
def run_example_5_execute_task():
    """
    Example 5: タスク実行フロー統合

    execute_task() を使って、検索→タスク実行→使用判定→強化→学び記録の
    一連のフローを統合実行します。

    このメソッドは以下の処理を自動で行います:
    1. search_memories() でメモリ検索（候補強化）
    2. task_func() を実行（検索結果を引数として渡す）
    3. identify_used_memories() で使用判定
    4. reinforce_used_memories() で使用強化
    5. record_learning() で学び記録（オプション）
    """
    print("\n" + "=" * 60)
    print("Example 5: タスク実行フロー統合（execute_task）")
    print("=" * 60)

    from src.search.ranking import ScoredMemory

    # タスク関数を定義
    # 検索結果を受け取り、タスクを実行して結果を返す
    def my_task_function(memories: List[ScoredMemory]) -> str:
        """
        サンプルタスク関数

        この関数は execute_task() から呼び出され、
        検索結果（memories）を受け取ってタスクを実行します。

        返り値に検索したメモリの内容が含まれていると、
        そのメモリは「使用された」と判定されます。
        """
        print(f"\n  [task_func] 検索結果 {len(memories)} 件を受け取りました")

        if memories:
            # 最も関連性の高いメモリの内容を使用
            top_memory = memories[0].memory
            print(f"  [task_func] 最関連メモリ: {top_memory.content[:40]}...")

            # タスク結果にメモリの内容を含める（使用判定のため）
            return f"パフォーマンス最適化の結果: pgvectorのインデックス設計を見直しました"
        else:
            return "検索結果がなかったため、デフォルトの処理を行いました"

    print("\nexecute_task() を呼び出します...")
    print("  - クエリ: 「データベースのパフォーマンス最適化」")
    print("  - タスク: サンプルタスク関数を実行")
    print("  - 学び記録: 有効（learning_content を指定）")
    print()

    # execute_task() を実行
    result = executor.execute_task(
        query="データベースのパフォーマンス最適化",
        agent_id=AGENT_ID,
        task_func=my_task_function,
        perspective="performance",
        learning_content="pgvectorのインデックス設計はデータ量に応じて選択すべき",
        learning_text="データベースパフォーマンス最適化タスクから得た知見",
    )

    # 実行結果を記録
    if result.recorded_memory_id:
        created_memory_ids.append(result.recorded_memory_id)

    # 結果の表示
    print("\n【execute_task() の実行結果】")
    print("-" * 50)
    print(f"  タスク結果: {result.task_result}")
    print(f"  検索されたメモリ: {len(result.searched_memories)} 件")
    print(f"  使用されたメモリ: {len(result.used_memory_ids)} 件")
    print(f"  記録された学び: {result.recorded_memory_id}")
    print(f"  実行日時: {result.executed_at}")
    print(f"  エラー: {len(result.errors)} 件")

    if result.searched_memories:
        print(f"\n【検索されたメモリの詳細】")
        for i, sm in enumerate(result.searched_memories[:3], 1):
            used_mark = "[使用]" if sm.memory.id in result.used_memory_ids else ""
            print(f"  [{i}] {sm.memory.content[:40]}... (score: {sm.final_score:.3f}) {used_mark}")

    if result.recorded_memory_id:
        new_memory = repository.get_by_id(result.recorded_memory_id)
        if new_memory:
            print(f"\n【新しく記録された学び】")
            print(f"  memory_id: {new_memory.id}")
            print(f"  content: {new_memory.content}")
            print(f"  learnings: {new_memory.learnings}")

    print(f"\n[OK] Example 5 完了: タスク実行フローを統合実行しました")


# ============================================
# クリーンアップ
# ============================================
def cleanup():
    """テストデータのクリーンアップ"""
    print("\n" + "=" * 60)
    print("クリーンアップ")
    print("=" * 60)

    # 作成したメモリを削除
    if created_memory_ids:
        print(f"\n作成したメモリ {len(created_memory_ids)} 件を削除します...")
        for memory_id in created_memory_ids:
            try:
                repository.delete(memory_id)
            except Exception:
                pass  # 削除失敗は無視
        print("[OK] メモリを削除しました")

    # データベース接続をクローズ
    if db:
        db.close()
        print("[OK] データベース接続をクローズしました")


# ============================================
# メイン処理
# ============================================
def main():
    """メイン処理: 各Exampleを順番に実行"""
    print("=" * 60)
    print("LLM永続メモリ Phase 1 - 基本操作サンプル")
    print("=" * 60)
    print()
    print("このサンプルでは以下の機能を順番に実行します:")
    print("  1. メモリの保存（複数件）")
    print("  2. メモリの検索")
    print("  3. 2段階強化")
    print("  4. 睡眠フェーズ")
    print("  5. タスク実行フロー統合")
    print()

    try:
        # 環境確認
        check_environment()

        # コンポーネント初期化
        init_components()

        # Example 1: メモリの保存
        run_example_1_save_memories()

        # Example 2: メモリの検索
        run_example_2_search()

        # Example 3: 2段階強化
        run_example_3_two_stage_reinforcement()

        # Example 4: 睡眠フェーズ
        run_example_4_sleep_phase()

        # Example 5: タスク実行フロー統合
        run_example_5_execute_task()

        print("\n" + "=" * 60)
        print("全Example完了!")
        print("=" * 60)
        print("\n次のステップ:")
        print("  - examples/claude_code_memory.py: Claude Code との連携例")
        print("  - examples/multi_agent_memory.py: マルチエージェント共有メモリ")
        print("  - docs/api-reference.ja.md: API リファレンス")

    except KeyboardInterrupt:
        print("\n\n[INFO] 処理を中断しました")

    except Exception as e:
        print(f"\n[ERROR] 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # クリーンアップ
        cleanup()


if __name__ == "__main__":
    main()
