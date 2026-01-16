#!/usr/bin/env python3
"""
LLM永続メモリ Phase 1 - クイックスタートサンプル

このスクリプトは、LLM永続メモリシステムの最小構成サンプルです。
コピペで動かせるように設計されています。

実行方法:
    # プロジェクトルートから実行
    cd /path/to/llm-persistent-memory
    source venv/bin/activate
    python examples/quickstart.py

前提条件:
    1. Docker で PostgreSQL + pgvector が起動していること
       docker compose -f docker/docker-compose.yml up -d

    2. .env ファイルに以下が設定されていること
       - POSTGRES_PASSWORD
       - AZURE_OPENAI_ENDPOINT (または OpenAIEmbeddingURI)
       - AZURE_OPENAI_API_KEY (または OpenAIEmbeddingKey)

    3. データベーススキーマが初期化されていること
       docker exec -i docker-postgres-1 psql -U agent -d agent_memory < src/db/schema.sql
"""

import os
import sys

# ============================================
# プロジェクトルートをPythonパスに追加
# （examples/ ディレクトリから実行しても動作するように）
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================
# Step 0: 環境変数の確認
# ============================================
def check_environment():
    """必要な環境変数が設定されているか確認します"""

    # .env ファイルがあれば読み込む
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[OK] .env ファイルを読み込みました")
    except ImportError:
        print("[INFO] python-dotenv がインストールされていません")
        print("       pip install python-dotenv でインストールできます")
        print("       .env ファイルを使用せず、環境変数を直接参照します")

    errors = []

    # PostgreSQL パスワード
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not postgres_password:
        errors.append(
            "POSTGRES_PASSWORD が設定されていません。\n"
            "  .env ファイルに POSTGRES_PASSWORD=your_password を追加してください"
        )

    # DATABASE_URL を設定（未設定の場合）
    if not os.getenv("DATABASE_URL") and postgres_password:
        os.environ["DATABASE_URL"] = (
            f"postgresql://agent:{postgres_password}@localhost:5432/agent_memory"
        )
        print(f"[OK] DATABASE_URL を設定しました")

    # Azure OpenAI Endpoint
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OpenAIEmbeddingURI")
    if not azure_endpoint:
        errors.append(
            "Azure OpenAI エンドポイントが設定されていません。\n"
            "  AZURE_OPENAI_ENDPOINT または OpenAIEmbeddingURI を設定してください\n"
            "  例: AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/"
        )

    # Azure OpenAI API Key
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OpenAIEmbeddingKey")
    if not azure_key:
        errors.append(
            "Azure OpenAI APIキーが設定されていません。\n"
            "  AZURE_OPENAI_API_KEY または OpenAIEmbeddingKey を設定してください"
        )

    # エラーがあれば表示して終了
    if errors:
        print("\n" + "=" * 50)
        print("環境変数エラー")
        print("=" * 50)
        for error in errors:
            print(f"\n[ERROR] {error}")
        print("\n詳細は docs/quickstart.ja.md を参照してください")
        sys.exit(1)

    print("[OK] 環境変数の確認が完了しました\n")


def main():
    """メイン処理: 記憶の保存と検索のデモ"""

    print("=" * 50)
    print("LLM永続メモリ Phase 1 - クイックスタート")
    print("=" * 50 + "\n")

    # ============================================
    # Step 0: 環境変数の確認
    # ============================================
    check_environment()

    # ============================================
    # Step 1: 接続確立
    # ============================================
    print("Step 1: コンポーネントの初期化...")

    try:
        # 設定の読み込み
        from src.config.phase1_config import Phase1Config
        config = Phase1Config()
        print("  - Phase1Config: OK")

        # データベース接続
        from src.db.connection import DatabaseConnection
        db = DatabaseConnection()
        print("  - DatabaseConnection: OK")

        # 接続テスト
        if not db.health_check():
            print("\n[ERROR] データベースに接続できません")
            print("  以下を確認してください:")
            print("  1. Docker コンテナが起動しているか")
            print("     docker compose -f docker/docker-compose.yml ps")
            print("  2. PostgreSQL が healthy 状態か")
            print("  3. POSTGRES_PASSWORD が正しいか")
            sys.exit(1)
        print("  - データベース接続テスト: OK")

        # Azure Embedding クライアント
        from src.embedding.azure_client import AzureEmbeddingClient
        embedding_client = AzureEmbeddingClient()
        print("  - AzureEmbeddingClient: OK")

        # リポジトリと各コンポーネント
        from src.core.memory_repository import MemoryRepository
        from src.core.strength_manager import StrengthManager
        from src.core.sleep_processor import SleepPhaseProcessor
        from src.search.vector_search import VectorSearch
        from src.search.ranking import MemoryRanker
        from src.core.task_executor import TaskExecutor

        repository = MemoryRepository(db, config)
        strength_manager = StrengthManager(repository, config)
        sleep_processor = SleepPhaseProcessor(db, config)
        vector_search = VectorSearch(db, embedding_client, config)
        ranker = MemoryRanker(config)
        print("  - Repository, StrengthManager, etc: OK")

        # TaskExecutor（メインAPI）の作成
        executor = TaskExecutor(
            vector_search=vector_search,
            ranker=ranker,
            strength_manager=strength_manager,
            sleep_processor=sleep_processor,
            repository=repository,
            config=config,
        )
        print("  - TaskExecutor: OK")
        print("\nStep 1 完了: 全コンポーネントの初期化に成功しました\n")

    except Exception as e:
        print(f"\n[ERROR] 初期化に失敗しました: {e}")
        print("\n詳細は docs/quickstart.ja.md を参照してください")
        sys.exit(1)

    # ============================================
    # Step 2: 記憶を1件保存
    # ============================================
    print("Step 2: 記憶の保存...")

    try:
        # 学びを記録
        # record_learning() は新しい記憶を保存し、memory_id を返します
        memory_id = executor.record_learning(
            agent_id="quickstart_agent",
            content="PostgreSQLのpgvectorは1万件未満ならインデックスなしで十分高速に動作する",
            learning="ベクトル検索のパフォーマンスに関する知見",
            perspective="performance",
        )
        print(f"  - 記憶を保存しました")
        print(f"  - memory_id: {memory_id}")
        print("\nStep 2 完了: 記憶の保存に成功しました\n")

    except Exception as e:
        print(f"\n[ERROR] 記憶の保存に失敗しました: {e}")
        sys.exit(1)

    # ============================================
    # Step 3: 検索で取得
    # ============================================
    print("Step 3: 記憶の検索...")

    try:
        # search_memories() はクエリに関連する記憶を検索します
        # 検索候補になった記憶は candidate_count がインクリメントされます（2段階強化の第1段階）
        results = executor.search_memories(
            query="ベクトル検索のパフォーマンス",
            agent_id="quickstart_agent",
        )
        print(f"  - 検索完了: {len(results)} 件の記憶が見つかりました")
        print("\nStep 3 完了: 記憶の検索に成功しました\n")

    except Exception as e:
        print(f"\n[ERROR] 検索に失敗しました: {e}")
        sys.exit(1)

    # ============================================
    # Step 4: 結果を表示
    # ============================================
    print("Step 4: 検索結果の表示...")
    print("-" * 50)

    if results:
        for i, scored_memory in enumerate(results, 1):
            memory = scored_memory.memory
            print(f"\n[{i}] 記憶ID: {memory.id}")
            print(f"    内容: {memory.content[:60]}...")
            print(f"    スコア: {scored_memory.final_score:.3f}")
            print(f"    強度: {memory.strength:.2f}")
            print(f"    アクセス回数: {memory.access_count}")
            print(f"    候補回数: {memory.candidate_count}")
    else:
        print("  検索結果がありません")

    print("\n" + "-" * 50)
    print("\nStep 4 完了: 結果表示に成功しました\n")

    # ============================================
    # クリーンアップ
    # ============================================
    print("クリーンアップ中...")
    db.close()
    print("データベース接続をクローズしました")

    print("\n" + "=" * 50)
    print("クイックスタート完了!")
    print("=" * 50)
    print("\n次のステップ:")
    print("  - docs/quickstart.ja.md: 詳細な使用方法")
    print("  - docs/api-reference.ja.md: API リファレンス")
    print("  - examples/claude_code_memory.py: Claude Code との連携例")


if __name__ == "__main__":
    main()
