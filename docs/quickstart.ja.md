# LLM永続メモリ Phase 1 クイックスタート

Phase 1 MVP を最短で動かすためのガイドです。5分で環境構築から動作確認まで完了できます。

---

## 前提条件

以下がインストールされていることを確認してください：

- **Docker** および **Docker Compose** (PostgreSQL + pgvector 用)
- **Python 3.12+**
- **Azure OpenAI アカウント** (text-embedding-3-small デプロイメント)

---

## 1. 環境構築

### 1.1 リポジトリのクローン

```bash
git clone https://github.com/MuneyukiHirao/llm-persistent-memory.git
cd llm-persistent-memory
```

### 1.2 .env ファイルの作成

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定します：

```bash
# PostgreSQL パスワード（任意の安全な文字列）
POSTGRES_PASSWORD=your_secure_password_here

# Azure OpenAI Embedding 設定
# 方式1: 標準的な変数名
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here

# 方式2: 代替の変数名（どちらか一方でOK）
# OpenAIEmbeddingURI=https://your-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15
# OpenAIEmbeddingKey=your_api_key_here
```

> **Note**: Azure OpenAI で `text-embedding-3-small` モデルをデプロイしておく必要があります。

### 1.3 Docker Compose 起動

PostgreSQL + pgvector を起動します：

```bash
cd docker
docker compose up -d
cd ..
```

起動確認：

```bash
docker compose -f docker/docker-compose.yml ps
# NAME             STATUS    PORTS
# docker-postgres-1  healthy   0.0.0.0:5432->5432/tcp
```

### 1.4 Python 仮想環境のセットアップ

```bash
# 仮想環境の作成（未作成の場合）
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate

# 依存パッケージのインストール
pip install psycopg2-binary openai anthropic
```

### 1.5 データベーススキーマの初期化

```bash
# DATABASE_URL を設定してスキーマを適用
export DATABASE_URL="postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_memory"

# psql でスキーマを適用（Docker コンテナ経由）
docker exec -i docker-postgres-1 psql -U agent -d agent_memory < src/db/schema.sql
```

---

## 2. 動作確認（5行で動くサンプル）

以下のコードをファイルに保存して実行するか、Python REPL で直接実行できます。

### 2.1 最小構成サンプル

```python
# quickstart_test.py
import os
from dotenv import load_dotenv

# .env ファイルを読み込む
load_dotenv()

# DATABASE_URL を設定
os.environ["DATABASE_URL"] = f"postgresql://agent:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/agent_memory"

# コンポーネントのインポート
from src.config.phase1_config import Phase1Config
from src.db.connection import DatabaseConnection
from src.embedding.azure_client import AzureEmbeddingClient
from src.core.memory_repository import MemoryRepository
from src.core.strength_manager import StrengthManager
from src.core.sleep_processor import SleepPhaseProcessor
from src.search.vector_search import VectorSearch
from src.search.ranking import MemoryRanker
from src.core.task_executor import TaskExecutor

# 初期化
config = Phase1Config()
db = DatabaseConnection()
embedding_client = AzureEmbeddingClient()
repository = MemoryRepository(db, config)
strength_manager = StrengthManager(repository, config)
sleep_processor = SleepPhaseProcessor(db, config)
vector_search = VectorSearch(db, embedding_client, config)
ranker = MemoryRanker(config)

# TaskExecutor の作成（メインAPI）
executor = TaskExecutor(
    vector_search=vector_search,
    ranker=ranker,
    strength_manager=strength_manager,
    sleep_processor=sleep_processor,
    repository=repository,
    config=config,
)

# 学びを記録
memory_id = executor.record_learning(
    agent_id="test_agent",
    content="PostgreSQLのpgvectorは1万件未満ならインデックスなしで十分高速",
    learning="ベクトル検索のパフォーマンス知見",
    perspective="performance",
)
print(f"記録完了: memory_id = {memory_id}")

# メモリを検索
results = executor.search_memories(
    query="ベクトル検索のパフォーマンス",
    agent_id="test_agent",
)
print(f"検索結果: {len(results)} 件")
for r in results:
    print(f"  - {r.memory.content[:50]}... (score: {r.final_score:.3f})")

# クリーンアップ
db.close()
```

### 2.2 実行

```bash
# python-dotenv が必要
pip install python-dotenv

# 実行
python quickstart_test.py
```

期待される出力：

```
記録完了: memory_id = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
検索結果: 1 件
  - PostgreSQLのpgvectorは1万件未満ならインデックスなしで十分高速... (score: 0.850)
```

---

## 3. 基本的な使い方

### 3.1 メモリの記録

```python
# 新しい学びを記録
memory_id = executor.record_learning(
    agent_id="my_agent",
    content="学びの内容（ベクトル検索の対象になるテキスト）",
    learning="観点別の学び内容",
    perspective="cost",  # 観点名（省略可）
)
```

### 3.2 メモリの検索

```python
# クエリに関連するメモリを検索
results = executor.search_memories(
    query="検索クエリ",
    agent_id="my_agent",
    perspective="cost",  # 観点でフィルタ（省略可）
)

# 結果を使用
for scored_memory in results:
    print(f"Content: {scored_memory.memory.content}")
    print(f"Score: {scored_memory.final_score}")
    print(f"Strength: {scored_memory.memory.strength}")
```

### 3.3 タスク実行フロー

```python
# メモリを参照しながらタスクを実行
def my_task(memories):
    # memories: 検索で見つかった関連メモリのリスト
    context = "\n".join([m.memory.content for m in memories])
    return f"タスク結果（参照: {len(memories)}件のメモリ）"

result = executor.execute_task(
    query="タスクに関連するクエリ",
    agent_id="my_agent",
    task_func=my_task,
    learning_content="このタスクで得た新しい学び",  # 省略可
    learning_text="学びの詳細説明",  # 省略可
)

print(f"タスク結果: {result.task_result}")
print(f"使用されたメモリ: {len(result.used_memory_ids)}件")
print(f"記録された学び: {result.recorded_memory_id}")
```

### 3.4 睡眠フェーズ（減衰処理）

```python
# タスク完了後に睡眠フェーズを実行
sleep_result = executor.run_sleep_phase(agent_id="my_agent")

print(f"減衰: {sleep_result.decayed_count}件")
print(f"アーカイブ: {sleep_result.archived_count}件")
print(f"統合: {sleep_result.consolidated_count}件")
```

---

## 4. 次のステップ

- [examples/basic_usage.py](../examples/basic_usage.py) - より詳細な使用例
- [api-reference.ja.md](./api-reference.ja.md) - 全API リファレンス
- [architecture.ja.md](./architecture.ja.md) - システムアーキテクチャの詳細
- [phase1-implementation-spec.ja.md](./phase1-implementation-spec.ja.md) - パラメータ設定の詳細

---

## トラブルシューティング

### Docker コンテナが起動しない

```bash
# ログを確認
docker compose -f docker/docker-compose.yml logs postgres

# コンテナを再作成
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up -d
```

### データベース接続エラー

```bash
# DATABASE_URL が正しく設定されているか確認
echo $DATABASE_URL

# PostgreSQL に接続できるか確認
docker exec -it docker-postgres-1 psql -U agent -d agent_memory -c "SELECT 1"
```

### Azure OpenAI エラー

```bash
# 環境変数を確認
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY

# または
echo $OpenAIEmbeddingURI
echo $OpenAIEmbeddingKey
```

- デプロイメント名が `text-embedding-3-small` になっているか確認
- API キーが有効か確認
- エンドポイント URL が正しいか確認

---

*作成日: 2025年1月13日*
*Phase 1 MVP バージョン*
