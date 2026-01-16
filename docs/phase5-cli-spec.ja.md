# Phase 5 実装仕様書: CLI インターフェース

## 概要

本ドキュメントは、永続的メモリシステムを**Pythonコードを書かずに**操作するためのCLIインターフェースの実装仕様を定義する。

**目標**: ターミナルからエージェントの登録、教育、タスク依頼を直感的に行えるようにする

---

## 1. 設計思想

### 1.1 核心原則

1. **Python不要**: ユーザーはPythonコードを書く必要がない
2. **宣言的設定**: YAMLファイルでエージェントを定義
3. **オーケストレーター中心**: タスクは基本的にオーケストレーター経由で処理
4. **既存実装の活用**: Phase 1-4 の実装をそのまま活用

### 1.2 設計方針

```
ユーザー（ターミナル）
       ↓
   CLI (agent コマンド)
       ↓
┌──────────────────────────────────┐
│  CLI レイヤー                      │
│  - コマンド解析                     │
│  - 入出力フォーマット               │
│  - エラーハンドリング               │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│  既存実装（Phase 1-4）              │
│  - AgentRegistry                   │
│  - EducationProcess                │
│  - Orchestrator                    │
│  - TaskExecutor                    │
│  - MemoryRepository                │
└──────────────────────────────────┘
       ↓
   PostgreSQL + pgvector
```

---

## 2. コマンド体系

### 2.1 メインコマンド構造

```bash
agent <サブコマンド> [オプション]
```

### 2.2 サブコマンド一覧

| サブコマンド | 説明 | 優先度 |
|-------------|------|--------|
| `init` | 環境初期化（DB接続確認、テーブル作成） | P0 |
| `register` | エージェント登録 | P0 |
| `list` | エージェント一覧表示 | P0 |
| `status` | エージェント状態確認 | P0 |
| `task` | タスク依頼（オーケストレーター経由） | P0 |
| `educate` | 教科書による教育 | P1 |
| `memory` | メモリ確認・操作 | P1 |
| `sleep` | 睡眠フェーズ実行 | P1 |
| `session` | セッション管理 | P2 |
| `config` | 設定管理 | P2 |

---

## 3. コマンド詳細設計

### 3.1 agent init

環境を初期化し、システムが使える状態にする。

```bash
# 基本使用
agent init

# オプション
agent init --force          # 既存データを削除して再初期化
agent init --check-only     # 接続確認のみ（変更なし）
```

**処理内容**:
1. 環境変数の確認（DATABASE_URL, AZURE_OPENAI_*）
2. DB接続テスト
3. 必要なテーブルが存在するか確認
4. 存在しない場合はマイグレーション実行

**出力例**:
```
✓ 環境変数を確認しました
✓ データベースに接続しました (PostgreSQL 16.11)
✓ pgvector 拡張が有効です (0.8.1)
✓ 必要なテーブルが存在します
✓ Azure OpenAI Embedding に接続できます

システムは使用可能です。
```

---

### 3.2 agent register

エージェントを登録する。

```bash
# YAMLファイルから登録
agent register -f agents/research_agent.yaml
agent register --file agents/research_agent.yaml

# コマンドラインから直接登録
agent register \
  --id research_agent \
  --name "研究エージェント" \
  --role "論文調査と技術リサーチ" \
  --perspectives "正確性,網羅性,関連性,引用,要約" \
  --capabilities "research,summarization,citation"

# オプション
agent register -f agents/*.yaml          # 複数ファイル一括登録
agent register -f agent.yaml --update    # 既存エージェントを更新
agent register -f agent.yaml --dry-run   # 登録内容を確認（実行しない）
```

**YAMLスキーマ** (`agents/research_agent.yaml`):
```yaml
# エージェント定義ファイル
agent_id: research_agent
name: 研究エージェント
role: 論文調査と技術リサーチを行う専門家

# 判断の観点（5つ程度推奨）
perspectives:
  - 正確性    # 情報の正確さ
  - 網羅性    # 必要な情報を網羅しているか
  - 関連性    # クエリとの関連度
  - 引用      # 出典の明示
  - 要約      # 簡潔にまとめる能力

# 能力タグ（ルーティング判断に使用）
capabilities:
  - research
  - summarization
  - citation
  - paper_analysis

# システムプロンプト
system_prompt: |
  あなたは研究専門のエージェントです。
  論文やドキュメントを調査し、正確で網羅的な情報を提供します。

  ## 行動指針
  - 出典を必ず明示する
  - 不確かな情報は「推測」と明記する
  - 複雑な概念はわかりやすく説明する

# オプション: 初期メモリ（エージェントの「経験」として登録）
initial_memories:
  - content: "論文のAbstractは目的・手法・結果・結論の順で読む"
    scope_level: universal
  - content: "arXivは査読前論文が多いため、引用時は注意が必要"
    scope_level: domain
    scope_domain: academic-research
```

**出力例**:
```
エージェントを登録しました:
  ID: research_agent
  名前: 研究エージェント
  観点: 正確性, 網羅性, 関連性, 引用, 要約
  能力: research, summarization, citation, paper_analysis

初期メモリ 2 件を登録しました。
```

---

### 3.3 agent list

登録済みエージェントの一覧を表示する。

```bash
# 基本使用
agent list

# オプション
agent list --format json         # JSON形式で出力
agent list --format table        # テーブル形式（デフォルト）
agent list --status active       # アクティブなエージェントのみ
agent list --verbose             # 詳細情報を含める
```

**出力例**:
```
登録済みエージェント (3件):

ID                 名前                  役割                       状態     メモリ数
─────────────────────────────────────────────────────────────────────────────────────
orchestrator_01    オーケストレーター     タスク振り分けと管理         active   45
research_agent     研究エージェント       論文調査と技術リサーチ       active   128
impl_agent         実装エージェント       コード実装とレビュー         active   89
```

---

### 3.4 agent status

特定エージェントの詳細状態を表示する。

```bash
# 基本使用
agent status research_agent

# オプション
agent status research_agent --memories        # 最新メモリも表示
agent status research_agent --memories --limit 5
agent status research_agent --statistics      # 統計情報を表示
```

**出力例**:
```
エージェント: research_agent

基本情報:
  名前: 研究エージェント
  役割: 論文調査と技術リサーチ
  状態: active
  作成日: 2026-01-15 10:00:00
  最終更新: 2026-01-16 14:30:00

観点:
  - 正確性
  - 網羅性
  - 関連性
  - 引用
  - 要約

能力タグ:
  - research, summarization, citation, paper_analysis

メモリ統計:
  総数: 128
  アクティブ: 120
  アーカイブ: 8
  平均強度: 1.24
  最高定着レベル: 3

最近のタスク (5件):
  2026-01-16 14:00 - RAG技術の調査 (成功)
  2026-01-16 12:30 - LLMファインチューニング調査 (成功)
  ...
```

---

### 3.5 agent task

オーケストレーター経由でタスクを依頼する。

```bash
# 基本使用（オーケストレーターが適切なエージェントを選択）
agent task "最新のRAG技術について調査して"

# 特定エージェントに直接依頼
agent task "論文を要約して" --agent research_agent

# オプション
agent task "..." --perspective "コスト"      # 重視する観点を指定
agent task "..." --context "前回の続き"      # 追加コンテキスト
agent task "..." --file input.txt            # ファイルから入力
agent task "..." --output result.md          # 結果をファイルに保存
agent task "..." --session <session_id>      # 既存セッションで継続
agent task "..." --wait                      # 完了まで待機（デフォルト）
agent task "..." --async                     # 非同期実行
agent task "..." --verbose                   # 詳細ログを表示
```

**実行例**:
```bash
$ agent task "Pythonでファイル操作のベストプラクティスをまとめて"

タスクを受け付けました (session: abc123)

ルーティング中...
  → research_agent を選択しました (スコア: 0.85)
  理由: 「ベストプラクティス」「まとめて」から調査・要約タスクと判断

実行中...

────────────────────────────────────────────
【結果】

# Pythonファイル操作のベストプラクティス

## 1. pathlib を使用する
Python 3.4以降は `pathlib` がおすすめです...

## 2. コンテキストマネージャーを使う
`with open()` を使ってファイルを自動的に閉じる...

## 3. エンコーディングを明示する
...
────────────────────────────────────────────

タスク完了 (1.2秒)
使用されたメモリ: 3件
新しい学び: 0件
```

---

### 3.6 agent educate

教科書を使ってエージェントを教育する。

```bash
# 基本使用
agent educate research_agent -f textbooks/research_basics.yaml

# オプション
agent educate research_agent -f textbook.yaml --quiz      # クイズを実行
agent educate research_agent -f textbook.yaml --dry-run   # 確認のみ
agent educate research_agent -f textbooks/*.yaml          # 複数教科書
```

**教科書YAMLスキーマ** (`textbooks/research_basics.yaml`):
```yaml
# 教科書定義ファイル
title: 研究の基礎
description: 論文の読み方と調査手法の基礎を学ぶ
scope_level: domain
scope_domain: academic-research

chapters:
  - title: 論文の読み方
    content: |
      論文を効率的に読むには、以下の順序がおすすめです：

      1. **Abstract**: 目的、手法、結果の概要を把握
      2. **Introduction**: 背景と研究の動機を理解
      3. **Conclusion**: 主要な発見と意義を確認
      4. **Methods**: 詳細な手法を理解（必要に応じて）
      5. **Results**: データと図表を精査

      最初から順に読むのではなく、必要な情報を素早く見つける
      スキミング技術が重要です。

    # 理解度確認のクイズ（オプション）
    quiz:
      - question: 論文を読む際、最初に確認すべきセクションは？
        answer: Abstract
        explanation: Abstractで論文の概要を把握してから詳細に進む

      - question: スキミング技術とは何か？
        answer: 必要な情報を素早く見つけるための読み方
        explanation: 最初から順に読むのではなく、重要な部分を優先的に読む

  - title: 信頼性の評価
    content: |
      論文の信頼性を評価する際のチェックポイント：

      - **査読の有無**: 査読付きジャーナルか、プレプリントか
      - **引用数**: 他の研究者にどれだけ参照されているか
      - **著者の実績**: 著者の過去の研究実績
      - **方法論の妥当性**: 実験設計や統計手法は適切か
      - **再現性**: 他の研究で結果が再現されているか
```

**出力例**:
```
教科書を読み込みました: 研究の基礎

チャプター 1/2: 論文の読み方
  → 5 チャンクに分割
  → メモリに登録中... 完了
  → クイズ実行中...
    Q: 論文を読む際、最初に確認すべきセクションは？
    A: Abstract ✓ 正解！
    Q: スキミング技術とは何か？
    A: 必要な情報を素早く見つけるための読み方 ✓ 正解！
  → 2段階強化を適用

チャプター 2/2: 信頼性の評価
  → 3 チャンクに分割
  → メモリに登録中... 完了

教育完了:
  登録メモリ: 8 件
  クイズ正解率: 100%
  推定定着レベル: 1
```

---

### 3.7 agent memory

エージェントのメモリを確認・操作する。

```bash
# メモリ一覧を表示
agent memory research_agent
agent memory research_agent --limit 20
agent memory research_agent --status active      # アクティブのみ
agent memory research_agent --status archived    # アーカイブのみ

# メモリを検索
agent memory research_agent --search "論文の読み方"
agent memory research_agent --search "..." --perspective "正確性"

# メモリを手動追加
agent memory research_agent --add "新しい知識をここに記述"
agent memory research_agent --add-file knowledge.txt

# メモリを削除（アーカイブ）
agent memory research_agent --archive <memory_id>

# メモリの詳細を表示
agent memory research_agent --show <memory_id>
```

**出力例**:
```bash
$ agent memory research_agent --limit 5

research_agent のメモリ (120件中 上位5件):

ID          強度   定着   最終アクセス      内容
──────────────────────────────────────────────────────────────────────
mem_001     1.45   3      2026-01-16 14:00  論文のAbstractは目的・手法・結果...
mem_002     1.32   2      2026-01-16 12:30  RAGはRetrieval-Augmented Gener...
mem_003     1.28   2      2026-01-16 10:00  arXivは査読前論文が多いため注意...
mem_004     1.15   1      2026-01-15 16:00  ベクトル検索の類似度閾値は0.3が...
mem_005     1.10   1      2026-01-15 14:00  LLMのハルシネーションを防ぐには...

--limit を増やすか --search で絞り込んでください
```

---

### 3.8 agent sleep

睡眠フェーズを手動で実行する。

```bash
# 特定エージェントの睡眠
agent sleep research_agent

# 全エージェントの睡眠
agent sleep --all

# オプション
agent sleep research_agent --dry-run    # 実行せずに影響を確認
agent sleep research_agent --verbose    # 詳細ログを表示
```

**出力例**:
```
research_agent の睡眠フェーズを実行中...

減衰処理:
  対象メモリ: 120 件
  処理済み: 120 件
  平均減衰率: 0.995

アーカイブ処理:
  閾値以下: 3 件
  アーカイブ: 3 件

統合処理:
  (Phase 1: スキップ)

睡眠フェーズ完了:
  アクティブメモリ: 117 件
  新規アーカイブ: 3 件
```

---

### 3.9 agent session

セッション（会話の継続）を管理する。

```bash
# 新規セッション開始
agent session start "新機能開発プロジェクト"

# セッション一覧
agent session list
agent session list --status in_progress

# セッション状態確認
agent session status <session_id>

# セッション再開
agent session resume <session_id>

# セッション終了
agent session close <session_id>
```

---

### 3.10 agent config

設定を管理する。

```bash
# 現在の設定を表示
agent config show

# 設定を変更
agent config set similarity_threshold 0.25
agent config set top_k_results 15

# 設定をリセット
agent config reset
agent config reset similarity_threshold
```

---

## 4. YAML スキーマ定義

### 4.1 エージェント定義スキーマ

```yaml
# JSON Schema (YAML形式で記述)
$schema: "http://json-schema.org/draft-07/schema#"
type: object
required:
  - agent_id
  - name
  - role
  - perspectives

properties:
  agent_id:
    type: string
    pattern: "^[a-z][a-z0-9_]*$"
    description: エージェントID（英小文字、数字、アンダースコア）

  name:
    type: string
    maxLength: 128
    description: エージェント名（日本語可）

  role:
    type: string
    maxLength: 256
    description: 役割の説明

  perspectives:
    type: array
    items:
      type: string
    minItems: 3
    maxItems: 7
    description: 判断の観点（5つ程度推奨）

  capabilities:
    type: array
    items:
      type: string
    description: 能力タグ（ルーティング判断に使用）

  system_prompt:
    type: string
    description: システムプロンプト

  initial_memories:
    type: array
    items:
      type: object
      properties:
        content:
          type: string
        scope_level:
          type: string
          enum: [universal, domain, project]
        scope_domain:
          type: string
        scope_project:
          type: string
```

### 4.2 教科書スキーマ

```yaml
$schema: "http://json-schema.org/draft-07/schema#"
type: object
required:
  - title
  - chapters

properties:
  title:
    type: string
    description: 教科書タイトル

  description:
    type: string
    description: 教科書の説明

  scope_level:
    type: string
    enum: [universal, domain, project]
    default: project

  scope_domain:
    type: string

  scope_project:
    type: string

  chapters:
    type: array
    items:
      type: object
      required:
        - title
        - content
      properties:
        title:
          type: string
        content:
          type: string
        quiz:
          type: array
          items:
            type: object
            required:
              - question
              - answer
            properties:
              question:
                type: string
              answer:
                type: string
              explanation:
                type: string
```

### 4.3 既存実装との連携マッピング

CLIコマンドは以下の既存実装クラスを直接活用する。

| CLIコマンド | 主要な既存クラス | モジュール |
|------------|-----------------|-----------|
| `agent init` | DatabaseConnection | src/db/connection.py |
| `agent register` | AgentRegistry, AgentDefinition | src/agents/agent_registry.py |
| `agent list` | AgentRegistry | src/agents/agent_registry.py |
| `agent status` | AgentRegistry, MemoryRepository | src/agents/, src/core/ |
| `agent task` | Orchestrator, Router, Evaluator | src/orchestrator/*.py |
| `agent educate` | EducationProcess, TextbookLoader | src/education/*.py |
| `agent memory` | MemoryRepository, VectorSearch | src/core/, src/search/ |
| `agent sleep` | SleepPhaseProcessor | src/core/sleep_processor.py |
| `agent session` | ProgressManager | src/orchestrator/progress_manager.py |

---

## 5. 既存実装の活用方法

### 5.1 CLIアプリケーション初期化

CLIアプリケーション起動時に、必要な依存関係を初期化する。

```python
# cli/main.py

import click
from src.db.connection import DatabaseConnection
from src.agents.agent_registry import AgentRegistry, AgentDefinition
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.router import Router
from src.orchestrator.evaluator import Evaluator
from src.core.task_executor import TaskExecutor
from src.core.memory_repository import MemoryRepository
from src.education.education_process import EducationProcess
from src.education.textbook import TextbookLoader
from src.embedding.azure_client import AzureEmbeddingClient
from src.search.vector_search import VectorSearch
from src.config.phase2_config import Phase2Config


class CLIContext:
    """CLI共通コンテキスト（依存関係を保持）"""

    def __init__(self):
        self.db = DatabaseConnection()
        self.config = Phase2Config()

        # 基盤コンポーネント
        self.agent_registry = AgentRegistry(self.db)
        self.memory_repository = MemoryRepository(self.db, self.config)
        self.embedding_client = AzureEmbeddingClient()
        self.vector_search = VectorSearch(
            self.db, self.embedding_client, self.config
        )

        # 高レベルコンポーネント（遅延初期化）
        self._orchestrator = None
        self._task_executor = None

    @property
    def task_executor(self) -> TaskExecutor:
        """TaskExecutor を遅延初期化"""
        if self._task_executor is None:
            self._task_executor = TaskExecutor(
                repository=self.memory_repository,
                vector_search=self.vector_search,
                embedding_client=self.embedding_client,
                config=self.config,
            )
        return self._task_executor

    @property
    def orchestrator(self) -> Orchestrator:
        """Orchestrator を遅延初期化"""
        if self._orchestrator is None:
            router = Router(self.agent_registry, config=self.config)
            evaluator = Evaluator(config=self.config)
            self._orchestrator = Orchestrator(
                agent_id="orchestrator_cli",
                router=router,
                evaluator=evaluator,
                task_executor=self.task_executor,
                config=self.config,
            )
        return self._orchestrator


# click の pass_context でCLIContextを共有
pass_context = click.make_pass_decorator(CLIContext, ensure=True)
```

### 5.2 コマンド別の実装フロー

#### agent register

```python
# cli/commands/register.py

@agent.command()
@click.option('-f', '--file', type=click.Path(exists=True), required=True)
@click.option('--update', is_flag=True, help='既存エージェントを更新')
@pass_context
def register(ctx: CLIContext, file: str, update: bool):
    """エージェントを登録"""

    # 1. YAMLファイルを読み込み
    with open(file, 'r') as f:
        data = yaml.safe_load(f)

    # 2. AgentDefinition を構築
    agent_def = AgentDefinition(
        agent_id=data['agent_id'],
        name=data['name'],
        role=data['role'],
        perspectives=data['perspectives'],
        system_prompt=data.get('system_prompt', ''),
        capabilities=data.get('capabilities', []),
    )

    # 3. 既存チェック
    existing = ctx.agent_registry.get_by_id(agent_def.agent_id)

    if existing and not update:
        raise click.ClickException(
            f"エージェント {agent_def.agent_id} は既に存在します。"
            f"更新する場合は --update を指定してください。"
        )

    # 4. 登録または更新
    if existing and update:
        ctx.agent_registry.update(agent_def)
        click.echo(f"エージェントを更新しました: {agent_def.agent_id}")
    else:
        ctx.agent_registry.register(agent_def)
        click.echo(f"エージェントを登録しました: {agent_def.agent_id}")

    # 5. 初期メモリがあれば登録
    initial_memories = data.get('initial_memories', [])
    for mem_data in initial_memories:
        memory = AgentMemory.create_from_education(
            agent_id=agent_def.agent_id,
            content=mem_data['content'],
            embedding=ctx.embedding_client.get_embedding(mem_data['content']),
            scope_level=mem_data.get('scope_level', 'project'),
        )
        ctx.memory_repository.create(memory)

    if initial_memories:
        click.echo(f"初期メモリ {len(initial_memories)} 件を登録しました。")
```

#### agent task

```python
# cli/commands/task.py

@agent.command()
@click.argument('task_description')
@click.option('--agent', 'agent_id', help='特定エージェントに直接依頼')
@click.option('--perspective', help='重視する観点')
@click.option('--verbose', is_flag=True, help='詳細ログを表示')
@pass_context
def task(ctx: CLIContext, task_description: str, agent_id: str,
         perspective: str, verbose: bool):
    """タスクを依頼（オーケストレーター経由）"""

    click.echo("タスクを受け付けました\n")

    # 1. 特定エージェント指定時は直接実行
    if agent_id:
        # TaskExecutor を使って直接実行
        result = ctx.task_executor.execute_task(
            agent_id=agent_id,
            task_description=task_description,
            perspective=perspective,
        )
        _display_result(result, verbose)
        return

    # 2. オーケストレーター経由で実行
    click.echo("ルーティング中...")

    orchestrator_result = ctx.orchestrator.process_request(
        task_summary=task_description,
        items=[],  # CLIでは論点リストなし
    )

    # 3. ルーティング結果を表示
    routing = orchestrator_result.routing_decision
    click.echo(
        f"  → {routing.selected_agent_id} を選択しました "
        f"(スコア: {routing.confidence:.2f})"
    )
    click.echo(f"  理由: {routing.selection_reason}\n")

    # 4. 結果を表示
    click.echo("─" * 40)
    click.echo("【結果】\n")
    click.echo(orchestrator_result.agent_result.get('output', ''))
    click.echo("─" * 40 + "\n")

    # 5. 完了メッセージ
    click.echo(f"タスク完了 (session: {orchestrator_result.session_id})")
```

#### agent educate

```python
# cli/commands/educate.py

@agent.command()
@click.argument('agent_id')
@click.option('-f', '--file', type=click.Path(exists=True), required=True)
@click.option('--quiz', is_flag=True, help='クイズを実行')
@pass_context
def educate(ctx: CLIContext, agent_id: str, file: str, quiz: bool):
    """教科書でエージェントを教育"""

    # 1. エージェント存在確認
    agent = ctx.agent_registry.get_by_id(agent_id)
    if not agent:
        raise click.ClickException(f"エージェント {agent_id} が見つかりません")

    # 2. 教科書を読み込み
    loader = TextbookLoader()
    textbook = loader.load(file)

    click.echo(f"教科書を読み込みました: {textbook.title}\n")

    # 3. EducationProcess を実行
    education_process = EducationProcess(
        agent_id=agent_id,
        textbook=textbook,
        repository=ctx.memory_repository,
        embedding_client=ctx.embedding_client,
        config=ctx.config,
    )

    result = education_process.run()

    # 4. 結果を表示
    click.echo(f"\n教育完了:")
    click.echo(f"  登録メモリ: {result.memories_created} 件")
    click.echo(f"  クイズ正解率: {result.pass_rate * 100:.0f}%")
    click.echo(f"  完了チャプター: {result.chapters_completed}")
```

### 5.3 既存実装クラスの役割

| クラス | 役割 | CLI での使用場面 |
|-------|------|-----------------|
| `AgentRegistry` | エージェント定義のCRUD | register, list, status |
| `AgentDefinition` | エージェント定義データクラス | register |
| `EducationProcess` | 教科書による教育フロー | educate |
| `TextbookLoader` | 教科書YAMLの読み込み | educate |
| `Orchestrator` | タスクルーティングと評価 | task |
| `Router` | エージェント選択ロジック | task（内部） |
| `Evaluator` | フィードバック評価 | task（内部） |
| `TaskExecutor` | 記憶検索・学び記録 | task, memory |
| `MemoryRepository` | 記憶のCRUD | memory, status |
| `VectorSearch` | ベクトル検索 | memory --search |
| `SleepPhaseProcessor` | 睡眠フェーズ処理 | sleep |
| `DatabaseConnection` | DB接続管理 | init, 全コマンド |

### 5.4 エラーハンドリングの統一

```python
# cli/utils/error_handler.py

import click
from psycopg2 import OperationalError


class CLIErrorHandler:
    """CLI用エラーハンドラー"""

    ERROR_CODES = {
        'general': 1,
        'argument': 2,
        'config': 3,
        'connection': 4,
        'auth': 5,
    }

    @staticmethod
    def handle_db_error(e: Exception) -> None:
        """DB関連エラーをハンドリング"""
        if isinstance(e, OperationalError):
            click.echo(
                "[接続エラー] データベースに接続できません\n\n"
                "ヒント:\n"
                "  - DATABASE_URL 環境変数を確認してください\n"
                "  - PostgreSQL が起動しているか確認してください\n"
                "  - docker compose up -d を実行してください",
                err=True,
            )
            raise SystemExit(CLIErrorHandler.ERROR_CODES['connection'])

    @staticmethod
    def handle_agent_not_found(agent_id: str) -> None:
        """エージェント未存在エラー"""
        click.echo(
            f"[エラー] エージェント '{agent_id}' が見つかりません\n\n"
            f"ヒント:\n"
            f"  - agent list で登録済みエージェントを確認してください\n"
            f"  - agent register -f <file.yaml> で登録してください",
            err=True,
        )
        raise SystemExit(CLIErrorHandler.ERROR_CODES['argument'])
```

---

## 6. ディレクトリ構成

### 6.1 CLIモジュール構成

```
src/
├── cli/
│   ├── __init__.py
│   ├── main.py              # エントリポイント
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── init.py          # agent init
│   │   ├── register.py      # agent register
│   │   ├── list_agents.py   # agent list
│   │   ├── status.py        # agent status
│   │   ├── task.py          # agent task
│   │   ├── educate.py       # agent educate
│   │   ├── memory.py        # agent memory
│   │   ├── sleep.py         # agent sleep
│   │   ├── session.py       # agent session
│   │   └── config.py        # agent config
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── output.py        # 出力フォーマット（テーブル、JSON等）
│   │   ├── yaml_loader.py   # YAML読み込み・バリデーション
│   │   └── progress.py      # 進捗表示
│   └── config/
│       ├── __init__.py
│       └── cli_config.py    # CLI固有設定
scripts/
├── agent                    # シェルスクリプト（エントリポイント）
└── install.sh               # インストールスクリプト
```

### 6.2 ユーザー向けディレクトリ（推奨構成）

```
project/
├── agents/                  # エージェント定義
│   ├── research_agent.yaml
│   ├── impl_agent.yaml
│   └── test_agent.yaml
├── textbooks/               # 教科書
│   ├── research_basics.yaml
│   └── coding_standards.yaml
├── .env                     # 環境変数
└── agent.yaml               # CLI設定（オプション）
```

---

## 7. 実装計画

### 7.1 Phase 5 実装順序

| 優先度 | コンポーネント | 依存 | 工数目安 |
|--------|---------------|------|----------|
| P0-1 | CLI基盤 (click) | - | 小 |
| P0-2 | agent init | DB接続 | 小 |
| P0-3 | agent register | AgentRegistry | 中 |
| P0-4 | agent list | AgentRegistry | 小 |
| P0-5 | agent status | AgentRegistry | 小 |
| P0-6 | agent task | Orchestrator | 大 |
| P1-1 | agent educate | EducationProcess | 中 |
| P1-2 | agent memory | MemoryRepository | 中 |
| P1-3 | agent sleep | SleepPhaseProcessor | 小 |
| P2-1 | agent session | ProgressManager | 中 |
| P2-2 | agent config | Phase1-3Config | 小 |

### 7.2 マイルストーン

| マイルストーン | 達成条件 |
|---------------|---------|
| M1: 基盤完了 | CLI基盤 + init + register が動作 |
| M2: 確認機能完了 | list + status が動作 |
| M3: タスク実行完了 | task がオーケストレーター経由で動作 |
| M4: 教育機能完了 | educate が動作 |
| M5: メモリ操作完了 | memory + sleep が動作 |
| M6: セッション管理完了 | session + config が動作 |

---

## 8. 技術選定

### 8.1 CLIフレームワーク

**選定: click**

| 候補 | メリット | デメリット |
|------|---------|-----------|
| **click** | デコレータベース、自動ヘルプ、豊富な機能 | やや冗長 |
| argparse | 標準ライブラリ | コード量多い |
| typer | 型ヒントベース、シンプル | 依存が多い |

```python
# click 使用例
import click

@click.group()
def agent():
    """エージェント管理CLI"""
    pass

@agent.command()
@click.option('-f', '--file', type=click.Path(exists=True), help='YAMLファイル')
def register(file):
    """エージェントを登録する"""
    pass

if __name__ == '__main__':
    agent()
```

### 8.2 依存ライブラリ

```
# requirements.txt に追加
click>=8.0.0
pyyaml>=6.0
rich>=13.0.0  # 出力フォーマット（テーブル、色付け等）
```

---

## 9. エラーハンドリング

### 9.1 エラーメッセージ設計

```
[エラータイプ] メッセージ

ヒント: 解決方法
```

**例**:
```
[接続エラー] データベースに接続できません

ヒント:
  - DATABASE_URL 環境変数を確認してください
  - PostgreSQL が起動しているか確認してください
  - docker compose up -d を実行してください
```

### 9.2 終了コード

| コード | 意味 |
|--------|------|
| 0 | 成功 |
| 1 | 一般エラー |
| 2 | 引数エラー |
| 3 | 設定エラー |
| 4 | 接続エラー |
| 5 | 認証エラー |

---

## 10. テスト戦略

### 10.1 単体テスト

| コンポーネント | テスト内容 |
|---------------|-----------|
| YAMLローダー | スキーマバリデーション、エラー検出 |
| 出力フォーマッター | テーブル/JSON出力の正確性 |
| コマンド解析 | オプション解析の正確性 |

### 10.2 統合テスト

| シナリオ | テスト内容 |
|---------|-----------|
| 新規セットアップ | init → register → task の一連の流れ |
| 教育フロー | register → educate → memory確認 |
| タスク実行 | task → 結果確認 → memory更新確認 |

---

## 11. 将来の拡張

### 11.1 REST API（Phase 6候補）

CLI実装完了後、同じコアロジックを使ってREST APIを構築：

```python
# FastAPI による REST API
from fastapi import FastAPI

app = FastAPI()

@app.post("/agents")
async def register_agent(agent: AgentDefinition):
    pass

@app.post("/tasks")
async def create_task(task: TaskRequest):
    pass
```

### 11.2 Web UI（将来）

REST API の上にReact/Vue.jsでフロントエンドを構築。

---

*本ドキュメントは Phase 1-4 の実装を基盤として、Python知識なしでエージェントを操作するためのCLI仕様書である。*

*作成日: 2026年1月16日*
