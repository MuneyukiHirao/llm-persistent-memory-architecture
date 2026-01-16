# Phase 2 実装仕様書: オーケストレーションと入力処理層

## 概要

本ドキュメントは、[LLMエージェントの永続的メモリアーキテクチャ設計](./architecture.ja.md)に基づくPhase 2 MVP（最小検証可能プロダクト）の実装仕様を定義する。

**検証目標**: オーケストレーションが機能するか（複数エージェント間のタスク振り分けと評価）

---

## 1. MVPの範囲

### 1.1 フェーズ分割（再掲）

| フェーズ | 検証対象 | 実装範囲 |
|---------|---------|---------|
| Phase 1 | 強度管理と減衰が「個性」を生むか | 単一エージェント + 外部メモリ（**完了**） |
| **Phase 2** | **オーケストレーションが機能するか** | **複数エージェント + ルーティング** |
| Phase 3 | スケーラビリティと実運用 | 負荷分散 + 監視 + 最適化 |

### 1.2 Phase 2 で検証する核心機能

1. **入力処理層**
   - ユーザー入力の前処理（論点数検出、概要生成）
   - 過大入力の検出と交渉（閾値超えで警告・オプション提示）
   - 曖昧な指示はそのままオーケストレーターへ（解釈しない）

2. **オーケストレーター**
   - 専門エージェントへのタスクルーティング
   - ユーザーフィードバックの観察と評価
   - 進捗監視と成果物の引き継ぎ
   - オーケストレーター自身の外部メモリによる学習

3. **複数エージェント間のルーティング**
   - エージェント適性判断（ルールベース）
   - タスク依頼と結果受領
   - 暗黙的フィードバックの検出

### 1.3 Phase 2 で実装しないもの

- 学習可能なルーティングモデル（ルールベースで開始）
- 高度な負荷分散アルゴリズム（ラウンドロビンで開始）
- 複雑なエージェント間通信プロトコル（直接呼び出しで開始）
- リアルタイム進捗監視（ポーリング方式で開始）
- 複数オーケストレーターの協調（単一オーケストレーターで開始）

---

## 2. 技術スタック

### 2.1 Phase 1 からの追加・変更

| コンポーネント | Phase 1 | Phase 2 追加 |
|---------------|---------|-------------|
| LLM（オーケストレーター） | - | Claude Sonnet 4 |
| LLM（入力処理層） | - | Claude Haiku（軽量・高速） |
| LLM（専門エージェント） | Claude Sonnet 4 | 同左（変更なし） |
| 状態管理 | JSONファイル | 同左 + DB テーブル追加 |

### 2.2 ディレクトリ構成（追加分）

```
src/
├── orchestrator/
│   ├── orchestrator.py       # オーケストレーター本体
│   ├── router.py             # ルーティングロジック
│   ├── evaluator.py          # 評価フロー（フィードバック検出）
│   └── progress_manager.py   # 進捗管理
├── input_processing/
│   ├── input_processor.py    # 入力処理層本体
│   ├── item_detector.py      # 論点数検出
│   └── summarizer.py         # 概要生成
├── agents/
│   ├── agent_registry.py     # エージェント登録・管理
│   ├── agent_executor.py     # エージェント実行
│   └── definitions/          # エージェント定義ファイル
│       ├── research_agent.yaml
│       ├── implementation_agent.yaml
│       └── test_agent.yaml
└── config/
    └── phase2_config.py      # Phase 2 設定パラメータ
```

---

## 3. データスキーマ

### 3.1 追加テーブル: agent_definitions

エージェント定義のDB化（Phase 1 ではコード内定義だったものをDB管理に移行）

```sql
CREATE TABLE agent_definitions (
    -- 識別子
    agent_id VARCHAR(64) PRIMARY KEY,

    -- 基本情報
    name VARCHAR(128) NOT NULL,
    role VARCHAR(256) NOT NULL,
    perspectives TEXT[] NOT NULL,  -- 観点（5つ程度）

    -- プロンプト
    system_prompt TEXT NOT NULL,

    -- ツール定義（JSON Schema形式）
    tools JSONB DEFAULT '[]',

    -- 能力タグ（ルーティング判断に使用）
    capabilities TEXT[] DEFAULT '{}',

    -- 状態
    status VARCHAR(16) DEFAULT 'active',  -- active / disabled

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_agent_definitions_status ON agent_definitions(status);
CREATE INDEX idx_agent_definitions_capabilities ON agent_definitions USING GIN(capabilities);
```

### 3.2 追加テーブル: routing_history

ルーティング履歴（評価・学習に使用）

```sql
CREATE TABLE routing_history (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,  -- セッション識別子

    -- ルーティング情報
    orchestrator_id VARCHAR(64) NOT NULL,
    task_summary TEXT NOT NULL,          -- タスクの概要
    selected_agent_id VARCHAR(64) NOT NULL,
    selection_reason TEXT,               -- 選択理由

    -- 候補エージェント（ルーティング判断の記録）
    candidate_agents JSONB,  -- [{"agent_id": "...", "score": 0.8, "reason": "..."}]

    -- 結果
    result_status VARCHAR(16),  -- success / partial_success / failure / timeout
    result_summary TEXT,
    user_feedback VARCHAR(32),  -- positive / neutral / negative / redo_requested

    -- タイムスタンプ
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,

    -- 外部キー
    CONSTRAINT fk_selected_agent FOREIGN KEY (selected_agent_id)
        REFERENCES agent_definitions(agent_id)
);

-- インデックス
CREATE INDEX idx_routing_history_session ON routing_history(session_id);
CREATE INDEX idx_routing_history_orchestrator ON routing_history(orchestrator_id);
CREATE INDEX idx_routing_history_agent ON routing_history(selected_agent_id);
CREATE INDEX idx_routing_history_result ON routing_history(result_status);
```

### 3.3 追加テーブル: session_state

オーケストレーターのセッション状態（中間睡眠からの復帰用）

```sql
CREATE TABLE session_state (
    -- 識別子
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    orchestrator_id VARCHAR(64) NOT NULL,

    -- 元のリクエスト
    user_request JSONB NOT NULL,  -- {original: "...", clarified: "..."}

    -- 進捗状態
    task_tree JSONB NOT NULL,            -- タスク依存関係と完了状況
    current_task JSONB,                  -- 現在実行中のタスク
    overall_progress_percent INT DEFAULT 0,

    -- 状態
    status VARCHAR(16) DEFAULT 'in_progress',  -- in_progress / paused / completed / failed

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_activity_at TIMESTAMP DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_session_state_orchestrator ON session_state(orchestrator_id);
CREATE INDEX idx_session_state_status ON session_state(status);
```

### 3.4 agent_memory テーブルの拡張（Phase 1 テーブルへの追加カラム）

```sql
-- オーケストレーターのメモリにルーティング情報を格納するための拡張
ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS routing_context JSONB;
    -- {"task_type": "...", "agent_selected": "...", "success": true}
```

---

## 4. パラメータ初期値

### 4.1 Phase 2 設定クラス

```python
# config/phase2_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config.phase1_config import Phase1Config


@dataclass
class Phase2Config(Phase1Config):
    """Phase 2 MVP パラメータ設定（Phase 1 設定を継承）"""

    # === 入力処理層 ===
    input_item_threshold: int = 10          # これ以上の論点で警告
    input_size_threshold: int = 5000        # トークン数、これ以上で概要生成
    summary_max_tokens: int = 1000          # 概要の最大トークン数

    # === オーケストレーター ===
    orchestrator_model: str = "claude-sonnet-4-20250514"
    input_processor_model: str = "claude-3-5-haiku-20241022"

    # オーケストレーターの睡眠トリガー
    orchestrator_context_threshold: float = 0.7   # コンテキスト使用率70%で中間睡眠
    orchestrator_idle_timeout_minutes: int = 60   # アイドル1時間で睡眠
    orchestrator_subtask_batch_size: int = 5      # 5サブタスク完了ごとに睡眠検討

    # === ルーティング ===
    routing_method: str = "rule_based"      # "rule_based" | "similarity" | "llm"
    routing_similarity_threshold: float = 0.5   # similarity方式での適性閾値
    max_routing_candidates: int = 3         # 候補エージェントの最大数

    # === 評価 ===
    feedback_detection_method: str = "keyword"  # "keyword" | "similarity" | "llm"
    implicit_feedback_enabled: bool = True      # 暗黙的フィードバック検出

    # === エージェント管理 ===
    agent_timeout_seconds: int = 300        # エージェント実行タイムアウト（5分）
    max_retry_count: int = 2                # 失敗時の最大リトライ回数

    # === 進捗管理 ===
    progress_check_interval_seconds: int = 30   # 進捗チェック間隔
    progress_state_file: str = "memory/progress_state.json"  # 進捗状態ファイル


# デフォルト設定のインスタンス
phase2_config = Phase2Config()
```

### 4.2 ルーティングパラメータ

```python
# === ルーティングスコア重み（rule_based方式） ===
ROUTING_SCORE_WEIGHTS = {
    "capability_match": 0.40,    # 能力タグのマッチ度
    "past_success_rate": 0.30,   # 過去の成功率
    "recent_activity": 0.20,     # 最近のアクティビティ（負荷考慮）
    "perspective_match": 0.10,   # 観点のマッチ度
}

# === フィードバック判定 ===
FEEDBACK_SIGNALS = {
    "positive": ["ありがとう", "良い", "完璧", "OK", "了解"],
    "negative": ["やり直し", "違う", "ダメ", "修正して"],
    "redo_requested": ["もう一度", "再度", "別のエージェント"],
}

# === タスクサイズ上限 ===
MAX_TASK_CONTEXT_TOKENS = 50000   # エージェントに渡すタスクの最大トークン数
TASK_SIZE_MARGIN = 0.7           # コンテキストウィンドウの70%以内に収める
```

### 4.3 エージェント定義パラメータ

```python
# === オーケストレーターの観点 ===
ORCHESTRATOR_PERSPECTIVES = [
    "ユーザー意図",       # ユーザーが本当に求めていることは何か
    "エージェント適性",   # どの専門エージェントに任せるべきか
    "タスク依存関係",     # どの順序で進めるべきか
    "タスク結果評価",     # エージェントの結果は期待を満たしているか
    "ユーザー満足度",     # ユーザーは満足しているか
]

# === 入力処理層の観点（軽量、判断のみ） ===
INPUT_PROCESSOR_PERSPECTIVES = [
    "論点数",            # 項目がいくつあるか
    "入力サイズ",        # 入力が大きすぎないか
    "曖昧さ",            # 曖昧な指示かどうか（判断のみ、解消はしない）
]
```

---

## 5. コンポーネント設計

### 5.1 入力処理層（InputProcessor）

#### 役割
ユーザー入力を前処理してオーケストレーターに渡す。軽量LLM（Haiku）で実装。

#### 処理フロー

```
ユーザー入力
    ↓
┌─────────────────────────────────────┐
│  1. 論点数検出（ItemDetector）       │
│     - 10個未満 → 次へ               │
│     - 10個以上 → 警告・オプション提示 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 入力サイズ確認                   │
│     - 5000トークン未満 → そのまま    │
│     - 5000トークン以上 → 概要生成    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. オーケストレーターへ渡す         │
│     - summary: 概要（必須）          │
│     - detail_refs: 詳細へのポインタ  │
│     - items: 論点リスト              │
└─────────────────────────────────────┘
```

#### 重要な設計判断
- **「解釈」しない**: 曖昧な指示もそのままオーケストレーターに渡す
- **「曖昧さの解消」はオーケストレーターの責務**: ユーザーへの質問で明確化

#### クラス設計

```python
@dataclass
class ProcessedInput:
    """入力処理層の出力"""
    summary: str                      # 概要（1000トークン以内）
    detail_refs: List[str]            # 詳細へのポインタ
    items: List[str]                  # 検出された論点
    item_count: int                   # 論点数
    original_size_tokens: int         # 元の入力サイズ
    needs_negotiation: bool           # 交渉が必要か
    negotiation_options: List[str]    # 交渉時の選択肢


class InputProcessor:
    """入力処理層"""

    def __init__(self, config: Phase2Config):
        self.config = config
        self.item_detector = ItemDetector(config)
        self.summarizer = Summarizer(config)

    def process(self, user_input: str) -> ProcessedInput:
        """ユーザー入力を処理"""
        # 1. 論点数を検出
        items = self.item_detector.detect(user_input)

        # 2. 論点数チェック
        if len(items) >= self.config.input_item_threshold:
            return ProcessedInput(
                summary=self._generate_summary(user_input),
                items=items,
                item_count=len(items),
                needs_negotiation=True,
                negotiation_options=[
                    f"優先度の高い{self.config.input_item_threshold}個を指定してください",
                    "全て処理します（時間がかかります）",
                    "カテゴリ別に分けて順次処理します",
                ],
                ...
            )

        # 3. 入力サイズチェック
        token_count = self._count_tokens(user_input)
        if token_count > self.config.input_size_threshold:
            summary = self.summarizer.summarize(user_input)
            detail_ref = self._store_detail(user_input)
            return ProcessedInput(
                summary=summary,
                detail_refs=[detail_ref],
                ...
            )

        # 4. 小さい入力はそのまま
        return ProcessedInput(
            summary=user_input,
            detail_refs=[],
            items=items,
            ...
        )
```

### 5.2 オーケストレーター（Orchestrator）

#### 役割
専門エージェントへのタスクルーティング、結果評価、進捗管理を行う。

#### 統一設計原則（architecture.ja.md より）
**オーケストレーターは専門エージェントと全く同じ仕組みで動く。違いは「役割」と「観点」だけ。**

```
すべてのエージェント（オーケストレーター含む）：
├── 外部メモリ（強度管理、減衰、定着レベル）
├── 観点（役割に応じた5つ程度）
├── タスクごとに睡眠
├── 容量制限と強制剪定
└── アーカイブと再活性化

唯一の違い = 観点（perspectives）と役割
```

#### 処理フロー

```
入力処理層からの ProcessedInput
    ↓
┌─────────────────────────────────────┐
│  1. タスク分析                       │
│     - 外部メモリ検索（過去の類似タスク）│
│     - タスクの性質を判断             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. ルーティング判断                 │
│     - 適切なエージェントを選択        │
│     - 過去のルーティング結果を参照    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. タスク委譲                       │
│     - エージェントにタスクを依頼      │
│     - 結果を待機（タイムアウト管理）  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. 結果評価                         │
│     - ユーザーに結果を提示           │
│     - フィードバックを観察           │
│     - 成功/失敗を判定し学びを記録    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. 睡眠（条件に応じて）             │
│     - タスク完了 → 睡眠              │
│     - コンテキスト70% → 中間睡眠     │
│     - アイドル1時間 → 睡眠           │
└─────────────────────────────────────┘
```

#### クラス設計

```python
class Orchestrator:
    """オーケストレーター"""

    def __init__(
        self,
        agent_id: str,
        router: Router,
        evaluator: Evaluator,
        progress_manager: ProgressManager,
        task_executor: TaskExecutor,  # Phase 1 の TaskExecutor を再利用
        config: Phase2Config,
    ):
        self.agent_id = agent_id
        self.router = router
        self.evaluator = evaluator
        self.progress_manager = progress_manager
        self.task_executor = task_executor
        self.config = config

        # オーケストレーター自身の外部メモリ（Phase 1 の仕組みを再利用）
        self.memory_search = task_executor.vector_search
        self.memory_repository = task_executor.repository

    async def process_request(
        self,
        processed_input: ProcessedInput,
        session_id: Optional[UUID] = None,
    ) -> OrchestratorResult:
        """リクエストを処理"""

        # セッション復帰または新規作成
        session = self._get_or_create_session(session_id, processed_input)

        try:
            # 1. 外部メモリで類似タスクを検索
            past_experiences = self.task_executor.search_memories(
                query=processed_input.summary,
                agent_id=self.agent_id,
                perspective="エージェント適性",
            )

            # 2. ルーティング判断
            routing_decision = await self.router.decide(
                task_summary=processed_input.summary,
                items=processed_input.items,
                past_experiences=past_experiences,
            )

            # 3. タスク委譲
            agent_result = await self._delegate_task(
                routing_decision=routing_decision,
                processed_input=processed_input,
            )

            # 4. 結果をユーザーに提示（呼び出し側で実行）

            return OrchestratorResult(
                routing_decision=routing_decision,
                agent_result=agent_result,
                session_id=session.session_id,
            )

        finally:
            # 睡眠判定
            if self._should_sleep():
                self._run_sleep_phase(session)

    def receive_feedback(
        self,
        session_id: UUID,
        user_response: str,
    ) -> FeedbackResult:
        """ユーザーフィードバックを受信して評価"""

        # フィードバックを評価
        feedback_type = self.evaluator.evaluate(user_response)

        # 学びを記録
        if feedback_type in ["positive", "negative", "redo_requested"]:
            self._record_routing_learning(session_id, feedback_type)

        return FeedbackResult(feedback_type=feedback_type)
```

### 5.3 ルーター（Router）

#### 役割
タスクに最適なエージェントを選択する。

#### ルーティングアルゴリズム（Phase 2: ルールベース）

```python
class Router:
    """ルーティングロジック"""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        routing_history_repository: RoutingHistoryRepository,
        config: Phase2Config,
    ):
        self.agent_registry = agent_registry
        self.routing_history = routing_history_repository
        self.config = config

    async def decide(
        self,
        task_summary: str,
        items: List[str],
        past_experiences: List[ScoredMemory],
    ) -> RoutingDecision:
        """ルーティング判断"""

        # 1. 全エージェントを取得
        all_agents = self.agent_registry.get_active_agents()

        # 2. 各エージェントのスコアを計算
        candidates = []
        for agent in all_agents:
            score = self._calculate_routing_score(
                agent=agent,
                task_summary=task_summary,
                past_experiences=past_experiences,
            )
            candidates.append({
                "agent_id": agent.agent_id,
                "score": score,
                "reason": self._generate_selection_reason(agent, task_summary),
            })

        # 3. スコア順にソート
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4. 上位を選択
        selected = candidates[0]

        return RoutingDecision(
            selected_agent_id=selected["agent_id"],
            selection_reason=selected["reason"],
            candidates=candidates[:self.config.max_routing_candidates],
        )

    def _calculate_routing_score(
        self,
        agent: AgentDefinition,
        task_summary: str,
        past_experiences: List[ScoredMemory],
    ) -> float:
        """ルーティングスコアを計算"""

        score = 0.0

        # 1. 能力タグのマッチ度（0.40）
        capability_score = self._match_capabilities(
            agent.capabilities,
            task_summary,
        )
        score += capability_score * ROUTING_SCORE_WEIGHTS["capability_match"]

        # 2. 過去の成功率（0.30）
        success_rate = self._get_past_success_rate(agent.agent_id)
        score += success_rate * ROUTING_SCORE_WEIGHTS["past_success_rate"]

        # 3. 最近のアクティビティ（0.20）- 負荷分散
        activity_score = self._get_activity_score(agent.agent_id)
        score += activity_score * ROUTING_SCORE_WEIGHTS["recent_activity"]

        # 4. 観点のマッチ度（0.10）
        perspective_score = self._match_perspectives(
            agent.perspectives,
            task_summary,
        )
        score += perspective_score * ROUTING_SCORE_WEIGHTS["perspective_match"]

        return score
```

### 5.4 評価者（Evaluator）

#### 役割
ユーザーフィードバックを観察し、ルーティング結果を評価する。

#### 暗黙的フィードバックの検出

| シグナル | 解釈 | 記憶への影響 |
|---------|------|-------------|
| 結果をそのまま使用 | 成功 | 「エージェント適性」観点で成功として記憶 |
| 軽微な修正後に使用 | 部分成功 | 部分成功として記憶 |
| 大幅に修正して使用 | 部分失敗 | 改善点として記憶 |
| やり直しを依頼 | 失敗 | 失敗パターンとして記憶 |
| 別のエージェントに再依頼 | ルーティング誤り | ルーティング改善の学びとして記憶 |

```python
class Evaluator:
    """評価フロー"""

    def __init__(self, config: Phase2Config):
        self.config = config

    def evaluate(self, user_response: str) -> str:
        """ユーザー応答からフィードバックタイプを判定"""

        response_lower = user_response.lower()

        # 明示的フィードバック（キーワードマッチ）
        for signal_type, keywords in FEEDBACK_SIGNALS.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return signal_type

        # 暗黙的フィードバック
        if self.config.implicit_feedback_enabled:
            return self._detect_implicit_feedback(user_response)

        return "neutral"

    def _detect_implicit_feedback(self, user_response: str) -> str:
        """暗黙的フィードバックを検出"""

        # 応答パターン分析
        if len(user_response) < 10:
            # 短い応答 = 受け入れの可能性高
            return "positive"

        if "修正" in user_response or "直して" in user_response:
            return "partial_failure"

        return "neutral"
```

### 5.5 進捗管理（ProgressManager）

#### 役割
オーケストレーターの進捗状態を管理し、中間睡眠からの復帰を可能にする。

#### 状態保存のタイミング
- タスク指示時
- タスク結果受領時
- 問題発生時
- ユーザー判断受領時

```python
class ProgressManager:
    """進捗管理"""

    def __init__(
        self,
        session_repository: SessionStateRepository,
        config: Phase2Config,
    ):
        self.session_repository = session_repository
        self.config = config

    def save_state(
        self,
        session_id: UUID,
        task_tree: Dict,
        current_task: Optional[Dict],
        progress_percent: int,
    ):
        """進捗状態を保存"""
        self.session_repository.update(
            session_id=session_id,
            task_tree=task_tree,
            current_task=current_task,
            overall_progress_percent=progress_percent,
            last_activity_at=datetime.now(),
        )

    def restore_state(self, session_id: UUID) -> Optional[SessionState]:
        """進捗状態を復元"""
        return self.session_repository.get_by_id(session_id)

    def generate_progress_report(self, session_id: UUID) -> str:
        """進捗レポートを生成"""
        state = self.restore_state(session_id)
        if not state:
            return "セッションが見つかりません"

        completed = self._count_completed_tasks(state.task_tree)
        total = self._count_total_tasks(state.task_tree)

        report = f"""【進捗報告】
進捗率: {state.overall_progress_percent}%

完了タスク: {completed}/{total}
{self._format_task_tree(state.task_tree)}

現在のタスク:
{state.current_task.get('description', 'なし') if state.current_task else 'なし'}
"""
        return report
```

---

## 6. 実装順序

### 6.1 Phase 2 実装順序

テストのしやすさを考慮し、以下の順序で実装する。

| 優先度 | コンポーネント | 依存 | 検証ポイント |
|--------|---------------|------|-------------|
| 1 | Phase2Config | Phase1Config | 設定パラメータの追加 |
| 2 | agent_definitions テーブル | - | エージェント定義のDB化 |
| 3 | AgentRegistry | agent_definitions | エージェント登録・検索 |
| 4 | Router（ルールベース） | AgentRegistry | 単純なルーティング |
| 5 | routing_history テーブル | agent_definitions | ルーティング履歴の保存 |
| 6 | Evaluator | - | フィードバック検出 |
| 7 | InputProcessor（論点数のみ） | - | 論点数検出 |
| 8 | Orchestrator（基本フロー） | Router, Evaluator | タスク委譲・評価 |
| 9 | session_state テーブル | - | セッション状態保存 |
| 10 | ProgressManager | session_state | 中間睡眠・復帰 |
| 11 | InputProcessor（概要生成） | - | 大きな入力の処理 |
| 12 | 統合テスト | 全コンポーネント | E2Eフロー検証 |

### 6.2 マイルストーン

| マイルストーン | 達成条件 |
|---------------|---------|
| M1: 基盤完了 | Phase2Config, agent_definitions, AgentRegistry が動作 |
| M2: ルーティング完了 | Router がエージェント選択を行える |
| M3: 評価完了 | Evaluator がフィードバックを検出できる |
| M4: オーケストレーター動作 | Orchestrator が一連のフローを実行できる |
| M5: 中間睡眠完了 | ProgressManager で状態保存・復帰ができる |
| M6: 入力処理完了 | InputProcessor が大きな入力を処理できる |
| M7: 統合完了 | E2Eテストが成功 |

---

## 7. 観測指標

### 7.1 オーケストレーション品質の測定

| 指標 | 計算方法 | 正常範囲 |
|------|---------|---------|
| ルーティング正解率 | success / total_routing | 0.7-0.9 |
| やり直し率 | redo_requested / total_tasks | 0.05-0.15 |
| 平均タスク完了時間 | avg(completed_at - started_at) | タスク種別による |
| エージェント利用分布 | 各エージェントの使用比率 | 均等 ± 20% |
| 中間睡眠回数 | sleep_count / total_sessions | 0.1-0.5 |

### 7.2 パラメータ調整の方向

| 観測結果 | 調整パラメータ | 方向 |
|---------|---------------|------|
| ルーティング正解率が低い | capability_match 重み | 上げる |
| 特定エージェントに偏る | recent_activity 重み | 上げる |
| やり直し率が高い | routing_similarity_threshold | 下げる |
| タスクが大きすぎる | TASK_SIZE_MARGIN | 下げる |
| 概要生成が多すぎる | input_size_threshold | 上げる |

### 7.3 ログ設計

```python
# オーケストレーターのログ形式
LOG_FORMAT = {
    "timestamp": "ISO8601",
    "level": "INFO/WARN/ERROR",
    "component": "orchestrator/router/evaluator/input_processor",
    "session_id": "UUID",
    "event": "routing_decision/task_delegated/feedback_received/sleep_triggered",
    "details": {
        "selected_agent": "agent_id",
        "score": 0.85,
        "reason": "...",
    },
}
```

---

## 8. Phase 1 モジュールとの連携

### 8.1 再利用するコンポーネント

| Phase 1 コンポーネント | Phase 2 での使用方法 |
|----------------------|---------------------|
| TaskExecutor | オーケストレーターの外部メモリ検索・学び記録 |
| MemoryRepository | 全エージェントのメモリ CRUD |
| StrengthManager | 全エージェントの強度管理 |
| VectorSearch | メモリ検索（ルーティング判断の補助） |
| MemoryRanker | 検索結果のランキング |
| SleepPhaseProcessor | 睡眠フェーズ処理 |
| Phase1Config | 基本設定パラメータ |
| DatabaseConnection | DB接続 |

### 8.2 拡張ポイント

```python
# オーケストレーターは TaskExecutor を内部で使用
orchestrator = Orchestrator(
    agent_id="orchestrator_01",
    router=router,
    evaluator=evaluator,
    progress_manager=progress_manager,
    task_executor=task_executor,  # Phase 1 の TaskExecutor
    config=phase2_config,
)

# オーケストレーター自身も外部メモリを持つ（Phase 1 と同じ仕組み）
# → search_memories(), record_learning(), run_sleep_phase() を使用
```

---

## 9. テスト戦略

### 9.1 単体テスト

| コンポーネント | テスト内容 |
|---------------|-----------|
| ItemDetector | 論点数検出の精度 |
| Router | スコア計算の正確性 |
| Evaluator | フィードバック検出の精度 |
| ProgressManager | 状態保存・復帰の正確性 |

### 9.2 統合テスト

| シナリオ | 検証内容 |
|---------|---------|
| 単純なタスク | ルーティング → 実行 → 評価 の基本フロー |
| 大きな入力 | 入力処理層 → オーケストレーター の連携 |
| やり直し要求 | フィードバック → 再ルーティング のフロー |
| 中間睡眠 | 状態保存 → 復帰 → 継続 のフロー |
| 複数タスク | タスク分割 → 順次実行 → 統合 のフロー |

### 9.3 E2Eテストシナリオ

```python
# E2E テスト例: 調査 → 実装 → テスト の連携
async def test_research_implement_test_flow():
    """調査・実装・テストの連携フローをテスト"""

    # 1. 入力
    user_input = "ユーザー認証機能を追加してください"

    # 2. 入力処理
    processed = input_processor.process(user_input)

    # 3. オーケストレーター実行
    result = await orchestrator.process_request(processed)

    # 4. 検証
    assert result.routing_decision.selected_agent_id in [
        "implementation_agent",
        "research_agent",
    ]

    # 5. フィードバック
    feedback = orchestrator.receive_feedback(
        session_id=result.session_id,
        user_response="ありがとう、良さそうです",
    )
    assert feedback.feedback_type == "positive"
```

---

## 10. 次のステップ

### 10.1 Phase 3 への移行条件

- Phase 2 の核心機能（入力処理層、オーケストレーター、ルーティング）が安定動作
- ルーティング正解率が 70% 以上
- やり直し率が 15% 以下
- 中間睡眠からの復帰が正常動作

### 10.2 Phase 3 で検討する機能

- 学習可能なルーティングモデル（ニューラルネット）
- 高度な負荷分散（キュー方式 → 動的スケジューリング）
- 複数オーケストレーターの協調
- リアルタイム進捗監視（WebSocket）
- A/Bテストによるパラメータ最適化

---

*本ドキュメントは [architecture.ja.md](./architecture.ja.md) および [phase1-implementation-spec.ja.md](./phase1-implementation-spec.ja.md) に基づいて作成された Phase 2 MVP の実装仕様書である。*

*作成日: 2026年1月14日*
