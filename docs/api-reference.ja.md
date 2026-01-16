# API リファレンス

LLM永続メモリシステム Phase 1 MVP の API リファレンスです。

## 目次

1. [TaskExecutor（メインAPI）](#1-taskexecutorメインapi)
2. [MemoryRepository（CRUD）](#2-memoryrepositorycrud)
3. [StrengthManager（2段階強化）](#3-strengthmanager2段階強化)
4. [VectorSearch（検索）](#4-vectorsearch検索)
5. [MemoryRanker（ランキング）](#5-memoryrankerランキング)
6. [SleepPhaseProcessor（睡眠フェーズ）](#6-sleepphaseprocessor睡眠フェーズ)
7. [AzureEmbeddingClient（Embedding）](#7-azureembeddingclientembedding)
8. [Phase1Config（設定）](#8-phase1config設定)
9. [AgentMemory（データモデル）](#9-agentmemoryデータモデル)
10. [DatabaseConnection（DB接続）](#10-databaseconnectiondb接続)
11. [パラメータ一覧](#11-パラメータ一覧)

---

## 1. TaskExecutor（メインAPI）

**ファイル**: `src/core/task_executor.py`

メモリ検索・2段階強化・学び記録のフローを統合管理するメインAPIクラス。
推奨されるエントリーポイントです。

### コンストラクタ

```python
TaskExecutor(
    vector_search: VectorSearch,
    ranker: MemoryRanker,
    strength_manager: StrengthManager,
    sleep_processor: SleepPhaseProcessor,
    repository: MemoryRepository,
    config: Optional[Phase1Config] = None,
)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `vector_search` | `VectorSearch` | ベクトル検索エンジン |
| `ranker` | `MemoryRanker` | スコア合成・ランキング |
| `strength_manager` | `StrengthManager` | 2段階強化管理 |
| `sleep_processor` | `SleepPhaseProcessor` | 睡眠フェーズ処理 |
| `repository` | `MemoryRepository` | CRUD操作 |
| `config` | `Optional[Phase1Config]` | 設定（省略時はデフォルト） |

### search_memories()

メモリ検索を実行し、候補強化（candidate_count++）を行う。

```python
def search_memories(
    self,
    query: str,
    agent_id: str,
    perspective: Optional[str] = None,
) -> List[ScoredMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `query` | `str` | 検索クエリ（テキスト） |
| `agent_id` | `str` | 検索対象のエージェントID |
| `perspective` | `Optional[str]` | 観点（指定時は観点別強度を考慮） |

**戻り値**: `List[ScoredMemory]` - スコア降順でソートされた検索結果

**例**:
```python
# メモリ検索
memories = executor.search_memories(
    query="緊急調達のコスト",
    agent_id="agent_01",
    perspective="コスト"
)
for mem in memories:
    print(f"Score: {mem.final_score:.3f}, Content: {mem.memory.content[:50]}")
```

### record_learning()

新しい学びをメモリとして保存する。

```python
def record_learning(
    self,
    agent_id: str,
    content: str,
    learning: str,
    perspective: Optional[str] = None,
) -> UUID
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | エージェントID |
| `content` | `str` | 学びの内容（メモリのメインコンテンツ） |
| `learning` | `str` | 観点別の学び内容（文字列） |
| `perspective` | `Optional[str]` | 観点名（未指定時は "general"） |

**戻り値**: `UUID` - 作成されたメモリのID

**例**:
```python
memory_id = executor.record_learning(
    agent_id="agent_01",
    content="緊急調達では1.5倍のコストを見込む必要がある",
    learning="納期短縮とコストのトレードオフを学んだ",
    perspective="コスト"
)
```

### execute_task()

メモリ検索→タスク実行→使用判定→使用強化→学び記録の統合フローを実行する。

```python
def execute_task(
    self,
    query: str,
    agent_id: str,
    task_func: Callable[[List[ScoredMemory]], Any],
    perspective: Optional[str] = None,
    extract_learning: bool = False,
    learning_content: Optional[str] = None,
    learning_text: Optional[str] = None,
) -> TaskExecutionResult
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `query` | `str` | 検索クエリ |
| `agent_id` | `str` | エージェントID |
| `task_func` | `Callable[[List[ScoredMemory]], Any]` | 実行するタスク関数 |
| `perspective` | `Optional[str]` | 観点（検索・強化に影響） |
| `extract_learning` | `bool` | タスク結果から学びを自動抽出するか |
| `learning_content` | `Optional[str]` | 記録する学びの content |
| `learning_text` | `Optional[str]` | 記録する学びの learning テキスト |

**戻り値**: `TaskExecutionResult` - タスク実行結果

**例**:
```python
def my_task(memories: List[ScoredMemory]) -> str:
    # メモリを参照してタスクを実行
    context = "\n".join([m.memory.content for m in memories])
    return f"タスク完了。参照メモリ: {len(memories)}件"

result = executor.execute_task(
    query="緊急調達のコスト",
    agent_id="agent_01",
    task_func=my_task,
    perspective="コスト",
    learning_content="緊急調達は1.5倍のコストがかかる",
    learning_text="納期短縮のトレードオフを学んだ",
)
print(f"タスク結果: {result.task_result}")
print(f"使用メモリ: {len(result.used_memory_ids)}件")
```

### run_sleep_phase()

睡眠フェーズを実行する（減衰・アーカイブ・統合処理）。

```python
def run_sleep_phase(self, agent_id: str) -> SleepPhaseResult
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | 処理対象のエージェントID |

**戻り値**: `SleepPhaseResult` - 処理結果

**例**:
```python
result = executor.run_sleep_phase("agent_01")
print(f"減衰: {result.decayed_count}件")
print(f"アーカイブ: {result.archived_count}件")
```

### reinforce_used_memories()

使用されたメモリを強化する（2段階強化の Stage 2）。

```python
def reinforce_used_memories(
    self,
    memory_ids: List[UUID],
    agent_id: str,
    perspective: Optional[str] = None,
) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_ids` | `List[UUID]` | 使用されたメモリのIDリスト |
| `agent_id` | `str` | エージェントID（ログ用） |
| `perspective` | `Optional[str]` | 観点（指定時は観点別強度も強化） |

**戻り値**: `int` - 強化されたメモリ数

### identify_used_memories()

タスク結果からメモリ使用を判定する（keyword方式）。

```python
def identify_used_memories(
    self,
    task_result: Any,
    candidates: List[ScoredMemory],
) -> List[UUID]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `task_result` | `Any` | タスク関数の実行結果 |
| `candidates` | `List[ScoredMemory]` | 検索候補のメモリリスト |

**戻り値**: `List[UUID]` - 使用されたと判定されたメモリのIDリスト

---

## 2. MemoryRepository（CRUD）

**ファイル**: `src/core/memory_repository.py`

PostgreSQL + pgvector に対する CRUD 操作を提供する低レベルAPI。

### コンストラクタ

```python
MemoryRepository(db: DatabaseConnection, config: Phase1Config)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `db` | `DatabaseConnection` | データベース接続 |
| `config` | `Phase1Config` | 設定パラメータ |

### create()

新規メモリを作成する。

```python
def create(self, memory: AgentMemory) -> AgentMemory
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory` | `AgentMemory` | 作成するメモリインスタンス |

**戻り値**: `AgentMemory` - 作成されたメモリ（DBから返却された値で更新済み）

**例**:
```python
memory = AgentMemory.create(
    agent_id="agent_01",
    content="緊急調達では15%のコスト増を見込む",
    tags=["コスト", "緊急調達"],
    source="task",
)
created = repository.create(memory)
print(f"Created: {created.id}")
```

### get_by_id()

IDでメモリを取得する。

```python
def get_by_id(self, memory_id: UUID) -> Optional[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | メモリのUUID |

**戻り値**: `Optional[AgentMemory]` - メモリ、見つからない場合は `None`

### get_by_agent_id()

エージェントIDでメモリを取得する。

```python
def get_by_agent_id(
    self, agent_id: str, status: str = "active"
) -> List[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | エージェントID |
| `status` | `str` | フィルタするステータス（デフォルト: "active"） |

**戻り値**: `List[AgentMemory]` - メモリのリスト（created_at 降順）

### update()

メモリを更新する。

```python
def update(self, memory: AgentMemory) -> AgentMemory
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory` | `AgentMemory` | 更新するメモリインスタンス |

**戻り値**: `AgentMemory` - 更新されたメモリ

**例外**: `ValueError` - 指定したIDのメモリが存在しない場合

### archive()

メモリをアーカイブ（論理削除）する。

```python
def archive(self, memory_id: UUID) -> bool
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | アーカイブするメモリのUUID |

**戻り値**: `bool` - `True`: アーカイブ成功、`False`: 対象のメモリが存在しない

### increment_candidate_count()

検索候補になった回数をインクリメントする（2段階強化の第1段階）。

```python
def increment_candidate_count(self, memory_id: UUID) -> None
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 対象メモリのUUID |

### increment_access_count()

使用回数をインクリメントし、強度を更新する（2段階強化の第2段階）。

```python
def increment_access_count(
    self, memory_id: UUID, strength_increment: float
) -> None
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 対象メモリのUUID |
| `strength_increment` | `float` | 強度の増分値 |

### update_perspective_strength()

観点別強度を更新する。

```python
def update_perspective_strength(
    self, memory_id: UUID, perspective: str, increment: float
) -> None
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 対象メモリのUUID |
| `perspective` | `str` | 観点名（例: "コスト", "納期"） |
| `increment` | `float` | 強度の増分値 |

### batch_increment_candidate_count()

複数メモリの candidate_count を一括インクリメントする。

```python
def batch_increment_candidate_count(self, memory_ids: List[UUID]) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_ids` | `List[UUID]` | 対象メモリのUUIDリスト |

**戻り値**: `int` - 更新された行数

### batch_update_strength()

複数メモリの strength を一括更新する。

```python
def batch_update_strength(
    self, updates: List[tuple[UUID, float]]
) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `updates` | `List[tuple[UUID, float]]` | (memory_id, new_strength) のタプルリスト |

**戻り値**: `int` - 更新された行数

### batch_archive()

複数メモリを一括アーカイブする。

```python
def batch_archive(self, memory_ids: List[UUID]) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_ids` | `List[UUID]` | アーカイブするメモリのUUIDリスト |

**戻り値**: `int` - アーカイブされた行数

### get_memories_for_decay()

減衰処理対象のメモリを取得する。

```python
def get_memories_for_decay(
    self, agent_id: str, batch_size: int = 100
) -> List[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | エージェントID |
| `batch_size` | `int` | 一度に取得する件数（デフォルト: 100） |

**戻り値**: `List[AgentMemory]` - メモリのリスト（strength 昇順）

### count_active_memories()

アクティブなメモリの件数を取得する。

```python
def count_active_memories(self, agent_id: str) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | エージェントID |

**戻り値**: `int` - アクティブなメモリの件数

### get_lowest_strength_memories()

最も強度の低いメモリを取得する。

```python
def get_lowest_strength_memories(
    self, agent_id: str, limit: int
) -> List[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | エージェントID |
| `limit` | `int` | 取得件数 |

**戻り値**: `List[AgentMemory]` - メモリのリスト（strength 昇順）

---

## 3. StrengthManager（2段階強化）

**ファイル**: `src/core/strength_manager.py`

2段階強化メカニズムとインパクトベースの強度管理を提供。

### コンストラクタ

```python
StrengthManager(repository: MemoryRepository, config: Phase1Config)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `repository` | `MemoryRepository` | リポジトリインスタンス |
| `config` | `Phase1Config` | 設定パラメータ |

### mark_as_candidate()

検索候補になったメモリをマークする（candidate_count++）。

```python
def mark_as_candidate(self, memory_ids: List[UUID]) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_ids` | `List[UUID]` | 検索候補になったメモリのUUIDリスト |

**戻り値**: `int` - 更新された行数

**例**:
```python
# 検索候補としてマーク（strength は変更しない）
updated = strength_manager.mark_as_candidate([memory_id1, memory_id2])
```

### mark_as_used()

実際に使用されたメモリを強化する。

```python
def mark_as_used(
    self, memory_id: UUID, perspective: Optional[str] = None
) -> Optional[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 使用されたメモリのUUID |
| `perspective` | `Optional[str]` | 強化する観点名（省略可） |

**戻り値**: `Optional[AgentMemory]` - 更新後のメモリ、存在しない場合は `None`

**処理内容**:
1. `access_count += 1`
2. `strength += 0.1` (strength_increment_on_use)
3. 観点指定時: `strength_by_perspective[perspective] += 0.15`
4. `last_accessed_at = now`
5. `consolidation_level` を更新

**例**:
```python
# 実際に使用されたことをマーク
updated = strength_manager.mark_as_used(memory_id, perspective="コスト")
print(f"New strength: {updated.strength}")
```

### apply_impact()

インパクトスコアを加算し、強度に反映する。

```python
def apply_impact(
    self, memory_id: UUID, impact_type: str
) -> Optional[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 対象メモリのUUID |
| `impact_type` | `str` | インパクトの種類 |

**impact_type の値**:
| 値 | 説明 | インパクト値 |
|----|------|-------------|
| `"user_positive"` | ユーザーから肯定的フィードバック | +2.0 |
| `"task_success"` | タスク成功に貢献 | +1.5 |
| `"prevented_error"` | エラー防止に貢献 | +2.0 |

**戻り値**: `Optional[AgentMemory]` - 更新後のメモリ

**例外**: `ValueError` - 不正な impact_type が指定された場合

**例**:
```python
# タスク成功に貢献したメモリを強化
updated = strength_manager.apply_impact(memory_id, "task_success")
print(f"Impact score: {updated.impact_score}")
```

### update_consolidation_level()

access_countに基づいて定着レベルを更新する。

```python
def update_consolidation_level(self, memory_id: UUID) -> Optional[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 対象メモリのUUID |

**戻り値**: `Optional[AgentMemory]` - 更新後のメモリ

**定着レベル閾値**（access_count）:
| レベル | 閾値 | 日次減衰率 |
|--------|------|-----------|
| Level 0 | 0回以上 | 5%/日 |
| Level 1 | 5回以上 | 3%/日 |
| Level 2 | 15回以上 | 2%/日 |
| Level 3 | 30回以上 | 1%/日 |
| Level 4 | 60回以上 | 0.5%/日 |
| Level 5 | 100回以上 | 0.2%/日 |

### reactivate()

アーカイブされたメモリを再活性化する。

```python
def reactivate(self, memory_id: UUID) -> Optional[AgentMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `memory_id` | `UUID` | 再活性化するメモリのUUID |

**戻り値**: `Optional[AgentMemory]` - 再活性化されたメモリ

**例外**: `ValueError` - メモリが既にアクティブな場合

**処理内容**:
1. `status = 'active'`
2. `strength = 0.5` (reactivation_strength)

### get_impact_value()

インパクトタイプに対応する値を取得する。

```python
def get_impact_value(self, impact_type: str) -> float
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `impact_type` | `str` | インパクトの種類 |

**戻り値**: `float` - インパクト値

---

## 4. VectorSearch（検索）

**ファイル**: `src/search/vector_search.py`

pgvector の cosine 距離を使用したベクトル検索エンジン（Stage 1: 関連性フィルタ）。

### コンストラクタ

```python
VectorSearch(
    db: DatabaseConnection,
    embedding_client: AzureEmbeddingClient,
    config: Optional[Phase1Config] = None,
)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `db` | `DatabaseConnection` | データベース接続 |
| `embedding_client` | `AzureEmbeddingClient` | Embeddingクライアント |
| `config` | `Optional[Phase1Config]` | 設定（省略時はデフォルト） |

### search_candidates()

ベクトル検索で候補を取得する。

```python
def search_candidates(
    self,
    query: str,
    agent_id: str,
    perspective: Optional[str] = None,
) -> List[Tuple[AgentMemory, float]]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `query` | `str` | 検索クエリ（テキスト） |
| `agent_id` | `str` | 検索対象のエージェントID |
| `perspective` | `Optional[str]` | 観点（将来の拡張用） |

**戻り値**: `List[Tuple[AgentMemory, float]]` - (メモリ, 類似度) のリスト。類似度の高い順にソート。

**例外**: `VectorSearchError` - エンベディング取得またはDB検索に失敗した場合

**例**:
```python
candidates = vector_search.search_candidates(
    query="緊急調達のコスト",
    agent_id="agent_01"
)
for memory, similarity in candidates:
    print(f"Similarity: {similarity:.3f}, Content: {memory.content[:50]}")
```

### search_by_embedding()

事前計算されたエンベディングでベクトル検索を行う。

```python
def search_by_embedding(
    self,
    query_embedding: List[float],
    agent_id: str,
    similarity_threshold: Optional[float] = None,
    candidate_limit: Optional[int] = None,
) -> List[Tuple[AgentMemory, float]]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `query_embedding` | `List[float]` | クエリのエンベディングベクトル |
| `agent_id` | `str` | 検索対象のエージェントID |
| `similarity_threshold` | `Optional[float]` | 類似度閾値（省略時は設定値） |
| `candidate_limit` | `Optional[int]` | 最大候補数（省略時は設定値） |

**戻り値**: `List[Tuple[AgentMemory, float]]` - (メモリ, 類似度) のリスト

### count_candidates()

類似度閾値以上の候補数をカウントする。

```python
def count_candidates(
    self,
    query: str,
    agent_id: str,
) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `query` | `str` | 検索クエリ |
| `agent_id` | `str` | 検索対象のエージェントID |

**戻り値**: `int` - 類似度閾値以上の候補数

---

## 5. MemoryRanker（ランキング）

**ファイル**: `src/search/ranking.py`

Stage 1 のベクトル検索で取得した候補に対して、複合スコアを計算してランキングを行う（Stage 2: 優先度ランキング）。

### ScoredMemory（データクラス）

スコア計算済みのメモリを表すデータクラス。

```python
@dataclass
class ScoredMemory:
    memory: AgentMemory          # 元の記憶データ
    similarity: float            # Stage 1 からの類似度（0-1）
    final_score: float           # 合成スコア（weighted sum）
    score_breakdown: Dict[str, float]  # スコア内訳（デバッグ用）
```

**score_breakdown の内容**:
```python
{
    "similarity_raw": 0.75,
    "similarity_weighted": 0.375,
    "strength_raw": 1.2,
    "strength_normalized": 0.6,
    "strength_weighted": 0.18,
    "recency_raw": 0.9,
    "recency_weighted": 0.18,
    "total": 0.735
}
```

### コンストラクタ

```python
MemoryRanker(config: Optional[Phase1Config] = None)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `config` | `Optional[Phase1Config]` | 設定（省略時はデフォルト） |

### rank()

候補をスコア合成してランキングする。

```python
def rank(
    self,
    candidates: List[Tuple[AgentMemory, float]],
    perspective: Optional[str] = None,
) -> List[ScoredMemory]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `candidates` | `List[Tuple[AgentMemory, float]]` | Stage 1 からの候補リスト |
| `perspective` | `Optional[str]` | 観点（指定時は観点別強度を使用） |

**戻り値**: `List[ScoredMemory]` - スコア降順でソートされた結果（top_k_results 件まで）

**スコア計算式**:
```
final_score = similarity × 0.50 + normalized_strength × 0.30 + recency_score × 0.20
```

**例**:
```python
ranker = MemoryRanker()
candidates = vector_search.search_candidates(query, agent_id)
ranked = ranker.rank(candidates, perspective="コスト")
for scored in ranked:
    print(f"Score: {scored.final_score:.3f}, Content: {scored.memory.content[:50]}")
```

---

## 6. SleepPhaseProcessor（睡眠フェーズ）

**ファイル**: `src/core/sleep_processor.py`

タスク完了時に実行される睡眠フェーズの処理を管理。強度減衰、閾値ベースのアーカイブ、類似メモリの統合レベル管理を行う。

### SleepPhaseResult（データクラス）

睡眠フェーズの処理結果を表すデータクラス。

```python
@dataclass
class SleepPhaseResult:
    agent_id: str                # 処理対象のエージェントID
    decayed_count: int           # 減衰処理されたメモリ数
    archived_count: int          # アーカイブされたメモリ数
    consolidated_count: int      # 統合レベルが更新されたメモリ数
    processed_at: datetime       # 処理実行日時
    errors: List[str]            # 処理中に発生したエラーのリスト
```

### コンストラクタ

```python
SleepPhaseProcessor(
    db: DatabaseConnection,
    config: Optional[Phase1Config] = None,
)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `db` | `DatabaseConnection` | データベース接続 |
| `config` | `Optional[Phase1Config]` | 設定（省略時はデフォルト） |

### process_all()

睡眠フェーズのメイン処理を実行する。

```python
def process_all(self, agent_id: str) -> SleepPhaseResult
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | 処理対象のエージェントID |

**戻り値**: `SleepPhaseResult` - 処理結果

**処理順序**:
1. 減衰対象メモリの取得（status="active"）
2. 各メモリに減衰を適用
3. 閾値以下のメモリをアーカイブ
4. 類似メモリのグループ化（Phase 1 ではスキップ）
5. 統合レベルの更新
6. 処理ログの記録

**例**:
```python
processor = SleepPhaseProcessor(db)
result = processor.process_all("agent_01")
print(f"減衰: {result.decayed_count}件")
print(f"アーカイブ: {result.archived_count}件")
```

### apply_decay_all()

全アクティブメモリに減衰を適用する。

```python
def apply_decay_all(
    self,
    agent_id: str,
    batch_size: int = 100,
) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | 処理対象のエージェントID |
| `batch_size` | `int` | 一度に処理するメモリ数（デフォルト: 100） |

**戻り値**: `int` - 減衰処理されたメモリ数

### archive_weak_memories()

閾値以下のメモリをアーカイブする。

```python
def archive_weak_memories(self, agent_id: str) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | 処理対象のエージェントID |

**戻り値**: `int` - アーカイブされたメモリ数

### consolidate_similar()

類似メモリの統合レベルを更新する（Phase 1 ではスキップ）。

```python
def consolidate_similar(self, agent_id: str) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `agent_id` | `str` | 処理対象のエージェントID |

**戻り値**: `int` - 統合レベルが更新されたメモリ数（Phase 1 では常に 0）

---

## 7. AzureEmbeddingClient（Embedding）

**ファイル**: `src/embedding/azure_client.py`

Azure OpenAI の text-embedding-3-small モデルを使用してテキストをベクトル化するクライアント。

### コンストラクタ

```python
AzureEmbeddingClient(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    deployment: Optional[str] = None,
)
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `endpoint` | `Optional[str]` | Azure OpenAI エンドポイント（省略時は環境変数から取得） |
| `api_key` | `Optional[str]` | Azure OpenAI APIキー（省略時は環境変数から取得） |
| `deployment` | `Optional[str]` | デプロイメント名（省略時は環境変数または設定から取得） |

**環境変数**:
| 変数名 | 説明 |
|--------|------|
| `AZURE_OPENAI_ENDPOINT` または `OpenAIEmbeddingURI` | エンドポイント |
| `AZURE_OPENAI_API_KEY` または `OpenAIEmbeddingKey` | APIキー |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | デプロイメント名 |

**例外**: `AzureEmbeddingError` - エンドポイントまたはAPIキーが設定されていない場合

### get_embedding()

単一テキストのエンベディングを取得する。

```python
def get_embedding(self, text: str) -> List[float]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `text` | `str` | エンベディングを取得するテキスト |

**戻り値**: `List[float]` - 1536次元のベクトル

**例外**: `AzureEmbeddingError` - API呼び出しに失敗した場合、または空のテキストが渡された場合

**例**:
```python
client = AzureEmbeddingClient()
embedding = client.get_embedding("緊急調達のコスト管理")
print(f"Dimension: {len(embedding)}")  # 1536
```

### get_embeddings()

複数テキストのエンベディングをバッチ取得する。

```python
def get_embeddings(self, texts: List[str]) -> List[List[float]]
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `texts` | `List[str]` | エンベディングを取得するテキストのリスト |

**戻り値**: `List[List[float]]` - 各テキストに対応する1536次元ベクトルのリスト

**例外**: `AzureEmbeddingError` - API呼び出しに失敗した場合、または空のリストが渡された場合

### is_available()

クライアントが利用可能かテストする。

```python
def is_available(self) -> bool
```

**戻り値**: `bool` - API呼び出しが成功した場合 `True`

### グローバル関数

```python
# グローバルなエンベディングクライアントを取得（シングルトン）
def get_embedding_client() -> AzureEmbeddingClient

# グローバルクライアントをリセット（テスト用）
def reset_client() -> None
```

---

## 8. Phase1Config（設定）

**ファイル**: `src/config/phase1_config.py`

Phase 1 MVP のパラメータ設定を管理するデータクラス。

### 属性一覧

**強度管理**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `initial_strength` | `float` | 1.0 | 新規記憶の初期強度 |
| `initial_strength_education` | `float` | 0.5 | 教育プロセスで読んだだけの記憶の初期強度 |
| `strength_increment_on_use` | `float` | 0.1 | 使用時の強化量 |
| `perspective_strength_increment` | `float` | 0.15 | 観点別強度の強化量 |
| `archive_threshold` | `float` | 0.1 | これ以下でアーカイブ |
| `reactivation_strength` | `float` | 0.5 | 再活性化時の初期強度 |

**減衰**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `expected_tasks_per_day` | `int` | 10 | 想定タスク数/日 |
| `consolidation_thresholds` | `List[int]` | [0, 5, 15, 30, 60, 100] | 定着レベル閾値 |
| `daily_decay_targets` | `Dict[int, float]` | {0: 0.95, ...} | 日次減衰目標 |

**検索**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `similarity_threshold` | `float` | 0.3 | 類似度の最低閾値 |
| `candidate_limit` | `int` | 50 | Stage 1の最大候補数 |
| `top_k_results` | `int` | 10 | コンテキストに渡す件数 |
| `score_weights` | `Dict[str, float]` | {"similarity": 0.50, ...} | スコア重み |

**インパクト**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `impact_user_positive` | `float` | 2.0 | ユーザー肯定時の加算量 |
| `impact_task_success` | `float` | 1.5 | タスク成功時の加算量 |
| `impact_prevented_error` | `float` | 2.0 | エラー防止時の加算量 |
| `impact_to_strength_ratio` | `float` | 0.2 | 強度へのインパクト反映率 |

**容量**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `max_active_memories` | `int` | 5000 | アクティブ記憶の最大件数 |

**エンベディング**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `embedding_model` | `str` | "text-embedding-3-small" | Embeddingモデル名 |
| `embedding_dimension` | `int` | 1536 | エンベディング次元数 |

**スコープ**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `current_project_id` | `str` | "llm-persistent-memory-phase1" | 現在のプロジェクトID |
| `related_domains` | `List[str]` | [...] | 関連ドメイン |
| `default_scope_level` | `str` | "project" | デフォルトスコープ |

### get_decay_rate()

定着レベルに応じたタスク単位の減衰率を取得する。

```python
def get_decay_rate(self, consolidation_level: int) -> float
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `consolidation_level` | `int` | 定着レベル (0-5) |

**戻り値**: `float` - タスク単位の減衰率

**計算式**: `decay_rate = daily_target ** (1 / expected_tasks_per_day)`

### get_consolidation_level()

access_countから定着レベルを計算する。

```python
def get_consolidation_level(self, access_count: int) -> int
```

**引数**:
| 引数 | 型 | 説明 |
|------|-----|------|
| `access_count` | `int` | 実際に使用された回数 |

**戻り値**: `int` - 定着レベル (0-5)

### グローバルインスタンス

```python
# デフォルト設定のインスタンス
config = Phase1Config()
```

---

## 9. AgentMemory（データモデル）

**ファイル**: `src/models/memory.py`

agent_memory テーブルに対応するデータクラス。

### 属性一覧

**識別子**:
| 属性 | 型 | 説明 |
|------|-----|------|
| `id` | `UUID` | 記憶の一意識別子 |
| `agent_id` | `str` | 所属エージェントのID |

**コンテンツ**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `content` | `str` | - | 記憶の内容（テキスト） |
| `embedding` | `Optional[List[float]]` | None | ベクトル表現 (1536次元) |
| `tags` | `List[str]` | [] | 分類タグ |

**スコープ**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `scope_level` | `str` | "project" | スコープレベル |
| `scope_domain` | `Optional[str]` | None | ドメイン名 |
| `scope_project` | `Optional[str]` | None | プロジェクトID |

**強度管理**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `strength` | `float` | 1.0 | 全体的な強度 |
| `strength_by_perspective` | `Dict[str, float]` | {} | 観点別の強度 |
| `access_count` | `int` | 0 | 実際に使用された回数 |
| `candidate_count` | `int` | 0 | 検索候補として参照された回数 |
| `last_accessed_at` | `Optional[datetime]` | None | 最後に使用された日時 |
| `impact_score` | `float` | 0.0 | インパクトスコア |
| `consolidation_level` | `int` | 0 | 定着レベル (0-5) |

**学び**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `learnings` | `Dict[str, str]` | {} | 観点別の学び内容 |

**状態**:
| 属性 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `status` | `str` | "active" | 状態 (active/archived) |
| `source` | `Optional[str]` | None | ソース (education/task/manual) |

**タイムスタンプ**:
| 属性 | 型 | 説明 |
|------|-----|------|
| `created_at` | `datetime` | 作成日時 |
| `updated_at` | `datetime` | 最終更新日時 |
| `last_decay_at` | `Optional[datetime]` | 最後の減衰処理日時 |

### AgentMemory.create()

デフォルト値を設定してインスタンスを生成するファクトリメソッド。

```python
@classmethod
def create(
    cls,
    agent_id: str,
    content: str,
    *,
    embedding: Optional[List[float]] = None,
    tags: Optional[List[str]] = None,
    scope_level: str = "project",
    scope_domain: Optional[str] = None,
    scope_project: Optional[str] = None,
    strength: float = 1.0,
    strength_by_perspective: Optional[Dict[str, float]] = None,
    learnings: Optional[Dict[str, str]] = None,
    source: Optional[str] = None,
) -> AgentMemory
```

**例**:
```python
memory = AgentMemory.create(
    agent_id="agent_01",
    content="緊急調達では15%のコスト増を見込む",
    tags=["コスト", "緊急調達"],
    scope_level="domain",
    scope_domain="procurement",
    source="task",
)
```

### AgentMemory.create_from_education()

教育プロセスからの記憶を生成するファクトリメソッド（初期強度: 0.5）。

```python
@classmethod
def create_from_education(
    cls,
    agent_id: str,
    content: str,
    **kwargs
) -> AgentMemory
```

### AgentMemory.from_row()

DBレコードからインスタンスを生成する。

```python
@classmethod
def from_row(cls, row: tuple | dict) -> AgentMemory
```

### copy_with()

指定したフィールドを変更した新しいインスタンスを生成する。

```python
def copy_with(self, **kwargs: Any) -> AgentMemory
```

**例**:
```python
updated = memory.copy_with(
    strength=memory.strength + 0.1,
    access_count=memory.access_count + 1,
    updated_at=datetime.now(),
)
```

### to_dict()

辞書形式に変換する。

```python
def to_dict(self) -> Dict[str, Any]
```

---

## 10. DatabaseConnection（DB接続）

**ファイル**: `src/db/connection.py`

PostgreSQLデータベース接続管理クラス。コネクションプールとコンテキストマネージャーによる安全な接続管理を提供。

### コンストラクタ

```python
DatabaseConnection(
    database_url: Optional[str] = None,
    min_connections: int = 1,
    max_connections: int = 10,
)
```

**引数**:
| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `database_url` | `Optional[str]` | None | PostgreSQL接続文字列（環境変数 `DATABASE_URL` から取得） |
| `min_connections` | `int` | 1 | プール内の最小接続数 |
| `max_connections` | `int` | 10 | プール内の最大接続数 |

**環境変数**:
- `DATABASE_URL`: PostgreSQL接続文字列（例: `postgresql://agent:password@localhost:5432/agent_memory`）

### get_connection()

データベース接続をコンテキストマネージャーとして取得する。

```python
@contextmanager
def get_connection(
    self, auto_commit: bool = True
) -> Generator[PsycopgConnection, None, None]
```

**引数**:
| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `auto_commit` | `bool` | True | コンテキスト終了時に自動commit |

**例**:
```python
with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM agent_memory WHERE agent_id = %s", (agent_id,))
        rows = cur.fetchall()
```

### get_cursor()

カーソルを直接取得するコンテキストマネージャー。

```python
@contextmanager
def get_cursor(
    self, auto_commit: bool = True
) -> Generator[Any, None, None]
```

**例**:
```python
with db.get_cursor() as cur:
    cur.execute("SELECT * FROM agent_memory")
    rows = cur.fetchall()
```

### health_check()

データベース接続の健全性をチェックする。

```python
def health_check(self) -> bool
```

**戻り値**: `bool` - 接続が正常な場合 `True`

### close()

コネクションプールをクローズする。

```python
def close(self) -> None
```

### グローバル関数

```python
# デフォルトのデータベース接続を取得（シングルトン）
def get_database() -> DatabaseConnection

# デフォルトのデータベース接続をクローズ
def close_database() -> None
```

---

## 11. パラメータ一覧

### Phase1Config パラメータテーブル

| カテゴリ | パラメータ | デフォルト値 | 推奨範囲 | 調整指針 |
|---------|-----------|-------------|---------|---------|
| **強度管理** | `initial_strength` | 1.0 | 0.5-1.5 | 新規記憶の重要度に応じて調整 |
| | `initial_strength_education` | 0.5 | 0.3-0.8 | 教育内容の定着度に応じて調整 |
| | `strength_increment_on_use` | 0.1 | 0.05-0.2 | 使用頻度と強化速度のバランス |
| | `perspective_strength_increment` | 0.15 | 0.1-0.25 | 観点の専門性に応じて調整 |
| | `archive_threshold` | 0.1 | 0.05-0.2 | 記憶の保持期間に影響 |
| | `reactivation_strength` | 0.5 | 0.3-0.7 | 再活性化記憶の初期重要度 |
| **減衰** | `expected_tasks_per_day` | 10 | 5-50 | 実際のタスク頻度に合わせる |
| **検索** | `similarity_threshold` | 0.3 | 0.2-0.5 | 低いほど候補が増加 |
| | `candidate_limit` | 50 | 20-100 | Stage 2 の負荷とトレードオフ |
| | `top_k_results` | 10 | 5-20 | コンテキストサイズとのバランス |
| **スコア重み** | `similarity` | 0.50 | 0.3-0.7 | 意味的関連性の重視度 |
| | `strength` | 0.30 | 0.2-0.4 | 記憶の定着度の重視度 |
| | `recency` | 0.20 | 0.1-0.3 | 新鮮さの重視度 |
| **インパクト** | `impact_user_positive` | 2.0 | 1.0-3.0 | ユーザーフィードバックの影響度 |
| | `impact_task_success` | 1.5 | 1.0-2.0 | タスク成功の影響度 |
| | `impact_prevented_error` | 2.0 | 1.0-3.0 | エラー防止の影響度 |
| | `impact_to_strength_ratio` | 0.2 | 0.1-0.3 | インパクトの強度への反映率 |
| **容量** | `max_active_memories` | 5000 | 1000-10000 | メモリ使用量とのトレードオフ |
| **エンベディング** | `embedding_dimension` | 1536 | - | text-embedding-3-small の固定値 |

### 定着レベルと減衰率

| レベル | access_count 閾値 | 日次減衰目標 | タスク単位減衰率（10タスク/日） |
|--------|-------------------|-------------|-------------------------------|
| 0 | 0回以上 | 0.95 (5%/日) | 0.9949 (0.51%/タスク) |
| 1 | 5回以上 | 0.97 (3%/日) | 0.9969 (0.31%/タスク) |
| 2 | 15回以上 | 0.98 (2%/日) | 0.9980 (0.20%/タスク) |
| 3 | 30回以上 | 0.99 (1%/日) | 0.9990 (0.10%/タスク) |
| 4 | 60回以上 | 0.995 (0.5%/日) | 0.9995 (0.05%/タスク) |
| 5 | 100回以上 | 0.998 (0.2%/日) | 0.9998 (0.02%/タスク) |

### スコープレベル

| レベル | 説明 | 検索範囲 |
|--------|------|---------|
| `universal` | 普遍的な知識 | 全エージェント共有 |
| `domain` | ドメイン固有の知識 | 同一ドメイン内で共有 |
| `project` | プロジェクト固有の知識 | 同一プロジェクト内のみ |

---

## 例外クラス

### VectorSearchError

ベクトル検索のエラー。

```python
from src.search.vector_search import VectorSearchError
```

### AzureEmbeddingError

Azure Embeddingクライアントのエラー。

```python
from src.embedding.azure_client import AzureEmbeddingError
```

---

## データクラス

### TaskExecutionResult

タスク実行結果を格納するデータクラス。

```python
from src.core.task_executor import TaskExecutionResult

@dataclass
class TaskExecutionResult:
    task_result: Any                          # タスク関数の実行結果
    searched_memories: List[ScoredMemory]     # 検索でヒットしたメモリ
    used_memory_ids: List[UUID]               # 実際に使用されたメモリのID
    recorded_memory_id: Optional[UUID]        # 新たに記録された学びのメモリID
    executed_at: datetime                     # 実行日時
    errors: List[str]                         # 処理中に発生したエラー

    def to_dict(self) -> Dict                 # 辞書形式に変換
```

---

## 関連ドキュメント

- [アーキテクチャ](architecture.ja.md)
- [Phase 1 実装仕様](phase1-implementation-spec.ja.md)
- [使用方法ガイド](usage-guide.ja.md)
