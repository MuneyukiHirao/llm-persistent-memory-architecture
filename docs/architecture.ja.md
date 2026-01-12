# LLMエージェントの永続的メモリアーキテクチャ設計

## 概要

本ドキュメントは、LLMエージェントが人間の組織のように協調し、長期的な専門性と個性を持って動作するためのアーキテクチャ設計について記述する。

この設計は、以下の本質的な問いから出発している：

> 「人間の組織（例：6万人規模のグローバル企業）は、各人がコンテキストウィンドウを共有しないにもかかわらず協調して働いている。なぜLLMエージェントには同じことができないのか？」

---

## 第1章：問題の本質

### 1.1 現在のLLMの忘却問題

現在のTransformerアーキテクチャには構造的な限界がある。

**Lost in the Middle問題**
- コンテキスト内の関連情報の位置により性能が大幅に変動する
- 関連情報が最初か最後にある場合は性能が高く、中央にある場合は大幅に低下
- RoPE（Rotary Position Embedding）が長期減衰効果を導入し、中央のコンテンツを軽視する原因となっている
- 100万トークンのコンテキストウィンドウがあっても、実効的に使えるのは一部のみ

**サブエージェント分割の限界**
- 各サブエージェントも同じTransformerアーキテクチャの制約を持つ
- エージェント間で共有される情報は要約・圧縮されるため情報欠落が発生
- 長期的な文脈の一貫性維持は依然として課題

### 1.2 人間の組織が持つ「見えないインフラ」

人間の組織が協調できている理由を分析すると、LLMに欠けているものが見える。

#### 1.2.1 共有された暗黙知と文化

人間の組織では：
- 社員が組織文化や業界の常識を**内面化**している
- 「言わなくてもわかる」前提が共有されている

LLMエージェントは：
- 汎用的な事前学習知識はあるが、**組織固有の文脈がない**
- 各エージェントが独立して解釈するため、微妙にずれる

#### 1.2.2 永続的なアイデンティティと関係性

人間は：
- 同一人物として継続的に存在する
- 過去の成功・失敗を覚えている
- 「この人に聞けばわかる」という組織知がある

LLMエージェントは：
- **毎回生まれ変わる** - セッションごとにリセット
- 関係性の蓄積がない
- 誰が何を知っているかの**メタ知識がない**

#### 1.2.3 非同期だが永続的な「組織の記憶」

人間の組織には：
- メール、議事録、設計ドキュメント
- 人事異動しても引き継ぎがある

LLMマルチエージェントは：
- 共有メモリを作っても**何が重要かの判断基準がない**
- すべてを保存すると情報過多、要約すると情報欠落

#### 1.2.4 修復機構の有無

**人間の組織**
- 「あれ、おかしいな」と気づける（メタ認知）
- 「ちょっと確認させて」と聞き返せる
- 失敗したら学習して次に活かす

**LLMマルチエージェント**
- 矛盾に気づかずに進んでしまう
- 「わからない」と言えない（自信過剰）
- コンテキストが切れたら文字通り「忘れる」

### 1.3 人間の組織の「非効率」には機能がある

一見非効率に見えるものが、実は重要な機能を果たしている：

| 人間の非効率 | 機能 |
|-------------|------|
| 重複開発 | 冗長性によるレジリエンス |
| 情報が行き届かない | 各部門の自律性と創発 |
| 放置 | 本当に重要なことのフィルタリング |
| セクショナリズム | 専門性の深化 |

LLMエージェントで「効率的な組織」を作ろうとすると、この「非効率の機能」を失ってしまう。

### 1.4 LLMの「壊れ」と人間の「壊れ」の本質的な違い

表面的には似ている：

| 人間組織の「壊れ」 | LLMの「壊れ」 |
|------------------|--------------|
| 伝言ゲームで変質 | サマリで情報欠落 |
| 忘れる | コンパクションで削除 |
| 誤解 | 解釈のドリフト |

**本質的な違いは「気づき」**：

```
【人間】
情報が欠落 → 「あれ、なんか変だな」→ 確認 → 修復

【LLM】
情報が欠落 → 気づかない → そのまま進む → 歪んだ結果
```

人間の「壊れ」には**違和感センサー**がついている。LLMのコンパクションは**サイレント**で、何が消えたかを知らない。

---

## 第2章：設計原則

### 2.1 核心的な洞察

#### エージェントの永続的アイデンティティ

人間は常に何かを経験し続けている。その経験が独自のものであり、過去の経験が現在の判断に影響する。

エージェントに必要なのは：
- 起動されたときだけ存在するのではなく、**経験の連続体としての自己**
- 独自の経験ログを持ち、起動時にそれを読み込む
- タスク実行中も情報を取り込み、終了時に経験を追記
- 次回起動時に「同じ自分」として継続

**重要な発想の転換**：
```
永続的なのはエージェント本体ではなく、外部メモリ

エージェント本体：毎回生まれ変わる（ステートレス）
外部メモリ：永続的（ステートフル）
アイデンティティ：外部メモリが担う
```

#### メモの限界とパラメータ不変の壁

パラメータを書き換えない限り、本当の意味での「専門性の内面化」はできない。

```
【人間】
経験 → 神経回路の変化 → 「身についた」専門性
        （パラメータ更新）

【現状のLLM】
経験 → メモに記録 → 毎回読み出し
        （パラメータ不変）
```

しかし、パラメータ不変でも**外部メモリの蓄積と強度管理**により、実用的な「個性」を実現できる。

### 2.2 脳のメカニズムに学ぶ

#### 睡眠中の記憶整理：全体を見ていない

人間の脳も全体像を見て整理していない。局所的なルールの組み合わせで、結果として整合性が出てくる。

**2つの同時進行プロセス**：

1. **グローバルダウンスケーリング**
   - 全シナプスを一律に弱める（単純なルール）
   - 「最近強化されたものほど弱めにくい」という局所ルール

2. **選択的リプレイ**
   - 海馬が特定のパターンを繰り返し再生
   - 再生されたパターンは強化される

この2つが同時に走ることで、全体を見なくても：
- 重要なものは残る（リプレイされたから）
- 不要なものは消える（ダウンスケーリングで）

#### LTP（長期増強）の原理

脳では、シナプスが発火するとそのシナプスが強化される（LTP）。これが「Use it or lose it」の実体。

現状のLLM/RAGでは情報が参照されても、参照された側の情報自体には何も起こらない。この「参照による強化」の仕組みが欠けている。

**本アーキテクチャの核心**：情報が参照されたら、その情報自体の強度カウンターがインクリメントされる。インパクトがあった情報は重要情報として多めにカウントされる。これにより「記憶の強化すべき情報」が自然とマーキングされる。

---

## 第3章：アーキテクチャ設計

### 3.1 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│  入力処理層（軽量LLM: Haiku等）                              │
├─────────────────────────────────────────────────────────────┤
│  大きな入力 → 機械的分割 → 要点抽出 → 概要生成              │
│  項目数検出 → 閾値超えなら警告・オプション提示              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  オーケストレーター                                          │
├─────────────────────────────────────────────────────────────┤
│  概要のみ読んでルーティング判断                              │
│  専用外部メモリ：エージェント専門性、担当履歴、負荷状況     │
│  観点：エージェント適性、過去ルーティング、現在負荷         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  専門エージェント群                                          │
├─────────────────────────────────────────────────────────────┤
│  各エージェント：                                            │
│  ├── 役割定義（調達、品質、顧客対応等）                     │
│  ├── 観点定義（役割に応じた5つ程度）                        │
│  ├── 専用外部メモリ（強度付き）                             │
│  └── インデックスメモ（コンテキスト内、ポインタ）           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  睡眠フェーズ（タスク完了時）                                │
├─────────────────────────────────────────────────────────────┤
│  学び保存 → 減衰処理 → アーカイブ → 剪定 → セッション終了   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 エージェント定義

各エージェントは以下を持つ：

```json
{
  "agent_id": "procurement_agent_01",
  "role": "調達エージェント",
  "perspectives": ["コスト", "納期", "サプライヤー", "品質", "代替"],
  "system_prompt": "あなたは調達専門のエージェントです。...",
  "external_memory_ref": "memory://procurement_agent_01"
}
```

**観点（perspectives）の意義**：
- エージェントが「何を気にする存在か」を定義
- 学び抽出の構造化に使用
- 検索時のフィルタリングに使用
- 観点の違いが視点の違いを生み、複数エージェントの協働で多角的な判断が可能になる

### 3.3 外部メモリ構造

#### 基本構造

```json
{
  "id": "mem_128",
  "content": "部品Aの納品が2週間遅延、サプライヤーYの工場火災",
  "learnings": {
    "コスト": "緊急調達で15%コスト増",
    "納期": "2週間バッファが必要",
    "サプライヤー": "サプライヤーYは単一拠点リスクあり",
    "代替": "部品AはサプライヤーZからも調達可能と判明"
  },
  "tags": ["部品A", "サプライヤーY", "遅延", "火災"],
  "embedding": [...],
  
  "strength": 1.0,
  "strength_by_perspective": {
    "コスト": 2.1,
    "納期": 0.8,
    "サプライヤー": 1.5,
    "品質": 0.3,
    "代替": 1.2
  },
  "access_count": 15,
  "candidate_count": 23,
  "last_access": "2025-01-10T14:30:00Z",
  "impact_score": 3.5,
  "created_at": "2024-12-01T09:00:00Z"
}
```

**注目すべきフィールド**：
- `access_count`：実際に使用された回数
- `candidate_count`：検索候補として参照されたが使用されなかった回数
- この2つを分けることで、ノイズの強化を防ぐ

### 3.4 検索アルゴリズム：2段階構造

#### 設計の核心：関連性フィルタと優先度ランキングの分離

検索は2つの段階に分かれる：

```
Stage 1: 関連性フィルタ（ベクトル検索）
    ↓
「今回のタスクに関係ありそうな情報」を絞り込む
類似度が閾値以下 = そもそも候補にならない
    ↓
Stage 2: 優先度ランキング（スコア合成）
    ↓
関連する候補の中で「どれをより重視するか」を決定
ここで access_count, strength が効く
```

**重要な設計判断**：`access_count` が非常に高い情報でも、類似度が低ければ検索結果に含まれない。**これは正しい動作**である。

```
例：
- 調達エージェントが過去100回「サプライヤーYの単一拠点リスク」を参照
- 今回のタスク：「マーケティング予算の承認」
- 類似度：低い

→ この情報は出てくるべきか？
→ 出てこなくて正解
```

人間の記憶も同じ動作をする：
- 「サプライヤー問題」を思い出すのは「調達の話」をしているとき
- マーケティングの話をしているときに、いくら重要でも調達の記憶は想起されない
- **これは正常な認知機能**

つまり：
```
access_count の役割：
× 「無関係でも引っ張り出す」
○ 「関連性がある候補の中で、どれを優先するか」
```

#### 検索の実装：学習可能なスコアラー

Stage 2の優先度ランキングでは、単純な線形加重ではなく、**ニューラルネットワークによる学習可能なスコアラー**を使用する。これにより、特徴量間の非線形な相互作用を学習できる。

**なぜ学習可能にするか**：

```
線形加重の限界：
- 「強度が高いが古い」vs「強度は低いが最近使った」の最適なトレードオフは？
- impact_scoreが高い情報は、類似度が少し低くても優先すべき？
- 原則タグがついた情報はどれくらい優先すべき？

→ これらの最適な組み合わせは、人間が事前に決められない
→ タスク成功/失敗のフィードバックから学習させる
```

**特徴量の設計**：

ニューラルネットワークへの入力は9次元の特徴ベクトル。これらの特徴量自体は固定であり、ネットワークが学習するのは「これらをどう組み合わせるか」という関数。

```python
def extract_features(memory, query, task_context, agent):
    """特徴量抽出（これらの値は固定、変わらない）"""
    return {
        # 1. 類似度（ベクトル検索の結果）
        'similarity': compute_similarity(memory.embedding, query.embedding),

        # 2. 全体の強度
        'strength': memory.strength,

        # 3. 観点別の強度（タスクの観点に対応）
        'perspective_strength': memory.strength_by_perspective.get(
            task_context.perspective, memory.strength
        ),

        # 4. 使用率（候補になった回数に対する実際の使用率）
        'access_ratio': memory.access_count / max(1, memory.candidate_count),

        # 5. 新鮮さ（最終アクセスからの経過時間）
        'recency': 1.0 / (1 + memory.days_since_last_access),

        # 6. インパクトスコア（過去の貢献度）
        'impact_score': memory.impact_score,

        # 7. タグ重複数（クエリとのタグ一致度）
        'tag_overlap': len(set(memory.tags) & set(query.tags)),

        # 8. 原則フラグ（基本原則かどうか）
        'is_principle': 1.0 if 'principle' in memory.tags else 0.0,

        # 9. 観点一致（タスクの観点についての学びを持っているか）
        'perspective_match': 1.0 if task_context.perspective in memory.learnings else 0.0,
    }
```

**スコアラーのアーキテクチャ**：

2層のMLP（Multi-Layer Perceptron）を使用。入力次元が小さく（9次元）、訓練データも限られるため、シンプルなアーキテクチャが適切。

```python
import torch
import torch.nn as nn

class Scorer(nn.Module):
    """
    学習可能なスコアラー

    アーキテクチャ: 9 → 16 → 1（約180パラメータ）
    - 入力: 9次元の特徴ベクトル
    - 隠れ層: 16ユニット（ReLU活性化）
    - 出力: 0-1のスコア（Sigmoid）
    """
    def __init__(self, input_dim=9, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 9×16 + 16 = 160パラメータ
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),           # 16×1 + 1 = 17パラメータ
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        features: [batch_size, 9] の特徴ベクトル
        returns: [batch_size, 1] のスコア（0-1）
        """
        return self.net(features)
```

**なぜ2層か**：

```
入力次元が小さい（9次元）場合、深いネットワークは不要：
- 9次元 → 3層以上にすると過学習しやすい
- 2層でも非線形な相互作用は学習可能
- 例：「強度×新鮮さ」「類似度×インパクト」のような掛け算的効果

パラメータ数も重要：
- 約180パラメータ = 数百〜数千のタスク結果で安定学習
- エージェントの初期段階でも比較的早く収束
- 過学習リスクが低い
```

**検索の実装**：

```python
def search_with_learned_scorer(query, perspective, agent_memory, scorer):
    # Stage 1: 関連性フィルタ（変更なし）
    candidates = vector_search(
        query,
        agent_memory,
        limit=50,
        similarity_threshold=0.3
    )

    # Stage 2: 学習済みスコアラーで優先度ランキング
    task_context = TaskContext(perspective=perspective)

    for memory in candidates:
        # 特徴量を抽出
        features = extract_features(memory, query, task_context, agent_memory.agent)
        feature_vector = torch.tensor([list(features.values())], dtype=torch.float32)

        # スコアラーで予測
        with torch.no_grad():
            memory.final_score = scorer(feature_vector).item()

    # 再ランキングして上位を返す
    ranked = sorted(candidates, key=lambda m: m.final_score, reverse=True)
    return ranked[:10]
```

**学習プロセス**：

タスクの成功/失敗をフィードバックとして、スコアラーを更新する。

```python
def train_scorer(scorer, optimizer, task_result, used_memories, unused_candidates):
    """
    タスク結果に基づいてスコアラーを学習

    正例: タスク成功時に実際に使われた情報 → 高スコアが正解
    負例: 候補になったが使われなかった情報 → 低スコアが正解
    """
    scorer.train()

    # 正例のラベル作成
    positive_features = [extract_features(m, ...) for m in used_memories]
    positive_labels = torch.ones(len(used_memories), 1)

    # 負例のラベル作成（一部サンプリング）
    negative_samples = random.sample(unused_candidates, min(len(unused_candidates), len(used_memories) * 2))
    negative_features = [extract_features(m, ...) for m in negative_samples]
    negative_labels = torch.zeros(len(negative_samples), 1)

    # タスク結果による重み付け
    if task_result == "success":
        weight = 1.0  # 成功時は通常の重み
    else:
        weight = 0.5  # 失敗時は弱めに学習（情報選択以外の原因かもしれない）

    # 損失計算と更新
    all_features = torch.tensor(positive_features + negative_features, dtype=torch.float32)
    all_labels = torch.cat([positive_labels, negative_labels])

    predictions = scorer(all_features)
    loss = nn.BCELoss()(predictions, all_labels) * weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**学習の安定化**：

```python
class ScorerWithFallback:
    """
    学習初期は線形スコアリングにフォールバック
    十分なデータが溜まったら学習済みスコアラーを使用
    """
    def __init__(self, scorer, min_training_samples=100):
        self.scorer = scorer
        self.min_training_samples = min_training_samples
        self.training_samples = 0

    def score(self, features):
        if self.training_samples < self.min_training_samples:
            # 学習初期は線形スコアリング（従来方式）
            return (
                features['similarity'] * 0.40 +
                features['strength'] * 0.40 +
                features['recency'] * 0.20
            )
        else:
            # 十分学習したらニューラルネットを使用
            feature_vector = torch.tensor([list(features.values())], dtype=torch.float32)
            with torch.no_grad():
                return self.scorer(feature_vector).item()
```

**設計の要点**：

| 要素 | 設計 | 理由 |
|------|------|------|
| 特徴量 | 9次元、固定 | ベクトル検索やメタデータから計算される値 |
| ネットワーク | 2層MLP | 入力次元が小さい、過学習防止 |
| パラメータ数 | 約180 | 数百タスクで学習可能 |
| 学習信号 | タスク成功/失敗 | 人間と同じ「使って良かった」フィードバック |
| フォールバック | 線形スコア | 学習初期の安定性確保 |

この設計により、スコアリングは運用しながら自動的に最適化される。エージェントの経験が蓄積されるにつれ、そのエージェントに最適なスコアリング関数が学習される。

#### 検索漏れ対策：クエリ拡張

ベクトル検索には表現の違いによる検索漏れのリスクがある：

```
クエリ：「部品の遅延対応」
記憶：「サプライチェーンの問題でサプライヤーYが2週間遅延した」

→ 意味的には関連するが、embeddingの類似度が低く出る可能性
→ これは access_count で救う問題ではない
→ クエリ拡張で対処する
```

```python
def expand_query(original_query, perspective):
    """クエリを拡張して検索漏れを防ぐ"""
    
    # 観点に関連するキーワードを追加
    perspective_keywords = {
        "調達": ["サプライヤー", "納期", "コスト", "発注", "遅延"],
        "品質": ["不良", "検査", "基準", "クレーム", "歩留まり"],
        "顧客": ["満足度", "対応", "信頼", "クレーム", "要望"],
    }
    
    expanded = original_query
    if perspective in perspective_keywords:
        expanded += " " + " ".join(perspective_keywords[perspective])
    
    return expanded


def search_with_expansion(query, perspective, agent_memory):
    """クエリ拡張による網羅的検索"""
    
    # 元のクエリで検索
    results1 = vector_search(query, agent_memory, limit=30, similarity_threshold=0.3)
    
    # 拡張クエリで検索
    expanded = expand_query(query, perspective)
    results2 = vector_search(expanded, agent_memory, limit=30, similarity_threshold=0.3)
    
    # マージして重複除去
    all_results = deduplicate(results1 + results2)
    
    # スコア合成で再ランキング
    return rerank_with_strength(all_results, perspective)
```

#### 常に参照すべき情報：基本原則タグ

一部の情報は、タスクの内容に関わらず常に参照すべき場合がある：

```
例：
「品質は絶対に妥協しない」という経営方針
→ どんなタスクでも頭に入れておくべき
→ でも「予算承認」のクエリには類似度が低い
```

これは **タグによる別管理** で対処する：

```json
{
  "id": "mem_001",
  "content": "品質は絶対に妥協しない",
  "tags": ["principle", "品質"],
  "strength": 10.0,
  "source": "policy"
}
```

```python
def retrieve_with_principles(query, perspective, agent_memory):
    """基本原則を常に含める検索"""
    
    # 通常の検索
    task_memories = search_with_expansion(query, perspective, agent_memory)
    
    # 基本原則は類似度に関係なく常に取得
    principles = agent_memory.get_by_tag("principle")
    
    # マージ（基本原則は常に含める、重複除去）
    return deduplicate(principles + task_memories)
```

#### スケーラビリティについて

**Q: 記憶が増えたらTop 1000取らないといけなくなる？**

**A: ならない。** 理由：

1. **関連性フィルタの目的**
   - 「重要な情報を漏らさない」ではなく
   - 「無関係な情報を除外する」

2. **スケールの違い**
   - Google検索: 数十億ページから検索
   - エージェントメモリ: 数千〜数万件
   - Top 50 でも十分な網羅率

3. **類似度が低い = 無関係**
   - これは正しい動作
   - access_count が高くても無関係なら出さない

4. **本当に漏れて困るケース**
   - クエリ拡張で対処
   - 基本原則はタグで別管理

#### 「崖」の問題と対策

検索結果を固定件数（例：Top 10）で切ると、スコアが連続的であるにもかかわらず不連続な扱いが生じる：

```
スコア:
Top 10: 0.51  ← コンテキストに入る → 使われる可能性あり → 強化される可能性あり
Top 11: 0.50  ← 入らない → 使われない → 強化されない → 減衰で弱くなる
```

これは「富める者はますます富む」格差拡大につながるリスクがある。

**緩和要因**：

単一クエリでは崖があっても、多数のクエリの平均では平滑化される：

```
クエリA: 情報Xは Top 8  → 入る
クエリB: 情報Xは Top 15 → 入らない
クエリC: 情報Xは Top 3  → 入る
クエリD: 情報Xは Top 12 → 入らない

→ 長期的には「本当に有用な情報」は様々なクエリで上位に来る
→ たまたま1回だけ Top 11 でも、他のクエリで挽回できる
```

人間の記憶も同じ動作をする：ある文脈では思い出さなくても、別の文脈では思い出す。

**対策オプション**：

問題が観測された場合の対策案：

```python
# 案1: スコア閾値方式（件数ではなくスコアで切る）
def select_by_threshold(ranked_memories, score_threshold=0.4, max_count=20):
    selected = []
    for memory in ranked_memories:
        if memory.final_score >= score_threshold and len(selected) < max_count:
            selected.append(memory)
    return selected
    # → 結果が5件のこともあれば15件のこともある


# 案2: 確率的選択（崖ではなく緩やかな傾斜）
import random

def select_probabilistic(ranked_memories, base=10, extended=20):
    selected = ranked_memories[:base]  # Top 10は確実
    
    for memory in ranked_memories[base:extended]:
        # スコアに応じた確率で追加
        if random.random() < memory.final_score:
            selected.append(memory)
    
    return selected
```

**推奨アプローチ**：

1. まずは固定件数（Top 20など、多めに）で運用開始
2. 偏りの監視を行う
3. 問題が観測されたら案1か案2に移行

```python
# 偏り監視メトリクス
def detect_never_used_memories(agent_memory):
    """候補になるのに使われない情報を検出"""
    warnings = []
    for memory in agent_memory.all():
        if memory.candidate_count > 50 and memory.access_count == 0:
            # 50回候補になったのに1回も使われていない
            warnings.append(memory)
    return warnings
```

### 3.5 強度管理と記憶の定着

#### 強度の役割の明確化

強度（strength）は2つの異なる役割を持つ：

| 役割 | 説明 | 発動タイミング |
|------|------|---------------|
| **存在可否** | アーカイブ判定（閾値以下で退避） | 睡眠フェーズ |
| **検索順位** | ランキングへの補助的影響 | 検索時（ただし類似度が主） |

**重要な設計判断**：検索順位は類似度（文脈マッチング）が主役であり、強度は補助的な役割に留める。これにより、定着した記憶が増えても類似度で差別化できる。

```
強度の本質的な役割：
× 検索で上位に来るかどうか
○ 存在し続けられるかどうか（忘却への耐性）
```

#### 記憶の定着（Consolidation）

人間の記憶と同様に、繰り返し使用された記憶は「定着」し、減衰しにくくなる。

**エピソード記憶から意味記憶への変換**：

```
エピソード記憶: 「2024年1月に田中さんがサプライヤーYの火災で困っていた」
     ↓ 繰り返し使用 + 睡眠での統合
意味記憶: 「サプライヤーYは単一拠点リスクがある」（文脈が剥離）
```

**定着レベルの管理**：

```python
class Memory:
    strength: float           # 現在の強度
    access_count: int         # 使用回数
    consolidation_level: int  # 定着レベル (0-5)
    status: str               # "active" | "archived"

# 定着レベルの閾値
CONSOLIDATION_THRESHOLDS = [0, 5, 15, 30, 60, 100]  # access_countの閾値

def update_consolidation(memory):
    """使用回数に応じて定着レベルを更新"""
    for level, threshold in enumerate(CONSOLIDATION_THRESHOLDS):
        if memory.access_count >= threshold:
            memory.consolidation_level = level
```

**定着レベルと減衰率の関係**：

減衰率は「1日あたりの目標減衰」を「想定タスク数」で割って算出する。例えば、1日10タスク想定で未定着記憶を5%/日減衰させたい場合、1タスクあたりの減衰率は `0.95^(1/10) ≈ 0.995` となる。

| 定着レベル | access_count | 減衰率/タスク | 1日10タスク時の日次減衰 |
|-----------|-------------|--------------|----------------------|
| 0 | 0-4 | 0.995 | 約5%/日 |
| 1 | 5-14 | 0.997 | 約3%/日 |
| 2 | 15-29 | 0.998 | 約2%/日 |
| 3 | 30-59 | 0.999 | 約1%/日 |
| 4 | 60-99 | 0.9995 | 約0.5%/日 |
| 5 | 100+ | 0.9998 | 約0.2%/日 |

```python
# 基本設計：1日の目標減衰率を想定タスク数で按分
EXPECTED_TASKS_PER_DAY = 10  # 運用に応じて調整

DAILY_DECAY_TARGETS = {
    0: 0.95,   # 未定着: 1日5%減衰が目標
    1: 0.97,
    2: 0.98,
    3: 0.99,
    4: 0.995,
    5: 0.998,  # 完全定着: ほぼ減衰しない
}

# タスクごとの減衰率を計算
DECAY_RATES = {
    level: target ** (1 / EXPECTED_TASKS_PER_DAY)
    for level, target in DAILY_DECAY_TARGETS.items()
}

def get_decay_rate(memory):
    return DECAY_RATES[memory.consolidation_level]
```

**パラメータ調整の指針**：
- `EXPECTED_TASKS_PER_DAY` を実際のタスク頻度に合わせて調整
- タスク頻度が高い環境では値を大きくする（各タスクの減衰を小さくする）
- タスク頻度が低い環境では値を小さくする（各タスクの減衰を大きくする）

#### 2段階の強化プロセス（参照と使用の分離）

**問題**：検索で10件返っても、LLMが実際に使うのは2件程度。全件を強化するとノイズが強化されていく。

**解決策**：「検索候補になった」と「実際に使用された」を分離する。

```python
def retrieve_and_track(query, perspective, agent_memory):
    """Step 1: 検索時（この時点では強化しない）"""
    results = search_with_expansion(query, perspective, agent_memory)
    
    # 候補として参照されたことだけ記録（軽いカウント）
    for memory in results:
        memory.candidate_count += 1
    
    return results


def finalize_task(task_context, agent_memory):
    """Step 2: タスク完了時（使用確認後に強化）"""
    
    # LLMの出力を分析して、実際に使われた情報を特定
    used_memories = identify_used_memories(
        task_context.llm_output,
        task_context.retrieved_memories
    )
    
    for memory in task_context.retrieved_memories:
        if memory in used_memories:
            # 実際に使われた → しっかり強化
            memory.access_count += 1
            memory.strength += 0.1
            memory.last_access = now()
            
            # 観点別の強化
            if task_context.perspective:
                perspective = task_context.perspective
                memory.strength_by_perspective[perspective] += 0.15
        # else: 参照されたが使われなかった → 何もしない（candidate_countは既に増加済み）
```

#### 「使用された」の判定方法

```python
def identify_used_memories(llm_output, retrieved_memories):
    """LLMの出力にどのメモリが反映されたかを判定"""
    used = []
    
    for memory in retrieved_memories:
        # 方法1: キーワードマッチング（簡易・低コスト）
        if any(tag in llm_output for tag in memory.tags):
            used.append(memory)
            continue
        
        # 方法2: 内容の類似度（中程度の精度）
        similarity = compute_similarity(memory.content, llm_output)
        if similarity > 0.3:
            used.append(memory)
            continue
    
    return used


def identify_used_memories_by_llm(llm_output, retrieved_memories, task):
    """方法3: LLMに聞く（最も正確だがコスト高、重要なタスクで使用）"""
    
    prompt = f"""
以下のタスクに対する回答を生成しました。

タスク: {task.summary}
回答: {llm_output}

参照した情報一覧:
{format_memories_with_index(retrieved_memories)}

上記の情報のうち、回答に実際に影響を与えたものの番号をカンマ区切りで答えてください。
影響を与えなかった情報は含めないでください。
"""
    
    response = lightweight_llm.complete(prompt)  # Haiku等
    used_indices = parse_indices(response)
    
    return [retrieved_memories[i] for i in used_indices]
```

#### インパクトによる強化

```python
def update_impact(memory, context):
    impact = 0
    
    # ユーザーからのポジティブフィードバック
    if context.user_feedback == "helpful":
        impact += 2.0
    
    # タスク成功に貢献した
    if context.task_result == "success" and memory in context.used_memories:
        impact += 1.5
    
    # エラーを防いだ
    if memory.prevented_error:
        impact += 2.0
    
    memory.impact_score += impact
    memory.strength += impact * 0.2
```

### 3.6 タスク実行フロー

```
1. タスク受信（概要 + 詳細へのポインタ）

2. 外部メモリから関連情報を検索
   - 概要をベースにクエリ生成
   - 役割の観点別に複数検索
   - 強度を考慮したスコア合成でランキング
   - この時点では candidate_count のみ増加

3. 必要な詳細を選択的にフェッチ
   - 概要と外部メモリ情報から判断
   - 全詳細を読まない（コンテキスト節約）

4. タスク実行（LLMによる処理）

5. 使用情報の特定と強度更新
   - LLM出力を分析して実際に使われた情報を特定
   - 使用された情報のみ access_count++, strength += 0.1
   - 該当観点の strength_by_perspective も更新
   - 使用されなかった情報は強化しない

6. 新しい学びの抽出・保存
   - LLMに観点別の学びを質問
   - 外部メモリに新規エントリ作成
   - 関連既存メモリへのリンク追加

7. 睡眠フェーズ（タスク完了時に必ず実行）
   - 定着レベル更新
   - 減衰処理
   - アーカイブ
   - 容量超過時は強制剪定

8. セッション終了（コンテキストクリア）
```

**重要な設計判断：毎タスク睡眠**

エージェントは各タスク完了時に必ず睡眠フェーズを実行する。これには以下の理由がある：

1. **コンテキスト分離**
   - ユーザーとの会話継続性はオーケストレーターが担う
   - エージェントには解釈済みの明確なタスクが渡される
   - エージェントが会話の文脈を覚える必要がない

2. **永続性の確保**
   - 学びが即座に外部メモリに保存される
   - コンパクション問題が完全に解消
   - タスク間で情報が失われない

3. **一貫した減衰**
   - 減衰がタスク単位で適用される
   - コンピュータ性能向上によるタスク量増加に対応
   - 減衰率のパラメータ調整で柔軟に対応可能

#### 学び抽出のプロンプト例

```
あなたは調達エージェントです。
このタスクの経験から、以下の観点で学びを1文ずつ抽出してください：
- コストへの影響
- 納期への影響
- サプライヤーとの関係
- 品質リスク
- 将来の代替手段
該当しない観点は省略してください。
```

### 3.7 睡眠フェーズ

#### なぜ睡眠が必要か

**減衰をリアルタイムで行う問題**：

```
09:00:00 減衰処理開始
09:00:01 メモリ1〜100を減衰済み
09:00:02 タスク到着、検索実行
         → メモリ50（減衰済み）とメモリ150（未減衰）を比較
         → メモリ150が「不当に」高く見える
         → 判断のブレが発生
```

減衰処理中にタスクが来ると、メモリの状態が不整合になり、判断の一貫性が崩れる。

**バックアップ方式の問題**：

```
09:00 バックアップ作成
09:05 本番で「重大クレーム」→ 関連メモリ大きく強化
09:10 減衰処理完了、バックアップにスイッチ
      → 重大クレームの強化が消えている
      → まずい判断をする
```

#### 睡眠フェーズの実装

睡眠フェーズでは、以下の3つの処理を順番に実行する：

1. **定着レベルに応じた減衰** - よく使われた記憶は減衰しにくい
2. **閾値ベースのアーカイブ** - 弱くなった記憶を退避
3. **容量ベースの強制剪定** - 上限を超えた場合の調整

```python
def sleep_phase(agent_memory):
    """
    各タスク完了時に実行

    設計根拠：
    - オーケストレーターが会話コンテキストを保持
    - エージェントは専門的判断のみを担当
    - 毎タスク睡眠により学びを即座に永続化
    """

    # 1. タスク受付停止（このエージェントインスタンスは終了予定）
    agent_memory.accepting_tasks = False

    # 2. 定着レベルの更新
    for memory in agent_memory.active():
        update_consolidation(memory)

    # 3. 定着レベルに応じた減衰適用
    for memory in agent_memory.active():
        decay_rate = get_decay_rate(memory)

        memory.strength *= decay_rate
        for perspective in memory.strength_by_perspective:
            memory.strength_by_perspective[perspective] *= decay_rate

        # 注：毎タスク睡眠なので「最近アクセス」の相殺は不要
        # 直前に使用された記憶は strength が増加済みのため、
        # 減衰との差し引きで自然と強化される

    # 4. 閾値以下をアーカイブ（論理的忘却）
    for memory in agent_memory.active():
        if memory.strength < 0.1 and all(
            s < 0.1 for s in memory.strength_by_perspective.values()
        ):
            memory.status = "archived"

    # 5. 容量ベースの強制剪定
    prune_if_over_capacity(agent_memory)

    # 6. タスク受付再開
    agent_memory.accepting_tasks = True
```

#### 容量管理と強制剪定

**なぜ容量制限が必要か**：

```
問題: 長期運用すると定着記憶が増え続ける
→ 検索が遅くなる
→ 類似した記憶が多すぎて差別化困難
→ 人間の脳と違って物理的制約がないから無限に増える

解決: 明示的な容量制限を設ける
```

**容量の計算方法**：

単純な件数ではなく、定着度に応じた「重み」で容量を計算する。定着した記憶ほど「場所を取る」という考え方。

```python
class MemoryCapacity:
    MAX_TOTAL_WEIGHT = 10000  # アクティブ記憶の総重み上限

    # 定着レベルごとの重み（定着するほど重い）
    CONSOLIDATION_WEIGHTS = {
        0: 1.0,   # 未定着: 軽い
        1: 2.0,
        2: 4.0,
        3: 8.0,
        4: 16.0,
        5: 32.0,  # 完全定着: 重い
    }

    def calculate_weight(self, memory):
        """記憶の「重み」を計算"""
        return self.CONSOLIDATION_WEIGHTS[memory.consolidation_level]

    def get_total_weight(self, agent_memory):
        """アクティブ記憶の総重みを計算"""
        return sum(
            self.calculate_weight(m)
            for m in agent_memory.active()
        )
```

**強制剪定の実装**：

```python
def prune_if_over_capacity(agent_memory):
    """容量超過時の強制剪定"""
    capacity = MemoryCapacity()
    total = capacity.get_total_weight(agent_memory)

    if total <= capacity.MAX_TOTAL_WEIGHT:
        return  # 容量内なら何もしない

    # 剪定候補をソート
    # 優先度: 定着度が低い → 最終アクセスが古い
    candidates = sorted(
        agent_memory.active(),
        key=lambda m: (m.consolidation_level, -m.days_since_last_access)
    )

    # 容量に収まるまでアーカイブ
    for memory in candidates:
        if total <= capacity.MAX_TOTAL_WEIGHT:
            break
        memory.status = "archived"
        total -= capacity.calculate_weight(memory)
```

**剪定の優先順位**：

```
1. 定着度が低い記憶を優先的にアーカイブ
2. 同じ定着度なら、最終アクセスが古いものを優先
3. 定着度5（完全定着）でも、容量のためには剪定される

→ 「絶対に忘れない」記憶は存在しない
→ ただし剪定されにくい（最後まで残る）
```

これは人間の脳のシナプス剪定と同じ原理：新しい記憶を作るには、使われていない古い記憶の「場所」を再利用する必要がある。

#### アーカイブと再活性化

**アーカイブは削除ではない**：

人間の記憶も「消えた」のではなく「アクセスできなくなった」だけの可能性がある。適切なキュー（手がかり）があれば思い出せる。

```python
def search(query, agent_memory):
    """通常検索: activeのみ"""
    candidates = vector_search(query, agent_memory, status="active")
    return candidates

def deep_recall(query, agent_memory):
    """深い想起: archivedも含めて検索"""
    # 「あの時の...なんだっけ」という明示的な想起
    candidates = vector_search(query, agent_memory, status="all")

    # 思い出せたらactiveに戻す（再活性化）
    for memory in candidates:
        if memory.status == "archived":
            memory.status = "active"
            memory.strength = 0.5  # 再活性化時の初期強度
            memory.consolidation_level = max(0, memory.consolidation_level - 2)

    return candidates
```

#### 設計のまとめ

| 処理 | タイミング | 対象 | 効果 |
|------|-----------|------|------|
| 強化 | タスク実行時 | 使用された記憶 | strength++, access_count++ |
| 定着更新 | 睡眠フェーズ | 全active記憶 | consolidation_level更新 |
| 減衰 | 睡眠フェーズ | 全active記憶 | 定着度に応じてstrength減少 |
| アーカイブ | 睡眠フェーズ | strength閾値以下 | status→archived |
| 強制剪定 | 睡眠フェーズ | 容量超過時 | 定着度低い順にarchived |
| 再活性化 | deep_recall時 | archived記憶 | status→active |

#### バックグラウンドリプレイは不要

人間の睡眠中リプレイの主目的は「海馬から新皮質への転送」。エージェントの外部メモリは最初から長期保存場所なので、この転送処理に相当するものは不要。

必要なのは：
1. 参照による強化（タスク実行時、使用確認後にリアルタイム）
2. 定着レベルに応じた減衰（タスク完了時、睡眠フェーズで実行）
3. 容量管理と強制剪定（タスク完了時、睡眠フェーズで実行）
4. インパクトによるボーナス（タスク実行時、リアルタイム）

この4つでよい。シンプルなほうが正しい。

#### 睡眠タイミングの設計

```
【従来の発想】
睡眠 = 1日1回のバッチ処理
    ↓
問題: コンピュータ性能向上でタスク量が増えると
      1日の間に大量の情報が蓄積
      減衰が追いつかない

【本アーキテクチャ】
睡眠 = 毎タスク完了時
    ↓
利点:
- タスク量に関係なく一定の減衰
- 学びの即座の永続化
- コンパクション問題の完全解消
- 会話コンテキストはオーケストレーターが管理
```

### 3.8 コンパクション問題への対応

#### 問題

エージェントが長時間稼働すると、コンテキストウィンドウが埋まりコンパクションが発生する。まだ外部メモリに書いていない情報が消えると、永久に失われる。

#### 解決策：1タスク1セッション

```
【従来の発想】
エージェント = 長時間稼働するプロセス
    ↓
コンパクション問題が発生

【新しい発想】
エージェント = タスクごとに起動・終了
外部メモリ = 永続的なアイデンティティ
    ↓
コンパクション問題が発生しない
```

タスク完了時に必ず学びを外部メモリに書き込み、セッションを終了することで、コンパクション前に重要情報を永続化する。

### 3.9 入力処理層

#### 概要と詳細の分離

オーケストレーターが大きな入力でコンテキストを溢れさせないための設計。

```python
class Task:
    summary: str          # 必須、1000トークン以内
    detail_refs: List[str] # 詳細へのポインタ（S3、DB等）
```

オーケストレーターは概要だけで判断し、詳細は専門エージェントが必要部分だけ見る。

#### 概要がない場合の生成

```
100ページ → 10ページずつに機械的分割
    ↓
各10ページから1文ずつ要点抽出（並列処理可能、軽量LLM）
    ↓
10文を結合 → 概要（約500トークン）
```

#### 過大入力の検出と交渉

```python
class InputProcessor:
    def process(self, input):
        items = self.detect_multiple_items(input)
        
        if len(items) > THRESHOLD:  # 例：20個
            return {
                "type": "negotiation_needed",
                "message": f"{len(items)}個の項目があります。",
                "options": [
                    "優先度の高い10個を指定してください",
                    "全て処理します（時間がかかります）",
                    "カテゴリ別に分けて順次処理します"
                ]
            }
        
        return {"type": "ok", "items": items}
```

---

## 第4章：専門性の形成

### 4.1 教育プロセス：専門性の種

新しいエージェントの外部メモリは空の状態から始まる。人間でいう「教育課程」に相当するプロセスが必要。

#### 教科書 + テスト方式

人間の教育と同じプロセスで専門性を形成する：

```
教科書（専門分野の本）+ 小テスト
    ↓
チャンクに分けて読む
    ↓
テストをタスクとして実行
    ↓
睡眠（減衰処理）
    ↓
繰り返し
```

#### 実装

```python
class EducationProcess:
    def __init__(self, agent, textbook_path, perspective):
        self.agent = agent
        self.textbook = load_textbook(textbook_path)
        self.perspective = perspective
    
    def run(self):
        chapters = self.textbook.chapters
        
        for chapter in chapters:
            # Step 1: 読む（チャンク分割して順次処理）
            self.read_chapter(chapter)
            
            # Step 2: 睡眠（減衰処理）
            self.agent.sleep()
            
            # Step 3: テスト
            self.take_test(chapter.test_questions)
            
            # Step 4: 睡眠
            self.agent.sleep()
    
    def read_chapter(self, chapter):
        for section in chapter.sections:
            # 読んだ内容を外部メモリに保存
            memory = {
                "content": section.text,
                "learnings": self.extract_learnings(section, self.perspective),
                "tags": extract_tags(section),
                "strength": 0.5,  # 初期強度は低め（まだ「使われていない」）
                "source": "education"
            }
            self.agent.external_memory.save(memory)
    
    def take_test(self, questions):
        for question in questions:
            # テストをタスクとして実行
            task = Task(
                summary=question.text,
                detail_refs=[]
            )
            
            result = self.agent.execute(task)
            
            # 正解判定
            if self.check_answer(result, question.answer):
                # 正解：使われた情報が自動的に強化される
                # （通常のタスク完了処理で実施済み）
                pass
            else:
                # 不正解：正解に関連する情報を明示的に強化（復習）
                self.reinforce_correct_knowledge(question)
    
    def reinforce_correct_knowledge(self, question):
        """不正解時の復習処理"""
        # 正解に関連するメモリを検索
        related = self.agent.external_memory.search(question.answer)
        
        for memory in related[:5]:
            # 「復習」として強化
            memory.strength += 0.2
            memory.access_count += 1
            memory.last_access = now()
```

#### プロセスの流れと効果

```
┌────────────────────────────────────────────────────────────┐
│  教育フェーズ                                               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Chapter 1                                                 │
│  ├── 読む → 外部メモリに保存（strength: 0.5）             │
│  ├── 睡眠 → 減衰（読んだだけの情報は弱くなる）           │
│  ├── テスト → 正解に使った情報が強化（strength: 0.6+）   │
│  └── 睡眠 → 減衰（使われなかった情報はさらに弱く）       │
│                                                            │
│  Chapter 2                                                 │
│  ├── 読む → 新しい内容を保存                              │
│  ├── 睡眠                                                  │
│  ├── テスト → Chapter 1の知識も使うかも → 再強化         │
│  └── 睡眠                                                  │
│                                                            │
│  ... 繰り返し ...                                          │
│                                                            │
│  結果：                                                    │
│  ├── テストで繰り返し使われた知識 → 高いstrength         │
│  ├── 読んだだけで使われなかった知識 → 低いstrength       │
│  └── 自然と「重要な知識」が残る                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 教科書の構造例

```yaml
textbook:
  title: "調達基礎"
  perspective: "調達"
  
  chapters:
    - title: "サプライヤー評価"
      sections:
        - title: "品質評価基準"
          text: "サプライヤーの品質評価では..."
        - title: "コスト分析"
          text: "総所有コスト（TCO）の観点から..."
      test_questions:
        - question: "サプライヤーAとBの情報が以下の通りです。どちらを選ぶべきですか？理由も述べてください。"
          answer_keywords: ["TCO", "品質", "リスク分散"]
          
    - title: "契約交渉"
      sections:
        - title: "価格交渉の基本"
          text: "..."
      test_questions:
        - question: "..."
```

#### 復習（Spaced Repetition）

人間の学習で効果的な「間隔を空けた復習」も実装可能：

```python
def spaced_repetition(agent, test_questions):
    """間隔を空けた復習スケジュール"""
    schedule = [1, 3, 7, 14, 30]  # 日数
    
    for days in schedule:
        wait(days)
        
        for question in test_questions:
            result = agent.execute(Task(summary=question.text))
            
            if check_answer(result, question.answer):
                # 正解：使われた情報が強化される
                # 次回の復習間隔を長くしても良い
                pass
            else:
                # 不正解：復習、次回の間隔を短くする
                reinforce_correct_knowledge(question)
        
        agent.sleep()
```

### 4.2 なぜ個性が生まれるか

**同じLLM（パラメータ同一）でも、外部メモリの違いで個性が出る**：

```
Agent A：調達タスクが多い
    → 調達関連の記憶が強化
    → 調達観点の strength_by_perspective が高い
    → 検索で調達関連が上位に来やすい
    → 新しいタスクでも調達視点で考える
    → 「調達に強いAgent A」という個性

Agent B：品質タスクが多い
    → 品質関連の記憶が強化
    → 「品質重視のAgent B」という個性
```

### 4.3 一貫性の形成

```
過去の判断：「納期より品質を優先した」→ 成功
    ↓
外部メモリに記録、impact_score が上がる
    ↓
類似状況で高い strength で参照される
    ↓
また品質優先の判断をする
    ↓
「品質重視」という性格の一貫性
```

### 4.4 視点の分化

```
【現状のマルチエージェント】
Agent A：システムプロンプト + 今回のタスク情報
Agent B：システムプロンプト + 今回のタスク情報
→ 全員が同じ視点、同じ結論に収束

【本アーキテクチャ】
Agent A：共通プロンプト + タスク + 調達専門外部メモリ（強度分布が調達寄り）
Agent B：共通プロンプト + タスク + 品質専門外部メモリ（強度分布が品質寄り）
→ 異なる視点、異なる提案
→ 議論が生まれる
→ 組織としての知恵
```

---

## 第5章：技術的考慮事項

### 5.1 技術的に曖昧な点

#### インパクトスコアの定義

- 「タスク成功に貢献した」の判定基準
- 間接的な影響の評価

**対応**：最初は単純に「使用された情報 +1」で始め、運用しながら洗練させる。

#### パラメータ調整

- 減衰率（0.95? 0.99?）
- 閾値（0.1? 0.05?）
- 強化量（+0.1? +0.2?）
- スコア合成の重み（類似度0.4, 強度0.4, 新鮮さ0.2）

**対応**：正解がないため、ドメインやタスク頻度に応じて運用しながら調整。

#### 崖の問題

検索結果を固定件数（Top N）で切ることで、境界付近の情報に不連続な扱いが生じる。Top 10 と Top 11 のスコア差が僅かでも、扱いは大きく異なる。

**緩和要因**：多数のクエリの平均では平滑化される。本当に有用な情報は様々なクエリで上位に来るため、1回の Top 11 は他のクエリで挽回される。

**対応**：
- まずは固定件数（Top 20など多め）で運用開始
- `candidate_count` と `access_count` の比率で偏りを監視
- 問題が観測されたらスコア閾値方式または確率的選択に移行

### 5.2 技術的な困難

**大きな困難はない**。必要な要素はすべて既存技術で実現可能。

| 要素 | 既存技術 |
|------|---------|
| ベクトルDB | Pinecone, Qdrant, Chroma等 |
| メタデータ管理 | 普通のRDB |
| LLM API | Claude, GPT等 |
| 定期バッチ処理 | cron, Cloud Scheduler等 |
| オーケストレーション | Python, LangGraph等 |

革新的な新技術は不要。**既存の部品の組み合わせ**で実現できる。

### 5.3 実装工数の見積もり

| フェーズ | 期間 | 内容 |
|---------|------|------|
| PoC | 1-2週間 | 単一エージェント、基本的な強化/減衰、動作確認 |
| 実用レベル | 1-2ヶ月 | 複数エージェント、パラメータ調整、エラーハンドリング、運用監視 |
| 教育プロセス | +1-2週間 | 教科書構造の設計、テスト問題作成、復習スケジュール |

### 5.4 残る制約

#### 「無茶な依頼」の限界

102個の深い設計質問を一発で投げるようなケースでは：
- 入力処理層で検出・警告は可能
- 分割処理のオプション提示は可能
- しかし「深さ」の事前推定は困難
- 発注側の適切な依頼設計が必要

**これは人間相手でも同じ制約**。完璧なシステムは作れないが、「黙ってフリーズ」より「無理ですと言う」「分割を提案する」ほうがマシ。

---

## 第6章：関連研究との比較

本アーキテクチャに関連する研究は複数存在する。本章では、それらの研究の概要と本アーキテクチャとの差分を明確にする。

### 6.1 MemoryBank（AAAI 2024）

**概要**：[MemoryBank](https://arxiv.org/abs/2305.10250)は、LLMに長期記憶を付与するための機構。Ebbinghaus忘却曲線理論に基づく減衰メカニズムを採用し、時間経過に応じて記憶を忘却・強化する。

**主な特徴**：
- メモリストレージ、メモリリトリーバー、メモリアップデーターの3つの柱
- Ebbinghaus忘却曲線（R = e^(-t/S)）による時間ベースの減衰
- SiliconFriendというAIコンパニオンでの実証

**本アーキテクチャとの差分**：

| 観点 | MemoryBank | 本アーキテクチャ |
|------|------------|-----------------|
| 減衰の基準 | 時間のみ | 時間 + 使用頻度 + インパクト |
| 強化の判定 | 参照されたら強化 | **実際に使用されたら強化**（2段階） |
| 観点別管理 | なし | **観点別の強度管理** |
| 睡眠フェーズ | なし | **一貫性のためのバッチ処理** |

MemoryBankは「時間が経ったら忘れる」という単純なモデル。本アーキテクチャは「使われないものは忘れ、使われるものは強化される」という脳のLTP原理をより忠実に模倣している。

### 6.2 LightMem（2025）

**概要**：[LightMem](https://arxiv.org/abs/2510.18866)は、Atkinson-Shiffrin記憶モデルに基づく3段階メモリシステム。効率性を重視し、トークン使用量を最大117倍削減。

**主な特徴**：
- 感覚記憶（Sensory Memory）：軽量圧縮によるフィルタリング
- 短期記憶（Short-term Memory）：トピックベースの整理・要約
- 長期記憶（Long-term Memory）：sleep-time updateによるオフライン統合

**本アーキテクチャとの差分**：

| 観点 | LightMem | 本アーキテクチャ |
|------|----------|-----------------|
| 睡眠フェーズ | あり（統合処理） | あり（減衰処理） |
| 強度管理 | なし | **使用ベースの強度管理** |
| 専門性形成 | なし | **教育プロセス** |
| 個性の発現 | なし | **観点別強度による個性** |

LightMemは効率性に焦点を当てており、記憶の「重要度」を使用頻度で管理する仕組みがない。

### 6.3 Language Models Need Sleep（OpenReview 2025）

**概要**：[Language Models Need Sleep](https://openreview.net/forum?id=iiZy6xyVVE)は、生物学的な「睡眠」パラダイムをLLMに導入する研究。Memory ConsolidationとDreamingの2段階で継続学習を実現。

**主な特徴**：
- Memory Consolidation：RLベースのKnowledge Seedingでパラメータ拡張
- Dreaming：合成データによる自己改善フェーズ
- 破滅的忘却への耐性向上

**本アーキテクチャとの差分**：

| 観点 | LM Need Sleep | 本アーキテクチャ |
|------|---------------|-----------------|
| パラメータ更新 | **する** | しない |
| 実装難易度 | 高い（モデル改変が必要） | 中程度（既存技術で可能） |
| 専門性の深さ | 深い（内面化） | 浅い（外部参照） |
| 実用化状況 | 研究段階 | **即座に実装可能** |

この研究はパラメータ更新を伴うため本当の意味での「学習」が可能だが、実装のハードルが高い。本アーキテクチャはパラメータ不変でも実用的な効果を狙う。

### 6.4 MemGPT / Letta

**概要**：[MemGPT](https://arxiv.org/abs/2310.08560)は、OSの仮想メモリ管理に着想を得たLLMメモリシステム。現在は[Letta](https://docs.letta.com/)として発展。

**主な特徴**：
- 2層メモリ階層：In-context（RAM相当）とOut-of-context（ディスク相当）
- 自己編集メモリ：LLM自身がメモリ移動を制御
- Heartbeat機構：マルチステップ推論のサポート
- Archival MemoryとRecall Memoryの分離

**本アーキテクチャとの差分**：

| 観点 | MemGPT/Letta | 本アーキテクチャ |
|------|--------------|-----------------|
| 忘却メカニズム | **なし** | 減衰 + 睡眠フェーズ |
| 強度管理 | なし | **使用ベースの強度管理** |
| 専門性形成 | なし | **教育プロセス** |
| メモリ構造 | 階層的（RAM/ディスク） | フラット（強度で管理） |

MemGPTは「無限メモリ」の錯覚を作ることに焦点を当てており、「何を忘れ、何を覚えるか」の判断メカニズムがない。

### 6.5 A-MEM（NeurIPS 2025）

**概要**：[A-MEM](https://arxiv.org/abs/2502.12110)は、Zettelkasten手法に基づく自己組織化メモリシステム。動的なインデックス作成とリンク生成が特徴。

**主な特徴**：
- Zettelkasten原理：相互接続された知識ネットワーク
- 動的インデックス生成：新規メモリ追加時に自動リンク
- メモリの進化：新規メモリが既存メモリを更新

**本アーキテクチャとの差分**：

| 観点 | A-MEM | 本アーキテクチャ |
|------|-------|-----------------|
| 忘却メカニズム | **なし** | 減衰 + 睡眠フェーズ |
| 強度管理 | なし | **使用ベースの強度管理** |
| 組織化方式 | リンクベース | 強度ベース |
| 検索方式 | グラフ走査 | ベクトル + 強度スコア合成 |

A-MEMはメモリの「組織化」に焦点を当てており、時間経過や使用頻度による「重要度の変化」を扱わない。

### 6.6 HippoRAG（NeurIPS 2024）

**概要**：[HippoRAG](https://arxiv.org/abs/2405.14831)は、海馬インデックス理論に基づくRAGフレームワーク。Knowledge GraphとPersonalized PageRankを組み合わせたマルチホップ推論が特徴。

**主な特徴**：
- LLMによるKnowledge Graph生成
- Personalized PageRankによるサブグラフ探索
- マルチホップ質問応答で最大20%の性能向上

**本アーキテクチャとの差分**：

| 観点 | HippoRAG | 本アーキテクチャ |
|------|----------|-----------------|
| 主目的 | 検索精度向上 | 記憶の永続性・個性形成 |
| 忘却メカニズム | **なし** | 減衰 + 睡眠フェーズ |
| 強度管理 | なし | **使用ベースの強度管理** |
| グラフ構造 | 必須 | 不要（ベクトルDBで十分） |

HippoRAGは「検索」の最適化に焦点を当てており、「記憶の管理」（何を忘れ、何を強化するか）は扱わない。

### 6.7 Google Titan（2024）

**概要**：[Titans](https://arxiv.org/abs/2501.00663)は、テスト時に学習するNeural Memory Moduleを導入したアーキテクチャ。2M以上のコンテキストウィンドウに対応。

**主な特徴**：
- 短期記憶（Attention）と長期記憶（Neural Memory）の分離
- Surprise-based learning：勾配を「驚き」シグナルとして記憶判定
- 適応的忘却：Weight decayによる古い記憶の減衰
- 3つのバリアント：MAC、MAG、MAL

**本アーキテクチャとの差分**：

| 観点 | Titan | 本アーキテクチャ |
|------|-------|-----------------|
| パラメータ更新 | **する**（Neural Memory） | しない |
| 実用化状況 | 研究段階 | **既存技術で即実装可能** |
| 専門性の深さ | 深い（内面化される） | 浅い（外部参照） |
| 実装複雑性 | 高い | 中程度 |
| 忘却判断 | 驚きメトリクス | **使用頻度 + インパクト** |

Titanは理想的なソリューションだが、実用化には時間がかかる。本アーキテクチャは「Titanが実用化されるまでの現実的な解」として位置づけられる。

### 6.8 継続学習研究（EWC、Synaptic Intelligence）

**概要**：[Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796)や[Synaptic Intelligence](https://www.pnas.org/doi/10.1073/pnas.1611835114)は、ニューラルネットワークの破滅的忘却を防ぐための正則化手法。

**主な特徴**：
- 重要なパラメータの変更にペナルティを課す
- シナプス可塑性の生物学的原理に基づく
- タスク間で共有構造を再利用

**本アーキテクチャとの差分**：

| 観点 | EWC/SI | 本アーキテクチャ |
|------|--------|-----------------|
| 対象 | モデルパラメータ | **外部メモリ** |
| 目的 | ファインチューニング時の忘却防止 | タスク実行中の記憶管理 |
| パラメータ更新 | する | **しない** |

これらの手法はモデル学習時の問題を扱うのに対し、本アーキテクチャは推論時の外部メモリ管理を扱う。

### 6.9 本アーキテクチャの独自性

上記の関連研究との比較から、本アーキテクチャの独自性は以下の点にある：

1. **2段階強化プロセス（候補 vs 使用）**
   - 検索候補になっただけでは強化しない
   - 実際にLLMが使用した情報のみを強化
   - ノイズの強化を防ぎ、本当に有用な情報を残す
   - **他の研究にはない新規性**

2. **観点別強度管理**
   - 同じ情報でも観点によって重要度が異なる
   - 専門エージェントの「個性」を外部メモリで実現
   - **他の研究にはない新規性**

3. **パラメータ不変での実用的効果**
   - 既存技術のみで実装可能
   - 今日から使える現実的なソリューション
   - Titanなどの将来技術への橋渡し

4. **教育プロセスによる専門性形成**
   - 人間と同じ「教科書＋テスト」プロセス
   - 睡眠フェーズを挟んだ繰り返し学習
   - 自然な形での専門知識の定着

### 6.10 関連研究のまとめ表

| 研究 | 忘却 | 使用強化 | 観点別 | パラメータ不変 | 睡眠 | 実装容易性 |
|------|:----:|:--------:|:------:|:--------------:|:----:|:----------:|
| MemoryBank | ○ | × | × | ○ | × | ○ |
| LightMem | △ | × | × | ○ | ○ | ○ |
| LM Need Sleep | ○ | △ | × | × | ○ | × |
| MemGPT/Letta | × | × | × | ○ | × | ○ |
| A-MEM | × | × | × | ○ | × | ○ |
| HippoRAG | × | × | × | ○ | × | ○ |
| Google Titan | ○ | ○ | × | × | × | × |
| **本アーキテクチャ** | **○** | **○** | **○** | **○** | **○** | **○** |

---

## 第7章：まとめ

### 7.1 設計の核心

1. **エージェント本体はステートレス、外部メモリがアイデンティティを担う**

2. **検索の2段階構造**：Stage 1（関連性フィルタ）で無関係な情報を除外し、Stage 2（優先度ランキング）で強度・新鮮さを加味。類似度が低い情報は access_count が高くても出さない（これは正しい動作）

3. **2段階の強化プロセス**：検索候補になっただけでは強化せず、実際に使用された情報のみ強化（ノイズ強化の防止）

4. **定期的な減衰**：睡眠フェーズで一括処理（一貫性確保）

5. **観点の事前定義**：役割に応じた5つ程度の観点で学びを構造化

6. **教育プロセス**：教科書+テストを睡眠を挟んで繰り返し、専門性の種を形成

7. **1タスク1セッション**：コンパクション前に学びを永続化

8. **概要と詳細の分離**：オーケストレーターは概要だけで判断

9. **例外対応**：クエリ拡張で検索漏れを防ぎ、基本原則はタグで別管理

### 7.2 人間の組織との対応

| 人間の組織 | 本アーキテクチャ |
|-----------|-----------------|
| 個人の記憶 | エージェント専用外部メモリ |
| 専門性 | 観点別の強度分布 |
| 経験の蓄積 | 使用された情報の強度増加 |
| 忘却 | 減衰処理 |
| 睡眠 | 睡眠フェーズ（減衰バッチ） |
| 教育・研修 | 教科書+テストプロセス |
| 復習 | Spaced Repetition |
| 部長の概要確認 | オーケストレーターの概要ルーティング |
| 担当者への詳細引き継ぎ | 詳細ポインタの受け渡し |

### 7.3 完璧は諦める

人間の組織も：
- 重要な経験を忘れることがある
- 後から「あのとき気づくべきだった」と思う
- 観点の漏れは普通にある
- **それでもなんとかやっている**

エージェントも同様に、完璧を目指さず「今より良い」を目指す。

---

## 付録A：用語集

| 用語 | 定義 |
|------|------|
| 外部メモリ | エージェント専用の永続的な記憶ストア（ベクトルDB + メタデータ） |
| 強度（strength） | メモリの重要度スコア。使用で増加、定期的に減衰。存在可否（アーカイブ判定）に影響 |
| 定着レベル（consolidation_level） | 記憶の定着度（0-5）。使用回数に応じて上昇し、減衰率を決定 |
| 減衰率（decay_rate） | 睡眠フェーズでの強度減少率。定着レベルが高いほど減衰しにくい |
| access_count | 実際に使用された回数。定着レベルの更新に使用 |
| candidate_count | 検索候補になったが使用されなかった回数 |
| 観点（perspective） | エージェントの役割に応じた判断の視点（例：コスト、納期） |
| 睡眠フェーズ | タスク受付を停止して減衰・アーカイブ・剪定を一括実行する期間 |
| インパクトスコア | フィードバックやタスク成功による追加の強化量 |
| コンパクション | コンテキストウィンドウが埋まった際の情報圧縮・削除 |
| LTP（長期増強） | 脳で、シナプス発火によりそのシナプスが強化される現象 |
| 関連性フィルタ | 検索Stage 1。ベクトル類似度で無関係な情報を除外する処理 |
| 優先度ランキング | 検索Stage 2。関連する候補内で類似度を主役に順位付け |
| クエリ拡張 | 観点キーワードを追加してベクトル検索の漏れを防ぐ手法 |
| 基本原則タグ | 類似度に関係なく常に参照すべき情報に付与する特別なタグ |
| 学習可能なスコアラー | 特徴量から検索スコアを予測するニューラルネットワーク |
| 崖の問題 | 固定件数で検索結果を切ることで境界付近の情報に不連続な扱いが生じる問題 |
| 教育プロセス | 教科書+テストにより専門性の種を形成するプロセス |
| アーカイブ | 論理的忘却。検索対象から外れるが削除はされない状態 |
| 再活性化 | アーカイブされた記憶を明示的な想起で active に戻すこと |
| 容量制限 | アクティブ記憶の総重みに対する上限。長期運用での破綻を防止 |
| 強制剪定 | 容量超過時に定着度が低い順にアーカイブする処理 |

## 付録B：関連研究

### B.1 LLMメモリシステム

| 研究 | 発表 | 概要 | 論文 |
|------|------|------|------|
| MemoryBank | AAAI 2024 | Ebbinghaus忘却曲線に基づく時間減衰メモリ | [arXiv:2305.10250](https://arxiv.org/abs/2305.10250) |
| LightMem | 2025 | Atkinson-Shiffrin 3段階メモリ、sleep-time update | [arXiv:2510.18866](https://arxiv.org/abs/2510.18866) |
| Language Models Need Sleep | OpenReview 2025 | Memory Consolidation + Dreamingによる継続学習 | [OpenReview](https://openreview.net/forum?id=iiZy6xyVVE) |
| MemGPT / Letta | NeurIPS 2023 | OS風仮想メモリ管理、自己編集メモリ | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) |
| A-MEM | NeurIPS 2025 | Zettelkasten手法による自己組織化メモリ | [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) |
| HippoRAG | NeurIPS 2024 | 海馬インデックス理論 + PageRank | [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) |
| Google Titan | 2024 | テスト時学習Neural Memory、Surprise-based learning、適応的減衰 | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| MemLong | 2024 | 外部メモリ検索による長文モデリング、検索頻度に基づく剪定 | [arXiv:2408.16967](https://arxiv.org/abs/2408.16967) |
| H-MEM | 2025 | 階層的メモリアーキテクチャ、位置インデックス付き多段検索 | [arXiv:2507.22925](https://arxiv.org/abs/2507.22925) |
| Mem0 | 2025 | 実用的なエージェント長期メモリ、グラフベース記憶表現 | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) |

### B.2 継続学習・破滅的忘却

| 研究 | 発表 | 概要 | 論文 |
|------|------|------|------|
| Elastic Weight Consolidation | PNAS 2017 | 重要パラメータの変更にペナルティを課す正則化 | [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) |
| Synaptic Intelligence | ICML 2017 | シナプス可塑性に基づく継続学習 | [PNAS](https://www.pnas.org/doi/10.1073/pnas.1611835114) |

### B.3 神経科学理論

| 研究 | 概要 |
|------|------|
| Synaptic Homeostasis Hypothesis (SHY) | 睡眠中にシナプス全体が縮小し、重要な記憶が相対的に強調される理論 |
| Long-Term Potentiation (LTP) | シナプス発火により当該シナプスが強化される現象。「Use it or lose it」の神経科学的基盤 |
| Ebbinghaus Forgetting Curve | 記憶保持率が時間とともに指数関数的に減衰する法則（R = e^(-t/S)） |
| Spaced Repetition | 間隔を空けた復習により記憶定着を促進する学習手法 |

### B.4 間隔反復・記憶強度モデル

| 研究 | 発表 | 概要 | 論文/リンク |
|------|------|------|------|
| FSRS | 2022- | DSRモデルに基づく機械学習最適化間隔反復スケジューラ | [GitHub](https://github.com/open-spaced-repetition/fsrs4anki) |
| DRL-SRS | Applied Sciences 2024 | 深層強化学習による間隔反復最適化、半減期回帰モデル | [MDPI](https://www.mdpi.com/2076-3417/14/13/5591) |

### B.5 サーベイ論文

| 研究 | 発表 | 概要 | 論文 |
|------|------|------|------|
| From Human Memory to AI Memory | 2025 | 人間の記憶とAIメモリの関係を体系的に整理したサーベイ | [arXiv:2504.15965](https://arxiv.org/abs/2504.15965) |
| Memory in the Age of AI Agents | 2025 | 102ページのエージェントメモリ統合フレームワーク（Forms, Functions, Dynamics） | [arXiv:2512.13564](https://arxiv.org/abs/2512.13564) |

---

*本ドキュメントは2025年1月11日の議論に基づいて作成された。*
