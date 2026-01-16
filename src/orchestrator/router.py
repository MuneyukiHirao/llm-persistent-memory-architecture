# ルーティングロジック
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.3
"""
ルーティングロジックモジュール

タスクに最適なエージェントを選択するルーティングを提供。

Phase 2 MVP:
- ルールベース方式（キーワードマッチ）
- 4要素によるスコア計算:
  - capability_match (0.40): 能力タグのマッチ度
  - past_success_rate (0.30): 過去の成功率
  - recent_activity (0.20): 最近のアクティビティ（負荷考慮）
  - perspective_match (0.10): 観点のマッチ度

設計方針（検索エンジンエージェント観点）:
- 検索精度: キーワードマッチの精度を重視
- レスポンス性能: シンプルなルールベースで高速処理
- スケーラビリティ: エージェント数増加時も線形スケール
- API連携: 外部APIに依存しないローカル処理
- フォールバック: エージェントが見つからない場合のデフォルト選択
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from src.agents.agent_registry import AgentRegistry, AgentDefinition
from src.config.phase2_config import ROUTING_SCORE_WEIGHTS


@dataclass
class RoutingDecision:
    """ルーティング判断結果

    Attributes:
        selected_agent_id: 選択されたエージェントのID
        selection_reason: 選択理由の説明
        candidates: 候補エージェントのリスト（スコア順）
        confidence: 判断の確信度（0.0-1.0）
    """

    selected_agent_id: str
    selection_reason: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "selected_agent_id": self.selected_agent_id,
            "selection_reason": self.selection_reason,
            "candidates": self.candidates,
            "confidence": self.confidence,
        }


class Router:
    """ルーティングロジック

    タスクに最適なエージェントを選択する。
    Phase 2 MVP ではルールベース方式を採用。

    使用例:
        registry = AgentRegistry(db)
        router = Router(registry)

        decision = router.decide(
            task_summary="ユーザー認証機能を実装",
            items=["認証フロー", "セッション管理"],
            past_experiences=[{"agent_id": "impl_agent", "success": True}],
        )

        print(f"選択: {decision.selected_agent_id}")
        print(f"理由: {decision.selection_reason}")
        print(f"確信度: {decision.confidence}")

    Attributes:
        agent_registry: AgentRegistry インスタンス
    """

    # capability キーワードマッピング（日本語 → capability タグ）
    # ルーティング判断に使用するキーワード辞書
    _CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
        # 調査・リサーチ系
        "research": [
            "調査", "調べ", "リサーチ", "検索", "探", "確認",
            "分析", "比較", "研究", "情報収集", "ドキュメント確認",
        ],
        "analysis": [
            "分析", "解析", "評価", "判定", "判断",
            "レビュー", "チェック", "検証",
        ],
        # 実装系
        "implementation": [
            "実装", "開発", "構築", "ビルド",
            "コーディング", "プログラミング", "新機能",
        ],
        "coding": [
            "コード", "プログラム", "関数", "クラス", "モジュール",
            "API", "エンドポイント", "機能",
        ],
        # テスト系
        "testing": [
            "テスト", "試験", "検証", "動作確認",
            "ユニットテスト", "統合テスト", "E2E", "pytest",
            "テストケース", "テストコード", "カバレッジ",
        ],
        "debugging": [
            "デバッグ", "バグ", "エラー", "問題", "修正",
            "トラブルシューティング", "原因調査",
        ],
        # ドキュメント系
        "documentation": [
            "ドキュメント", "文書", "README", "仕様書",
            "説明", "コメント", "記述",
        ],
        # 設計系
        "design": [
            "設計", "アーキテクチャ", "構造", "構成",
            "パターン", "モデル", "スキーマ",
        ],
        # インフラ系
        "infrastructure": [
            "インフラ", "デプロイ", "CI/CD", "Docker",
            "Kubernetes", "サーバー", "環境構築",
        ],
        "database": [
            "データベース", "DB", "SQL", "テーブル",
            "マイグレーション", "スキーマ", "クエリ",
        ],
    }

    # 観点キーワードマッピング
    _PERSPECTIVE_KEYWORDS: Dict[str, List[str]] = {
        "正確性": ["正確", "精度", "品質", "確実"],
        "効率性": ["効率", "高速", "パフォーマンス", "最適化"],
        "網羅性": ["網羅", "全て", "すべて", "完全"],
        "保守性": ["保守", "メンテナンス", "可読性", "拡張"],
        "安全性": ["安全", "セキュリティ", "脆弱性", "認証"],
    }

    def __init__(self, agent_registry: AgentRegistry):
        """Router を初期化

        Args:
            agent_registry: AgentRegistry インスタンス
        """
        self.agent_registry = agent_registry

        # 過去の実行履歴キャッシュ（シンプル実装）
        # 本番ではDBから取得する
        self._execution_history: Dict[str, List[Dict]] = {}

    def decide(
        self,
        task_summary: str,
        items: Optional[List[str]] = None,
        past_experiences: Optional[List[Dict]] = None,
    ) -> RoutingDecision:
        """ルーティング判断を行う

        Args:
            task_summary: タスクの概要
            items: 論点リスト（オプション）
            past_experiences: 過去の経験（メモリから取得）

        Returns:
            RoutingDecision インスタンス

        Note:
            - アクティブなエージェントがいない場合は空のRoutingDecisionを返す
            - items は task_summary に追加してマッチング精度を上げる
        """
        # 1. 全アクティブエージェントを取得
        all_agents = self.agent_registry.get_active_agents()

        if not all_agents:
            return RoutingDecision(
                selected_agent_id="",
                selection_reason="アクティブなエージェントが見つかりません",
                candidates=[],
                confidence=0.0,
            )

        # items を task_summary に結合してマッチング精度を向上
        full_task_text = task_summary
        if items:
            full_task_text = f"{task_summary} {' '.join(items)}"

        # 2. 各エージェントのスコアを計算
        candidates = []
        for agent in all_agents:
            score = self._calculate_routing_score(
                agent=agent,
                task_summary=full_task_text,
                past_experiences=past_experiences,
            )
            reason = self._generate_selection_reason(agent, task_summary)
            candidates.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "score": round(score, 4),
                "reason": reason,
            })

        # 3. スコア順にソート
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4. 上位を選択
        if not candidates:
            return RoutingDecision(
                selected_agent_id="",
                selection_reason="候補エージェントがありません",
                candidates=[],
                confidence=0.0,
            )

        selected = candidates[0]

        # 確信度を計算（上位2つのスコア差から）
        confidence = self._calculate_confidence(candidates)

        return RoutingDecision(
            selected_agent_id=selected["agent_id"],
            selection_reason=selected["reason"],
            candidates=candidates[:3],  # 上位3件を保持
            confidence=confidence,
        )

    def _calculate_routing_score(
        self,
        agent: AgentDefinition,
        task_summary: str,
        past_experiences: Optional[List[Dict]] = None,
    ) -> float:
        """ルーティングスコアを計算

        仕様書セクション4.2に基づく重み付けスコア:
        - capability_match: 0.40
        - past_success_rate: 0.30
        - recent_activity: 0.20
        - perspective_match: 0.10

        Args:
            agent: 評価対象のエージェント
            task_summary: タスクの概要（items含む）
            past_experiences: 過去の経験リスト

        Returns:
            0.0-1.0 のスコア
        """
        score = 0.0

        # 1. 能力タグのマッチ度（0.40）
        capability_score = self._match_capabilities(
            agent.capabilities,
            task_summary,
        )
        score += capability_score * ROUTING_SCORE_WEIGHTS["capability_match"]

        # 2. 過去の成功率（0.30）
        success_rate = self._get_past_success_rate(
            agent.agent_id,
            past_experiences,
        )
        score += success_rate * ROUTING_SCORE_WEIGHTS["past_success_rate"]

        # 3. 最近のアクティビティ（0.20）- 負荷分散
        # アクティビティが少ないほどスコアが高い（負荷分散）
        activity_score = self._get_activity_score(agent.agent_id)
        score += activity_score * ROUTING_SCORE_WEIGHTS["recent_activity"]

        # 4. 観点のマッチ度（0.10）
        perspective_score = self._match_perspectives(
            agent.perspectives,
            task_summary,
        )
        score += perspective_score * ROUTING_SCORE_WEIGHTS["perspective_match"]

        return min(score, 1.0)

    def _match_capabilities(
        self,
        capabilities: List[str],
        task_summary: str,
    ) -> float:
        """タスクとcapabilitiesのマッチ度を計算

        キーワードベースのマッチング。
        タスク概要に含まれるキーワードと、エージェントのcapabilitiesを照合。

        Args:
            capabilities: エージェントの能力タグリスト
            task_summary: タスクの概要

        Returns:
            0.0-1.0 のマッチスコア
        """
        if not capabilities:
            return 0.0

        task_lower = task_summary.lower()
        matched_count = 0
        total_weight = 0

        for capability in capabilities:
            cap_lower = capability.lower()

            # 直接マッチ
            if cap_lower in task_lower:
                matched_count += 1
                total_weight += 1
                continue

            # キーワード辞書によるマッチ
            keywords = self._CAPABILITY_KEYWORDS.get(cap_lower, [])
            for keyword in keywords:
                if keyword in task_summary:
                    matched_count += 1
                    total_weight += 1
                    break

        if not capabilities:
            return 0.0

        # マッチした割合を返す
        return matched_count / len(capabilities)

    def _get_past_success_rate(
        self,
        agent_id: str,
        past_experiences: Optional[List[Dict]] = None,
    ) -> float:
        """過去の成功率を取得

        past_experiencesから該当エージェントの成功率を計算。
        データがない場合はデフォルト値（0.5）を返す。

        Args:
            agent_id: エージェントID
            past_experiences: 過去の経験リスト

        Returns:
            0.0-1.0 の成功率
        """
        if not past_experiences:
            return 0.5  # デフォルト値

        # 該当エージェントの履歴をフィルタ
        agent_experiences = [
            exp for exp in past_experiences
            if exp.get("agent_id") == agent_id
            or exp.get("selected_agent_id") == agent_id
        ]

        if not agent_experiences:
            return 0.5  # デフォルト値

        # 成功率を計算
        success_count = sum(
            1 for exp in agent_experiences
            if exp.get("success") is True
            or exp.get("result_status") == "success"
            or exp.get("user_feedback") == "positive"
        )

        return success_count / len(agent_experiences)

    def _get_activity_score(self, agent_id: str) -> float:
        """最近のアクティビティスコアを取得

        負荷分散のため、最近の使用頻度が低いエージェントに高いスコアを与える。
        Phase 2 MVP ではシンプルな実装。

        Args:
            agent_id: エージェントID

        Returns:
            0.0-1.0 のスコア（使用頻度が低いほど高い）
        """
        history = self._execution_history.get(agent_id, [])

        if not history:
            return 1.0  # 履歴なし = 最も優先

        # 直近1時間のタスク数をカウント
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(
            1 for h in history
            if h.get("timestamp", datetime.min) > one_hour_ago
        )

        # 5タスク以上で0.0、0タスクで1.0
        return max(0.0, 1.0 - (recent_count / 5))

    def _match_perspectives(
        self,
        perspectives: List[str],
        task_summary: str,
    ) -> float:
        """観点のマッチ度を計算

        エージェントの観点とタスクの関連性を評価。

        Args:
            perspectives: エージェントの観点リスト
            task_summary: タスクの概要

        Returns:
            0.0-1.0 のマッチスコア
        """
        if not perspectives:
            return 0.0

        matched_count = 0

        for perspective in perspectives:
            # 観点がタスクに直接含まれているか
            if perspective in task_summary:
                matched_count += 1
                continue

            # キーワード辞書によるマッチ
            keywords = self._PERSPECTIVE_KEYWORDS.get(perspective, [])
            for keyword in keywords:
                if keyword in task_summary:
                    matched_count += 1
                    break

        return matched_count / len(perspectives)

    def _generate_selection_reason(
        self,
        agent: AgentDefinition,
        task_summary: str,
    ) -> str:
        """選択理由を生成

        なぜこのエージェントが選択されたかの説明を生成。

        Args:
            agent: 選択されたエージェント
            task_summary: タスクの概要

        Returns:
            選択理由の文字列
        """
        # マッチした capabilities を抽出
        matched_caps = []
        task_lower = task_summary.lower()

        for capability in agent.capabilities:
            cap_lower = capability.lower()
            if cap_lower in task_lower:
                matched_caps.append(capability)
                continue

            keywords = self._CAPABILITY_KEYWORDS.get(cap_lower, [])
            for keyword in keywords:
                if keyword in task_summary:
                    matched_caps.append(capability)
                    break

        if matched_caps:
            caps_str = ", ".join(matched_caps[:3])
            return f"{agent.name}は{caps_str}の能力を持ち、このタスクに適しています"

        return f"{agent.name}（{agent.role[:30]}...）がこのタスクを担当します"

    def _calculate_confidence(self, candidates: List[Dict]) -> float:
        """確信度を計算

        上位2つのスコア差から確信度を算出。
        差が大きいほど確信度が高い。

        Args:
            candidates: スコア順にソートされた候補リスト

        Returns:
            0.0-1.0 の確信度
        """
        if len(candidates) < 2:
            return 1.0 if candidates else 0.0

        top_score = candidates[0]["score"]
        second_score = candidates[1]["score"]

        if top_score == 0:
            return 0.0

        # スコア差を確信度に変換
        # 差が0.2以上で確信度1.0
        diff = top_score - second_score
        return min(diff / 0.2, 1.0)

    def record_execution(
        self,
        agent_id: str,
        success: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """実行履歴を記録

        負荷分散とスコア計算に使用。

        Args:
            agent_id: 実行したエージェントのID
            success: 成功したかどうか
            timestamp: 実行時刻（省略時は現在時刻）
        """
        if agent_id not in self._execution_history:
            self._execution_history[agent_id] = []

        self._execution_history[agent_id].append({
            "timestamp": timestamp or datetime.now(),
            "success": success,
        })

        # 古い履歴を削除（24時間以上前）
        one_day_ago = datetime.now() - timedelta(days=1)
        self._execution_history[agent_id] = [
            h for h in self._execution_history[agent_id]
            if h.get("timestamp", datetime.min) > one_day_ago
        ]

    def clear_execution_history(self) -> None:
        """実行履歴をクリア（テスト用）"""
        self._execution_history.clear()
