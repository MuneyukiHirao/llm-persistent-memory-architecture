# src/monitoring/websocket_server.py
"""WebSocketサーバーモジュール

Phase 3 MVP のリアルタイム通知基盤。
FastAPI WebSocket互換インターフェースを採用し、依存性を分離。

実装仕様: docs/phase3-implementation-spec.ja.md セクション5.5

機能:
- クライアントとのWebSocket接続管理
- セッション単位での購読/購読解除
- リアルタイムイベント通知（進捗更新、タスク開始/完了等）
- ユーザー単位でのブロードキャスト
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable
import asyncio
import json
import logging

from src.config.phase3_config import (
    Phase3Config,
    WEBSOCKET_EVENT_TYPES,
    WEBSOCKET_CONFIG,
)


logger = logging.getLogger(__name__)


# =============================================================================
# WebSocket 抽象インターフェース（依存性分離）
# =============================================================================


class WebSocketError(Exception):
    """WebSocket関連のエラー"""
    pass


class WebSocketDisconnect(WebSocketError):
    """WebSocket切断エラー"""
    pass


class WebSocketConnectionLimitError(WebSocketError):
    """接続上限エラー"""
    pass


@runtime_checkable
class WebSocketProtocol(Protocol):
    """WebSocket インターフェース（FastAPI WebSocket互換）

    FastAPI の WebSocket クラスと互換性のあるインターフェース。
    テスト時やFastAPI以外の環境でもモック可能。
    """

    async def accept(self) -> None:
        """接続を受け入れる"""
        ...

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """接続を閉じる"""
        ...

    async def send_json(self, data: Dict[str, Any]) -> None:
        """JSON形式でデータを送信"""
        ...

    async def receive_json(self) -> Dict[str, Any]:
        """JSON形式でデータを受信"""
        ...


@dataclass
class WebSocketMessage:
    """WebSocketメッセージ

    サーバーからクライアントへ送信されるメッセージの構造。
    """
    event: str
    session_id: Optional[str]
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "event": self.event,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# =============================================================================
# WebSocketServer 本体
# =============================================================================


class WebSocketServer:
    """WebSocketサーバー

    リアルタイムで進捗やイベントをクライアントに通知する。

    仕様書参照: docs/phase3-implementation-spec.ja.md セクション5.5

    イベントフロー:
    1. クライアントがWebSocket接続
    2. セッション購読（subscribe）
    3. サーバーからイベント通知（task_started, progress_update等）
    4. 購読解除（unsubscribe）または切断

    使用例:
        server = WebSocketServer(config)

        # 接続ハンドリング（FastAPIルーターから呼び出し）
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            await server.handle_connection(websocket, user_id)

        # イベント通知（タスク処理から呼び出し）
        await server.broadcast(session_id, "task_started", {"task_id": "xxx"})

    Attributes:
        config: Phase3Config インスタンス
        _connections: セッションID → WebSocket接続のマッピング
        _user_connections: ユーザーID → WebSocket接続のマッピング
        _websocket_to_sessions: WebSocket → 購読中セッションIDのマッピング
        _websocket_to_user: WebSocket → ユーザーIDのマッピング
        _lock: 接続管理用の非同期ロック
    """

    def __init__(self, config: Phase3Config) -> None:
        """初期化

        Args:
            config: Phase3Config インスタンス
        """
        self.config = config

        # セッション購読管理: session_id -> Set[WebSocket]
        self._connections: Dict[str, Set[WebSocketProtocol]] = {}

        # ユーザー接続管理: user_id -> Set[WebSocket]
        self._user_connections: Dict[str, Set[WebSocketProtocol]] = {}

        # 逆引き用: WebSocket -> Set[session_id]
        self._websocket_to_sessions: Dict[WebSocketProtocol, Set[str]] = {}

        # WebSocket -> user_id
        self._websocket_to_user: Dict[WebSocketProtocol, str] = {}

        # 並行アクセス制御
        self._lock = asyncio.Lock()

        logger.info(
            "WebSocketServer initialized (max_connections=%d, ping_interval=%d)",
            config.websocket_max_connections,
            config.websocket_ping_interval,
        )

    # =========================================================================
    # 接続管理メソッド（async）
    # =========================================================================

    async def handle_connection(
        self,
        websocket: WebSocketProtocol,
        user_id: str,
    ) -> None:
        """WebSocket接続を処理

        接続の受け入れから切断までのライフサイクルを管理。
        受信ループを実行し、クライアントからのメッセージを処理する。

        Args:
            websocket: WebSocket接続
            user_id: ユーザーID

        Raises:
            WebSocketConnectionLimitError: 接続上限に達した場合
        """
        # 接続上限チェック
        total_connections = self.get_connection_count()
        if total_connections >= self.config.websocket_max_connections:
            logger.warning(
                "Connection limit reached (current=%d, max=%d), rejecting user=%s",
                total_connections,
                self.config.websocket_max_connections,
                user_id,
            )
            raise WebSocketConnectionLimitError(
                f"Connection limit reached: {self.config.websocket_max_connections}"
            )

        # ユーザーあたりの接続数チェック
        max_per_user = WEBSOCKET_CONFIG.get("max_connections_per_user", 5)
        async with self._lock:
            user_conn_count = len(self._user_connections.get(user_id, set()))
            if user_conn_count >= max_per_user:
                logger.warning(
                    "User connection limit reached (user=%s, current=%d, max=%d)",
                    user_id,
                    user_conn_count,
                    max_per_user,
                )
                raise WebSocketConnectionLimitError(
                    f"User connection limit reached: {max_per_user}"
                )

        # 接続受け入れ
        await websocket.accept()

        # ユーザー接続を記録
        async with self._lock:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(websocket)

            self._websocket_to_sessions[websocket] = set()
            self._websocket_to_user[websocket] = user_id

        logger.info(
            "WebSocket connection accepted (user=%s, total_connections=%d)",
            user_id,
            self.get_connection_count(),
        )

        try:
            # 受信ループ
            while True:
                try:
                    message = await websocket.receive_json()
                    await self._handle_message(websocket, user_id, message)
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected (user=%s)", user_id)
                    break
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON received from user=%s: %s",
                        user_id,
                        str(e),
                    )
                    # 不正なJSONは無視して継続
                except Exception as e:
                    logger.error(
                        "Error receiving message from user=%s: %s",
                        user_id,
                        str(e),
                    )
                    break
        finally:
            await self._cleanup_connection(websocket, user_id)

    async def _handle_message(
        self,
        websocket: WebSocketProtocol,
        user_id: str,
        message: Dict[str, Any],
    ) -> None:
        """メッセージを処理

        クライアントからのメッセージを処理し、適切なアクションを実行。

        サポートするアクション:
        - subscribe: セッション購読
        - unsubscribe: セッション購読解除
        - ping: ハートビート応答

        Args:
            websocket: WebSocket接続
            user_id: ユーザーID
            message: 受信したメッセージ（JSON形式）
        """
        action = message.get("action")

        if action == "subscribe":
            session_id = message.get("session_id")
            if session_id:
                await self._subscribe(websocket, session_id)
                logger.debug(
                    "User %s subscribed to session %s",
                    user_id,
                    session_id,
                )
            else:
                logger.warning(
                    "Subscribe action without session_id from user=%s",
                    user_id,
                )

        elif action == "unsubscribe":
            session_id = message.get("session_id")
            if session_id:
                await self._unsubscribe(websocket, session_id)
                logger.debug(
                    "User %s unsubscribed from session %s",
                    user_id,
                    session_id,
                )

        elif action == "ping":
            # ハートビート応答
            try:
                await websocket.send_json({
                    "event": "pong",
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.warning("Failed to send pong to user=%s: %s", user_id, str(e))

        else:
            logger.debug(
                "Unknown action '%s' from user=%s",
                action,
                user_id,
            )

    async def _subscribe(
        self,
        websocket: WebSocketProtocol,
        session_id: str,
    ) -> None:
        """セッション購読

        指定されたセッションのイベント通知を購読する。

        Args:
            websocket: WebSocket接続
            session_id: セッションID
        """
        async with self._lock:
            # セッション→接続のマッピングに追加
            if session_id not in self._connections:
                self._connections[session_id] = set()
            self._connections[session_id].add(websocket)

            # 逆引き用にも追加
            if websocket in self._websocket_to_sessions:
                self._websocket_to_sessions[websocket].add(session_id)

        # 購読成功を通知
        try:
            await websocket.send_json({
                "event": "subscribed",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.warning(
                "Failed to send subscription confirmation for session=%s: %s",
                session_id,
                str(e),
            )

    async def _unsubscribe(
        self,
        websocket: WebSocketProtocol,
        session_id: str,
    ) -> None:
        """セッション購読解除

        指定されたセッションのイベント通知購読を解除する。

        Args:
            websocket: WebSocket接続
            session_id: セッションID
        """
        async with self._lock:
            # セッション→接続のマッピングから削除
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                # 空になったら削除
                if not self._connections[session_id]:
                    del self._connections[session_id]

            # 逆引き用からも削除
            if websocket in self._websocket_to_sessions:
                self._websocket_to_sessions[websocket].discard(session_id)

        # 購読解除を通知
        try:
            await websocket.send_json({
                "event": "unsubscribed",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.warning(
                "Failed to send unsubscription confirmation for session=%s: %s",
                session_id,
                str(e),
            )

    async def _cleanup_connection(
        self,
        websocket: WebSocketProtocol,
        user_id: str,
    ) -> None:
        """接続クリーンアップ

        切断された接続に関連するすべてのデータを削除する。

        Args:
            websocket: WebSocket接続
            user_id: ユーザーID
        """
        async with self._lock:
            # 購読していたセッションから削除
            if websocket in self._websocket_to_sessions:
                subscribed_sessions = self._websocket_to_sessions[websocket].copy()
                for session_id in subscribed_sessions:
                    if session_id in self._connections:
                        self._connections[session_id].discard(websocket)
                        if not self._connections[session_id]:
                            del self._connections[session_id]
                del self._websocket_to_sessions[websocket]

            # ユーザー接続から削除
            if user_id in self._user_connections:
                self._user_connections[user_id].discard(websocket)
                if not self._user_connections[user_id]:
                    del self._user_connections[user_id]

            # WebSocket→ユーザーのマッピングから削除
            if websocket in self._websocket_to_user:
                del self._websocket_to_user[websocket]

        logger.info(
            "WebSocket connection cleaned up (user=%s, remaining_connections=%d)",
            user_id,
            self.get_connection_count(),
        )

    # =========================================================================
    # 通知メソッド（async）
    # =========================================================================

    async def broadcast(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> int:
        """セッション購読者にイベントをブロードキャスト

        指定されたセッションを購読しているすべてのクライアントに
        イベントを通知する。

        Args:
            session_id: セッションID
            event_type: イベントタイプ（WEBSOCKET_EVENT_TYPESの値）
            data: イベントデータ

        Returns:
            int: 送信に成功した接続数

        使用例:
            await server.broadcast(
                session_id="xxx-xxx",
                event_type="task_started",
                data={"task_id": "task_1", "agent_id": "agent_a"}
            )
        """
        message = WebSocketMessage(
            event=event_type,
            session_id=session_id,
            data=data,
        )

        async with self._lock:
            connections = self._connections.get(session_id, set()).copy()

        if not connections:
            logger.debug(
                "No subscribers for session=%s, event=%s",
                session_id,
                event_type,
            )
            return 0

        success_count = 0
        failed_websockets: Set[WebSocketProtocol] = set()

        for ws in connections:
            try:
                await ws.send_json(message.to_dict())
                success_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to send event to websocket for session=%s: %s",
                    session_id,
                    str(e),
                )
                failed_websockets.add(ws)

        # 送信失敗した接続をクリーンアップ（別タスクで非同期実行）
        if failed_websockets:
            for ws in failed_websockets:
                user_id = self._websocket_to_user.get(ws)
                if user_id:
                    asyncio.create_task(self._cleanup_connection(ws, user_id))

        logger.debug(
            "Broadcast to session=%s, event=%s, success=%d/%d",
            session_id,
            event_type,
            success_count,
            len(connections),
        )

        return success_count

    async def broadcast_to_user(
        self,
        user_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> int:
        """ユーザーにイベントをブロードキャスト

        指定されたユーザーのすべての接続にイベントを通知する。

        Args:
            user_id: ユーザーID
            event_type: イベントタイプ
            data: イベントデータ

        Returns:
            int: 送信に成功した接続数

        使用例:
            await server.broadcast_to_user(
                user_id="user_123",
                event_type="alert",
                data={"message": "Session expired", "severity": "warning"}
            )
        """
        message = WebSocketMessage(
            event=event_type,
            session_id=None,
            data=data,
        )

        async with self._lock:
            connections = self._user_connections.get(user_id, set()).copy()

        if not connections:
            logger.debug(
                "No connections for user=%s, event=%s",
                user_id,
                event_type,
            )
            return 0

        success_count = 0
        failed_websockets: Set[WebSocketProtocol] = set()

        for ws in connections:
            try:
                await ws.send_json(message.to_dict())
                success_count += 1
            except Exception as e:
                logger.warning(
                    "Failed to send event to user=%s: %s",
                    user_id,
                    str(e),
                )
                failed_websockets.add(ws)

        # 送信失敗した接続をクリーンアップ
        if failed_websockets:
            for ws in failed_websockets:
                asyncio.create_task(self._cleanup_connection(ws, user_id))

        logger.debug(
            "Broadcast to user=%s, event=%s, success=%d/%d",
            user_id,
            event_type,
            success_count,
            len(connections),
        )

        return success_count

    # =========================================================================
    # 統計メソッド
    # =========================================================================

    def get_connection_count(self) -> int:
        """総接続数を取得

        現在アクティブなWebSocket接続の総数を返す。

        Returns:
            int: 総接続数
        """
        return len(self._websocket_to_user)

    def get_session_subscribers(self, session_id: str) -> int:
        """セッション購読者数を取得

        指定されたセッションを購読している接続の数を返す。

        Args:
            session_id: セッションID

        Returns:
            int: 購読者数
        """
        return len(self._connections.get(session_id, set()))

    def get_user_connections(self, user_id: str) -> int:
        """ユーザーの接続数を取得

        指定されたユーザーの接続数を返す。

        Args:
            user_id: ユーザーID

        Returns:
            int: 接続数
        """
        return len(self._user_connections.get(user_id, set()))

    def get_active_sessions(self) -> int:
        """アクティブセッション数を取得

        少なくとも1人以上の購読者がいるセッションの数を返す。

        Returns:
            int: アクティブセッション数
        """
        return len(self._connections)

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得

        WebSocketサーバーの現在の状態を統計情報として返す。

        Returns:
            Dict[str, Any]: 統計情報
        """
        return {
            "total_connections": self.get_connection_count(),
            "active_sessions": self.get_active_sessions(),
            "unique_users": len(self._user_connections),
            "max_connections": self.config.websocket_max_connections,
            "utilization": (
                self.get_connection_count() / self.config.websocket_max_connections
                if self.config.websocket_max_connections > 0
                else 0.0
            ),
        }


# =============================================================================
# ヘルパー関数
# =============================================================================


def validate_event_type(event_type: str) -> bool:
    """イベントタイプの妥当性を検証

    Args:
        event_type: イベントタイプ

    Returns:
        bool: 有効なイベントタイプの場合True
    """
    return event_type in WEBSOCKET_EVENT_TYPES


def get_event_description(event_type: str) -> Optional[str]:
    """イベントタイプの説明を取得

    Args:
        event_type: イベントタイプ

    Returns:
        Optional[str]: イベントの説明、または無効な場合はNone
    """
    return WEBSOCKET_EVENT_TYPES.get(event_type)
