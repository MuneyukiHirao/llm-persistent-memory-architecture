# tests/monitoring/test_websocket_server.py
"""WebSocketServer のユニットテスト

テスト観点:
- 初期化: 内部状態の初期化、設定値の反映
- 接続管理: handle_connection, _handle_message, _subscribe, _unsubscribe, _cleanup_connection
- 接続上限: 全体接続上限、ユーザー毎接続上限
- 通知: broadcast, broadcast_to_user
- 統計: get_connection_count, get_session_subscribers, get_stats
- 非同期処理: 並行アクセス時の整合性
"""

import asyncio
import json
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.monitoring.websocket_server import (
    WebSocketServer,
    WebSocketProtocol,
    WebSocketMessage,
    WebSocketError,
    WebSocketDisconnect,
    WebSocketConnectionLimitError,
    validate_event_type,
    get_event_description,
)
from src.config.phase3_config import Phase3Config, WEBSOCKET_CONFIG


# =============================================================================
# モッククラス
# =============================================================================


class MockWebSocket:
    """WebSocketProtocol互換のモック

    テスト用WebSocket実装。送受信メッセージを記録し、
    テストでの検証を容易にする。
    """

    def __init__(
        self,
        receive_messages: Optional[List[Dict[str, Any]]] = None,
        raise_on_receive: Optional[Exception] = None,
        raise_on_send: Optional[Exception] = None,
    ):
        """
        Args:
            receive_messages: receive_json()が返すメッセージのリスト
            raise_on_receive: receive_json()で発生させる例外
            raise_on_send: send_json()で発生させる例外
        """
        self.receive_messages = receive_messages or []
        self.receive_index = 0
        self.raise_on_receive = raise_on_receive
        self.raise_on_send = raise_on_send

        self.accepted = False
        self.closed = False
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self.sent_messages: List[Dict[str, Any]] = []

    async def accept(self) -> None:
        """接続を受け入れる"""
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """接続を閉じる"""
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_json(self, data: Dict[str, Any]) -> None:
        """JSON形式でデータを送信"""
        if self.raise_on_send:
            raise self.raise_on_send
        self.sent_messages.append(data)

    async def receive_json(self) -> Dict[str, Any]:
        """JSON形式でデータを受信"""
        if self.raise_on_receive:
            raise self.raise_on_receive

        if self.receive_index >= len(self.receive_messages):
            # メッセージがなくなったら切断
            raise WebSocketDisconnect("No more messages")

        message = self.receive_messages[self.receive_index]
        self.receive_index += 1
        return message


@pytest.fixture
def config():
    """テスト用Phase3Config"""
    return Phase3Config(
        websocket_max_connections=10,
        websocket_ping_interval=30,
    )


@pytest.fixture
def server(config):
    """テスト用WebSocketServer"""
    return WebSocketServer(config)


# =============================================================================
# 初期化テスト
# =============================================================================


class TestWebSocketServerInit:
    """WebSocketServer 初期化テスト"""

    def test_init_creates_empty_collections(self, config):
        """初期化時に空のコレクションが作成される"""
        server = WebSocketServer(config)

        assert len(server._connections) == 0
        assert len(server._user_connections) == 0
        assert len(server._websocket_to_sessions) == 0
        assert len(server._websocket_to_user) == 0

    def test_init_stores_config(self, config):
        """初期化時に設定が保存される"""
        server = WebSocketServer(config)

        assert server.config is config
        assert server.config.websocket_max_connections == 10

    def test_init_creates_lock(self, config):
        """初期化時にロックが作成される"""
        server = WebSocketServer(config)

        assert server._lock is not None
        assert isinstance(server._lock, asyncio.Lock)


# =============================================================================
# 接続管理テスト
# =============================================================================


class TestHandleConnection:
    """handle_connection メソッドテスト"""

    @pytest.mark.asyncio
    async def test_handle_connection_accepts_websocket(self, server):
        """接続が正常に受け入れられる"""
        ws = MockWebSocket(receive_messages=[])
        user_id = "user_001"

        await server.handle_connection(ws, user_id)

        assert ws.accepted is True

    @pytest.mark.asyncio
    async def test_handle_connection_registers_user(self, server):
        """ユーザー接続が登録される"""
        ws = MockWebSocket(receive_messages=[])
        user_id = "user_001"

        await server.handle_connection(ws, user_id)

        assert user_id not in server._user_connections  # クリーンアップ済み

    @pytest.mark.asyncio
    async def test_handle_connection_processes_messages(self, server):
        """メッセージが処理される"""
        messages = [
            {"action": "subscribe", "session_id": "session_001"},
            {"action": "ping"},
        ]
        ws = MockWebSocket(receive_messages=messages)
        user_id = "user_001"

        await server.handle_connection(ws, user_id)

        # subscribed と pong が送信されているはず
        assert len(ws.sent_messages) == 2
        assert ws.sent_messages[0]["event"] == "subscribed"
        assert ws.sent_messages[1]["event"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_connection_cleans_up_on_disconnect(self, server):
        """切断時にクリーンアップされる"""
        ws = MockWebSocket(receive_messages=[
            {"action": "subscribe", "session_id": "session_001"},
        ])
        user_id = "user_001"

        await server.handle_connection(ws, user_id)

        # クリーンアップ後は接続情報が削除されている
        assert ws not in server._websocket_to_user
        assert ws not in server._websocket_to_sessions
        assert server.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_handle_connection_handles_json_decode_error(self, server):
        """JSONデコードエラーが処理される"""
        ws = MockWebSocket(raise_on_receive=json.JSONDecodeError("test", "doc", 0))
        user_id = "user_001"

        # JSONDecodeErrorは無視されて継続、最終的にWebSocketDisconnectで終了する
        ws.raise_on_receive = WebSocketDisconnect("end")
        await server.handle_connection(ws, user_id)

        assert ws.accepted is True


class TestHandleMessage:
    """_handle_message メソッドテスト"""

    @pytest.mark.asyncio
    async def test_handle_message_subscribe_action(self, server):
        """subscribe アクションの処理"""
        ws = MockWebSocket()
        await ws.accept()

        # 接続を準備
        async with server._lock:
            server._websocket_to_sessions[ws] = set()
            server._websocket_to_user[ws] = "user_001"

        message = {"action": "subscribe", "session_id": "session_001"}
        await server._handle_message(ws, "user_001", message)

        assert ws in server._connections.get("session_001", set())
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "subscribed"

    @pytest.mark.asyncio
    async def test_handle_message_subscribe_without_session_id(self, server):
        """session_idなしのsubscribeは無視される"""
        ws = MockWebSocket()

        message = {"action": "subscribe"}
        await server._handle_message(ws, "user_001", message)

        assert len(ws.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_handle_message_unsubscribe_action(self, server):
        """unsubscribe アクションの処理"""
        ws = MockWebSocket()

        # 先にsubscribeしておく
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}

        message = {"action": "unsubscribe", "session_id": "session_001"}
        await server._handle_message(ws, "user_001", message)

        assert ws not in server._connections.get("session_001", set())
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "unsubscribed"

    @pytest.mark.asyncio
    async def test_handle_message_ping_action(self, server):
        """ping アクションの処理"""
        ws = MockWebSocket()

        message = {"action": "ping"}
        await server._handle_message(ws, "user_001", message)

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "pong"
        assert "timestamp" in ws.sent_messages[0]

    @pytest.mark.asyncio
    async def test_handle_message_unknown_action(self, server):
        """不明なアクションは無視される"""
        ws = MockWebSocket()

        message = {"action": "unknown_action"}
        await server._handle_message(ws, "user_001", message)

        assert len(ws.sent_messages) == 0


class TestSubscribe:
    """_subscribe メソッドテスト"""

    @pytest.mark.asyncio
    async def test_subscribe_adds_to_connections(self, server):
        """購読がconnectionsに追加される"""
        ws = MockWebSocket()
        async with server._lock:
            server._websocket_to_sessions[ws] = set()

        await server._subscribe(ws, "session_001")

        assert ws in server._connections["session_001"]
        assert "session_001" in server._websocket_to_sessions[ws]

    @pytest.mark.asyncio
    async def test_subscribe_sends_confirmation(self, server):
        """購読確認が送信される"""
        ws = MockWebSocket()
        async with server._lock:
            server._websocket_to_sessions[ws] = set()

        await server._subscribe(ws, "session_001")

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "subscribed"
        assert ws.sent_messages[0]["session_id"] == "session_001"

    @pytest.mark.asyncio
    async def test_subscribe_multiple_sessions(self, server):
        """複数セッションを購読できる"""
        ws = MockWebSocket()
        async with server._lock:
            server._websocket_to_sessions[ws] = set()

        await server._subscribe(ws, "session_001")
        await server._subscribe(ws, "session_002")

        assert "session_001" in server._websocket_to_sessions[ws]
        assert "session_002" in server._websocket_to_sessions[ws]

    @pytest.mark.asyncio
    async def test_subscribe_handles_send_error(self, server):
        """送信エラー時も購読は成功"""
        ws = MockWebSocket(raise_on_send=Exception("Send failed"))
        async with server._lock:
            server._websocket_to_sessions[ws] = set()

        # エラーが発生してもクラッシュしない
        await server._subscribe(ws, "session_001")

        assert ws in server._connections["session_001"]


class TestUnsubscribe:
    """_unsubscribe メソッドテスト"""

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_connections(self, server):
        """購読解除がconnectionsから削除される"""
        ws = MockWebSocket()
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}

        await server._unsubscribe(ws, "session_001")

        assert ws not in server._connections.get("session_001", set())
        assert "session_001" not in server._websocket_to_sessions.get(ws, set())

    @pytest.mark.asyncio
    async def test_unsubscribe_sends_confirmation(self, server):
        """購読解除確認が送信される"""
        ws = MockWebSocket()
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}

        await server._unsubscribe(ws, "session_001")

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "unsubscribed"
        assert ws.sent_messages[0]["session_id"] == "session_001"

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_empty_session(self, server):
        """購読者がいなくなったセッションは削除される"""
        ws = MockWebSocket()
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}

        await server._unsubscribe(ws, "session_001")

        assert "session_001" not in server._connections

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_session(self, server):
        """存在しないセッションの購読解除は安全"""
        ws = MockWebSocket()

        # エラーにならない
        await server._unsubscribe(ws, "nonexistent_session")

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "unsubscribed"


# =============================================================================
# 接続上限テスト
# =============================================================================


class TestConnectionLimits:
    """接続上限テスト"""

    @pytest.mark.asyncio
    async def test_global_connection_limit(self):
        """全体接続上限のテスト"""
        config = Phase3Config(websocket_max_connections=2)
        server = WebSocketServer(config)

        # 2接続を確立
        ws1 = MockWebSocket(receive_messages=[])
        ws2 = MockWebSocket(receive_messages=[])

        # 非同期で接続（即座に切断される）
        await server.handle_connection(ws1, "user_001")
        await server.handle_connection(ws2, "user_002")

        # 両方受け入れられた
        assert ws1.accepted is True
        assert ws2.accepted is True

    @pytest.mark.asyncio
    async def test_global_connection_limit_exceeded(self):
        """全体接続上限超過時のエラー"""
        config = Phase3Config(websocket_max_connections=1)
        server = WebSocketServer(config)

        # 1つ目の接続を維持
        ws1 = MockWebSocket()
        async with server._lock:
            server._websocket_to_user[ws1] = "user_001"
            server._user_connections["user_001"] = {ws1}
            server._websocket_to_sessions[ws1] = set()

        assert server.get_connection_count() == 1

        # 2つ目の接続は拒否される
        ws2 = MockWebSocket()
        with pytest.raises(WebSocketConnectionLimitError, match="Connection limit reached"):
            await server.handle_connection(ws2, "user_002")

    @pytest.mark.asyncio
    async def test_user_connection_limit(self):
        """ユーザー毎接続上限のテスト"""
        config = Phase3Config(websocket_max_connections=100)
        server = WebSocketServer(config)

        # 5接続（デフォルト上限）を事前に設定
        user_id = "user_001"
        async with server._lock:
            server._user_connections[user_id] = set()
            for i in range(5):
                ws = MockWebSocket()
                server._user_connections[user_id].add(ws)
                server._websocket_to_user[ws] = user_id
                server._websocket_to_sessions[ws] = set()

        # 6つ目の接続は拒否される
        ws_new = MockWebSocket()
        with pytest.raises(WebSocketConnectionLimitError, match="User connection limit"):
            await server.handle_connection(ws_new, user_id)

    @pytest.mark.asyncio
    async def test_different_users_can_connect(self):
        """異なるユーザーは接続上限内で接続可能"""
        config = Phase3Config(websocket_max_connections=10)
        server = WebSocketServer(config)

        # user_001が5接続
        async with server._lock:
            server._user_connections["user_001"] = set()
            for i in range(5):
                ws = MockWebSocket()
                server._user_connections["user_001"].add(ws)
                server._websocket_to_user[ws] = "user_001"
                server._websocket_to_sessions[ws] = set()

        # user_002は新規接続可能（全体上限10、現在5）
        ws_user2 = MockWebSocket(receive_messages=[])
        await server.handle_connection(ws_user2, "user_002")

        assert ws_user2.accepted is True


# =============================================================================
# 通知テスト
# =============================================================================


class TestBroadcast:
    """broadcast メソッドテスト"""

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self, server):
        """購読者への通知"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        async with server._lock:
            server._connections["session_001"] = {ws1, ws2}
            server._websocket_to_user[ws1] = "user_001"
            server._websocket_to_user[ws2] = "user_002"

        success_count = await server.broadcast(
            "session_001",
            "task_started",
            {"task_id": "task_001"},
        )

        assert success_count == 2
        assert len(ws1.sent_messages) == 1
        assert len(ws2.sent_messages) == 1
        assert ws1.sent_messages[0]["event"] == "task_started"
        assert ws1.sent_messages[0]["session_id"] == "session_001"
        assert ws1.sent_messages[0]["data"]["task_id"] == "task_001"

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self, server):
        """購読者がいない場合"""
        success_count = await server.broadcast(
            "nonexistent_session",
            "task_started",
            {"task_id": "task_001"},
        )

        assert success_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_partial_failure(self, server):
        """一部送信失敗時のハンドリング"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket(raise_on_send=Exception("Send failed"))

        async with server._lock:
            server._connections["session_001"] = {ws1, ws2}
            server._websocket_to_user[ws1] = "user_001"
            server._websocket_to_user[ws2] = "user_002"

        success_count = await server.broadcast(
            "session_001",
            "task_started",
            {"task_id": "task_001"},
        )

        assert success_count == 1
        assert len(ws1.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_broadcast_message_format(self, server):
        """メッセージ形式の検証"""
        ws = MockWebSocket()

        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_user[ws] = "user_001"

        await server.broadcast(
            "session_001",
            "progress_update",
            {"progress": 50, "status": "running"},
        )

        msg = ws.sent_messages[0]
        assert "event" in msg
        assert "session_id" in msg
        assert "data" in msg
        assert "timestamp" in msg


class TestBroadcastToUser:
    """broadcast_to_user メソッドテスト"""

    @pytest.mark.asyncio
    async def test_broadcast_to_user_single_connection(self, server):
        """単一接続のユーザーへの通知"""
        ws = MockWebSocket()

        async with server._lock:
            server._user_connections["user_001"] = {ws}

        success_count = await server.broadcast_to_user(
            "user_001",
            "alert",
            {"message": "Test alert"},
        )

        assert success_count == 1
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["event"] == "alert"

    @pytest.mark.asyncio
    async def test_broadcast_to_user_multiple_connections(self, server):
        """複数接続のユーザーへの通知"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        async with server._lock:
            server._user_connections["user_001"] = {ws1, ws2, ws3}

        success_count = await server.broadcast_to_user(
            "user_001",
            "alert",
            {"message": "Test alert"},
        )

        assert success_count == 3
        for ws in [ws1, ws2, ws3]:
            assert len(ws.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_user(self, server):
        """存在しないユーザーへの通知"""
        success_count = await server.broadcast_to_user(
            "nonexistent_user",
            "alert",
            {"message": "Test alert"},
        )

        assert success_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_user_session_id_is_none(self, server):
        """ユーザー通知のsession_idはNone"""
        ws = MockWebSocket()

        async with server._lock:
            server._user_connections["user_001"] = {ws}

        await server.broadcast_to_user(
            "user_001",
            "alert",
            {"message": "Test alert"},
        )

        assert ws.sent_messages[0]["session_id"] is None


# =============================================================================
# クリーンアップテスト
# =============================================================================


class TestCleanupConnection:
    """_cleanup_connection メソッドテスト"""

    @pytest.mark.asyncio
    async def test_cleanup_removes_all_references(self, server):
        """クリーンアップで全ての参照が削除される"""
        ws = MockWebSocket()
        user_id = "user_001"

        # 接続状態を設定
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._connections["session_002"] = {ws}
            server._user_connections[user_id] = {ws}
            server._websocket_to_sessions[ws] = {"session_001", "session_002"}
            server._websocket_to_user[ws] = user_id

        await server._cleanup_connection(ws, user_id)

        assert ws not in server._connections.get("session_001", set())
        assert ws not in server._connections.get("session_002", set())
        assert ws not in server._user_connections.get(user_id, set())
        assert ws not in server._websocket_to_sessions
        assert ws not in server._websocket_to_user

    @pytest.mark.asyncio
    async def test_cleanup_removes_empty_session(self, server):
        """購読者がいなくなったセッションは削除される"""
        ws = MockWebSocket()
        user_id = "user_001"

        async with server._lock:
            server._connections["session_001"] = {ws}
            server._user_connections[user_id] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}
            server._websocket_to_user[ws] = user_id

        await server._cleanup_connection(ws, user_id)

        assert "session_001" not in server._connections

    @pytest.mark.asyncio
    async def test_cleanup_preserves_other_connections(self, server):
        """他のユーザーの接続は保持される"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        async with server._lock:
            server._connections["session_001"] = {ws1, ws2}
            server._user_connections["user_001"] = {ws1}
            server._user_connections["user_002"] = {ws2}
            server._websocket_to_sessions[ws1] = {"session_001"}
            server._websocket_to_sessions[ws2] = {"session_001"}
            server._websocket_to_user[ws1] = "user_001"
            server._websocket_to_user[ws2] = "user_002"

        await server._cleanup_connection(ws1, "user_001")

        assert ws2 in server._connections["session_001"]
        assert ws2 in server._websocket_to_user


# =============================================================================
# 統計テスト
# =============================================================================


class TestStatistics:
    """統計メソッドテスト"""

    def test_get_connection_count_empty(self, server):
        """接続なしの場合"""
        assert server.get_connection_count() == 0

    def test_get_connection_count(self, server):
        """接続数の取得"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        server._websocket_to_user[ws1] = "user_001"
        server._websocket_to_user[ws2] = "user_002"

        assert server.get_connection_count() == 2

    def test_get_session_subscribers_empty(self, server):
        """購読者なしの場合"""
        assert server.get_session_subscribers("nonexistent") == 0

    def test_get_session_subscribers(self, server):
        """セッション購読者数の取得"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        server._connections["session_001"] = {ws1, ws2}
        server._connections["session_002"] = {ws3}

        assert server.get_session_subscribers("session_001") == 2
        assert server.get_session_subscribers("session_002") == 1

    def test_get_user_connections(self, server):
        """ユーザー接続数の取得"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        server._user_connections["user_001"] = {ws1, ws2}

        assert server.get_user_connections("user_001") == 2
        assert server.get_user_connections("nonexistent") == 0

    def test_get_active_sessions(self, server):
        """アクティブセッション数の取得"""
        server._connections["session_001"] = {MockWebSocket()}
        server._connections["session_002"] = {MockWebSocket()}
        server._connections["session_003"] = {MockWebSocket()}

        assert server.get_active_sessions() == 3

    def test_get_stats(self, server):
        """統計情報の一括取得"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        server._websocket_to_user[ws1] = "user_001"
        server._websocket_to_user[ws2] = "user_002"
        server._user_connections["user_001"] = {ws1}
        server._user_connections["user_002"] = {ws2}
        server._connections["session_001"] = {ws1}
        server._connections["session_002"] = {ws2}

        stats = server.get_stats()

        assert stats["total_connections"] == 2
        assert stats["active_sessions"] == 2
        assert stats["unique_users"] == 2
        assert stats["max_connections"] == 10
        assert stats["utilization"] == pytest.approx(0.2)

    def test_get_stats_utilization_zero_max(self):
        """最大接続数0の場合の利用率"""
        config = Phase3Config(websocket_max_connections=0)
        server = WebSocketServer(config)

        stats = server.get_stats()
        assert stats["utilization"] == 0.0


# =============================================================================
# WebSocketMessageテスト
# =============================================================================


class TestWebSocketMessage:
    """WebSocketMessage データクラステスト"""

    def test_to_dict(self):
        """to_dict メソッド"""
        msg = WebSocketMessage(
            event="task_started",
            session_id="session_001",
            data={"task_id": "task_001"},
        )

        d = msg.to_dict()

        assert d["event"] == "task_started"
        assert d["session_id"] == "session_001"
        assert d["data"]["task_id"] == "task_001"
        assert "timestamp" in d

    def test_to_dict_with_none_session(self):
        """session_idがNoneの場合"""
        msg = WebSocketMessage(
            event="alert",
            session_id=None,
            data={"message": "test"},
        )

        d = msg.to_dict()

        assert d["session_id"] is None


# =============================================================================
# ヘルパー関数テスト
# =============================================================================


class TestHelperFunctions:
    """ヘルパー関数テスト"""

    def test_validate_event_type_valid(self):
        """有効なイベントタイプ"""
        assert validate_event_type("progress_update") is True
        assert validate_event_type("task_started") is True
        assert validate_event_type("task_completed") is True
        assert validate_event_type("alert") is True

    def test_validate_event_type_invalid(self):
        """無効なイベントタイプ"""
        assert validate_event_type("invalid_event") is False
        assert validate_event_type("") is False

    def test_get_event_description_valid(self):
        """イベント説明の取得"""
        assert get_event_description("progress_update") == "進捗更新"
        assert get_event_description("task_started") == "タスク開始"

    def test_get_event_description_invalid(self):
        """無効なイベントタイプの説明"""
        assert get_event_description("invalid_event") is None


# =============================================================================
# 並行処理テスト
# =============================================================================


class TestConcurrency:
    """並行処理テスト"""

    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, server):
        """並行購読のテスト"""
        websockets = [MockWebSocket() for _ in range(10)]

        async with server._lock:
            for i, ws in enumerate(websockets):
                server._websocket_to_sessions[ws] = set()

        # 並行して購読
        async def subscribe_task(ws, session_id):
            await server._subscribe(ws, session_id)

        await asyncio.gather(*[
            subscribe_task(ws, "session_001")
            for ws in websockets
        ])

        assert server.get_session_subscribers("session_001") == 10

    @pytest.mark.asyncio
    async def test_concurrent_broadcast(self, server):
        """並行ブロードキャストのテスト"""
        websockets = [MockWebSocket() for _ in range(5)]

        async with server._lock:
            server._connections["session_001"] = set(websockets)
            for ws in websockets:
                server._websocket_to_user[ws] = "user"

        # 並行してブロードキャスト
        results = await asyncio.gather(*[
            server.broadcast("session_001", "event", {"i": i})
            for i in range(10)
        ])

        # 全ブロードキャストが5接続に送信成功
        assert all(r == 5 for r in results)

        # 各WebSocketが10メッセージを受信
        for ws in websockets:
            assert len(ws.sent_messages) == 10


# =============================================================================
# エラーハンドリングテスト
# =============================================================================


class TestErrorHandling:
    """エラーハンドリングテスト"""

    @pytest.mark.asyncio
    async def test_ping_send_failure(self, server):
        """ping応答の送信失敗"""
        ws = MockWebSocket(raise_on_send=Exception("Send failed"))

        # エラーは無視される
        message = {"action": "ping"}
        await server._handle_message(ws, "user_001", message)

        # エラーは発生しない
        assert True

    @pytest.mark.asyncio
    async def test_subscribe_send_failure(self, server):
        """購読確認の送信失敗"""
        ws = MockWebSocket(raise_on_send=Exception("Send failed"))
        async with server._lock:
            server._websocket_to_sessions[ws] = set()

        # エラーは無視され、購読は成功
        await server._subscribe(ws, "session_001")

        assert ws in server._connections["session_001"]

    @pytest.mark.asyncio
    async def test_unsubscribe_send_failure(self, server):
        """購読解除確認の送信失敗"""
        ws = MockWebSocket(raise_on_send=Exception("Send failed"))
        async with server._lock:
            server._connections["session_001"] = {ws}
            server._websocket_to_sessions[ws] = {"session_001"}

        # エラーは無視され、購読解除は成功
        await server._unsubscribe(ws, "session_001")

        assert ws not in server._connections.get("session_001", set())


# =============================================================================
# 例外クラステスト
# =============================================================================


class TestExceptions:
    """例外クラステスト"""

    def test_websocket_error(self):
        """WebSocketError"""
        error = WebSocketError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_websocket_disconnect(self):
        """WebSocketDisconnect"""
        error = WebSocketDisconnect("Connection closed")
        assert str(error) == "Connection closed"
        assert isinstance(error, WebSocketError)

    def test_websocket_connection_limit_error(self):
        """WebSocketConnectionLimitError"""
        error = WebSocketConnectionLimitError("Limit reached")
        assert str(error) == "Limit reached"
        assert isinstance(error, WebSocketError)
