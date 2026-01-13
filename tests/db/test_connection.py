# DatabaseConnection のテスト
"""
DatabaseConnection クラスの単体テスト

テスト観点:
- DatabaseConnection インスタンス化
- コネクションプール（ThreadedConnectionPool）の動作
- コンテキストマネージャー（with文）の正常動作
- 接続エラー時の例外処理
- プールからの取得・返却
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import psycopg2
from psycopg2 import pool

from src.db.connection import (
    DatabaseConnection,
    get_database,
    close_database,
)


# =============================================================================
# DatabaseConnection インスタンス化のテスト
# =============================================================================


class TestDatabaseConnectionInitialization:
    """DatabaseConnection の初期化テスト"""

    def test_init_with_explicit_url(self):
        """明示的なdatabase_urlでインスタンス化できる"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        assert db.database_url == url
        assert db.min_connections == 1
        assert db.max_connections == 10

    def test_init_with_custom_pool_size(self):
        """カスタムのプールサイズでインスタンス化できる"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(
            database_url=url,
            min_connections=5,
            max_connections=20,
        )

        assert db.min_connections == 5
        assert db.max_connections == 20

    def test_init_from_environment_variable(self):
        """環境変数DATABASE_URLからURLを取得できる"""
        test_url = "postgresql://env:pass@localhost:5432/envdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            db = DatabaseConnection()

            assert db.database_url == test_url

    def test_init_explicit_url_overrides_env(self):
        """明示的なURLは環境変数より優先される"""
        explicit_url = "postgresql://explicit:pass@localhost:5432/explicitdb"
        env_url = "postgresql://env:pass@localhost:5432/envdb"

        with patch.dict(os.environ, {"DATABASE_URL": env_url}):
            db = DatabaseConnection(database_url=explicit_url)

            assert db.database_url == explicit_url

    def test_init_without_url_raises_value_error(self):
        """URLが設定されていない場合はValueErrorが発生"""
        with patch.dict(os.environ, {}, clear=True):
            # DATABASE_URL を削除
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]

            with pytest.raises(ValueError) as exc_info:
                DatabaseConnection()

            assert "DATABASE_URL" in str(exc_info.value)

    def test_init_pool_is_none_initially(self):
        """初期状態ではプールはNone（遅延初期化）"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        assert db._pool is None


# =============================================================================
# コネクションプールの動作テスト
# =============================================================================


class TestConnectionPool:
    """コネクションプールの動作テスト"""

    @pytest.fixture
    def mock_pool(self):
        """モック ThreadedConnectionPool"""
        mock = MagicMock(spec=pool.ThreadedConnectionPool)
        return mock

    @pytest.fixture
    def mock_connection(self):
        """モック psycopg2 connection"""
        mock = MagicMock()
        mock.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock

    def test_get_pool_creates_pool_on_first_call(self, mock_pool):
        """_get_pool()は初回呼び出し時にプールを作成する"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        with patch(
            "src.db.connection.pool.ThreadedConnectionPool", return_value=mock_pool
        ) as mock_class:
            result = db._get_pool()

            mock_class.assert_called_once_with(
                minconn=1,
                maxconn=10,
                dsn=url,
            )
            assert result == mock_pool

    def test_get_pool_reuses_existing_pool(self, mock_pool):
        """_get_pool()は既存のプールを再利用する"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)
        db._pool = mock_pool

        result = db._get_pool()

        assert result == mock_pool

    def test_get_pool_uses_custom_pool_size(self, mock_pool):
        """カスタムプールサイズが正しく設定される"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(
            database_url=url,
            min_connections=3,
            max_connections=15,
        )

        with patch(
            "src.db.connection.pool.ThreadedConnectionPool", return_value=mock_pool
        ) as mock_class:
            db._get_pool()

            mock_class.assert_called_once_with(
                minconn=3,
                maxconn=15,
                dsn=url,
            )


# =============================================================================
# コンテキストマネージャーの正常動作テスト
# =============================================================================


class TestContextManager:
    """コンテキストマネージャーの正常動作テスト"""

    @pytest.fixture
    def mock_pool(self):
        """モック ThreadedConnectionPool"""
        mock = MagicMock(spec=pool.ThreadedConnectionPool)
        return mock

    @pytest.fixture
    def mock_connection(self):
        """モック psycopg2 connection"""
        mock = MagicMock()
        mock_cursor = MagicMock()
        mock.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock

    @pytest.fixture
    def db_with_mock_pool(self, mock_pool, mock_connection):
        """モックプール付きのDatabaseConnection"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_pool.getconn.return_value = mock_connection
        db._pool = mock_pool

        return db, mock_pool, mock_connection

    def test_get_connection_yields_connection(self, db_with_mock_pool):
        """get_connection()はコネクションを返す"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with db.get_connection() as conn:
            assert conn == mock_connection

    def test_get_connection_commits_on_success(self, db_with_mock_pool):
        """正常終了時はcommitが呼ばれる"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with db.get_connection() as conn:
            pass  # 正常終了

        mock_connection.commit.assert_called_once()
        mock_connection.rollback.assert_not_called()

    def test_get_connection_auto_commit_false_skips_commit(self, db_with_mock_pool):
        """auto_commit=Falseの場合はcommitが呼ばれない"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with db.get_connection(auto_commit=False) as conn:
            pass  # 正常終了

        mock_connection.commit.assert_not_called()
        mock_connection.rollback.assert_not_called()

    def test_get_connection_rollbacks_on_exception(self, db_with_mock_pool):
        """例外発生時はrollbackが呼ばれる"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with pytest.raises(ValueError):
            with db.get_connection() as conn:
                raise ValueError("Test error")

        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()

    def test_get_connection_returns_connection_to_pool(self, db_with_mock_pool):
        """コンテキスト終了時にコネクションがプールに返却される"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with db.get_connection() as conn:
            pass

        mock_pool.putconn.assert_called_once_with(mock_connection)

    def test_get_connection_returns_to_pool_even_on_exception(self, db_with_mock_pool):
        """例外発生時もコネクションがプールに返却される"""
        db, mock_pool, mock_connection = db_with_mock_pool

        with pytest.raises(ValueError):
            with db.get_connection() as conn:
                raise ValueError("Test error")

        mock_pool.putconn.assert_called_once_with(mock_connection)

    def test_get_cursor_yields_cursor(self, db_with_mock_pool):
        """get_cursor()はカーソルを返す"""
        db, mock_pool, mock_connection = db_with_mock_pool
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with db.get_cursor() as cur:
            assert cur == mock_cursor

    def test_get_cursor_commits_on_success(self, db_with_mock_pool):
        """get_cursor()正常終了時はcommitが呼ばれる"""
        db, mock_pool, mock_connection = db_with_mock_pool
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with db.get_cursor() as cur:
            pass

        mock_connection.commit.assert_called_once()


# =============================================================================
# 接続エラー時の例外処理テスト
# =============================================================================


class TestConnectionErrors:
    """接続エラー時の例外処理テスト"""

    def test_pool_creation_error_propagates(self):
        """プール作成時のエラーが伝播する"""
        url = "postgresql://invalid:url@localhost:5432/db"
        db = DatabaseConnection(database_url=url)

        with patch(
            "src.db.connection.pool.ThreadedConnectionPool",
            side_effect=psycopg2.OperationalError("Connection refused"),
        ):
            with pytest.raises(psycopg2.OperationalError) as exc_info:
                db._get_pool()

            assert "Connection refused" in str(exc_info.value)

    def test_getconn_error_propagates(self):
        """プールからの取得エラーが伝播する"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
        mock_pool.getconn.side_effect = pool.PoolError("No connections available")
        db._pool = mock_pool

        with pytest.raises(pool.PoolError) as exc_info:
            with db.get_connection() as conn:
                pass

        assert "No connections" in str(exc_info.value)

    def test_database_error_causes_rollback(self):
        """データベースエラー発生時はrollbackが呼ばれる"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_connection = MagicMock()
        mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
        mock_pool.getconn.return_value = mock_connection
        db._pool = mock_pool

        with pytest.raises(psycopg2.DatabaseError):
            with db.get_connection() as conn:
                raise psycopg2.DatabaseError("Query failed")

        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()


# =============================================================================
# プールからの取得・返却テスト
# =============================================================================


class TestPoolConnectionManagement:
    """プールからの取得・返却テスト"""

    @pytest.fixture
    def db_with_mock(self):
        """モックプール付きのDatabaseConnection"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_connection = MagicMock()
        mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
        mock_pool.getconn.return_value = mock_connection
        db._pool = mock_pool

        return db, mock_pool, mock_connection

    def test_getconn_called_once_per_context(self, db_with_mock):
        """コンテキストごとにgetconnが1回呼ばれる"""
        db, mock_pool, _ = db_with_mock

        with db.get_connection() as conn1:
            pass

        with db.get_connection() as conn2:
            pass

        assert mock_pool.getconn.call_count == 2

    def test_putconn_called_once_per_context(self, db_with_mock):
        """コンテキストごとにputconnが1回呼ばれる"""
        db, mock_pool, mock_connection = db_with_mock

        with db.get_connection() as conn:
            pass

        mock_pool.putconn.assert_called_once_with(mock_connection)

    def test_close_closes_all_pool_connections(self, db_with_mock):
        """close()は全プール接続をクローズする"""
        db, mock_pool, _ = db_with_mock

        db.close()

        mock_pool.closeall.assert_called_once()
        assert db._pool is None

    def test_close_is_idempotent(self, db_with_mock):
        """close()は複数回呼んでも安全"""
        db, mock_pool, _ = db_with_mock

        db.close()
        db.close()  # 2回目

        mock_pool.closeall.assert_called_once()

    def test_close_on_uninitialized_pool(self):
        """未初期化プールのclose()は安全"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        # 例外が発生しないことを確認
        db.close()

        assert db._pool is None


# =============================================================================
# DatabaseConnectionのコンテキストマネージャーとしての使用テスト
# =============================================================================


class TestDatabaseConnectionAsContextManager:
    """DatabaseConnectionをコンテキストマネージャーとして使用するテスト"""

    @pytest.fixture
    def db_with_mock(self):
        """モックプール付きのDatabaseConnection"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_connection = MagicMock()
        mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
        mock_pool.getconn.return_value = mock_connection
        db._pool = mock_pool

        return db, mock_pool, mock_connection

    def test_enter_returns_self(self, db_with_mock):
        """__enter__はselfを返す"""
        db, _, _ = db_with_mock

        with db as entered:
            assert entered is db

    def test_exit_closes_pool(self, db_with_mock):
        """__exit__はプールをクローズする"""
        db, mock_pool, _ = db_with_mock

        with db:
            pass

        mock_pool.closeall.assert_called_once()


# =============================================================================
# health_check のテスト
# =============================================================================


class TestHealthCheck:
    """health_check メソッドのテスト"""

    @pytest.fixture
    def db_with_mock(self):
        """モックプール付きのDatabaseConnection"""
        url = "postgresql://user:pass@localhost:5432/testdb"
        db = DatabaseConnection(database_url=url)

        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
        mock_pool.getconn.return_value = mock_connection
        db._pool = mock_pool

        return db, mock_pool, mock_connection, mock_cursor

    def test_health_check_returns_true_on_success(self, db_with_mock):
        """正常時はTrueを返す"""
        db, _, _, mock_cursor = db_with_mock
        mock_cursor.fetchone.return_value = (1,)

        result = db.health_check()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    def test_health_check_returns_false_on_query_failure(self, db_with_mock):
        """クエリ失敗時はFalseを返す"""
        db, _, _, mock_cursor = db_with_mock
        mock_cursor.execute.side_effect = psycopg2.Error("Query failed")

        result = db.health_check()

        assert result is False

    def test_health_check_returns_false_on_unexpected_result(self, db_with_mock):
        """予期しない結果の場合はFalseを返す"""
        db, _, _, mock_cursor = db_with_mock
        mock_cursor.fetchone.return_value = None

        result = db.health_check()

        assert result is False

    def test_health_check_returns_false_on_wrong_value(self, db_with_mock):
        """値が1でない場合はFalseを返す"""
        db, _, _, mock_cursor = db_with_mock
        mock_cursor.fetchone.return_value = (0,)

        result = db.health_check()

        assert result is False


# =============================================================================
# シングルトン関数のテスト
# =============================================================================


class TestSingletonFunctions:
    """get_database, close_database のテスト"""

    def teardown_method(self):
        """テスト後にシングルトンをリセット"""
        close_database()

    def test_get_database_returns_database_connection(self):
        """get_database()はDatabaseConnectionを返す"""
        test_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            db = get_database()

            assert isinstance(db, DatabaseConnection)
            assert db.database_url == test_url

    def test_get_database_returns_same_instance(self):
        """get_database()は同じインスタンスを返す（シングルトン）"""
        test_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            db1 = get_database()
            db2 = get_database()

            assert db1 is db2

    def test_close_database_closes_singleton(self):
        """close_database()はシングルトンをクローズする"""
        test_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            db = get_database()

            # モックプールを設定
            mock_pool = MagicMock(spec=pool.ThreadedConnectionPool)
            db._pool = mock_pool

            close_database()

            mock_pool.closeall.assert_called_once()

    def test_close_database_allows_new_instance(self):
        """close_database()後は新しいインスタンスが作成される"""
        test_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            db1 = get_database()
            close_database()
            db2 = get_database()

            assert db1 is not db2

    def test_close_database_is_idempotent(self):
        """close_database()は複数回呼んでも安全"""
        test_url = "postgresql://user:pass@localhost:5432/testdb"

        with patch.dict(os.environ, {"DATABASE_URL": test_url}):
            get_database()

            # 例外が発生しないことを確認
            close_database()
            close_database()
