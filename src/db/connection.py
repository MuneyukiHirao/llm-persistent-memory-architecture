# PostgreSQL接続管理
# 接続: postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_memory
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション2.1
"""
PostgreSQL接続管理モジュール

コネクションプールとコンテキストマネージャーによる安全な接続管理を提供。

設計方針（メモリ管理エージェント観点）:
- 原子性: コンテキストマネージャーでcommit/rollbackを確実に制御
- 効率性: コネクションプールで接続のオーバーヘッドを削減
- テスト容易性: database_urlを外部から注入可能にし、テスト時のモック化を容易に
"""

import os
from contextlib import contextmanager
from typing import Optional, Generator, Any

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PsycopgConnection


class DatabaseConnection:
    """PostgreSQLデータベース接続管理クラス

    コネクションプールを使用した効率的な接続管理と、
    コンテキストマネージャーによるトランザクション安全性を提供。

    使用例:
        db = DatabaseConnection()

        # 基本的な使用方法
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_memory WHERE agent_id = %s", (agent_id,))
                rows = cur.fetchall()
        # コンテキスト終了時に自動commit（例外時はrollback）

        # 手動トランザクション制御
        with db.get_connection(auto_commit=False) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO ...")
                cur.execute("UPDATE ...")
            conn.commit()  # 明示的にcommit

    Attributes:
        database_url: 接続文字列（環境変数DATABASE_URLから取得、または直接指定）
        min_connections: プール内の最小接続数
        max_connections: プール内の最大接続数
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """DatabaseConnectionを初期化

        Args:
            database_url: PostgreSQL接続文字列。
                         Noneの場合は環境変数DATABASE_URLを使用。
            min_connections: プール内の最小接続数（デフォルト: 1）
            max_connections: プール内の最大接続数（デフォルト: 10）

        Raises:
            ValueError: database_urlが設定されていない場合
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Please set it or provide database_url parameter."
            )

        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Optional[pool.ThreadedConnectionPool] = None

    def _get_pool(self) -> pool.ThreadedConnectionPool:
        """コネクションプールを取得（遅延初期化）

        Returns:
            ThreadedConnectionPool: スレッドセーフなコネクションプール
        """
        if self._pool is None:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                dsn=self.database_url,
            )
        return self._pool

    @contextmanager
    def get_connection(
        self, auto_commit: bool = True
    ) -> Generator[PsycopgConnection, None, None]:
        """データベース接続をコンテキストマネージャーとして取得

        トランザクションの整合性を保証するコンテキストマネージャー。
        正常終了時はcommit、例外発生時はrollbackを自動実行。

        Args:
            auto_commit: Trueの場合、コンテキスト終了時に自動commit。
                        Falseの場合、手動でcommit/rollbackが必要。

        Yields:
            psycopg2 connection: データベース接続

        Raises:
            psycopg2.Error: データベース操作エラー

        使用例:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        """
        connection = self._get_pool().getconn()
        try:
            yield connection
            if auto_commit:
                connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._get_pool().putconn(connection)

    @contextmanager
    def get_cursor(
        self, auto_commit: bool = True
    ) -> Generator[Any, None, None]:
        """カーソルを直接取得するコンテキストマネージャー

        より簡潔な操作のためのショートカット。

        Args:
            auto_commit: Trueの場合、コンテキスト終了時に自動commit。

        Yields:
            psycopg2 cursor: データベースカーソル

        使用例:
            with db.get_cursor() as cur:
                cur.execute("SELECT * FROM agent_memory")
                rows = cur.fetchall()
        """
        with self.get_connection(auto_commit=auto_commit) as conn:
            with conn.cursor() as cur:
                yield cur

    def close(self) -> None:
        """コネクションプールをクローズ

        アプリケーション終了時に呼び出すことで、
        すべての接続を適切に解放。
        """
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    def health_check(self) -> bool:
        """データベース接続の健全性をチェック

        Returns:
            bool: 接続が正常な場合True
        """
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None and result[0] == 1
        except Exception:
            return False

    def __enter__(self) -> "DatabaseConnection":
        """コンテキストマネージャーとしての入口"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """コンテキストマネージャーとしての出口

        コンテキスト終了時に自動的にプールをクローズ。
        """
        self.close()


# シングルトンインスタンス（オプション）
# アプリケーション全体で共有する場合に使用
_default_connection: Optional[DatabaseConnection] = None


def get_database() -> DatabaseConnection:
    """デフォルトのデータベース接続を取得

    シングルトンパターンでアプリケーション全体で
    同一のコネクションプールを共有。

    Returns:
        DatabaseConnection: データベース接続インスタンス
    """
    global _default_connection
    if _default_connection is None:
        _default_connection = DatabaseConnection()
    return _default_connection


def close_database() -> None:
    """デフォルトのデータベース接続をクローズ

    アプリケーション終了時に呼び出す。
    """
    global _default_connection
    if _default_connection is not None:
        _default_connection.close()
        _default_connection = None
