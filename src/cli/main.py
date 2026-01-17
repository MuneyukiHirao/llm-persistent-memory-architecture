#!/usr/bin/env python3
from __future__ import annotations
"""
Phase 5 CLI メインエントリーポイント

永続的メモリシステムを Pythonコードを書かずに操作するための CLI インターフェース。
"""

import click
import sys
import os
import re
import threading
from typing import Optional
from uuid import UUID, uuid4

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.db.connection import DatabaseConnection
from src.agents.agent_registry import AgentRegistry
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.router import Router
from src.orchestrator.evaluator import Evaluator
from src.core.task_executor import TaskExecutor
from src.core.memory_repository import MemoryRepository
from src.embedding.azure_client import AzureEmbeddingClient
from src.search.vector_search import VectorSearch
from src.config.phase2_config import Phase2Config
from src.cli.utils.output import echo_json, echo_table
from src.cli.utils.progress import Spinner

# コマンドモジュールインポート
from src.cli.commands.register import register_agent_command
from src.cli.commands.status import status_command
from src.cli.commands.memory import memory_command
from src.cli.commands.sleep import sleep_command
from src.cli.commands.educate import educate_command
from src.cli.commands.session import session_command
from src.cli.commands.config import config_command
from src.cli.commands.chat import chat_command


class CLIContext:
    """CLI共通コンテキスト（依存関係を保持）"""

    def __init__(self):
        self.db: Optional[DatabaseConnection] = None
        self.config: Optional[Phase2Config] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.memory_repository: Optional[MemoryRepository] = None
        self.embedding_client: Optional[AzureEmbeddingClient] = None
        self.vector_search: Optional[VectorSearch] = None
        self._task_executor: Optional[TaskExecutor] = None
        self._llm_task_executor = None
        self._orchestrator: Optional[Orchestrator] = None
        self._initialized = False

    def initialize(self):
        """遅延初期化（必要時に呼び出される）"""
        if self._initialized:
            return

        try:
            self.db = DatabaseConnection()
            self.config = Phase2Config()

            # 基盤コンポーネント
            self.agent_registry = AgentRegistry(self.db)
            self.memory_repository = MemoryRepository(self.db, self.config)
            self.embedding_client = AzureEmbeddingClient()
            self.vector_search = VectorSearch(
                self.db, self.embedding_client, self.config
            )

            self._initialized = True

        except Exception as e:
            click.echo(f"[初期化エラー] システムの初期化に失敗しました: {e}", err=True)
            sys.exit(1)

    @property
    def task_executor(self) -> TaskExecutor:
        """TaskExecutor を遅延初期化"""
        self.initialize()
        if self._task_executor is None:
            from src.search.ranking import MemoryRanker
            from src.core.strength_manager import StrengthManager
            from src.core.sleep_processor import SleepPhaseProcessor

            # 必要なコンポーネントを初期化
            ranker = MemoryRanker(self.config)
            strength_manager = StrengthManager(self.memory_repository, self.config)
            sleep_processor = SleepPhaseProcessor(self.db, self.config)

            self._task_executor = TaskExecutor(
                vector_search=self.vector_search,
                ranker=ranker,
                strength_manager=strength_manager,
                sleep_processor=sleep_processor,
                repository=self.memory_repository,
                config=self.config,
            )
        return self._task_executor

    @property
    def llm_task_executor(self):
        """LLMTaskExecutor を遅延初期化"""
        self.initialize()
        if self._llm_task_executor is None:
            from pathlib import Path
            from src.llm.claude_client import ClaudeClient
            from src.llm.tool_executor import ToolExecutor
            from src.llm.llm_task_executor import LLMTaskExecutor
            from src.config.llm_config import llm_config
            from src.llm.tools import file_tools
            from src.llm.tools.bash_tools import get_bash_execute_tool, bash_execute

            # プロジェクトルートを設定
            project_root = Path.cwd()
            file_tools.set_project_root(str(project_root))

            claude_client = ClaudeClient(config=llm_config)
            tool_executor = ToolExecutor()

            # ファイル操作ツールを登録（ラッパー関数経由）
            def file_read_handler(input_data: dict) -> str:
                result = file_tools.file_read(input_data["path"])
                if result.get("success"):
                    return result["data"]["content"]
                return f"Error: {result.get('error', 'Unknown error')}"

            def file_write_handler(input_data: dict) -> str:
                result = file_tools.file_write(input_data["path"], input_data["content"])
                if result.get("success"):
                    return f"File written successfully: {result['data']['path']}"
                return f"Error: {result.get('error', 'Unknown error')}"

            def file_list_handler(input_data: dict) -> str:
                result = file_tools.file_list(
                    input_data["path"],
                    input_data.get("pattern")
                )
                if result.get("success"):
                    data = result["data"]
                    return f"Files: {', '.join(data['files'])}\nDirectories: {', '.join(data['directories'])}"
                return f"Error: {result.get('error', 'Unknown error')}"

            def bash_execute_handler(input_data: dict) -> str:
                result = bash_execute(
                    input_data["command"],
                    timeout=input_data.get("timeout", 30),
                    working_dir=input_data.get("working_dir")
                )
                if result.get("success"):
                    data = result["data"]
                    output = data["stdout"]
                    if data.get("stderr"):
                        output += f"\n[stderr]: {data['stderr']}"
                    return output
                return f"Error: {result.get('error', 'Unknown error')}"

            tool_executor.register_tool(
                file_tools.get_file_read_tool(),
                file_read_handler
            )
            tool_executor.register_tool(
                file_tools.get_file_write_tool(),
                file_write_handler
            )
            tool_executor.register_tool(
                file_tools.get_file_list_tool(),
                file_list_handler
            )

            # Bashツールを登録
            tool_executor.register_tool(
                get_bash_execute_tool(),
                bash_execute_handler
            )

            self._llm_task_executor = LLMTaskExecutor(
                claude_client=claude_client,
                tool_executor=tool_executor,
                task_executor=self.task_executor,
            )
        return self._llm_task_executor

    @property
    def orchestrator(self) -> Orchestrator:
        """Orchestrator を遅延初期化"""
        self.initialize()
        if self._orchestrator is None:
            from src.llm.claude_client import ClaudeClient
            from src.agents.meta_agent import MetaAgent
            from src.config.llm_config import llm_config

            # LLMTaskExecutorを再利用
            llm_task_executor = self.llm_task_executor

            # ClaudeClient を取得（LLMTaskExecutorから）
            claude_client = llm_task_executor.claude_client

            # MetaAgentを初期化（オプション）
            meta_agent = MetaAgent(
                agent_registry=self.agent_registry,
                claude_client=claude_client,
            )

            router = Router(self.agent_registry)
            evaluator = Evaluator(config=self.config)
            self._orchestrator = Orchestrator(
                agent_id="orchestrator_cli",
                router=router,
                evaluator=evaluator,
                task_executor=self.task_executor,
                config=self.config,
                llm_task_executor=llm_task_executor,
                meta_agent=meta_agent,
                db=self.db,  # セッション永続化のためにDBを渡す
            )
        return self._orchestrator


# click の pass_context でCLIContextを共有
pass_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group()
@click.version_option(version="1.0.0", prog_name="agent")
@pass_context
def agent(ctx: CLIContext):
    """
    永続的メモリシステム CLI
    
    エージェントの登録、教育、タスク依頼をターミナルから直感的に行えます。
    """
    pass


@agent.command()
@click.option('--force', is_flag=True, help='既存データを削除して再初期化')
@click.option('--check-only', is_flag=True, help='接続確認のみ（変更なし）')
@pass_context
def init(ctx: CLIContext, force: bool, check_only: bool):
    """環境を初期化し、システムが使える状態にする"""
    if force and check_only:
        click.echo("[エラー] --force と --check-only は同時に指定できません", err=True)
        sys.exit(2)

    click.echo("システムを初期化中...")

    try:
        ctx.initialize()

        with ctx.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                click.echo(f"✓ データベースに接続しました ({version.split()[0]} {version.split()[1]})")

        with ctx.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                if result:
                    click.echo(f"✓ pgvector 拡張が有効です ({result[0]})")
                else:
                    click.echo("⚠ pgvector 拡張が見つかりません", err=True)

        # 必須環境変数チェック
        if not os.getenv("DATABASE_URL"):
            click.echo("⚠ DATABASE_URL が設定されていません", err=True)

        # Azure OpenAI Embedding: 2つの設定方式をサポート
        # 方式1: OpenAIEmbeddingURI + OpenAIEmbeddingKey
        # 方式2: AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        has_method1 = os.getenv("OpenAIEmbeddingURI") and os.getenv("OpenAIEmbeddingKey")
        has_method2 = (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            and os.getenv("AZURE_OPENAI_API_KEY")
            and os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        )

        if not (has_method1 or has_method2):
            click.echo("⚠ Azure OpenAI Embedding の環境変数が設定されていません", err=True)
            click.echo("  方式1: OpenAIEmbeddingURI + OpenAIEmbeddingKey", err=True)
            click.echo("  方式2: AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", err=True)

        table_names = _load_schema_table_names()
        with ctx.db.get_connection() as conn:
            existing_tables = _fetch_existing_tables(conn)

        missing_tables = [t for t in table_names if t not in existing_tables]
        if missing_tables:
            click.echo(f"⚠ 必要なテーブルが不足しています: {', '.join(missing_tables)}", err=True)
        else:
            click.echo("✓ 必要なテーブルが存在します")

        if check_only:
            click.echo("\nチェックのみ完了しました。")
            return

        if force:
            if not click.confirm("既存データを削除して再初期化します。続行しますか？"):
                click.echo("初期化をキャンセルしました")
                return
            with ctx.db.get_connection() as conn:
                _drop_tables(conn, table_names)
            existing_tables = set()
            missing_tables = list(table_names)

        if missing_tables:
            if existing_tables:
                click.echo("[エラー] 既存テーブルが部分的に存在します。--force で再初期化してください。", err=True)
                sys.exit(1)

            with ctx.db.get_connection() as conn:
                _apply_schema(conn)
                _apply_migrations(conn)

            click.echo("✓ スキーマとマイグレーションを適用しました")

        try:
            test_embedding = ctx.embedding_client.get_embedding("test")
            if test_embedding and len(test_embedding) > 0:
                click.echo("✓ Azure OpenAI Embedding に接続できます")
        except Exception as e:
            click.echo(f"⚠ Azure OpenAI Embedding 接続エラー: {e}", err=True)

        click.echo("\nシステムは使用可能です。")

    except Exception as e:
        click.echo(f"[エラー] 初期化に失敗しました: {e}", err=True)
        sys.exit(1)


@agent.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='出力形式')
@click.option('--status', type=click.Choice(['active', 'disabled', 'all']), default='active', help='ステータスでフィルタ')
@click.option('--verbose', is_flag=True, help='詳細情報を含める')
@pass_context
def list(ctx: CLIContext, output_format: str, status: str, verbose: bool):
    """登録済みエージェントの一覧を表示する"""
    ctx.initialize()

    try:
        if status == 'active':
            agents = ctx.agent_registry.get_active_agents()
        elif status == 'all':
            agents = ctx.agent_registry.get_all()
        else:
            agents = [a for a in ctx.agent_registry.get_all() if a.status == 'disabled']

        if not agents:
            click.echo("登録済みエージェントはありません。")
            click.echo("\nヒント: agent register -f <agent.yaml> でエージェントを登録してください")
            return

        if output_format == 'json':
            payload = []
            for agent_def in agents:
                data = agent_def.to_dict()
                data["memory_count"] = _get_memory_count(ctx, agent_def.agent_id)
                if not verbose:
                    data.pop("system_prompt", None)
                    data.pop("tools", None)
                payload.append(data)
            echo_json(payload)
            return

        click.echo(f"登録済みエージェント ({len(agents)}件):\n")
        headers = ["ID", "名前", "役割", "状態", "メモリ数"]
        rows = []
        for agent_def in agents:
            memory_count = _get_memory_count(ctx, agent_def.agent_id)
            rows.append([
                agent_def.agent_id,
                agent_def.name,
                agent_def.role[:28],
                agent_def.status,
                memory_count,
            ])
        echo_table(headers, rows)

        if verbose:
            for agent_def in agents:
                click.echo("")
                click.echo(f"[{agent_def.agent_id}]")
                click.echo(f"  観点: {', '.join(agent_def.perspectives)}")
                if agent_def.capabilities:
                    click.echo(f"  能力: {', '.join(agent_def.capabilities)}")

    except Exception as e:
        click.echo(f"[エラー] エージェント一覧の取得に失敗しました: {e}", err=True)
        sys.exit(1)


@agent.command()
@click.argument('task_description')
@click.option('--agent', 'agent_id', help='特定エージェントに直接依頼')
@click.option('--perspective', help='重視する観点')
@click.option('--context', help='追加コンテキスト')
@click.option('--file', 'input_file', type=click.Path(exists=True), help='ファイルから入力')
@click.option('--output', 'output_file', help='結果をファイルに保存')
@click.option('--session', 'session_id', help='既存セッションで継続')
@click.option('--wait/--async', 'wait_for_completion', default=True, help='完了まで待機（デフォルト）')
@click.option('--verbose', is_flag=True, help='詳細ログを表示')
@pass_context
def task(ctx: CLIContext, task_description: str, agent_id: Optional[str], 
         perspective: Optional[str], context: Optional[str], input_file: Optional[str],
         output_file: Optional[str], session_id: Optional[str], wait_for_completion: bool,
         verbose: bool):
    """タスクを依頼（オーケストレーター経由）"""
    ctx.initialize()
    
    if input_file:
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
        except Exception as e:
            click.echo(f"[エラー] 入力ファイルの読み込みに失敗しました: {e}", err=True)
            sys.exit(2)
        if file_content:
            task_description = f"{task_description}\\n\\n{file_content}"

    if context:
        task_description = f"{task_description}\\n\\n{context}"

    click.echo("タスクを受け付けました\\n")
    
    try:
        session_uuid = None
        if session_id:
            try:
                session_uuid = UUID(session_id)
            except ValueError:
                click.echo(f"[エラー] 無効なセッションID形式です: {session_id}", err=True)
                sys.exit(2)

        # 特定エージェント指定時は直接実行
        if agent_id:
            # エージェント存在確認
            agent_def = ctx.agent_registry.get_by_id(agent_id)
            if not agent_def:
                click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
                click.echo("\\nヒント: agent list で登録済みエージェントを確認してください", err=True)
                sys.exit(2)
            
            if not wait_for_completion:
                def _background_direct():
                    ctx.llm_task_executor.execute_task_with_tools(
                        agent_id=agent_id,
                        system_prompt=agent_def.system_prompt,
                        task_description=task_description,
                        perspective=perspective,
                    )

                threading.Thread(target=_background_direct, daemon=True).start()
                click.echo("非同期で実行しました。完了を待たずに終了します。")
                return

            # スピナー付きでタスク実行
            with Spinner(f"{agent_def.name}が処理中"):
                result = ctx.llm_task_executor.execute_task_with_tools(
                    agent_id=agent_id,
                    system_prompt=agent_def.system_prompt,
                    task_description=task_description,
                    perspective=perspective,
                )

            _display_task_result(result, verbose, output_file)
            return
        
        # オーケストレーター経由で実行
        if not wait_for_completion:
            async_session_id = session_uuid or uuid4()

            def _background_orchestrate():
                ctx.orchestrator.process_request(
                    task_summary=task_description,
                    items=[],
                    session_id=async_session_id,
                )

            threading.Thread(target=_background_orchestrate, daemon=True).start()
            click.echo(f"非同期で実行しました (session: {async_session_id})")
            return

        # スピナー付きでオーケストレーター実行
        with Spinner("オーケストレーターが処理中"):
            orchestrator_result = ctx.orchestrator.process_request(
                task_summary=task_description,
                items=[],  # CLIでは論点リストなし
                session_id=session_uuid,
            )
        
        # ルーティング結果を表示
        if hasattr(orchestrator_result, 'routing_decision') and orchestrator_result.routing_decision:
            routing = orchestrator_result.routing_decision
            click.echo(
                f"  → {routing.selected_agent_id} を選択しました "
                f"(スコア: {routing.confidence:.2f})"
            )
            if hasattr(routing, 'selection_reason'):
                click.echo(f"  理由: {routing.selection_reason}\\n")
        
        # 結果を表示
        click.echo("─" * 40)
        click.echo("【結果】\\n")
        
        output = "結果が取得できませんでした"
        if hasattr(orchestrator_result, 'agent_result') and orchestrator_result.agent_result:
            output = orchestrator_result.agent_result.get('output', output)
        click.echo(output)
        
        click.echo("\\n" + "─" * 40)
        
        # 完了メッセージ
        session_id = getattr(orchestrator_result, 'session_id', 'unknown')
        click.echo(f"\\nタスク完了 (session: {session_id})")

        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output)
                click.echo(f"結果を保存しました: {output_file}")
            except Exception as e:
                click.echo(f"[エラー] 出力ファイルの書き込みに失敗しました: {e}", err=True)
        
    except Exception as e:
        click.echo(f"[エラー] タスク実行に失敗しました: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def _display_task_result(result, verbose: bool, output_file: Optional[str] = None):
    """タスク結果を表示する"""
    from src.llm.llm_task_executor import LLMTaskResult

    click.echo("─" * 40)
    click.echo("【結果】\\n")

    # LLMTaskResult の場合
    if isinstance(result, LLMTaskResult):
        output = result.content
        click.echo(output)
        click.echo("\\n" + "─" * 40)

        if verbose:
            click.echo("\\n[詳細情報]")
            click.echo(f"  トークン使用量: {result.total_tokens}")
            click.echo(f"  イテレーション回数: {result.iterations}")
            click.echo(f"  検索されたメモリ: {result.searched_memories_count}件")
            click.echo(f"  ツール呼び出し回数: {len(result.tool_calls)}")
            click.echo(f"  停止理由: {result.stop_reason}")

    # 辞書形式の場合（従来のフォーマット）
    elif isinstance(result, dict):
        output = result.get('output', result.get('response', result.get('content', str(result))))
        click.echo(output)
        click.echo("\\n" + "─" * 40)

        if verbose:
            click.echo("\\n[詳細情報]")
            for key, value in result.items():
                if key not in ('output', 'response', 'content'):
                    click.echo(f"  {key}: {value}")

    # その他（文字列など）
    else:
        output = str(result)
        click.echo(output)
        click.echo("\\n" + "─" * 40)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            click.echo(f"結果を保存しました: {output_file}")
        except Exception as e:
            click.echo(f"[エラー] 出力ファイルの書き込みに失敗しました: {e}", err=True)


def _get_memory_count(ctx: CLIContext, agent_id: str):
    try:
        memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")
        return len(memories)
    except Exception:
        return "N/A"


def _load_schema_table_names() -> list[str]:
    schema_path = os.path.join(os.path.dirname(__file__), "../db/schema.sql")
    schema_path = os.path.abspath(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    return re.findall(r"CREATE TABLE\\s+([a-zA-Z0-9_]+)", sql)


def _fetch_existing_tables(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        return {row[0] for row in cur.fetchall()}


def _drop_tables(conn, table_names: list[str]) -> None:
    if not table_names:
        return
    with conn.cursor() as cur:
        for table in table_names:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.commit()


def _apply_schema(conn) -> None:
    schema_path = os.path.join(os.path.dirname(__file__), "../db/schema.sql")
    schema_path = os.path.abspath(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _apply_migrations(conn) -> None:
    migrations_dir = os.path.join(os.path.dirname(__file__), "../db/migrations")
    migrations_dir = os.path.abspath(migrations_dir)
    if not os.path.isdir(migrations_dir):
        return

    files = sorted(
        f for f in os.listdir(migrations_dir) if f.endswith(".sql")
    )
    for filename in files:
        if not _migration_needed(conn, filename):
            continue
        path = os.path.join(migrations_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        sql = _extract_up_sql(content)
        if not sql.strip():
            continue
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def _migration_needed(conn, filename: str) -> bool:
    if filename.startswith("001_"):
        return not _column_exists(conn, "agent_memory", "learning")
    if filename.startswith("002_"):
        return not _column_exists(conn, "agent_memory", "next_review_at")
    return True


def _column_exists(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            """,
            (table, column),
        )
        return cur.fetchone() is not None


def _extract_up_sql(content: str) -> str:
    parts = content.split("-- DOWN Migration", 1)
    up_section = parts[0]
    if "-- UP Migration" in up_section:
        up_section = up_section.split("-- UP Migration", 1)[1]
    return up_section


# 各コマンドを追加
register_agent_command(agent, pass_context)
status_command(agent, pass_context)
memory_command(agent, pass_context)
sleep_command(agent, pass_context)
educate_command(agent, pass_context)
session_command(agent, pass_context)
config_command(agent, pass_context)
chat_command(agent, pass_context)


if __name__ == '__main__':
    agent()
