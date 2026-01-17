"""
セッション管理コマンド実装

セッションの作成、一覧表示、状態確認、再開、終了を行う。
"""

import click
import sys
import os
from typing import Optional
from uuid import UUID

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.orchestrator.progress_manager import SessionStateRepository, ProgressManager


def session_command(agent_group, pass_context):
    """session コマンドを agent グループに追加"""

    @agent_group.group()
    @pass_context
    def session(ctx):
        """セッション管理コマンド

        セッションの作成、一覧表示、状態確認、再開、終了を行います。
        """
        pass

    @session.command()
    @click.argument('name')
    @pass_context
    def start(ctx, name: str):
        """新規セッションを開始する

        NAME: セッション名（プロジェクト名など）

        例:
          agent session start "新機能開発プロジェクト"
        """
        ctx.initialize()

        try:
            # ProgressManager を初期化
            session_repository = SessionStateRepository(ctx.db)
            progress_manager = ProgressManager(session_repository, ctx.config)

            # 新規セッションを作成
            session_id = progress_manager.create_session(
                orchestrator_id="orchestrator_cli",
                user_request={
                    "original": name,
                    "clarified": name,
                },
                task_tree={
                    "tasks": [],
                    "name": name,
                },
            )

            click.echo(f"セッションを開始しました\n")
            click.echo(f"  セッションID: {session_id}")
            click.echo(f"  名前: {name}")
            click.echo(f"  状態: in_progress")
            click.echo(f"\n次のステップ:")
            click.echo(f"  agent session status {session_id}  # 状態確認")
            click.echo(f"  agent session close {session_id}   # 終了")

        except Exception as e:
            click.echo(f"[エラー] セッションの開始に失敗しました: {e}", err=True)
            sys.exit(1)

    @session.command()
    @click.option('--status', 'filter_status', type=click.Choice(['in_progress', 'paused', 'completed', 'failed']),
                  help='ステータスでフィルタ')
    @pass_context
    def list(ctx, filter_status: Optional[str]):
        """セッション一覧を表示する

        例:
          agent session list
          agent session list --status in_progress
        """
        ctx.initialize()

        try:
            # SessionStateRepository を初期化
            session_repository = SessionStateRepository(ctx.db)

            # セッション一覧を取得
            if filter_status:
                sessions = session_repository.list_by_status(filter_status)
            else:
                # 全ステータスを取得して結合
                progress_manager = ProgressManager(session_repository, ctx.config)
                in_progress = session_repository.list_by_status("in_progress")
                paused = session_repository.list_by_status("paused")
                completed = session_repository.list_by_status("completed")
                failed = session_repository.list_by_status("failed")
                sessions = in_progress + paused + completed + failed
                sessions.sort(key=lambda s: s.last_activity_at, reverse=True)

            if not sessions:
                if filter_status:
                    click.echo(f"ステータス '{filter_status}' のセッションはありません。")
                else:
                    click.echo("セッションはありません。")
                click.echo("\nヒント: agent session start <名前> でセッションを開始してください")
                return

            # ヘッダー
            click.echo(f"セッション一覧 ({len(sessions)}件):\n")
            click.echo(f"{'ID':<38} {'名前':<24} {'進捗':<6} {'状態':<12} {'最終更新':<16}")
            click.echo("-" * 100)

            for sess in sessions:
                # セッション名を取得
                session_name = sess.user_request.get("original", "不明")[:22]
                if len(sess.user_request.get("original", "")) > 22:
                    session_name += ".."

                # 最終更新日時をフォーマット
                last_update = sess.last_activity_at.strftime("%m-%d %H:%M")

                # 進捗率を表示
                progress = f"{sess.overall_progress_percent}%"

                click.echo(
                    f"{str(sess.session_id):<38} "
                    f"{session_name:<24} "
                    f"{progress:<6} "
                    f"{sess.status:<12} "
                    f"{last_update:<16}"
                )

        except Exception as e:
            click.echo(f"[エラー] セッション一覧の取得に失敗しました: {e}", err=True)
            sys.exit(1)

    @session.command()
    @click.argument('session_id')
    @pass_context
    def status(ctx, session_id: str):
        """セッションの状態を表示する

        SESSION_ID: セッション識別子（UUID）

        例:
          agent session status 12345678-1234-1234-1234-123456789012
        """
        ctx.initialize()

        try:
            # UUID バリデーション
            try:
                uuid = UUID(session_id)
            except ValueError:
                click.echo(f"[エラー] 無効なセッションID形式です: {session_id}", err=True)
                click.echo("\nヒント: agent session list でセッションIDを確認してください", err=True)
                sys.exit(2)

            # ProgressManager を初期化
            session_repository = SessionStateRepository(ctx.db)
            progress_manager = ProgressManager(session_repository, ctx.config)

            # セッション状態を取得
            state = progress_manager.restore_state(uuid)

            if not state:
                click.echo(f"[エラー] セッションが見つかりません: {session_id}", err=True)
                click.echo("\nヒント: agent session list でセッション一覧を確認してください", err=True)
                sys.exit(2)

            # 基本情報を表示
            click.echo(f"セッション: {session_id}\n")

            click.echo("基本情報:")
            click.echo(f"  名前: {state.user_request.get('original', '不明')}")
            click.echo(f"  オーケストレーター: {state.orchestrator_id}")
            click.echo(f"  状態: {state.status}")
            click.echo(f"  進捗率: {state.overall_progress_percent}%")

            click.echo("\n日時情報:")
            click.echo(f"  作成日時: {state.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"  更新日時: {state.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"  最終アクティビティ: {state.last_activity_at.strftime('%Y-%m-%d %H:%M:%S')}")

            # タスク情報
            tasks = state.task_tree.get("tasks", [])
            if tasks:
                completed_count = sum(1 for t in tasks if t.get("status") == "completed")
                click.echo(f"\nタスク情報:")
                click.echo(f"  完了: {completed_count}/{len(tasks)}")

                click.echo("\n  タスク一覧:")
                for task in tasks:
                    status_icon = _get_status_icon(task.get("status", "pending"))
                    description = task.get("description", "不明")[:50]
                    click.echo(f"    {status_icon} {description}")

            # 現在のタスク
            if state.current_task:
                click.echo(f"\n現在のタスク:")
                click.echo(f"  {state.current_task.get('description', '不明')}")

            # エラー情報
            if state.task_tree.get("error"):
                error = state.task_tree["error"]
                click.echo(f"\nエラー情報:")
                click.echo(f"  メッセージ: {error.get('message', '不明')}")
                click.echo(f"  発生日時: {error.get('occurred_at', '不明')}")

            # 次のアクションを提案
            click.echo("\n" + "-" * 60)
            if state.status == "paused":
                click.echo(f"agent session resume {session_id}  # 再開")
            elif state.status == "in_progress":
                click.echo(f"agent session close {session_id}   # 終了")

        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"[エラー] セッション状態の取得に失敗しました: {e}", err=True)
            sys.exit(1)

    @session.command()
    @click.argument('session_id')
    @pass_context
    def resume(ctx, session_id: str):
        """セッションを再開する

        SESSION_ID: セッション識別子（UUID）

        一時停止（paused）状態のセッションを再開します。

        例:
          agent session resume 12345678-1234-1234-1234-123456789012
        """
        ctx.initialize()

        try:
            # UUID バリデーション
            try:
                uuid = UUID(session_id)
            except ValueError:
                click.echo(f"[エラー] 無効なセッションID形式です: {session_id}", err=True)
                click.echo("\nヒント: agent session list でセッションIDを確認してください", err=True)
                sys.exit(2)

            # ProgressManager を初期化
            session_repository = SessionStateRepository(ctx.db)
            progress_manager = ProgressManager(session_repository, ctx.config)

            # セッション存在確認
            state = progress_manager.restore_state(uuid)
            if not state:
                click.echo(f"[エラー] セッションが見つかりません: {session_id}", err=True)
                sys.exit(2)

            # 状態チェック
            if state.status == "completed":
                click.echo(f"[エラー] セッションは既に完了しています", err=True)
                sys.exit(1)
            if state.status == "failed":
                click.echo(f"[エラー] セッションは失敗状態です。新しいセッションを開始してください", err=True)
                sys.exit(1)
            if state.status == "in_progress":
                click.echo(f"セッションは既に進行中です")
                return

            # セッションを再開
            success = progress_manager.resume_session(uuid)

            if success:
                click.echo(f"セッションを再開しました\n")
                click.echo(f"  セッションID: {session_id}")
                click.echo(f"  名前: {state.user_request.get('original', '不明')}")
                click.echo(f"  状態: in_progress")
                click.echo(f"  進捗率: {state.overall_progress_percent}%")
            else:
                click.echo(f"[エラー] セッションの再開に失敗しました", err=True)
                sys.exit(1)

        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"[エラー] セッションの再開に失敗しました: {e}", err=True)
            sys.exit(1)

    @session.command()
    @click.argument('session_id')
    @click.option('--force', is_flag=True, help='強制終了（エラー状態として終了）')
    @pass_context
    def close(ctx, session_id: str, force: bool):
        """セッションを終了する

        SESSION_ID: セッション識別子（UUID）

        セッションを完了状態（completed）に変更します。
        --force オプションで失敗状態（failed）として終了できます。

        例:
          agent session close 12345678-1234-1234-1234-123456789012
          agent session close 12345678-1234-1234-1234-123456789012 --force
        """
        ctx.initialize()

        try:
            # UUID バリデーション
            try:
                uuid = UUID(session_id)
            except ValueError:
                click.echo(f"[エラー] 無効なセッションID形式です: {session_id}", err=True)
                click.echo("\nヒント: agent session list でセッションIDを確認してください", err=True)
                sys.exit(2)

            # ProgressManager を初期化
            session_repository = SessionStateRepository(ctx.db)
            progress_manager = ProgressManager(session_repository, ctx.config)

            # セッション存在確認
            state = progress_manager.restore_state(uuid)
            if not state:
                click.echo(f"[エラー] セッションが見つかりません: {session_id}", err=True)
                sys.exit(2)

            # 既に終了しているか確認
            if state.status == "completed" and not force:
                click.echo(f"セッションは既に完了しています")
                return
            if state.status == "failed" and not force:
                click.echo(f"セッションは既に失敗状態です")
                return

            # セッションを終了
            if force:
                success = progress_manager.fail_session(uuid, error_message="ユーザーによる強制終了")
                new_status = "failed"
            else:
                success = progress_manager.complete_session(uuid)
                new_status = "completed"

            if success:
                click.echo(f"セッションを終了しました\n")
                click.echo(f"  セッションID: {session_id}")
                click.echo(f"  名前: {state.user_request.get('original', '不明')}")
                click.echo(f"  状態: {new_status}")
                if new_status == "completed":
                    click.echo(f"  進捗率: 100%")
            else:
                click.echo(f"[エラー] セッションの終了に失敗しました", err=True)
                sys.exit(1)

        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"[エラー] セッションの終了に失敗しました: {e}", err=True)
            sys.exit(1)


def _get_status_icon(status: str) -> str:
    """ステータスアイコンを取得

    Args:
        status: タスクステータス

    Returns:
        ステータスアイコン
    """
    icons = {
        "pending": "[ ]",
        "in_progress": "[>]",
        "completed": "[x]",
        "failed": "[!]",
        "skipped": "[-]",
    }
    return icons.get(status, "[?]")
