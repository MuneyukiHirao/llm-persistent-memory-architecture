"""
対話モード (agent chat) コマンド実装
"""

import click
import sys
import os
import logging
from uuid import uuid4, UUID
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.orchestrator.progress_manager import ProgressManager, SessionStateRepository
from src.cli.utils.progress import Spinner

logger = logging.getLogger(__name__)


def chat_command(agent_group, pass_context):
    """chat コマンドを agent グループに追加"""

    @agent_group.command()
    @click.option('--session', 'session_id', help='既存セッションで継続')
    @click.option('--agent', 'agent_id', help='特定エージェントに直接依頼')
    @click.option('--verbose', is_flag=True, help='詳細ログを表示')
    @pass_context
    def chat(ctx, session_id: Optional[str], agent_id: Optional[str], verbose: bool):
        """対話モードでエージェントと会話する

        REPLループを開始し、ユーザーからの入力を受け付けます。
        'exit' または 'quit' で終了、Ctrl+C でも終了できます。

        使用例:

          # 対話モード起動（新規セッション）
          agent chat

          # 既存セッションで継続
          agent chat --session <uuid>

          # 特定エージェントと対話
          agent chat --agent dev_orchestrator
        """
        ctx.initialize()

        # セッションIDの処理
        session_uuid = None
        session_resumed = False
        if session_id:
            # 既存セッションのロード
            session_uuid = _load_or_create_session(ctx, session_id)
            session_resumed = True
        else:
            # 前回のセッションを自動ロード（新機能）
            last_session_id = _get_last_session(ctx)
            if last_session_id:
                session_uuid = last_session_id
                session_resumed = True
                click.echo(f"前回のセッション {session_uuid} を自動的に継続します。\n")
            else:
                session_uuid = uuid4()
                click.echo(f"新しいセッション {session_uuid} を開始しました。\n")

        # 再開されたセッションの場合は、セッションを in_progress 状態に更新
        if session_resumed:
            try:
                session_repository = SessionStateRepository(ctx.db)
                progress_manager = ProgressManager(
                    session_repository=session_repository,
                    config=ctx.config,
                )
                progress_manager.resume_session(session_uuid)
            except Exception as e:
                logger.warning(f"セッション再開に失敗: {e}")

        # エージェント指定時は存在確認
        if agent_id:
            agent_def = ctx.agent_registry.get_by_id(agent_id)
            if not agent_def:
                click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
                click.echo("\nヒント: agent list で登録済みエージェントを確認してください", err=True)
                sys.exit(2)
            click.echo(f"エージェント: {agent_def.name} ({agent_id})")
        else:
            click.echo("モード: オーケストレーター経由（自動ルーティング）")

        click.echo("対話モードを開始しました。'exit' または 'quit' で終了します。\n")

        # セッションの初期状態を保存（新規セッションの場合）
        if not session_resumed:
            _save_session(ctx, session_uuid, "対話モード開始")

        try:
            while True:
                # プロンプトを表示してユーザー入力を受け付ける
                try:
                    user_input = click.prompt("agent>", type=str, default="", show_default=False)
                except (EOFError, click.Abort):
                    # Ctrl+D または入力が閉じられた場合
                    click.echo("\n対話モードを終了します。")
                    break

                # 終了コマンドチェック
                if user_input.strip().lower() in ['exit', 'quit']:
                    # セッションを閉じる
                    _close_session(ctx, session_uuid)
                    click.echo("対話モードを終了します。")
                    break

                # 空入力はスキップ
                if not user_input.strip():
                    continue

                # タスク実行前にセッション情報を更新
                _update_session(ctx, session_uuid, user_input)

                # タスクを実行
                try:
                    if agent_id:
                        # 特定エージェントに直接依頼
                        agent_def = ctx.agent_registry.get_by_id(agent_id)

                        with Spinner(f"{agent_def.name}が処理中"):
                            result = ctx.llm_task_executor.execute_task_with_tools(
                                agent_id=agent_id,
                                system_prompt=agent_def.system_prompt,
                                task_description=user_input,
                                perspective=None,
                            )

                        # 結果表示
                        _display_result(result, verbose)

                        # タスク完了後にセッション情報を更新
                        from src.llm.llm_task_executor import LLMTaskResult
                        if isinstance(result, LLMTaskResult):
                            output = result.content
                        elif isinstance(result, dict):
                            output = result.get('output', '結果が取得できませんでした')
                        else:
                            output = str(result)
                        _update_session_result(ctx, session_uuid, output)

                    else:
                        # オーケストレーター経由で実行
                        with Spinner("オーケストレーターが処理中"):
                            orchestrator_result = ctx.orchestrator.process_request(
                                task_summary=user_input,
                                items=[],
                                session_id=session_uuid,
                            )

                        # ルーティング結果を表示（verbose時のみ）
                        if verbose and hasattr(orchestrator_result, 'routing_decision') and orchestrator_result.routing_decision:
                            routing = orchestrator_result.routing_decision
                            click.echo(
                                f"\n[ルーティング] {routing.selected_agent_id} を選択 "
                                f"(スコア: {routing.confidence:.2f})"
                            )
                            if hasattr(routing, 'selection_reason'):
                                click.echo(f"[理由] {routing.selection_reason}")

                        # 結果表示
                        _display_orchestrator_result(orchestrator_result, verbose)

                        # タスク完了後にセッション情報を更新
                        output = "結果が取得できませんでした"
                        if hasattr(orchestrator_result, 'agent_result') and orchestrator_result.agent_result:
                            output = orchestrator_result.agent_result.get('output', output)
                        _update_session_result(ctx, session_uuid, output)

                except Exception as e:
                    click.echo(f"\n[エラー] タスク実行に失敗しました: {e}", err=True)
                    if verbose:
                        import traceback
                        click.echo(traceback.format_exc(), err=True)
                    click.echo("")  # 空行を追加して次の入力へ

        except KeyboardInterrupt:
            # Ctrl+C で終了
            # セッションを一時停止（中断）
            _pause_session(ctx, session_uuid)
            click.echo("\n\n対話モードを終了します。")
        except Exception as e:
            click.echo(f"\n[エラー] 予期しないエラーが発生しました: {e}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)


def _display_result(result, verbose: bool):
    """タスク実行結果を表示（直接エージェント実行時）"""
    from src.llm.llm_task_executor import LLMTaskResult

    click.echo("")

    # LLMTaskResult の場合
    if isinstance(result, LLMTaskResult):
        output = result.content
        click.echo(output)

        if verbose:
            click.echo("\n[詳細情報]")
            click.echo(f"  トークン使用量: {result.total_tokens}")
            click.echo(f"  イテレーション回数: {result.iterations}")
            click.echo(f"  検索されたメモリ: {result.searched_memories_count}件")
            click.echo(f"  ツール呼び出し回数: {len(result.tool_calls)}")
            click.echo(f"  停止理由: {result.stop_reason}")

    # 辞書形式の場合（従来のフォーマット）
    elif isinstance(result, dict):
        output = result.get('output', result.get('response', str(result)))
        click.echo(output)

        if verbose:
            click.echo("\n[詳細情報]")
            for key, value in result.items():
                if key != 'output':
                    click.echo(f"  {key}: {value}")

    # その他（文字列など）
    else:
        output = str(result)
        click.echo(output)

    click.echo("")  # 空行を追加


def _display_orchestrator_result(orchestrator_result, verbose: bool):
    """オーケストレーター実行結果を表示"""
    click.echo("")

    output = "結果が取得できませんでした"
    if hasattr(orchestrator_result, 'agent_result') and orchestrator_result.agent_result:
        output = orchestrator_result.agent_result.get('output', output)

    click.echo(output)

    if verbose:
        # セッションIDを表示
        session_id = getattr(orchestrator_result, 'session_id', 'unknown')
        click.echo(f"\n[セッション] {session_id}")

        # その他の詳細情報
        if hasattr(orchestrator_result, 'agent_result') and orchestrator_result.agent_result:
            click.echo("[詳細情報]")
            for key, value in orchestrator_result.agent_result.items():
                if key != 'output':
                    click.echo(f"  {key}: {value}")

    click.echo("")  # 空行を追加


def _get_last_session(ctx) -> Optional[UUID]:
    """前回のセッションIDを取得

    Args:
        ctx: CLIContext

    Returns:
        最新のセッションID、見つからない場合は None
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )

        # 最新のセッションを取得（status='paused' または 'in_progress'）
        sessions = progress_manager.get_recent_sessions(
            orchestrator_id="orchestrator_cli",
            limit=1,
        )

        if sessions:
            return sessions[0].session_id

    except Exception as e:
        logger.warning(f"前回セッションの取得に失敗: {e}")

    return None


def _load_or_create_session(ctx, session_id: str) -> UUID:
    """セッションをロードまたは作成

    Args:
        ctx: CLIContext
        session_id: セッションID文字列

    Returns:
        セッションUUID
    """
    try:
        session_uuid = UUID(session_id)

        # セッション存在確認
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )
        session_state = progress_manager.restore_state(session_uuid)

        if session_state:
            click.echo(f"セッションを復元しました: {session_uuid}")
            return session_uuid
        else:
            click.echo(f"セッションが見つかりません。新規作成します。")
            return uuid4()

    except ValueError:
        click.echo(f"無効なセッションID形式です。新規作成します。")
        return uuid4()


def _save_session(ctx, session_id: UUID, user_input: str):
    """セッション情報を保存

    Args:
        ctx: CLIContext
        session_id: セッションUUID
        user_input: ユーザー入力
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )

        # セッション情報を作成・更新
        progress_manager.save_state(
            session_id=session_id,
            task_tree={"tasks": []},
            current_task={"description": user_input},
            progress_percent=0,
        )

    except Exception as e:
        logger.warning(f"セッション保存に失敗: {e}")


def _update_session(ctx, session_id: UUID, user_input: str):
    """セッション情報を更新（タスク開始時）

    Args:
        ctx: CLIContext
        session_id: セッションUUID
        user_input: ユーザー入力
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )

        # 既存のセッション情報を取得
        state = progress_manager.restore_state(session_id)
        if state:
            # タスクツリーを更新
            task_tree = state.task_tree
            if "tasks" not in task_tree:
                task_tree["tasks"] = []

            # 新しいタスクを追加
            task_tree["tasks"].append({
                "description": user_input,
                "status": "in_progress",
            })

            # 進捗を更新
            progress_manager.save_state(
                session_id=session_id,
                task_tree=task_tree,
                current_task={"description": user_input},
                progress_percent=state.overall_progress_percent,
            )
        else:
            # セッションが存在しない場合は新規作成
            _save_session(ctx, session_id, user_input)

    except Exception as e:
        logger.warning(f"セッション更新に失敗: {e}")


def _update_session_result(ctx, session_id: UUID, output: str):
    """セッション情報を更新（タスク完了時）

    Args:
        ctx: CLIContext
        session_id: セッションUUID
        output: タスク出力
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )

        # 既存のセッション情報を取得
        state = progress_manager.restore_state(session_id)
        if state:
            # タスクツリーを更新（最後のタスクを完了にマーク）
            task_tree = state.task_tree
            if "tasks" in task_tree and task_tree["tasks"]:
                task_tree["tasks"][-1]["status"] = "completed"
                task_tree["tasks"][-1]["output"] = output

            # 進捗率を計算
            total = len(task_tree.get("tasks", []))
            completed = sum(1 for t in task_tree.get("tasks", []) if t.get("status") == "completed")
            progress_percent = int((completed / total * 100)) if total > 0 else 0

            # 進捗を更新
            progress_manager.save_state(
                session_id=session_id,
                task_tree=task_tree,
                current_task=None,
                progress_percent=progress_percent,
            )

    except Exception as e:
        logger.warning(f"セッション結果更新に失敗: {e}")


def _close_session(ctx, session_id: UUID):
    """セッションを閉じる（完了状態に）

    Args:
        ctx: CLIContext
        session_id: セッションUUID
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )
        progress_manager.close_session(session_id, status="completed")
    except Exception as e:
        logger.warning(f"セッション終了に失敗: {e}")


def _pause_session(ctx, session_id: UUID):
    """セッションを一時停止（中断状態に）

    Args:
        ctx: CLIContext
        session_id: セッションUUID
    """
    try:
        session_repository = SessionStateRepository(ctx.db)
        progress_manager = ProgressManager(
            session_repository=session_repository,
            config=ctx.config,
        )
        progress_manager.close_session(session_id, status="paused")
    except Exception as e:
        logger.warning(f"セッション一時停止に失敗: {e}")
