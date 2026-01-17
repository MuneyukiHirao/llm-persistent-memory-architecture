#!/usr/bin/env python3
"""
セッション自動継続機能の統合テスト

手動で実行して動作を確認するためのスクリプト
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from uuid import uuid4
from src.orchestrator.progress_manager import ProgressManager, SessionStateRepository
from src.db.connection import DatabaseConnection
from src.config.phase2_config import Phase2Config


def test_session_lifecycle():
    """セッションのライフサイクルをテスト"""
    print("=" * 60)
    print("セッション自動継続機能の統合テスト")
    print("=" * 60)

    # データベース接続
    db = DatabaseConnection()
    session_repository = SessionStateRepository(db)
    progress_manager = ProgressManager(session_repository, Phase2Config())

    # 1. 新規セッション作成
    print("\n1. 新規セッション作成")
    session_id = progress_manager.create_session(
        orchestrator_id="orchestrator_cli",
        user_request={
            "original": "テスト会話セッション",
            "clarified": "セッション自動継続のテスト",
        },
        task_tree={
            "tasks": [],
            "conversation_history": [],
        },
    )
    print(f"   セッションID: {session_id}")

    # 2. セッション状態を保存
    print("\n2. セッション状態を保存")
    progress_manager.save_state(
        session_id=session_id,
        task_tree={
            "tasks": [
                {"description": "タスク1", "status": "in_progress"}
            ],
            "conversation_history": [
                {
                    "user_input": "最初の質問",
                    "agent_output": "最初の回答",
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        },
        current_task={"description": "タスク1"},
        progress_percent=50,
    )
    print("   保存完了")

    # 3. セッションを復元
    print("\n3. セッションを復元")
    state = progress_manager.restore_state(session_id)
    if state:
        print(f"   セッションID: {state.session_id}")
        print(f"   ユーザーリクエスト: {state.user_request}")
        print(f"   進捗率: {state.overall_progress_percent}%")
        print(f"   会話履歴: {len(state.task_tree.get('conversation_history', []))}件")
        if state.task_tree.get("conversation_history"):
            history = state.task_tree["conversation_history"][0]
            print(f"     - ユーザー入力: {history['user_input']}")
            print(f"     - エージェント出力: {history['agent_output']}")
    else:
        print("   セッションが見つかりません")
        return False

    # 4. 最新のセッションを取得
    print("\n4. 最新のセッションを取得")
    recent_sessions = progress_manager.get_recent_sessions(
        orchestrator_id="orchestrator_cli",
        limit=1,
    )
    if recent_sessions:
        latest = recent_sessions[0]
        print(f"   最新セッションID: {latest.session_id}")
        print(f"   ステータス: {latest.status}")
        print(f"   最終アクティビティ: {latest.last_activity_at}")
    else:
        print("   セッションが見つかりません")

    # 5. セッションを一時停止
    print("\n5. セッションを一時停止")
    progress_manager.pause_session(session_id)
    state = progress_manager.restore_state(session_id)
    print(f"   ステータス: {state.status}")

    # 6. セッションを再開
    print("\n6. セッションを再開")
    progress_manager.resume_session(session_id)
    state = progress_manager.restore_state(session_id)
    print(f"   ステータス: {state.status}")

    # 7. セッションを完了
    print("\n7. セッションを完了")
    progress_manager.close_session(session_id, status="completed")
    state = progress_manager.restore_state(session_id)
    print(f"   ステータス: {state.status}")
    print(f"   進捗率: {state.overall_progress_percent}%")

    print("\n" + "=" * 60)
    print("テスト完了！")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_session_lifecycle()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
