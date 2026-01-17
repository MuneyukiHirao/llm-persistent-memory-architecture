"""
エージェント状態確認コマンド実装
"""

import click
import sys
import os
from datetime import datetime
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


def status_command(agent_group, pass_context):
    """status コマンドを agent グループに追加"""
    
    @agent_group.command()
    @click.argument('agent_id')
    @click.option('--memories', is_flag=True, help='最新メモリも表示')
    @click.option('--limit', default=5, help='表示するメモリ数（--memories指定時）')
    @click.option('--statistics', is_flag=True, help='統計情報を表示')
    @pass_context
    def status(ctx, agent_id: str, memories: bool, limit: int, statistics: bool):
        """特定エージェントの詳細状態を表示する"""
        ctx.initialize()
        
        try:
            # エージェント存在確認
            agent_def = ctx.agent_registry.get_by_id(agent_id)
            if not agent_def:
                click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
                click.echo("\\nヒント: agent list で登録済みエージェントを確認してください", err=True)
                sys.exit(2)
            
            # 基本情報を表示
            click.echo(f"エージェント: {agent_id}\\n")
            
            click.echo("基本情報:")
            click.echo(f"  名前: {agent_def.name}")
            click.echo(f"  役割: {agent_def.role}")
            click.echo(f"  状態: active")  # TODO: 実際の状態管理実装後に更新
            
            # 作成日・更新日（TODO: エージェント定義に追加後に実装）
            # click.echo(f"  作成日: {agent_def.created_at}")
            # click.echo(f"  最終更新: {agent_def.updated_at}")
            
            click.echo("\\n観点:")
            for perspective in agent_def.perspectives:
                click.echo(f"  - {perspective}")
            
            if agent_def.capabilities:
                click.echo("\\n能力タグ:")
                capabilities_str = ", ".join(agent_def.capabilities)
                click.echo(f"  - {capabilities_str}")
            
            # メモリ統計
            try:
                all_memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")
                active_memories = [m for m in all_memories if m.status == 'active']
                archived_memories = [m for m in all_memories if m.status == 'archived']
                
                click.echo("\\nメモリ統計:")
                click.echo(f"  総数: {len(all_memories)}")
                click.echo(f"  アクティブ: {len(active_memories)}")
                click.echo(f"  アーカイブ: {len(archived_memories)}")
                
                if active_memories:
                    avg_strength = sum(m.strength for m in active_memories) / len(active_memories)
                    max_consolidation = max(m.consolidation_level for m in active_memories)
                    click.echo(f"  平均強度: {avg_strength:.2f}")
                    click.echo(f"  最高定着レベル: {max_consolidation}")
                
            except Exception as e:
                click.echo(f"\\nメモリ統計: 取得エラー ({e})")
            
            # 統計情報表示
            if statistics:
                _display_statistics(ctx, agent_id)
            
            # 最新メモリ表示
            if memories:
                _display_recent_memories(ctx, agent_id, limit)
                
        except Exception as e:
            click.echo(f"[エラー] エージェント状態の取得に失敗しました: {e}", err=True)
            sys.exit(1)


def _display_statistics(ctx, agent_id: str):
    """統計情報を表示"""
    click.echo("\\n詳細統計:")
    
    try:
        all_memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")
        
        if not all_memories:
            click.echo("  統計データがありません")
            return
        
        # スコープ別統計
        scope_stats = {}
        for memory in all_memories:
            scope = memory.scope_level
            if scope not in scope_stats:
                scope_stats[scope] = {'count': 0, 'avg_strength': 0, 'total_strength': 0}
            scope_stats[scope]['count'] += 1
            scope_stats[scope]['total_strength'] += memory.strength
        
        for scope, stats in scope_stats.items():
            stats['avg_strength'] = stats['total_strength'] / stats['count']
        
        click.echo("  スコープ別:")
        for scope, stats in scope_stats.items():
            click.echo(f"    {scope}: {stats['count']}件 (平均強度: {stats['avg_strength']:.2f})")
        
        # アクセス統計
        recent_access = [m for m in all_memories if m.last_accessed_at]
        if recent_access:
            recent_access.sort(key=lambda x: x.last_accessed_at, reverse=True)
            latest_access = recent_access[0].last_accessed_at
            click.echo(f"  最終アクセス: {latest_access}")
        
    except Exception as e:
        click.echo(f"  統計計算エラー: {e}")


def _display_recent_memories(ctx, agent_id: str, limit: int):
    """最新メモリを表示"""
    click.echo(f"\\n最新メモリ ({limit}件):")
    
    try:
        memories = ctx.memory_repository.search_by_agent(agent_id, limit=limit)
        
        if not memories:
            click.echo("  メモリがありません")
            return
        
        # 強度順でソート
        memories.sort(key=lambda x: x.strength, reverse=True)
        
        # ヘッダー
        click.echo(f"{'ID':<12} {'強度':<6} {'定着':<4} {'最終アクセス':<16} {'内容'}")
        click.echo("─" * 70)
        
        for memory in memories[:limit]:
            # IDを短縮
            short_id = memory.memory_id[:8] + "..." if len(memory.memory_id) > 11 else memory.memory_id
            
            # 最終アクセス日時
            if memory.last_accessed_at:
                access_time = memory.last_accessed_at.strftime("%m-%d %H:%M")
            else:
                access_time = "未アクセス"
            
            # 内容を短縮
            content_preview = memory.content[:30] + "..." if len(memory.content) > 33 else memory.content
            
            click.echo(
                f"{short_id:<12} "
                f"{memory.strength:<6.2f} "
                f"{memory.consolidation_level:<4} "
                f"{access_time:<16} "
                f"{content_preview}"
            )
            
    except Exception as e:
        click.echo(f"  メモリ取得エラー: {e}")