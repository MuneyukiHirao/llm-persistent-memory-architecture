"""
睡眠フェーズコマンド実装
"""

import click
import sys
import os
from typing import List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.core.sleep_processor import SleepPhaseProcessor


def sleep_command(agent_group, pass_context):
    """sleep コマンドを agent グループに追加"""
    
    @agent_group.command()
    @click.argument('agent_id', required=False)
    @click.option('--all', is_flag=True, help='全エージェントの睡眠を実行')
    @click.option('--dry-run', is_flag=True, help='実行せずに影響を確認')
    @click.option('--verbose', is_flag=True, help='詳細ログを表示')
    @pass_context
    def sleep(ctx, agent_id: str, all: bool, dry_run: bool, verbose: bool):
        """睡眠フェーズを手動で実行する"""
        ctx.initialize()
        
        try:
            if all:
                # 全エージェントの睡眠
                _sleep_all_agents(ctx, dry_run, verbose)
            elif agent_id:
                # 特定エージェントの睡眠
                _sleep_single_agent(ctx, agent_id, dry_run, verbose)
            else:
                click.echo("[エラー] エージェントIDを指定するか --all オプションを使用してください", err=True)
                click.echo("\\n使用例:", err=True)
                click.echo("  agent sleep research_agent", err=True)
                click.echo("  agent sleep --all", err=True)
                sys.exit(2)
                
        except Exception as e:
            click.echo(f"[エラー] 睡眠フェーズの実行に失敗しました: {e}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)


def _sleep_single_agent(ctx, agent_id: str, dry_run: bool, verbose: bool):
    """特定エージェントの睡眠フェーズを実行"""
    # エージェント存在確認
    agent_def = ctx.agent_registry.get_by_id(agent_id)
    if not agent_def:
        click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
        click.echo("\\nヒント: agent list で登録済みエージェントを確認してください", err=True)
        sys.exit(2)
    
    click.echo(f"{agent_id} の睡眠フェーズを実行中...\\n")
    
    # 睡眠前の状態を取得
    pre_memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")
    pre_active = [m for m in pre_memories if m.status == 'active']
    pre_archived = [m for m in pre_memories if m.status == 'archived']
    
    if verbose:
        click.echo(f"睡眠前の状態:")
        click.echo(f"  アクティブメモリ: {len(pre_active)}件")
        click.echo(f"  アーカイブメモリ: {len(pre_archived)}件")
        click.echo()
    
    if dry_run:
        click.echo("[DRY RUN] 実際の処理は行いません\\n")
        _simulate_sleep_process(ctx, agent_id, pre_active, verbose)
        return
    
    # 実際の睡眠フェーズ実行
    try:
        sleep_processor = SleepPhaseProcessor(
            repository=ctx.memory_repository,
            config=ctx.config
        )
        
        result = sleep_processor.process_agent(agent_id)
        
        # 結果表示
        click.echo("減衰処理:")
        click.echo(f"  対象メモリ: {result.get('processed_count', len(pre_active))} 件")
        click.echo(f"  処理済み: {result.get('processed_count', len(pre_active))} 件")
        
        if 'average_decay_rate' in result:
            click.echo(f"  平均減衰率: {result['average_decay_rate']:.3f}")
        
        click.echo("\\nアーカイブ処理:")
        archived_count = result.get('archived_count', 0)
        click.echo(f"  新規アーカイブ: {archived_count} 件")
        
        # Phase 1では統合処理はスキップ
        click.echo("\\n統合処理:")
        click.echo("  (Phase 1: スキップ)")
        
        # 睡眠後の状態を取得
        post_memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")
        post_active = [m for m in post_memories if m.status == 'active']
        post_archived = [m for m in post_memories if m.status == 'archived']
        
        click.echo("\\n睡眠フェーズ完了:")
        click.echo(f"  アクティブメモリ: {len(post_active)} 件")
        click.echo(f"  新規アーカイブ: {len(post_archived) - len(pre_archived)} 件")
        
        if verbose:
            click.echo(f"\\n詳細:")
            click.echo(f"  睡眠前アクティブ: {len(pre_active)} → 睡眠後アクティブ: {len(post_active)}")
            click.echo(f"  睡眠前アーカイブ: {len(pre_archived)} → 睡眠後アーカイブ: {len(post_archived)}")
        
    except Exception as e:
        click.echo(f"睡眠フェーズ実行エラー: {e}")
        raise


def _sleep_all_agents(ctx, dry_run: bool, verbose: bool):
    """全エージェントの睡眠フェーズを実行"""
    # 全エージェント取得
    agents = ctx.agent_registry.list_all()
    
    if not agents:
        click.echo("登録済みエージェントがありません")
        return
    
    click.echo(f"全エージェント ({len(agents)}件) の睡眠フェーズを実行中...\\n")
    
    success_count = 0
    error_count = 0
    
    for agent_def in agents:
        try:
            click.echo(f"[{agent_def.agent_id}] 処理中...")
            
            if dry_run:
                click.echo(f"  [DRY RUN] 実際の処理は行いません")
            else:
                _sleep_single_agent(ctx, agent_def.agent_id, dry_run=False, verbose=False)
            
            success_count += 1
            click.echo(f"  ✓ 完了\\n")
            
        except Exception as e:
            error_count += 1
            click.echo(f"  ✗ エラー: {e}\\n")
            if verbose:
                import traceback
                click.echo(traceback.format_exc())
    
    # サマリー表示
    click.echo("─" * 40)
    click.echo(f"全エージェント睡眠フェーズ完了:")
    click.echo(f"  成功: {success_count}件")
    click.echo(f"  エラー: {error_count}件")


def _simulate_sleep_process(ctx, agent_id: str, active_memories: List, verbose: bool):
    """睡眠処理をシミュレート（dry-run用）"""
    if not active_memories:
        click.echo("処理対象のアクティブメモリがありません")
        return
    
    # 減衰シミュレーション
    threshold = ctx.config.memory_archive_threshold
    
    # 強度が閾値以下になる予想メモリ数を計算
    weak_memories = [m for m in active_memories if m.strength <= threshold * 1.1]  # 概算
    
    click.echo("予想される処理:")
    click.echo(f"  減衰処理対象: {len(active_memories)} 件")
    click.echo(f"  アーカイブ候補: 約 {len(weak_memories)} 件")
    
    if verbose:
        click.echo(f"\\n詳細:")
        click.echo(f"  現在の強度分布:")
        
        # 強度別の分布を表示
        strength_ranges = [
            (0.0, 0.1, "極弱"),
            (0.1, 0.3, "弱"),
            (0.3, 0.7, "中"),
            (0.7, 1.0, "強"),
            (1.0, float('inf'), "極強")
        ]
        
        for min_val, max_val, label in strength_ranges:
            count = len([m for m in active_memories if min_val <= m.strength < max_val])
            if count > 0:
                click.echo(f"    {label}({min_val}-{max_val if max_val != float('inf') else '∞'}): {count}件")