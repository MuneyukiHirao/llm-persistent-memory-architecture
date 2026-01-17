"""
メモリ確認・操作コマンド実装
"""

import click
import sys
import os
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.models.memory import AgentMemory


def memory_command(agent_group, pass_context):
    """memory コマンドを agent グループに追加"""
    
    @agent_group.command()
    @click.argument('agent_id')
    @click.option('--limit', default=10, help='表示するメモリ数')
    @click.option('--status', type=click.Choice(['active', 'archived']), help='ステータスでフィルタ')
    @click.option('--search', help='メモリを検索')
    @click.option('--perspective', help='重視する観点（検索時）')
    @click.option('--add', help='メモリを手動追加')
    @click.option('--add-file', type=click.Path(exists=True), help='ファイルからメモリを追加')
    @click.option('--archive', help='メモリをアーカイブ（memory_id指定）')
    @click.option('--show', help='メモリの詳細を表示（memory_id指定）')
    @pass_context
    def memory(ctx, agent_id: str, limit: int, status: Optional[str], search: Optional[str],
               perspective: Optional[str], add: Optional[str], add_file: Optional[str], 
               archive: Optional[str], show: Optional[str]):
        """エージェントのメモリを確認・操作する"""
        ctx.initialize()
        
        try:
            # エージェント存在確認
            agent_def = ctx.agent_registry.get_by_id(agent_id)
            if not agent_def:
                click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
                click.echo("\\nヒント: agent list で登録済みエージェントを確認してください", err=True)
                sys.exit(2)
            
            # メモリ詳細表示
            if show:
                _show_memory_detail(ctx, agent_id, show)
                return
            
            # メモリアーカイブ
            if archive:
                _archive_memory(ctx, agent_id, archive)
                return
            
            # メモリ追加
            if add:
                _add_memory(ctx, agent_id, add)
                return
            
            # ファイルからメモリ追加
            if add_file:
                _add_memory_from_file(ctx, agent_id, add_file)
                return
            
            # メモリ検索
            if search:
                _search_memories(ctx, agent_id, search, perspective, limit)
                return
            
            # メモリ一覧表示
            _list_memories(ctx, agent_id, limit, status)
            
        except Exception as e:
            click.echo(f"[エラー] メモリ操作に失敗しました: {e}", err=True)
            sys.exit(1)


def _list_memories(ctx, agent_id: str, limit: int, status_filter: Optional[str]):
    """メモリ一覧を表示"""
    try:
        memories = ctx.memory_repository.get_by_agent_id(agent_id, status="active")  # フィルタ後に制限
        
        # ステータスフィルタ
        if status_filter:
            memories = [m for m in memories if m.status == status_filter]
        
        # 制限適用
        memories = memories[:limit]
        
        if not memories:
            filter_msg = f" ({status_filter})" if status_filter else ""
            click.echo(f"{agent_id} のメモリ{filter_msg}はありません")
            return
        
        # 強度順でソート
        memories.sort(key=lambda x: x.strength, reverse=True)
        
        total_count = len(ctx.memory_repository.get_by_agent_id(agent_id, status="active"))
        filter_msg = f" ({status_filter})" if status_filter else ""
        click.echo(f"{agent_id} のメモリ{filter_msg} ({total_count}件中 上位{len(memories)}件):\\n")
        
        # ヘッダー
        click.echo(f"{'ID':<12} {'強度':<6} {'定着':<4} {'最終アクセス':<16} {'内容'}")
        click.echo("─" * 80)
        
        for memory in memories:
            # IDを短縮
            id_str = str(memory.id)
            short_id = id_str[:8] + "..." if len(id_str) > 11 else id_str
            
            # 最終アクセス日時
            if memory.last_accessed_at:
                access_time = memory.last_accessed_at.strftime("%m-%d %H:%M")
            else:
                access_time = "未アクセス"
            
            # 内容を短縮
            content_preview = memory.content[:40] + "..." if len(memory.content) > 43 else memory.content
            
            click.echo(
                f"{short_id:<12} "
                f"{memory.strength:<6.2f} "
                f"{memory.consolidation_level:<4} "
                f"{access_time:<16} "
                f"{content_preview}"
            )
        
        if total_count > limit:
            click.echo(f"\\n--limit を増やすか --search で絞り込んでください")
            
    except Exception as e:
        click.echo(f"メモリ一覧取得エラー: {e}")


def _search_memories(ctx, agent_id: str, query: str, perspective: Optional[str], limit: int):
    """メモリを検索"""
    try:
        click.echo(f"'{query}' でメモリを検索中...\\n")
        
        # ベクトル検索実行
        results = ctx.vector_search.search(
            agent_id=agent_id,
            query=query,
            top_k=limit,
            perspective=perspective
        )
        
        if not results:
            click.echo("検索結果が見つかりませんでした")
            return
        
        click.echo(f"検索結果 ({len(results)}件):\\n")
        
        # ヘッダー
        click.echo(f"{'スコア':<8} {'ID':<12} {'強度':<6} {'内容'}")
        click.echo("─" * 70)
        
        for result in results:
            memory = result.memory
            score = result.similarity_score
            
            # IDを短縮
            id_str = str(memory.id)
            short_id = id_str[:8] + "..." if len(id_str) > 11 else id_str
            
            # 内容を短縮
            content_preview = memory.content[:35] + "..." if len(memory.content) > 38 else memory.content
            
            click.echo(
                f"{score:<8.3f} "
                f"{short_id:<12} "
                f"{memory.strength:<6.2f} "
                f"{content_preview}"
            )
            
    except Exception as e:
        click.echo(f"検索エラー: {e}")


def _add_memory(ctx, agent_id: str, content: str):
    """メモリを手動追加"""
    try:
        click.echo(f"メモリを追加中...\\n")
        
        # エンベディング生成
        embedding = ctx.embedding_client.get_embedding(content)
        
        # メモリ作成
        memory = AgentMemory.create_from_education(
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            scope_level='project',  # デフォルト
        )
        
        # メモリ登録
        ctx.memory_repository.create(memory)
        
        click.echo(f"✓ メモリを追加しました")
        click.echo(f"  ID: {str(memory.id)[:8]}...")
        click.echo(f"  内容: {content[:50]}{'...' if len(content) > 50 else ''}")
        
    except Exception as e:
        click.echo(f"メモリ追加エラー: {e}")


def _add_memory_from_file(ctx, agent_id: str, file_path: str):
    """ファイルからメモリを追加"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            click.echo("[エラー] ファイルが空です", err=True)
            return
        
        _add_memory(ctx, agent_id, content)
        
    except Exception as e:
        click.echo(f"ファイル読み込みエラー: {e}")


def _archive_memory(ctx, agent_id: str, memory_id: str):
    """メモリをアーカイブ"""
    try:
        # メモリ存在確認
        memory = ctx.memory_repository.get_by_id(memory_id)
        if not memory:
            click.echo(f"[エラー] メモリ '{memory_id}' が見つかりません", err=True)
            return
        
        if memory.agent_id != agent_id:
            click.echo(f"[エラー] メモリ '{memory_id}' は {agent_id} のものではありません", err=True)
            return
        
        # アーカイブ実行
        memory.status = 'archived'
        ctx.memory_repository.update(memory)
        
        click.echo(f"✓ メモリをアーカイブしました: {memory_id[:8]}...")
        
    except Exception as e:
        click.echo(f"アーカイブエラー: {e}")


def _show_memory_detail(ctx, agent_id: str, memory_id: str):
    """メモリの詳細を表示"""
    try:
        # メモリ取得
        memory = ctx.memory_repository.get_by_id(memory_id)
        if not memory:
            click.echo(f"[エラー] メモリ '{memory_id}' が見つかりません", err=True)
            return
        
        if memory.agent_id != agent_id:
            click.echo(f"[エラー] メモリ '{memory_id}' は {agent_id} のものではありません", err=True)
            return
        
        # 詳細表示
        click.echo(f"メモリ詳細: {memory_id}\\n")
        
        click.echo("基本情報:")
        click.echo(f"  エージェント: {memory.agent_id}")
        click.echo(f"  ステータス: {memory.status}")
        click.echo(f"  強度: {memory.strength:.3f}")
        click.echo(f"  定着レベル: {memory.consolidation_level}")
        click.echo(f"  アクセス回数: {memory.access_count}")
        
        if memory.created_at:
            click.echo(f"  作成日時: {memory.created_at}")
        if memory.last_accessed_at:
            click.echo(f"  最終アクセス: {memory.last_accessed_at}")
        
        click.echo("\\nスコープ情報:")
        click.echo(f"  レベル: {memory.scope_level}")
        if memory.scope_domain:
            click.echo(f"  ドメイン: {memory.scope_domain}")
        if memory.scope_project:
            click.echo(f"  プロジェクト: {memory.scope_project}")
        
        click.echo("\\n内容:")
        click.echo(f"  {memory.content}")
        
    except Exception as e:
        click.echo(f"メモリ詳細取得エラー: {e}")