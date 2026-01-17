"""
エージェント登録コマンド実装
"""

import click
import sys
import os
from typing import List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents.agent_registry import AgentDefinition
from src.models.memory import AgentMemory
from src.cli.utils.yaml_loader import load_yaml, validate_agent_definition, YamlValidationError


def register_agent_command(agent_group, pass_context):
    """register コマンドを agent グループに追加"""
    
    @agent_group.command()
    @click.option('-f', '--file', 'files', multiple=True, type=click.Path(exists=True), help='エージェント定義YAMLファイル')
    @click.option('--id', 'agent_id', help='エージェントID（CLI登録用）')
    @click.option('--name', help='エージェント名（CLI登録用）')
    @click.option('--role', help='エージェント役割（CLI登録用）')
    @click.option('--perspectives', help='観点（カンマ区切り）')
    @click.option('--capabilities', help='能力タグ（カンマ区切り）')
    @click.option('--update', is_flag=True, help='既存エージェントを更新')
    @click.option('--dry-run', is_flag=True, help='登録内容を確認（実行しない）')
    @pass_context
    def register(ctx, files: List[str], agent_id: str, name: str, role: str,
                 perspectives: str, capabilities: str, update: bool, dry_run: bool):
        """エージェントを登録する"""
        ctx.initialize()
        
        try:
            if files and any([agent_id, name, role, perspectives, capabilities]):
                click.echo("[エラー] --file とCLI直接指定は同時に使用できません", err=True)
                sys.exit(2)

            if files:
                for file_path in files:
                    data = load_yaml(file_path)
                    validate_agent_definition(data)
                    _register_agent(ctx, data, update, dry_run)
                return

            if not agent_id or not name or not role or not perspectives:
                click.echo("[エラー] CLI登録は --id/--name/--role/--perspectives が必須です", err=True)
                sys.exit(2)

            data = {
                "agent_id": agent_id,
                "name": name,
                "role": role,
                "perspectives": _split_csv(perspectives),
                "capabilities": _split_csv(capabilities) if capabilities else [],
            }
            validate_agent_definition(data)
            _register_agent(ctx, data, update, dry_run)

        except YamlValidationError as e:
            click.echo(f"[エラー] YAML検証に失敗しました: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"[エラー] エージェント登録に失敗しました: {e}", err=True)
            sys.exit(1)


def _register_agent(ctx, data: dict, update: bool, dry_run: bool) -> None:
    """登録処理を実行"""
    agent_def = AgentDefinition(
        agent_id=data["agent_id"],
        name=data["name"],
        role=data["role"],
        perspectives=data["perspectives"],
        system_prompt=data.get("system_prompt", ""),
        capabilities=data.get("capabilities", []),
    )

    if dry_run:
        _display_agent_definition(agent_def, data.get("initial_memories", []))
        click.echo("\n[DRY RUN] 実際の登録は行いませんでした")
        return

    existing = ctx.agent_registry.get_by_id(agent_def.agent_id)
    if existing and not update:
        click.echo(
            f"[エラー] エージェント {agent_def.agent_id} は既に存在します。\n"
            f"更新する場合は --update を指定してください。",
            err=True,
        )
        sys.exit(2)

    if existing and update:
        ctx.agent_registry.update(agent_def)
        click.echo(f"エージェントを更新しました: {agent_def.agent_id}")
    else:
        ctx.agent_registry.register(agent_def)
        click.echo(f"エージェントを登録しました: {agent_def.agent_id}")

    click.echo(f"  名前: {agent_def.name}")
    click.echo(f"  役割: {agent_def.role}")
    click.echo(f"  観点: {', '.join(agent_def.perspectives)}")
    if agent_def.capabilities:
        click.echo(f"  能力: {', '.join(agent_def.capabilities)}")

    initial_memories = data.get("initial_memories", [])
    if initial_memories:
        registered_count = 0
        for mem_data in initial_memories:
            try:
                embedding = ctx.embedding_client.get_embedding(mem_data["content"])
                memory = AgentMemory.create_from_education(
                    agent_id=agent_def.agent_id,
                    content=mem_data["content"],
                    embedding=embedding,
                    scope_level=mem_data.get("scope_level", "project"),
                    scope_domain=mem_data.get("scope_domain"),
                    scope_project=mem_data.get("scope_project"),
                )
                ctx.memory_repository.create(memory)
                registered_count += 1
            except Exception as e:
                click.echo(f"  ⚠ 初期メモリ登録エラー: {e}", err=True)

        if registered_count > 0:
            click.echo(f"\n初期メモリ {registered_count} 件を登録しました。")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _display_agent_definition(agent_def: AgentDefinition, initial_memories: list):
    """エージェント定義を表示する（dry-run用）"""
    click.echo("登録予定のエージェント:")
    click.echo(f"  ID: {agent_def.agent_id}")
    click.echo(f"  名前: {agent_def.name}")
    click.echo(f"  役割: {agent_def.role}")
    click.echo(f"  観点: {', '.join(agent_def.perspectives)}")
    
    if agent_def.capabilities:
        click.echo(f"  能力: {', '.join(agent_def.capabilities)}")
    
    if agent_def.system_prompt:
        click.echo(f"  システムプロンプト: {agent_def.system_prompt[:100]}{'...' if len(agent_def.system_prompt) > 100 else ''}")
    
    if initial_memories:
        click.echo(f"  初期メモリ: {len(initial_memories)} 件")
        for i, mem in enumerate(initial_memories[:3], 1):  # 最初の3件のみ表示
            content_preview = mem['content'][:50] + '...' if len(mem['content']) > 50 else mem['content']
            click.echo(f"    {i}. {content_preview}")
        if len(initial_memories) > 3:
            click.echo(f"    ... 他 {len(initial_memories) - 3} 件")
