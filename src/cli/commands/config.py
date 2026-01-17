"""
設定管理コマンド実装
"""

import click
import sys
import os
from typing import Optional
from dataclasses import fields, is_dataclass

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.config.phase2_config import Phase2Config
from src.config.llm_config import LLMConfig


# 設定可能なキーとその説明（日本語）
CONFIG_DESCRIPTIONS = {
    # Phase2Config から
    "similarity_threshold": "類似度閾値（0.0-1.0）",
    "top_k_results": "検索結果の最大取得件数",
    "candidate_limit": "Stage 1 の最大候補数",
    "archive_threshold": "アーカイブ閾値（これ以下で記憶をアーカイブ）",
    "initial_strength": "新規記憶の初期強度",
    "strength_increment_on_use": "使用時の強度増加量",
    "max_active_memories": "アクティブ記憶の最大件数",
    "embedding_model": "エンベディングモデル名",
    "embedding_dimension": "エンベディング次元数",
    "default_scope_level": "デフォルトスコープ（universal/domain/project）",
    "current_project_id": "現在のプロジェクトID",
    "orchestrator_model": "オーケストレーター用LLMモデル",
    "input_processor_model": "入力処理層用LLMモデル",
    "routing_method": "ルーティング方式（rule_based/similarity/llm）",
    "routing_similarity_threshold": "ルーティング類似度閾値",
    # LLMConfig から
    "model_name": "LLMモデル名",
    "max_tokens": "最大出力トークン数",
    "temperature": "サンプリング温度（0.0-1.0）",
    "timeout_seconds": "APIタイムアウト（秒）",
    "max_retries": "最大リトライ回数",
}

# 設定の型マッピング
CONFIG_TYPES = {
    "similarity_threshold": float,
    "top_k_results": int,
    "candidate_limit": int,
    "archive_threshold": float,
    "initial_strength": float,
    "strength_increment_on_use": float,
    "max_active_memories": int,
    "embedding_dimension": int,
    "routing_similarity_threshold": float,
    "max_tokens": int,
    "temperature": float,
    "timeout_seconds": int,
    "max_retries": int,
}


def config_command(agent_group, pass_context):
    """config コマンドを agent グループに追加"""

    @agent_group.group()
    @pass_context
    def config(ctx):
        """設定の表示と変更を行う

        \b
        サブコマンド:
          show   - 現在の設定を表示
          set    - 設定値を変更
          reset  - 設定をデフォルトにリセット

        \b
        例:
          agent config show
          agent config set similarity_threshold 0.25
          agent config reset
        """
        pass

    @config.command()
    @click.option('--all', 'show_all', is_flag=True, help='すべての設定を表示')
    @click.option('--json', 'as_json', is_flag=True, help='JSON形式で出力')
    @pass_context
    def show(ctx, show_all: bool, as_json: bool):
        """現在の設定を表示する

        主要な設定項目を表示します。--all オプションですべての設定を表示できます。

        \b
        例:
          agent config show          # 主要設定を表示
          agent config show --all    # すべての設定を表示
          agent config show --json   # JSON形式で出力
        """
        ctx.initialize()

        try:
            phase2_config = ctx.config
            llm_config = LLMConfig()

            if as_json:
                _display_config_json(phase2_config, llm_config, show_all)
            else:
                _display_config_table(phase2_config, llm_config, show_all)

        except Exception as e:
            click.echo(f"[エラー] 設定の取得に失敗しました: {e}", err=True)
            sys.exit(1)

    @config.command()
    @click.argument('key')
    @click.argument('value')
    @pass_context
    def set(ctx, key: str, value: str):
        """設定値を変更する

        指定したキーの設定値を変更します。
        変更は現在のセッションでのみ有効です。
        永続的に変更するには環境変数を設定してください。

        \b
        設定可能なキー:
          similarity_threshold    - 類似度閾値（0.0-1.0）
          top_k_results          - 検索結果の最大取得件数
          model_name             - LLMモデル名
          max_tokens             - 最大出力トークン数
          temperature            - サンプリング温度（0.0-1.0）

        \b
        例:
          agent config set similarity_threshold 0.25
          agent config set top_k_results 15
          agent config set temperature 0.5
        """
        ctx.initialize()

        try:
            # キーの検証
            if key not in CONFIG_DESCRIPTIONS:
                click.echo(f"[エラー] 不明な設定キー: {key}", err=True)
                click.echo("\n設定可能なキー:", err=True)
                for k, desc in sorted(CONFIG_DESCRIPTIONS.items()):
                    click.echo(f"  {k}: {desc}", err=True)
                sys.exit(1)

            # 値の型変換
            converted_value = _convert_value(key, value)

            # 値の検証
            validation_error = _validate_value(key, converted_value)
            if validation_error:
                click.echo(f"[エラー] {validation_error}", err=True)
                sys.exit(1)

            # 設定を更新（Phase2Config または LLMConfig のどちらか）
            config_updated = False

            # Phase2Config の属性かチェック
            if hasattr(ctx.config, key):
                old_value = getattr(ctx.config, key)
                setattr(ctx.config, key, converted_value)
                config_updated = True
                click.echo(f"設定を更新しました:")
                click.echo(f"  {key}: {old_value} -> {converted_value}")

            # LLMConfig の属性かチェック
            llm_config = LLMConfig()
            if hasattr(llm_config, key):
                old_value = getattr(llm_config, key)
                # 注: LLMConfig はグローバルインスタンスではないため、
                # セッション中の変更は限定的
                click.echo(f"\n[注意] LLM設定 ({key}) はセッション中のみ有効です。")
                click.echo(f"永続的に変更するには環境変数を設定してください:")
                if key == "model_name":
                    click.echo(f"  export CLAUDE_MODEL={value}")
                config_updated = True

            if not config_updated:
                click.echo(f"[エラー] 設定の更新に失敗しました", err=True)
                sys.exit(1)

            click.echo("\n[注意] この変更は現在のセッションでのみ有効です。")

        except ValueError as e:
            click.echo(f"[エラー] 値の変換に失敗しました: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"[エラー] 設定の変更に失敗しました: {e}", err=True)
            sys.exit(1)

    @config.command()
    @click.argument('key', required=False)
    @click.option('--yes', '-y', is_flag=True, help='確認をスキップ')
    @pass_context
    def reset(ctx, key: Optional[str], yes: bool):
        """設定をデフォルトにリセットする

        特定のキーを指定すると、そのキーのみリセットします。
        キーを指定しない場合は、すべての設定をリセットします。

        \b
        例:
          agent config reset                       # すべてリセット
          agent config reset similarity_threshold  # 特定キーのみリセット
          agent config reset -y                    # 確認なしでリセット
        """
        ctx.initialize()

        try:
            default_config = Phase2Config()
            default_llm_config = LLMConfig()

            if key:
                # 特定キーのリセット
                if key not in CONFIG_DESCRIPTIONS:
                    click.echo(f"[エラー] 不明な設定キー: {key}", err=True)
                    sys.exit(1)

                # デフォルト値を取得
                if hasattr(default_config, key):
                    default_value = getattr(default_config, key)
                    current_value = getattr(ctx.config, key)

                    if not yes:
                        click.confirm(
                            f"{key} を {current_value} から {default_value} にリセットしますか？",
                            abort=True
                        )

                    setattr(ctx.config, key, default_value)
                    click.echo(f"設定をリセットしました:")
                    click.echo(f"  {key}: {current_value} -> {default_value}")

                elif hasattr(default_llm_config, key):
                    default_value = getattr(default_llm_config, key)
                    click.echo(f"LLM設定 ({key}) のデフォルト値: {default_value}")
                    click.echo("\n[注意] LLM設定のリセットは環境変数の削除が必要です。")
                else:
                    click.echo(f"[エラー] キー {key} が見つかりません", err=True)
                    sys.exit(1)
            else:
                # すべてリセット
                if not yes:
                    click.confirm(
                        "すべての設定をデフォルトにリセットしますか？",
                        abort=True
                    )

                # Phase2Config の主要設定をリセット
                reset_keys = [
                    "similarity_threshold",
                    "top_k_results",
                    "candidate_limit",
                    "archive_threshold",
                    "initial_strength",
                    "max_active_memories",
                    "routing_similarity_threshold",
                ]

                click.echo("設定をリセットしました:\n")
                for k in reset_keys:
                    if hasattr(ctx.config, k) and hasattr(default_config, k):
                        old_value = getattr(ctx.config, k)
                        new_value = getattr(default_config, k)
                        setattr(ctx.config, k, new_value)
                        if old_value != new_value:
                            click.echo(f"  {k}: {old_value} -> {new_value}")
                        else:
                            click.echo(f"  {k}: {new_value} (変更なし)")

                click.echo("\n[注意] この変更は現在のセッションでのみ有効です。")

        except click.Abort:
            click.echo("リセットをキャンセルしました")
            sys.exit(0)
        except Exception as e:
            click.echo(f"[エラー] 設定のリセットに失敗しました: {e}", err=True)
            sys.exit(1)


def _display_config_table(phase2_config: Phase2Config, llm_config: LLMConfig, show_all: bool):
    """設定をテーブル形式で表示"""
    click.echo("現在の設定:\n")

    # 主要設定（常に表示）
    click.echo("【検索設定】")
    click.echo(f"  similarity_threshold:  {phase2_config.similarity_threshold}  ({CONFIG_DESCRIPTIONS['similarity_threshold']})")
    click.echo(f"  top_k_results:         {phase2_config.top_k_results}  ({CONFIG_DESCRIPTIONS['top_k_results']})")
    click.echo(f"  candidate_limit:       {phase2_config.candidate_limit}  ({CONFIG_DESCRIPTIONS['candidate_limit']})")

    click.echo("\n【記憶管理】")
    click.echo(f"  initial_strength:      {phase2_config.initial_strength}  ({CONFIG_DESCRIPTIONS['initial_strength']})")
    click.echo(f"  archive_threshold:     {phase2_config.archive_threshold}  ({CONFIG_DESCRIPTIONS['archive_threshold']})")
    click.echo(f"  max_active_memories:   {phase2_config.max_active_memories}  ({CONFIG_DESCRIPTIONS['max_active_memories']})")

    click.echo("\n【LLM設定】")
    click.echo(f"  model_name:            {llm_config.model_name}  ({CONFIG_DESCRIPTIONS['model_name']})")
    click.echo(f"  max_tokens:            {llm_config.max_tokens}  ({CONFIG_DESCRIPTIONS['max_tokens']})")
    click.echo(f"  temperature:           {llm_config.temperature}  ({CONFIG_DESCRIPTIONS['temperature']})")

    if show_all:
        click.echo("\n【オーケストレーター設定】")
        click.echo(f"  orchestrator_model:        {phase2_config.orchestrator_model}")
        click.echo(f"  input_processor_model:     {phase2_config.input_processor_model}")
        click.echo(f"  routing_method:            {phase2_config.routing_method}")
        click.echo(f"  routing_similarity_threshold: {phase2_config.routing_similarity_threshold}")

        click.echo("\n【エンベディング設定】")
        click.echo(f"  embedding_model:      {phase2_config.embedding_model}")
        click.echo(f"  embedding_dimension:  {phase2_config.embedding_dimension}")

        click.echo("\n【スコープ設定】")
        click.echo(f"  default_scope_level:  {phase2_config.default_scope_level}")
        click.echo(f"  current_project_id:   {phase2_config.current_project_id}")

        click.echo("\n【その他】")
        click.echo(f"  timeout_seconds:      {llm_config.timeout_seconds}")
        click.echo(f"  max_retries:          {llm_config.max_retries}")
        click.echo(f"  enable_rate_limiting: {llm_config.enable_rate_limiting}")

    click.echo("\n" + "-" * 60)
    click.echo("ヒント: --all オプションですべての設定を表示できます")
    click.echo("       agent config set <key> <value> で設定を変更できます")


def _display_config_json(phase2_config: Phase2Config, llm_config: LLMConfig, show_all: bool):
    """設定をJSON形式で表示"""
    import json

    config_dict = {
        "search": {
            "similarity_threshold": phase2_config.similarity_threshold,
            "top_k_results": phase2_config.top_k_results,
            "candidate_limit": phase2_config.candidate_limit,
        },
        "memory": {
            "initial_strength": phase2_config.initial_strength,
            "archive_threshold": phase2_config.archive_threshold,
            "max_active_memories": phase2_config.max_active_memories,
        },
        "llm": {
            "model_name": llm_config.model_name,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature,
        },
    }

    if show_all:
        config_dict["orchestrator"] = {
            "orchestrator_model": phase2_config.orchestrator_model,
            "input_processor_model": phase2_config.input_processor_model,
            "routing_method": phase2_config.routing_method,
            "routing_similarity_threshold": phase2_config.routing_similarity_threshold,
        }
        config_dict["embedding"] = {
            "embedding_model": phase2_config.embedding_model,
            "embedding_dimension": phase2_config.embedding_dimension,
        }
        config_dict["scope"] = {
            "default_scope_level": phase2_config.default_scope_level,
            "current_project_id": phase2_config.current_project_id,
        }
        config_dict["other"] = {
            "timeout_seconds": llm_config.timeout_seconds,
            "max_retries": llm_config.max_retries,
            "enable_rate_limiting": llm_config.enable_rate_limiting,
        }

    click.echo(json.dumps(config_dict, indent=2, ensure_ascii=False))


def _convert_value(key: str, value: str):
    """文字列値を適切な型に変換"""
    if key in CONFIG_TYPES:
        target_type = CONFIG_TYPES[key]
        if target_type == float:
            return float(value)
        elif target_type == int:
            return int(value)

    # 文字列のまま返す
    return value


def _validate_value(key: str, value) -> Optional[str]:
    """設定値を検証し、エラーメッセージを返す（問題なければNone）"""

    if key == "similarity_threshold":
        if not (0.0 <= value <= 1.0):
            return f"similarity_threshold は 0.0-1.0 の範囲である必要があります: {value}"

    elif key == "temperature":
        if not (0.0 <= value <= 1.0):
            return f"temperature は 0.0-1.0 の範囲である必要があります: {value}"

    elif key == "top_k_results":
        if value <= 0:
            return f"top_k_results は正の整数である必要があります: {value}"
        if value > 100:
            return f"top_k_results は 100 以下である必要があります: {value}"

    elif key == "max_tokens":
        if value <= 0:
            return f"max_tokens は正の整数である必要があります: {value}"

    elif key == "archive_threshold":
        if not (0.0 <= value <= 1.0):
            return f"archive_threshold は 0.0-1.0 の範囲である必要があります: {value}"

    elif key == "routing_similarity_threshold":
        if not (0.0 <= value <= 1.0):
            return f"routing_similarity_threshold は 0.0-1.0 の範囲である必要があります: {value}"

    elif key == "default_scope_level":
        valid_scopes = ["universal", "domain", "project"]
        if value not in valid_scopes:
            return f"default_scope_level は {valid_scopes} のいずれかである必要があります: {value}"

    elif key == "routing_method":
        valid_methods = ["rule_based", "similarity", "llm"]
        if value not in valid_methods:
            return f"routing_method は {valid_methods} のいずれかである必要があります: {value}"

    return None
