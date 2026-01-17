"""
教育プロセスコマンド実装
"""

import click
import sys
import os
from typing import List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.education.education_process import EducationProcess, EducationResult
from src.education.textbook import Chapter, Quiz, Textbook, TextbookLoader
from src.cli.utils.yaml_loader import load_yaml, validate_textbook, YamlValidationError


def educate_command(agent_group, pass_context):
    """educate コマンドを agent グループに追加"""
    
    @agent_group.command()
    @click.argument('agent_id')
    @click.option('-f', '--file', 'files', multiple=True, type=click.Path(exists=True), required=True, help='教科書YAMLファイル')
    @click.option('--quiz', is_flag=True, help='クイズを実行')
    @click.option('--dry-run', is_flag=True, help='実行せずに確認のみ')
    @click.option('--verbose', is_flag=True, help='詳細な進捗を表示')
    @pass_context
    def educate(ctx, agent_id: str, files: List[str], quiz: bool, dry_run: bool, verbose: bool):
        """エージェントに教科書を使って教育を実施する
        
        教科書（YAML形式）を読み込み、以下のプロセスで教育を実施します：
        
        \b
        1. 各章のコンテンツをチャンク分割して記憶として保存
        2. クイズを実行して理解度を確認
        3. 正解した場合は関連記憶を強化
        
        教科書ファイルは以下の形式で記述してください：
        
        \b
        textbook:
          title: "教科書のタイトル"
          perspective: "観点名"
          chapters:
            - title: "第1章"
              content: "章の内容..."
              quiz:
                - question: "テスト問題"
                  expected_keywords: ["キーワード1", "キーワード2"]
        
        例:
          agent educate memory_agent -f textbooks/memory_management.yaml
        """
        ctx.initialize()
        
        try:
            # エージェント存在確認
            agent_def = ctx.agent_registry.get_by_id(agent_id)
            if not agent_def:
                click.echo(f"[エラー] エージェント '{agent_id}' が見つかりません", err=True)
                click.echo("\nヒント: agent list で登録済みエージェントを確認してください", err=True)
                sys.exit(2)
            
            click.echo(f"エージェント: {agent_id}")
            click.echo(f"教科書ファイル数: {len(files)}\n")

            all_results: List[EducationResult] = []

            for file_path in files:
                click.echo(f"教科書を読み込み中: {file_path}")
                textbook = _load_textbook(file_path)

                if not quiz:
                    textbook = Textbook(
                        title=textbook.title,
                        perspective=textbook.perspective,
                        chapters=[
                            Chapter(title=ch.title, content=ch.content, quiz=[])
                            for ch in textbook.chapters
                        ],
                    )

                click.echo(f"✓ 教科書を読み込みました: {textbook.title}")
                click.echo(f"  観点: {textbook.perspective}")
                click.echo(f"  章数: {len(textbook.chapters)}\n")

                if dry_run:
                    click.echo("[DRY RUN] 実際の教育は行いません\n")
                    continue

                click.echo("教育プロセスを開始します...\n")
                education_process = EducationProcess(
                    agent_id=agent_id,
                    textbook=textbook,
                    repository=ctx.memory_repository,
                    embedding_client=ctx.embedding_client,
                    config=ctx.config,
                )

                result = _run_with_progress(education_process, textbook, verbose)
                _display_result(result, agent_id)
                all_results.append(result)

        except YamlValidationError as e:
            click.echo(f"\n[エラー] 教科書の形式が正しくありません: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"\n[エラー] 教育プロセスに失敗しました: {e}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            sys.exit(1)


def _load_textbook(path: str) -> Textbook:
    """教科書を読み込み、Textbookに変換"""
    data = load_yaml(path)

    # 既存のtextbook形式に対応
    if "textbook" in data:
        loader = TextbookLoader()
        textbook = loader.load(path)
        if not loader.validate(textbook):
            raise ValueError("教科書の形式が正しくありません")
        return textbook

    validate_textbook(data)

    perspective = data.get("scope_domain") or "education"
    chapters = []
    for chapter_data in data["chapters"]:
        quiz_items = []
        for quiz in chapter_data.get("quiz", []) or []:
            answer = quiz.get("answer", "")
            expected_keywords = [answer] if answer else []
            quiz_items.append(Quiz(question=quiz.get("question", ""), expected_keywords=expected_keywords))
        chapters.append(Chapter(title=chapter_data["title"], content=chapter_data["content"], quiz=quiz_items))

    return Textbook(
        title=data["title"],
        perspective=perspective,
        chapters=chapters,
    )


def _run_with_progress(education_process: EducationProcess, textbook, verbose: bool) -> EducationResult:
    """進捗表示付きで教育プロセスを実行
    
    Args:
        education_process: EducationProcess インスタンス
        textbook: Textbook インスタンス
        verbose: 詳細表示フラグ
        
    Returns:
        EducationResult: 教育プロセスの実行結果
    """
    chapters_completed = 0
    memories_created = 0
    tests_passed = 0
    tests_total = 0
    
    total_chapters = len(textbook.chapters)
    
    for i, chapter in enumerate(textbook.chapters, 1):
        # 章の開始
        click.echo(f"[{i}/{total_chapters}] {chapter.title}")
        
        # Step 1: 読む
        if verbose:
            click.echo("  📖 コンテンツを読み込み中...")
        
        try:
            memory_ids = education_process.read_chapter(chapter)
            memories_created += len(memory_ids)
            
            if verbose:
                click.echo(f"  ✓ {len(memory_ids)} 個の記憶を作成しました")
            else:
                click.echo(f"  ✓ 読み込み完了 ({len(memory_ids)} 記憶)")
        
        except Exception as e:
            click.echo(f"  ✗ 読み込みエラー: {e}", err=True)
            continue
        
        # Step 2: テスト
        if chapter.quiz:
            if verbose:
                click.echo(f"  📝 テストを実行中... ({len(chapter.quiz)} 問)")
            
            try:
                passed = education_process.run_test(chapter, memory_ids)
                tests_passed += passed
                tests_total += len(chapter.quiz)
                
                pass_rate = (passed / len(chapter.quiz)) * 100 if len(chapter.quiz) > 0 else 0
                
                if verbose:
                    click.echo(f"  ✓ テスト完了: {passed}/{len(chapter.quiz)} 問正解 ({pass_rate:.0f}%)")
                else:
                    status_icon = "✓" if pass_rate >= 70 else "⚠"
                    click.echo(f"  {status_icon} テスト: {passed}/{len(chapter.quiz)} 問正解 ({pass_rate:.0f}%)")
            
            except Exception as e:
                click.echo(f"  ✗ テストエラー: {e}", err=True)
        else:
            if verbose:
                click.echo("  (テストなし)")
        
        chapters_completed += 1
        click.echo("")  # 空行
    
    return EducationResult(
        chapters_completed=chapters_completed,
        memories_created=memories_created,
        tests_passed=tests_passed,
        tests_total=tests_total,
    )


def _display_result(result: EducationResult, agent_id: str):
    """教育プロセスの結果を表示
    
    Args:
        result: EducationResult インスタンス
        agent_id: エージェントID
    """
    click.echo("─" * 60)
    click.echo("【教育プロセス完了】\n")
    
    click.echo(f"エージェント: {agent_id}")
    click.echo(f"完了した章: {result.chapters_completed}")
    click.echo(f"作成した記憶: {result.memories_created}")
    
    if result.tests_total > 0:
        pass_rate = result.pass_rate * 100
        click.echo(f"テスト結果: {result.tests_passed}/{result.tests_total} 問正解 ({pass_rate:.1f}%)")
        
        # 合格率に応じたメッセージ
        if pass_rate >= 90:
            click.echo("\n🎉 優秀な成績です！")
        elif pass_rate >= 70:
            click.echo("\n✓ 合格レベルに達しています")
        elif pass_rate >= 50:
            click.echo("\n⚠ もう少し理解を深める必要があります")
        else:
            click.echo("\n⚠ 再学習をお勧めします")
    else:
        click.echo("テスト: なし")
    
    click.echo("\n" + "─" * 60)
    click.echo("\n次のステップ:")
    click.echo(f"  agent status {agent_id} --memories  # 記憶を確認")
    click.echo(f"  agent sleep {agent_id}              # 睡眠フェーズで定着")
