# Phase 1: 画像認識機能 実装仕様書

## 1. 概要

### 1.1 目的
llm-persistent-memoryシステムに画像認識機能を追加し、エージェントがローカル画像ファイルを読み込んで解釈できるようにする。

### 1.2 Phase 1 のスコープ
- ✅ ローカル画像ファイル（PNG, JPEG, WebP, GIF）の読み込み
- ✅ base64エンコードしてClaude APIに送信
- ✅ エージェントによる画像解釈と応答
- ✅ CLIからの画像指定インターフェース
- ❌ 複数画像の同時処理（Phase 2以降）
- ❌ URL経由の画像取得（Phase 3以降）
- ❌ 画像生成機能（Phase 4以降）

### 1.3 成功基準
- [ ] `agent task "この画像に何が写っていますか？" --image test.png` が動作する
- [ ] 画像情報がメモリに保存される
- [ ] セキュリティチェックが機能する（パストラバーサル、サイズ制限）
- [ ] 既存のテキストベース機能に影響を与えない

---

## 2. アーキテクチャ

### 2.1 データフロー

```
┌─────────────┐
│ CLI         │
│ --image     │
│ path.png    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ LLMTaskExecutor │
│ execute_task()  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────────┐
│ ClaudeClient    │      │ file_read_image()│
│ send_message()  │─────▶│ tool             │
└──────┬──────────┘      └──────────────────┘
       │                          │
       │                          ▼
       │                  ┌──────────────────┐
       │                  │ 1. Path validate │
       │                  │ 2. Read binary   │
       │                  │ 3. Base64 encode │
       │                  │ 4. Media type    │
       │                  └──────────────────┘
       ▼
┌─────────────────┐
│ Claude API      │
│ (multimodal)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Response        │
│ + Memory Save   │
└─────────────────┘
```

### 2.2 関連コンポーネント

| コンポーネント | 役割 | 変更 |
|--------------|------|------|
| `src/cli/commands/agent_commands.py` | CLI引数処理 | `--image` オプション追加 |
| `src/llm/task_executor.py` | タスク実行 | `image_path` パラメータ追加 |
| `src/llm/claude_client.py` | API通信 | 画像contentブロック対応 |
| `src/llm/tools/file_tools.py` | ファイル操作 | `file_read_image()` 追加 |

---

## 3. 実装詳細

### 3.1 新規ツール: `file_read_image()`

**ファイル**: `src/llm/tools/file_tools.py`

```python
def file_read_image(path: str) -> dict:
    """
    画像ファイルを読み込み、base64エンコードして返す
    
    Args:
        path: 画像ファイルパス（相対パスまたは絶対パス）
        
    Returns:
        {
            "media_type": "image/png",
            "data": "base64_encoded_string"
        }
        
    Raises:
        FileNotFoundError: ファイルが存在しない
        ValueError: サポートされていない画像形式
        PermissionError: パストラバーサル検出
    """
```

#### 3.1.1 実装ステップ

1. **パス検証**（既存の `_validate_file_path()` を再利用）
   ```python
   resolved_path = _validate_file_path(path, require_exists=True)
   ```

2. **拡張子チェックとメディアタイプ判定**
   ```python
   SUPPORTED_IMAGE_TYPES = {
       '.png': 'image/png',
       '.jpg': 'image/jpeg',
       '.jpeg': 'image/jpeg',
       '.webp': 'image/webp',
       '.gif': 'image/gif'
   }
   
   ext = resolved_path.suffix.lower()
   if ext not in SUPPORTED_IMAGE_TYPES:
       raise ValueError(f"Unsupported image format: {ext}")
   media_type = SUPPORTED_IMAGE_TYPES[ext]
   ```

3. **ファイルサイズチェック**
   ```python
   MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
   file_size = resolved_path.stat().st_size
   if file_size > MAX_IMAGE_SIZE:
       raise ValueError(f"Image too large: {file_size} bytes (max: {MAX_IMAGE_SIZE})")
   ```

4. **バイナリ読み込みとbase64エンコード**
   ```python
   import base64
   
   with open(resolved_path, 'rb') as f:
       image_data = f.read()
   
   encoded_data = base64.standard_b64encode(image_data).decode('utf-8')
   ```

5. **戻り値**
   ```python
   return {
       "media_type": media_type,
       "data": encoded_data
   }
   ```

### 3.2 ClaudeClient の修正

**ファイル**: `src/llm/claude_client.py`

#### 3.2.1 `send_message()` メソッドの拡張

```python
def send_message(
    self,
    message: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
    image_data: Optional[Dict[str, str]] = None  # 追加
) -> ClaudeResponse:
    """
    Args:
        image_data: {"media_type": "image/png", "data": "base64..."} 形式
    """
```

#### 3.2.2 メッセージ構築ロジック

```python
# ユーザーメッセージの構築
if image_data:
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data["media_type"],
                    "data": image_data["data"]
                }
            },
            {
                "type": "text",
                "text": message
            }
        ]
    }
else:
    # 既存のテキストのみの処理
    user_message = {
        "role": "user",
        "content": message
    }
```

### 3.3 LLMTaskExecutor の修正

**ファイル**: `src/llm/task_executor.py`

```python
def execute_task(
    self,
    task: str,
    context: Optional[str] = None,
    image_path: Optional[str] = None  # 追加
) -> str:
    """
    Args:
        image_path: 画像ファイルパス（オプション）
    """
    
    # 画像データの読み込み
    image_data = None
    if image_path:
        from llm.tools.file_tools import file_read_image
        image_data = file_read_image(image_path)
    
    # Claude APIへの送信
    response = self.client.send_message(
        message=task,
        system_prompt=self._build_system_prompt(context),
        conversation_history=self.conversation_history,
        tools=self.tools,
        image_data=image_data  # 追加
    )
```

### 3.4 CLI インターフェース

**ファイル**: `src/cli/commands/agent_commands.py`

```python
@click.command()
@click.argument('task')
@click.option('--context', '-c', help='追加のコンテキスト情報')
@click.option('--image', '-i', type=click.Path(exists=True), help='画像ファイルパス')
@click.option('--verbose', '-v', is_flag=True, help='詳細な出力')
def task(task: str, context: Optional[str], image: Optional[str], verbose: bool):
    """タスクを実行"""
    
    executor = LLMTaskExecutor(
        api_key=get_api_key(),
        memory_manager=memory_manager,
        verbose=verbose
    )
    
    result = executor.execute_task(
        task=task,
        context=context,
        image_path=image  # 追加
    )
    
    click.echo(result)
```

---

## 4. セキュリティ考慮事項

### 4.1 パストラバーサル防止

既存の `_validate_file_path()` を再利用：

```python
# src/llm/tools/file_tools.py の既存実装
def _validate_file_path(path: str, require_exists: bool = False) -> Path:
    """
    - 絶対パスへの解決
    - WORKSPACE_ROOT外へのアクセス禁止
    - シンボリックリンクの解決と検証
    """
```

### 4.2 ファイルサイズ制限

```python
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

# Claude API の実際の制限:
# - 画像1枚あたり最大5MB
# - リクエスト全体で最大100MB（複数画像の場合）
```

### 4.3 サポート形式の制限

```python
# ホワイトリスト方式
SUPPORTED_IMAGE_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif'
}

# 実ファイル内容の検証は Phase 2 で検討
# （拡張子偽装攻撃への対策）
```

### 4.4 エラーハンドリング

```python
try:
    image_data = file_read_image(image_path)
except FileNotFoundError:
    logger.error(f"Image file not found: {image_path}")
    raise
except ValueError as e:
    logger.error(f"Invalid image: {e}")
    raise
except PermissionError as e:
    logger.error(f"Access denied: {e}")
    raise
```

---

## 5. テスト戦略

### 5.1 単体テスト

**ファイル**: `tests/llm/tools/test_file_tools_image.py`

```python
class TestFileReadImage:
    def test_read_valid_png(self, tmp_path):
        """正常系: PNG画像の読み込み"""
        
    def test_read_valid_jpeg(self, tmp_path):
        """正常系: JPEG画像の読み込み"""
        
    def test_unsupported_format(self, tmp_path):
        """異常系: サポート外の形式（.bmp）"""
        
    def test_file_not_found(self):
        """異常系: ファイルが存在しない"""
        
    def test_path_traversal(self, tmp_path):
        """セキュリティ: パストラバーサル"""
        
    def test_file_too_large(self, tmp_path):
        """異常系: ファイルサイズ超過"""
        
    def test_base64_encoding(self, tmp_path):
        """base64エンコードの正確性"""
```

### 5.2 統合テスト

**ファイル**: `tests/integration/test_image_recognition.py`

```python
class TestImageRecognition:
    def test_task_with_image(self, mock_claude_api):
        """エンドツーエンド: 画像付きタスク実行"""
        
    def test_image_content_in_api_request(self, mock_claude_api):
        """APIリクエストに画像contentブロックが含まれる"""
        
    def test_memory_saves_image_context(self):
        """画像に関する情報がメモリに保存される"""
```

### 5.3 手動テスト

```bash
# 1. 基本的な画像認識
agent task "この画像に何が写っていますか？" --image tests/fixtures/sample.png

# 2. コンテキスト付き
agent task "この図の問題点を指摘してください" --image diagram.png --context "システム設計図"

# 3. エラーケース
agent task "..." --image nonexistent.png  # FileNotFoundError
agent task "..." --image ../../../etc/passwd  # PermissionError
agent task "..." --image large.bmp  # ValueError (unsupported format)
```

---

## 6. 実装タスク分解

### Task 1: `file_read_image()` ツールの実装
**優先度**: 高  
**見積もり**: 2時間

- [ ] `src/llm/tools/file_tools.py` に関数追加
- [ ] パス検証、サイズチェック、base64エンコード実装
- [ ] 単体テスト作成（`tests/llm/tools/test_file_tools_image.py`）
- [ ] テストフィクスチャ画像の準備（`tests/fixtures/`）

**完了条件**: 全単体テストがパス

---

### Task 2: ClaudeClient の画像対応
**優先度**: 高  
**見積もり**: 1.5時間

- [ ] `send_message()` に `image_data` パラメータ追加
- [ ] マルチモーダルcontentブロックの構築ロジック実装
- [ ] 既存のテキストのみのケースが壊れていないことを確認

**完了条件**: 既存テストがパス + 画像contentブロックが正しく構築される

---

### Task 3: LLMTaskExecutor の画像対応
**優先度**: 高  
**見積もり**: 1時間

- [ ] `execute_task()` に `image_path` パラメータ追加
- [ ] `file_read_image()` の呼び出し実装
- [ ] エラーハンドリング実装

**完了条件**: 画像パスを渡すとClaudeClientに正しく伝わる

---

### Task 4: CLI インターフェース実装
**優先度**: 高  
**見積もり**: 1時間

- [ ] `agent task` コマンドに `--image` オプション追加
- [ ] パス存在チェック（`click.Path(exists=True)`）
- [ ] ヘルプメッセージ更新

**完了条件**: `agent task "..." --image path.png` が動作

---

### Task 5: 統合テスト作成
**優先度**: 中  
**見積もり**: 2時間

- [ ] `tests/integration/test_image_recognition.py` 作成
- [ ] モックAPIを使ったエンドツーエンドテスト
- [ ] メモリ保存の確認テスト

**完了条件**: 統合テストがパス

---

### Task 6: ドキュメント更新
**優先度**: 中  
**見積もり**: 1時間

- [ ] `README.md` に画像認識機能の説明追加
- [ ] `docs/usage.ja.md` に使用例追加
- [ ] `CHANGELOG.md` 更新

**完了条件**: ドキュメントレビュー完了

---

### Task 7: 手動テストと修正
**優先度**: 中  
**見積もり**: 1.5時間

- [ ] 実際の画像ファイルでテスト
- [ ] エラーメッセージの改善
- [ ] パフォーマンス確認（大きめの画像で）

**完了条件**: 全ての手動テストケースが期待通り動作

---

## 7. 将来の拡張

### Phase 2: 複数画像対応（予定）
- `--image` オプションの複数指定
- 会話履歴内の画像保持
- 画像間の比較・分析

```bash
agent task "これらの画像の違いは？" --image before.png --image after.png
```

### Phase 3: URL経由の画像取得（予定）
- HTTP/HTTPS URLからの画像ダウンロード
- キャッシュ機構
- タイムアウト・リトライ処理

```bash
agent task "このWebページのスクリーンショットを分析" --image-url https://example.com/screenshot.png
```

### Phase 4: 画像生成機能（予定）
- DALL-E等の画像生成APIとの統合
- 生成画像の自動保存
- プロンプトテンプレート

```bash
agent generate-image "夕暮れの富士山" --style realistic
```

### Phase 5: OCR・ドキュメント解析（予定）
- PDF/スキャン画像からのテキスト抽出
- 表・グラフの構造化データ化
- 手書き文字認識

---

## 8. 実装チェックリスト

### コード実装
- [ ] `file_read_image()` 実装
- [ ] `ClaudeClient.send_message()` 修正
- [ ] `LLMTaskExecutor.execute_task()` 修正
- [ ] CLI `--image` オプション追加

### テスト
- [ ] 単体テスト（`test_file_tools_image.py`）
- [ ] 統合テスト（`test_image_recognition.py`）
- [ ] 手動テスト実施

### ドキュメント
- [ ] README.md 更新
- [ ] usage.ja.md 更新
- [ ] CHANGELOG.md 更新
- [ ] この仕様書のレビュー

### セキュリティ
- [ ] パストラバーサル対策確認
- [ ] ファイルサイズ制限確認
- [ ] エラーハンドリング確認

### リリース準備
- [ ] 全テストパス
- [ ] コードレビュー完了
- [ ] バージョン番号更新（例: 0.2.0）

---

## 9. 参考情報

### Claude API 画像仕様
- サポート形式: PNG, JPEG, WebP, GIF（非アニメーション）
- 最大サイズ: 5MB/画像
- base64エンコード必須
- メディアタイプ指定必須

### 既存コードパターン
```python
# src/llm/tools/file_tools.py のセキュリティモデル
WORKSPACE_ROOT = Path.cwd()

def _validate_file_path(path: str, require_exists: bool = False) -> Path:
    # このパターンを踏襲
```

### 関連Issue/PR
- （実装後に記載）

---

**作成日**: 2024-XX-XX  
**バージョン**: 1.0  
**ステータス**: Draft
\n────────────────────────────────────────
\n[詳細情報]
  トークン使用量: 6223
  イテレーション回数: 1
  検索されたメモリ: 0件
  ツール呼び出し回数: 0
