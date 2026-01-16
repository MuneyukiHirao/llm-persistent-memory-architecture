"""入力処理層パッケージ

ユーザー入力を前処理してオーケストレーターに渡す。

使用例:
    from src.input_processing import InputProcessor, ProcessedInput

    processor = InputProcessor()
    result = processor.process("タスク1を実行してください")

    if result.needs_negotiation:
        print(f"論点数が多すぎます: {result.item_count}個")
        for option in result.negotiation_options:
            print(f"  - {option}")

仕様書参照: docs/phase2-implementation-spec.ja.md セクション5.1
"""

from src.input_processing.input_processor import InputProcessor, ProcessedInput
from src.input_processing.item_detector import ItemDetector
from src.input_processing.summarizer import Summarizer

__all__ = [
    "InputProcessor",
    "ProcessedInput",
    "ItemDetector",
    "Summarizer",
]
