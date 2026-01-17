# レート制限機能の単体テスト
# 実装仕様: docs/phase1-implementation-spec.ja.md

import pytest
import time
import threading
from dataclasses import asdict

from src.config.llm_config import RateLimitConfig
from src.llm.rate_limiter import RateLimiter, UsageRecord


class TestUsageRecordDataclass:
    """UsageRecord dataclass のテスト"""

    def test_usage_record_creation_minimal(self):
        """最小限のUsageRecordを作成できることを確認"""
        record = UsageRecord(timestamp=1234567890.0)
        assert record.timestamp == 1234567890.0
        assert record.requests == 1
        assert record.input_tokens == 0
        assert record.output_tokens == 0

    def test_usage_record_creation_with_tokens(self):
        """トークン情報付きのUsageRecordを作成できることを確認"""
        record = UsageRecord(
            timestamp=1234567890.0,
            requests=1,
            input_tokens=100,
            output_tokens=50,
        )
        assert record.timestamp == 1234567890.0
        assert record.requests == 1
        assert record.input_tokens == 100
        assert record.output_tokens == 50

    def test_usage_record_to_dict(self):
        """UsageRecordを辞書に変換できることを確認"""
        record = UsageRecord(
            timestamp=1234567890.0,
            requests=1,
            input_tokens=200,
            output_tokens=100,
        )
        data = asdict(record)
        assert data == {
            "timestamp": 1234567890.0,
            "requests": 1,
            "input_tokens": 200,
            "output_tokens": 100,
        }


class TestRateLimitConfigDefaults:
    """RateLimitConfig のデフォルト値テスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されることを確認"""
        config = RateLimitConfig()
        
        # Tier 2 の制限値
        assert config.requests_per_minute == 1000
        assert config.input_tokens_per_minute == 450000
        assert config.output_tokens_per_minute == 90000
        
        # その他の設定
        assert config.window_seconds == 60
        assert config.safety_margin == 0.9

    def test_custom_values(self):
        """カスタム値を設定できることを確認"""
        config = RateLimitConfig(
            requests_per_minute=100,
            input_tokens_per_minute=80000,
            output_tokens_per_minute=16000,
            window_seconds=120,
            safety_margin=0.8,
        )
        
        assert config.requests_per_minute == 100
        assert config.input_tokens_per_minute == 80000
        assert config.output_tokens_per_minute == 16000
        assert config.window_seconds == 120
        assert config.safety_margin == 0.8


class TestRateLimiterInitialization:
    """RateLimiter 初期化テスト"""

    def test_init_with_default_config(self):
        """デフォルト設定でRateLimiterを初期化できることを確認"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        assert limiter.config == config
        assert limiter._usage_history == []
        assert limiter._lock is not None

    def test_init_applies_safety_margin(self):
        """安全マージンが適用されることを確認"""
        config = RateLimitConfig(
            requests_per_minute=100,
            input_tokens_per_minute=10000,
            output_tokens_per_minute=2000,
            safety_margin=0.8,
        )
        limiter = RateLimiter(config)
        
        # 安全マージン 0.8 が適用される
        assert limiter._max_requests == 80  # 100 * 0.8
        assert limiter._max_input_tokens == 8000  # 10000 * 0.8
        assert limiter._max_output_tokens == 1600  # 2000 * 0.8


class TestRateLimiterBasicFunctions:
    """RateLimiter 基本機能テスト"""

    @pytest.fixture
    def limiter(self):
        """テスト用のRateLimiterを作成"""
        config = RateLimitConfig(
            requests_per_minute=10,
            input_tokens_per_minute=1000,
            output_tokens_per_minute=500,
            window_seconds=60,
            safety_margin=1.0,  # テストのためマージンなし
        )
        return RateLimiter(config)

    def test_record_usage(self, limiter):
        """使用量を記録できることを確認"""
        limiter.record_usage(input_tokens=100, output_tokens=50)
        
        assert len(limiter._usage_history) == 1
        record = limiter._usage_history[0]
        assert record.requests == 1
        assert record.input_tokens == 100
        assert record.output_tokens == 50

    def test_record_usage_multiple_times(self, limiter):
        """複数回の使用量を記録できることを確認"""
        limiter.record_usage(input_tokens=100, output_tokens=50)
        limiter.record_usage(input_tokens=200, output_tokens=100)
        limiter.record_usage(input_tokens=150, output_tokens=75)
        
        assert len(limiter._usage_history) == 3

    def test_get_current_usage_empty(self, limiter):
        """使用量がない場合のget_current_usageを確認"""
        usage = limiter.get_current_usage()
        
        assert usage.requests == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_get_current_usage_with_records(self, limiter):
        """記録がある場合のget_current_usageを確認"""
        limiter.record_usage(input_tokens=100, output_tokens=50)
        limiter.record_usage(input_tokens=200, output_tokens=100)
        
        usage = limiter.get_current_usage()
        
        assert usage.requests == 2
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150

    def test_wait_if_needed_no_wait(self, limiter):
        """制限に達していない場合は待機しないことを確認"""
        start_time = time.time()
        limiter.wait_if_needed(estimated_input_tokens=100, estimated_output_tokens=50)
        elapsed_time = time.time() - start_time
        
        # 待機しないので即座に返る（0.1秒未満）
        assert elapsed_time < 0.1

    def test_wait_if_needed_with_existing_usage(self, limiter):
        """既存の使用量があっても制限内なら待機しないことを確認"""
        # 制限の半分程度を使用
        limiter.record_usage(input_tokens=500, output_tokens=250)
        
        start_time = time.time()
        limiter.wait_if_needed(estimated_input_tokens=100, estimated_output_tokens=50)
        elapsed_time = time.time() - start_time
        
        # まだ制限内なので待機しない
        assert elapsed_time < 0.1


class TestRateLimiterCleanup:
    """RateLimiter クリーンアップ機能テスト"""

    def test_cleanup_old_records(self):
        """古い記録がクリーンアップされることを確認"""
        config = RateLimitConfig(window_seconds=2)  # 2秒の時間窓
        limiter = RateLimiter(config)
        
        # 記録を追加
        limiter.record_usage(input_tokens=100, output_tokens=50)
        assert len(limiter._usage_history) == 1
        
        # 3秒待つ（時間窓外）
        time.sleep(3)
        
        # クリーンアップを実行
        limiter.get_current_usage()
        
        # 古い記録が削除される
        assert len(limiter._usage_history) == 0

    def test_cleanup_keeps_recent_records(self):
        """新しい記録は保持されることを確認"""
        config = RateLimitConfig(window_seconds=60)
        limiter = RateLimiter(config)

        # 古い記録を手動で追加
        old_time = time.time() - 70  # 70秒前
        limiter._usage_history.append(
            UsageRecord(timestamp=old_time, input_tokens=100, output_tokens=50)
        )

        # 新しい記録を追加（この時点でクリーンアップが自動実行される）
        limiter.record_usage(input_tokens=200, output_tokens=100)

        # record_usage() 内で自動的にクリーンアップされるため、古い記録は既に削除されている
        assert len(limiter._usage_history) == 1

        # クリーンアップを実行
        usage = limiter.get_current_usage()

        # 新しい記録のみが残っている
        assert len(limiter._usage_history) == 1
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100

    def test_cleanup_partial_records(self):
        """一部の記録がクリーンアップされることを確認"""
        config = RateLimitConfig(window_seconds=5)
        limiter = RateLimiter(config)
        
        # 複数の記録を時間差で追加
        current_time = time.time()
        limiter._usage_history.append(
            UsageRecord(timestamp=current_time - 10, input_tokens=100)
        )
        limiter._usage_history.append(
            UsageRecord(timestamp=current_time - 3, input_tokens=200)
        )
        limiter._usage_history.append(
            UsageRecord(timestamp=current_time - 1, input_tokens=300)
        )
        
        assert len(limiter._usage_history) == 3
        
        # クリーンアップ実行
        limiter._cleanup_old_records(current_time)
        
        # 10秒前の記録のみ削除される
        assert len(limiter._usage_history) == 2
        assert limiter._usage_history[0].input_tokens == 200
        assert limiter._usage_history[1].input_tokens == 300


class TestRateLimiterWaitLogic:
    """RateLimiter 待機ロジックテスト"""

    def test_wait_for_request_limit(self):
        """リクエスト数制限で待機することを確認"""
        config = RateLimitConfig(
            requests_per_minute=2,
            input_tokens_per_minute=100000,
            output_tokens_per_minute=100000,
            window_seconds=2,  # 短い時間窓でテスト
            safety_margin=1.0,
        )
        limiter = RateLimiter(config)
        
        # 制限まで使用
        limiter.record_usage(input_tokens=100, output_tokens=50)
        limiter.record_usage(input_tokens=100, output_tokens=50)
        
        # 次のリクエストは待機が必要
        start_time = time.time()
        limiter.wait_if_needed(estimated_input_tokens=100, estimated_output_tokens=50)
        elapsed_time = time.time() - start_time
        
        # 少なくとも待機したことを確認（完全に2秒待つ必要はない）
        assert elapsed_time > 0.5

    def test_wait_for_input_token_limit(self):
        """入力トークン制限で待機することを確認"""
        config = RateLimitConfig(
            requests_per_minute=100,
            input_tokens_per_minute=500,
            output_tokens_per_minute=100000,
            window_seconds=2,
            safety_margin=1.0,
        )
        limiter = RateLimiter(config)
        
        # 制限近くまで使用
        limiter.record_usage(input_tokens=400, output_tokens=50)
        
        # 次のリクエストで制限を超える
        start_time = time.time()
        limiter.wait_if_needed(estimated_input_tokens=200, estimated_output_tokens=50)
        elapsed_time = time.time() - start_time
        
        # 待機したことを確認
        assert elapsed_time > 0.5

    def test_wait_for_output_token_limit(self):
        """出力トークン制限で待機することを確認"""
        config = RateLimitConfig(
            requests_per_minute=100,
            input_tokens_per_minute=100000,
            output_tokens_per_minute=300,
            window_seconds=2,
            safety_margin=1.0,
        )
        limiter = RateLimiter(config)
        
        # 制限近くまで使用
        limiter.record_usage(input_tokens=100, output_tokens=250)
        
        # 次のリクエストで制限を超える
        start_time = time.time()
        limiter.wait_if_needed(estimated_input_tokens=100, estimated_output_tokens=100)
        elapsed_time = time.time() - start_time
        
        # 待機したことを確認
        assert elapsed_time > 0.5


class TestRateLimiterThreadSafety:
    """RateLimiter スレッドセーフ性テスト"""

    def test_concurrent_record_usage(self):
        """複数スレッドから同時にrecord_usageを呼べることを確認"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        def record_multiple_times():
            for _ in range(10):
                limiter.record_usage(input_tokens=10, output_tokens=5)
        
        # 5つのスレッドを起動
        threads = [threading.Thread(target=record_multiple_times) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 全ての記録が保存される（5スレッド × 10回 = 50回）
        assert len(limiter._usage_history) == 50
        
        usage = limiter.get_current_usage()
        assert usage.requests == 50
        assert usage.input_tokens == 500
        assert usage.output_tokens == 250

    def test_concurrent_get_current_usage(self):
        """複数スレッドから同時にget_current_usageを呼べることを確認"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        # 初期データを設定
        limiter.record_usage(input_tokens=100, output_tokens=50)
        
        results = []
        
        def get_usage():
            usage = limiter.get_current_usage()
            results.append(usage)
        
        # 10個のスレッドを起動
        threads = [threading.Thread(target=get_usage) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 全てのスレッドが結果を取得
        assert len(results) == 10
        
        # 全ての結果が一貫している
        for usage in results:
            assert usage.requests == 1
            assert usage.input_tokens == 100
            assert usage.output_tokens == 50

    def test_concurrent_wait_if_needed(self):
        """複数スレッドから同時にwait_if_neededを呼べることを確認"""
        config = RateLimitConfig(
            requests_per_minute=100,
            input_tokens_per_minute=100000,
            output_tokens_per_minute=100000,
            safety_margin=1.0,
        )
        limiter = RateLimiter(config)
        
        completed = []
        
        def wait_and_complete():
            limiter.wait_if_needed(estimated_input_tokens=100, estimated_output_tokens=50)
            completed.append(True)
        
        # 5つのスレッドを起動
        threads = [threading.Thread(target=wait_and_complete) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 全てのスレッドが完了
        assert len(completed) == 5


class TestRateLimiterEdgeCases:
    """RateLimiter エッジケーステスト"""

    def test_zero_estimated_tokens(self):
        """推定トークン数が0の場合も正常に動作することを確認"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        # エラーが発生しないことを確認
        limiter.wait_if_needed(estimated_input_tokens=0, estimated_output_tokens=0)
        limiter.record_usage(input_tokens=0, output_tokens=0)
        
        usage = limiter.get_current_usage()
        assert usage.requests == 1
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_large_token_counts(self):
        """大きなトークン数も正常に処理できることを確認"""
        config = RateLimitConfig(
            input_tokens_per_minute=100000,
            output_tokens_per_minute=50000,
        )
        limiter = RateLimiter(config)
        
        limiter.record_usage(input_tokens=50000, output_tokens=25000)
        
        usage = limiter.get_current_usage()
        assert usage.input_tokens == 50000
        assert usage.output_tokens == 25000

    def test_empty_usage_history_calculations(self):
        """使用履歴が空の場合の計算が正しいことを確認"""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        # 内部メソッドを直接テスト
        wait_time = limiter._calculate_wait_time_for_requests()
        assert wait_time == 0.0
        
        wait_time = limiter._calculate_wait_time_for_input_tokens(1000)
        assert wait_time == 0.0
        
        wait_time = limiter._calculate_wait_time_for_output_tokens(500)
        assert wait_time == 0.0


class TestRateLimiterIntegration:
    """RateLimiter 統合テスト"""

    def test_realistic_usage_pattern(self):
        """現実的な使用パターンをシミュレート"""
        config = RateLimitConfig(
            requests_per_minute=10,
            input_tokens_per_minute=5000,
            output_tokens_per_minute=2000,
            window_seconds=5,  # テスト用に短縮
            safety_margin=0.9,
        )
        limiter = RateLimiter(config)
        
        # 複数回のAPI呼び出しをシミュレート
        for i in range(5):
            limiter.wait_if_needed(estimated_input_tokens=500, estimated_output_tokens=200)
            limiter.record_usage(input_tokens=500, output_tokens=200)
        
        usage = limiter.get_current_usage()
        
        # 5回のリクエストが記録される
        assert usage.requests == 5
        assert usage.input_tokens == 2500
        assert usage.output_tokens == 1000
        
        # まだ制限内
        assert usage.requests <= limiter._max_requests
        assert usage.input_tokens <= limiter._max_input_tokens
        assert usage.output_tokens <= limiter._max_output_tokens

    def test_wait_and_cleanup_cycle(self):
        """待機とクリーンアップのサイクルが正しく動作することを確認"""
        config = RateLimitConfig(
            requests_per_minute=3,
            window_seconds=2,
            safety_margin=1.0,
        )
        limiter = RateLimiter(config)
        
        # 制限まで使用
        for _ in range(3):
            limiter.record_usage(input_tokens=100, output_tokens=50)
        
        # 使用量を確認
        usage = limiter.get_current_usage()
        assert usage.requests == 3
        
        # 2秒待つ（時間窓が経過）
        time.sleep(2.5)
        
        # クリーンアップ後は使用量がリセット
        usage = limiter.get_current_usage()
        assert usage.requests == 0
        
        # 再び記録できる
        limiter.record_usage(input_tokens=100, output_tokens=50)
        usage = limiter.get_current_usage()
        assert usage.requests == 1
