# レート制限機能
# Anthropic API の Tier 1 制限を守るための proactive rate limiting

import logging
import threading
import time
from dataclasses import dataclass
from typing import List

from src.config.llm_config import RateLimitConfig

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """使用量記録
    
    タイムスタンプ付きの使用量を記録します。
    """
    
    timestamp: float
    """記録時刻（UNIX時刻）"""
    
    requests: int = 1
    """リクエスト数（通常は1）"""
    
    input_tokens: int = 0
    """入力トークン数"""
    
    output_tokens: int = 0
    """出力トークン数"""


class RateLimiter:
    """レート制限機能
    
    sliding window 方式でAPIの使用量を追跡し、
    制限に達する前に待機を行います。
    
    スレッドセーフな実装で、複数のリクエストが同時に実行されても
    正確に制限を守ります。
    
    使用例:
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        # リクエスト前にチェック
        limiter.wait_if_needed(estimated_input_tokens=1000)
        
        # API呼び出し実行
        response = api_call()
        
        # 実際の使用量を記録
        limiter.record_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
    """
    
    def __init__(self, config: RateLimitConfig):
        """レート制限機能を初期化
        
        Args:
            config: レート制限設定
        """
        self.config = config
        self._usage_history: List[UsageRecord] = []
        self._lock = threading.Lock()
        
        # 制限値（安全マージンを適用）
        self._max_requests = int(config.requests_per_minute * config.safety_margin)
        self._max_input_tokens = int(config.input_tokens_per_minute * config.safety_margin)
        self._max_output_tokens = int(config.output_tokens_per_minute * config.safety_margin)
        
        logger.info(
            f"RateLimiter 初期化: requests={self._max_requests}/min, "
            f"input_tokens={self._max_input_tokens}/min, "
            f"output_tokens={self._max_output_tokens}/min"
        )
    
    def wait_if_needed(self, estimated_input_tokens: int = 0, estimated_output_tokens: int = 0) -> None:
        """必要に応じて待機
        
        現在の使用量と推定使用量をチェックし、
        制限に達する場合は適切な時間だけ待機します。
        
        Args:
            estimated_input_tokens: 推定入力トークン数
            estimated_output_tokens: 推定出力トークン数
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_records(current_time)
            
            # 現在の使用量を計算
            current_usage = self._calculate_current_usage()
            
            # 予想使用量（現在 + 推定）
            predicted_requests = current_usage.requests + 1
            predicted_input_tokens = current_usage.input_tokens + estimated_input_tokens
            predicted_output_tokens = current_usage.output_tokens + estimated_output_tokens
            
            # 制限チェック
            wait_times = []
            
            # リクエスト数制限
            if predicted_requests > self._max_requests:
                wait_times.append(self._calculate_wait_time_for_requests())
            
            # 入力トークン制限
            if predicted_input_tokens > self._max_input_tokens:
                wait_times.append(self._calculate_wait_time_for_input_tokens(estimated_input_tokens))
            
            # 出力トークン制限
            if predicted_output_tokens > self._max_output_tokens:
                wait_times.append(self._calculate_wait_time_for_output_tokens(estimated_output_tokens))
            
            # 最大待機時間を適用
            if wait_times:
                wait_time = max(wait_times)
                logger.info(
                    f"レート制限により {wait_time:.1f}秒待機 "
                    f"(requests: {predicted_requests}/{self._max_requests}, "
                    f"input_tokens: {predicted_input_tokens}/{self._max_input_tokens}, "
                    f"output_tokens: {predicted_output_tokens}/{self._max_output_tokens})"
                )
                time.sleep(wait_time)
    
    def record_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """使用量を記録
        
        API呼び出し後に実際の使用量を記録します。
        
        Args:
            input_tokens: 実際の入力トークン数
            output_tokens: 実際の出力トークン数
        """
        with self._lock:
            current_time = time.time()
            
            record = UsageRecord(
                timestamp=current_time,
                requests=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            self._usage_history.append(record)
            self._cleanup_old_records(current_time)
            
            logger.debug(
                f"使用量記録: input_tokens={input_tokens}, output_tokens={output_tokens}"
            )
    
    def get_current_usage(self) -> UsageRecord:
        """現在の使用量を取得
        
        Returns:
            現在の1分間の使用量
        """
        with self._lock:
            self._cleanup_old_records(time.time())
            return self._calculate_current_usage()
    
    def _cleanup_old_records(self, current_time: float) -> None:
        """古い記録をクリーンアップ
        
        時間窓外の記録を削除します。
        
        Args:
            current_time: 現在時刻
        """
        cutoff_time = current_time - self.config.window_seconds
        self._usage_history = [
            record for record in self._usage_history
            if record.timestamp > cutoff_time
        ]
    
    def _calculate_current_usage(self) -> UsageRecord:
        """現在の使用量を計算
        
        Returns:
            時間窓内の合計使用量
        """
        if not self._usage_history:
            return UsageRecord(timestamp=time.time(), requests=0)
        
        total_requests = sum(record.requests for record in self._usage_history)
        total_input_tokens = sum(record.input_tokens for record in self._usage_history)
        total_output_tokens = sum(record.output_tokens for record in self._usage_history)
        
        return UsageRecord(
            timestamp=time.time(),
            requests=total_requests,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens
        )
    
    def _calculate_wait_time_for_requests(self) -> float:
        """リクエスト数制限の待機時間を計算
        
        Returns:
            待機時間（秒）
        """
        if not self._usage_history:
            return 0.0
        
        # 最も古い記録が時間窓から出るまでの時間
        oldest_record = min(self._usage_history, key=lambda r: r.timestamp)
        return oldest_record.timestamp + self.config.window_seconds - time.time()
    
    def _calculate_wait_time_for_input_tokens(self, estimated_tokens: int) -> float:
        """入力トークン制限の待機時間を計算
        
        Args:
            estimated_tokens: 推定入力トークン数
            
        Returns:
            待機時間（秒）
        """
        return self._calculate_wait_time_for_tokens(estimated_tokens, 'input')
    
    def _calculate_wait_time_for_output_tokens(self, estimated_tokens: int) -> float:
        """出力トークン制限の待機時間を計算
        
        Args:
            estimated_tokens: 推定出力トークン数
            
        Returns:
            待機時間（秒）
        """
        return self._calculate_wait_time_for_tokens(estimated_tokens, 'output')
    
    def _calculate_wait_time_for_tokens(self, estimated_tokens: int, token_type: str) -> float:
        """トークン制限の待機時間を計算
        
        Args:
            estimated_tokens: 推定トークン数
            token_type: 'input' または 'output'
            
        Returns:
            待機時間（秒）
        """
        if not self._usage_history:
            return 0.0
        
        # トークン数の上限
        max_tokens = (
            self._max_input_tokens if token_type == 'input' 
            else self._max_output_tokens
        )
        
        # 現在の使用量
        current_tokens = sum(
            getattr(record, f'{token_type}_tokens') 
            for record in self._usage_history
        )
        
        # 超過分を計算
        excess_tokens = current_tokens + estimated_tokens - max_tokens
        if excess_tokens <= 0:
            return 0.0
        
        # 超過分を解消するのに必要な時間を計算
        # 古い記録から順に削除していき、いつ制限内に収まるかを計算
        sorted_records = sorted(self._usage_history, key=lambda r: r.timestamp)
        tokens_to_remove = 0
        
        for record in sorted_records:
            tokens_to_remove += getattr(record, f'{token_type}_tokens')
            if tokens_to_remove >= excess_tokens:
                # この記録が時間窓から出るまでの時間
                return record.timestamp + self.config.window_seconds - time.time()
        
        # 全記録を削除しても足りない場合は最大待機時間
        return self.config.window_seconds
