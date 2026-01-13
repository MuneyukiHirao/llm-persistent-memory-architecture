# Azure OpenAI Embedding クライアント
# 環境変数:
#   - AZURE_OPENAI_ENDPOINT または OpenAIEmbeddingURI: Azure OpenAIのエンドポイント
#   - AZURE_OPENAI_API_KEY または OpenAIEmbeddingKey: Azure OpenAIのAPIキー
#   - AZURE_OPENAI_EMBEDDING_DEPLOYMENT: デプロイメント名 (デフォルト: text-embedding-3-small)

import os
import logging
from typing import List, Optional

from openai import AzureOpenAI

from src.config.phase1_config import config

logger = logging.getLogger(__name__)


class AzureEmbeddingError(Exception):
    """Azure Embeddingクライアントのエラー"""
    pass


class AzureEmbeddingClient:
    """Azure OpenAI Embedding クライアント

    text-embedding-3-small モデルを使用してテキストをベクトル化します。
    次元: 1536

    使用例:
        client = AzureEmbeddingClient()
        embedding = client.get_embedding("検索対象のテキスト")
        embeddings = client.get_embeddings(["テキスト1", "テキスト2"])
    """

    # Azure OpenAI APIバージョン
    API_VERSION = "2024-02-01"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        """Azure OpenAI Embedding クライアントを初期化

        Args:
            endpoint: Azure OpenAI エンドポイント (省略時は環境変数から取得)
            api_key: Azure OpenAI APIキー (省略時は環境変数から取得)
            deployment: デプロイメント名 (省略時は環境変数または設定から取得)

        Raises:
            AzureEmbeddingError: エンドポイントまたはAPIキーが設定されていない場合
        """
        # エンドポイント取得（優先順位: 引数 > AZURE_OPENAI_ENDPOINT > OpenAIEmbeddingURI）
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OpenAIEmbeddingURI")
        if not self.endpoint:
            raise AzureEmbeddingError(
                "Azure OpenAI エンドポイントが設定されていません。"
                "AZURE_OPENAI_ENDPOINT または OpenAIEmbeddingURI 環境変数を設定してください。"
            )

        # APIキー取得（優先順位: 引数 > AZURE_OPENAI_API_KEY > OpenAIEmbeddingKey）
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OpenAIEmbeddingKey")
        if not self.api_key:
            raise AzureEmbeddingError(
                "Azure OpenAI APIキーが設定されていません。"
                "AZURE_OPENAI_API_KEY または OpenAIEmbeddingKey 環境変数を設定してください。"
            )

        # デプロイメント名取得
        self.deployment = (
            deployment
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or config.embedding_model
        )

        # 期待される次元数（検証用）
        self.expected_dimension = config.embedding_dimension

        # Azure OpenAI クライアント初期化
        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.API_VERSION,
        )

        logger.info(
            f"AzureEmbeddingClient 初期化完了: "
            f"endpoint={self._mask_endpoint(self.endpoint)}, "
            f"deployment={self.deployment}"
        )

    def _mask_endpoint(self, endpoint: str) -> str:
        """エンドポイントをログ用にマスク"""
        if len(endpoint) > 30:
            return endpoint[:15] + "..." + endpoint[-10:]
        return endpoint

    def get_embedding(self, text: str) -> List[float]:
        """単一テキストのエンベディングを取得

        Args:
            text: エンベディングを取得するテキスト

        Returns:
            1536次元のベクトル（floatのリスト）

        Raises:
            AzureEmbeddingError: API呼び出しに失敗した場合
        """
        if not text or not text.strip():
            raise AzureEmbeddingError("空のテキストはエンベディングできません")

        try:
            response = self._client.embeddings.create(
                model=self.deployment,
                input=text,
            )

            embedding = response.data[0].embedding

            # 次元数の検証
            if len(embedding) != self.expected_dimension:
                logger.warning(
                    f"エンベディング次元数が期待値と異なります: "
                    f"got={len(embedding)}, expected={self.expected_dimension}"
                )

            return embedding

        except Exception as e:
            logger.error(f"エンベディング取得に失敗: {e}")
            raise AzureEmbeddingError(f"エンベディング取得に失敗しました: {e}") from e

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """複数テキストのエンベディングをバッチ取得

        Azure OpenAI の embeddings API は一度に複数のテキストを処理できます。

        Args:
            texts: エンベディングを取得するテキストのリスト

        Returns:
            各テキストに対応する1536次元ベクトルのリスト

        Raises:
            AzureEmbeddingError: API呼び出しに失敗した場合、または空のリストが渡された場合
        """
        if not texts:
            raise AzureEmbeddingError("テキストリストが空です")

        # 空文字列のチェック
        valid_texts = []
        empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
            else:
                empty_indices.append(i)
                logger.warning(f"インデックス {i} のテキストが空のためスキップします")

        if not valid_texts:
            raise AzureEmbeddingError("有効なテキストがありません（全て空文字列）")

        try:
            response = self._client.embeddings.create(
                model=self.deployment,
                input=valid_texts,
            )

            # レスポンスからエンベディングを抽出（インデックス順にソート）
            embeddings_data = sorted(response.data, key=lambda x: x.index)
            valid_embeddings = [item.embedding for item in embeddings_data]

            # 空文字列の位置にはゼロベクトルを挿入
            if empty_indices:
                zero_vector = [0.0] * self.expected_dimension
                result = []
                valid_idx = 0
                for i in range(len(texts)):
                    if i in empty_indices:
                        result.append(zero_vector)
                    else:
                        result.append(valid_embeddings[valid_idx])
                        valid_idx += 1
                return result

            return valid_embeddings

        except Exception as e:
            logger.error(f"バッチエンベディング取得に失敗: {e}")
            raise AzureEmbeddingError(f"バッチエンベディング取得に失敗しました: {e}") from e

    def is_available(self) -> bool:
        """クライアントが利用可能かテスト

        Returns:
            True: API呼び出しが成功した場合
            False: API呼び出しが失敗した場合
        """
        try:
            self.get_embedding("test")
            return True
        except AzureEmbeddingError:
            return False


# シングルトンインスタンス（遅延初期化）
_client_instance: Optional[AzureEmbeddingClient] = None


def get_embedding_client() -> AzureEmbeddingClient:
    """グローバルなエンベディングクライアントを取得

    初回呼び出し時にクライアントを初期化します。

    Returns:
        AzureEmbeddingClient インスタンス

    Raises:
        AzureEmbeddingError: クライアントの初期化に失敗した場合
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = AzureEmbeddingClient()
    return _client_instance


def reset_client() -> None:
    """グローバルクライアントをリセット（テスト用）"""
    global _client_instance
    _client_instance = None
