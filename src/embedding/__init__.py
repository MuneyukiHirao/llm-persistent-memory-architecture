# Embedding モジュール
# Azure OpenAI Embedding クライアント

from src.embedding.azure_client import (
    AzureEmbeddingClient,
    AzureEmbeddingError,
    get_embedding_client,
    reset_client,
)

__all__ = [
    "AzureEmbeddingClient",
    "AzureEmbeddingError",
    "get_embedding_client",
    "reset_client",
]
