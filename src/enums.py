from enum import Enum


class FileType(Enum):
    PDF = "pdf"
    PARQUET = "parquet"


class EmbeddingsService(Enum):
    COHERE = "cohere"
    GEMINI = "gemini"


class LLMService(Enum):
    COHERE = "cohere"
    GEMINI = "gemini"


class VectorStoreService(Enum):
    PGVECTOR = "pgvector"
    MILVUS = "milvus"
