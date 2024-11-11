from enums import Enum


class FileType(Enum):
    PDF = "pdf"
    PARQUET = "parquet"


class EmbeddingsService(Enum):
    COHERE = "cohere"
    GEMINI = "gemini"
