import os

from typing import List
from .enums import FileType, EmbeddingsService
from langchain_postgres.vectorstores import PGVector
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from src.doc_reader.pdf_reader import PDFReader
from src.doc_reader.parquet_reader import ParquetReader


class Ingestion:
    def __init__(self, embeddings_service: str):
        if embeddings_service == EmbeddingsService.COHERE.value:
            self.embeddings = CohereEmbeddings(
                model=os.environ["COHERE_EMBEDDING_MODEL_NAME"],
                cohere_api_key=os.environ["COHERE_API_KEY"],
            )
        elif embeddings_service == EmbeddingsService.GEMINI.value:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=os.environ["GEMINI_EMBEDDING_MODEL_NAME"],
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
        else:
            raise ValueError("Unsupported embeddings service")

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="docstore",
            connection=os.environ["DATABASE_URI"],
            use_jsonb=True,
        )

    def ingest_document(
        self,
        file_path: str,
        category: str,
        sub_category: str,
        exclude_columns: List[str] | None = None,
    ):
        self.file_path = file_path
        try:
            self.doc_reader = self.__get_reader_by_filetype()
            if isinstance(self.doc_reader, PDFReader):
                docs, ids = self.doc_reader.load_document(
                    file_path, category=category, sub_category=sub_category
                )
            elif isinstance(self.doc_reader, ParquetReader):
                docs, ids = self.doc_reader.load_document(
                    file_path,
                    category=category,
                    sub_category=sub_category,
                    exclude_columns=exclude_columns,
                )
            _ids = self.vector_store.add_documents(documents=docs, ids=ids)
            if _ids != ids:
                raise ValueError("Failed to ingest all documents")
        except ValueError as e:
            raise e

    def __get_reader_by_filetype(self) -> PDFReader | ParquetReader:
        extension = str(os.path.basename(self.file_path)).split(".")[-1].lower()
        if extension == FileType.PDF.value:
            return PDFReader()
        elif extension == FileType.PARQUET.value:
            return ParquetReader()
        else:
            raise ValueError(f"Unsupported file type: {extension}")
