import os
import time
from typing import List

from src.doc_reader.parquet_reader import ParquetReader
from src.doc_reader.pdf_reader import PDFReader
from src.services.logger_service import LoggerService

from .enums import FileType
from .services.embeddings_factory import EmbeddingsFactory
from .services.vectorstore_factory import VectorStoreFactory

logger = LoggerService.get_logger(__name__)


class Ingestion:
    def __init__(self, embeddings_service: str, vectorstore_service: str) -> None:
        """
        Initializes the ingestion process with the specified embeddings and vector store services.
        Args:
            embeddings_service (str): The name of the embeddings service to use.
                Valid options include:
                - 'cohere
                - 'gemini
            vectorstore_service (str): The name of the vector store service to use.
                Valid options include:
                - 'pgvector'
                - 'milvus'
        Raises:
            ValueError: If an invalid embeddings_service or vectorstore_service is provided.
        """
        LoggerService.log_system_info()
        self.embeddings = EmbeddingsFactory.get_embeddings(embeddings_service)

        self.vector_store = VectorStoreFactory.get_vectorstore(
            vectorstore_service=vectorstore_service,
            embeddings=self.embeddings,
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
            _ids = []
            if isinstance(self.doc_reader, PDFReader):
                docs, ids = self.doc_reader.load_document(
                    file_path,
                    category=category,
                    sub_category=sub_category,
                )
                logger.info("Ingesting document to vectorstore")
                _ids = self.vector_store.add_documents(
                    documents=docs,
                    ids=ids,
                )
            elif isinstance(self.doc_reader, ParquetReader):
                docs, ids = self.doc_reader.load_document(
                    file_path,
                    category=category,
                    sub_category=sub_category,
                    exclude_columns=exclude_columns,
                )
                logger.info("Ingesting document to vectorstore using chunks.")
                chunk_size = 100
                for i in range(0, len(docs), chunk_size):
                    chunk_docs = docs[i : i + chunk_size]
                    chunk_ids = ids[i : i + chunk_size]
                    _ids.extend(
                        self.vector_store.add_documents(
                            documents=chunk_docs,
                            ids=chunk_ids,
                        )
                    )
                    logger.info("\tChunk Ingested.")
                    time.sleep(60)
        except Exception as e:
            logger.error("Ingestion failed. Error: %s", e)

    def __get_reader_by_filetype(self) -> PDFReader | ParquetReader:
        extension = str(os.path.basename(self.file_path)).split(".")[-1].lower()
        if extension == FileType.PDF.value:
            logger.info("Using PDF Reader")
            return PDFReader()
        elif extension == FileType.PARQUET.value:
            logger.info("Using Parquet Reader")
            return ParquetReader()
        else:
            raise ValueError(f"Unsupported file type: {extension}")
