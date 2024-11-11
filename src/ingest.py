import os
from typing import List

from src.doc_reader.parquet_reader import ParquetReader
from src.doc_reader.pdf_reader import PDFReader

from .enums import FileType
from .services.embeddings_factory import EmbeddingsFactory
from .services.vectorstore_factory import VectorStoreFactory


class Ingestion:
    def __init__(self, embeddings_service: str, vectorstore_service: str) -> None:
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
            _ids = []
            try:
                _ids = self.vector_store.add_documents(documents=docs, ids=ids)
            except Exception:
                chunk_size = 500
                for i in range(0, len(docs), chunk_size):
                    chunk_docs = docs[i : i + chunk_size]
                    chunk_ids = ids[i : i + chunk_size]
                    _ids.extend(
                        self.vector_store.add_documents(
                            documents=chunk_docs, ids=chunk_ids
                        )
                    )
            if _ids != ids:
                raise ValueError("Failed to add documents to vectorstore")
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
