import os
import pandas as pd

from hashlib import sha256
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

from src.doc_reader.base import BaseDocumentReader


class ParquetReader(BaseDocumentReader):
    """Custom PDF Loader to embed metadata with the pdfs."""

    def __init__(self) -> None:
        super().__init__()
        self.file_name = ""
        self.total_pages = 0

    def load_document(self, file_path, category, sub_category, **kwargs):
        """
        Loads a document from a parquet file, processes its content, and splits it into chunks.
        Args:
            file_path (str): The path to the parquet file containing the document data.
            category (str): The category of the document.
            sub_category (str): The sub-category of the document.
            **kwargs: Additional keyword arguments.
                - exclude_columns (list, optional): List of columns to exclude from the parquet file.
        Returns:
            tuple: A tuple containing:
                - final_chunks (list): A list of Document objects, each representing a chunk of the document.
                - ids (list): A list of SHA-256 hash strings, each representing the unique identifier of a chunk.
        Raises:
            FileNotFoundError: If the specified file_path does not exist.
            ValueError: If the parquet file does not contain the required columns.
        """
        try:
            self.file_name = os.path.basename(file_path)
            data = pd.read_parquet(file_path, engine="pyarrow")
            excluded_columns = kwargs.get("exclude_columns", None)
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=200,
            )
            self.total_pages = len(data)
            chunks = []
            ids = []
            for idx, page in data.iterrows():
                chunks.append(
                    Document(
                        page_content="\n\n".join(
                            [
                                f"{column}: {page[column]}"
                                for column in page.index
                                if excluded_columns and column not in excluded_columns
                            ]
                        ),
                        metadata=dict(
                            {
                                "file_name": self.file_name,
                                "page_no": str(idx + 1),
                                "total_pages": str(self.total_pages),
                                "category": category,
                                "sub_category": sub_category,
                            }
                        ),
                    )
                )
            final_chunks = text_splitter.split_documents(chunks)
            ids = [
                sha256(
                    (chunk.page_content + chunk.metadata["page_no"]).encode()
                ).hexdigest()
                for chunk in final_chunks
            ]
            return final_chunks, ids
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e
