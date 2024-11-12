import os

from hashlib import sha256
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document

from src.doc_reader.base import BaseDocumentReader


class PDFReader(BaseDocumentReader):
    """Custom PDF Loader to embed metadata with the pdfs."""

    def __init__(self) -> None:
        self.file_name = ""
        self.total_pages = 0

    def load_document(self, file_path, category, sub_category):
        self.file_name = os.path.basename(file_path)
        loader = PyMuPDFLoader(file_path)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
        )
        pages = loader.load()
        self.total_pages = len(pages)
        chunks = []
        ids = []
        for idx, page in enumerate(pages):
            chunks.append(
                Document(
                    page_content=page.page_content,
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
