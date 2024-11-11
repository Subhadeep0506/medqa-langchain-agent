import os

from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_cohere.embeddings import CohereEmbeddings
from src.doc_reader.pdf_reader import PDFReader
from src.doc_reader.parquet_reader import ParquetReader

load_dotenv()

embeddings = CohereEmbeddings(
    model=os.environ["COHERE_EMBEDDING_MODEL_NAME"],
    cohere_api_key=os.environ["COHERE_API_KEY"],
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="docstore",
    connection=os.environ["DATABASE_URI"],
    use_jsonb=True,
)

reader = PDFReader()
docs, ids = reader.load_pdf(
    file_path="./data/clinical_medicine_ashok_chandra.pdf",
    category="docs",
    sub_category="medical",
)
_ids = vector_store.add_documents(
    documents=docs,
    ids=ids,
)

reader = ParquetReader()

docs, ids = reader.load_document(
    file_path="./data/medqa.parquet",
    category="records",
    sub_category="medical",
)
