from dotenv import load_dotenv

from src.ingest import Ingestion

_ = load_dotenv()


ingestion = Ingestion(
    embeddings_service="gemini",
    vectorstore_service="pgvector",
)

ingestion.ingest_document(
    file_path="data/medqa.parquet",
    category="test",
    sub_category="test",
)
