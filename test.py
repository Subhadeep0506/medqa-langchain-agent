from dotenv import load_dotenv

from src.ingest import Ingestion

_ = load_dotenv()


ingestion = Ingestion(
    embeddings_service="cohere",
    vectorstore_service="milvus",
)

ingestion.ingest_document(
    file_path="data/medqa.parquet",
    category="medical",
    sub_category="conversation",
    exclude_columns=["instruction"],
)
