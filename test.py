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

"""
schema = client.create_schema(enable_dynamic_field=True, description="")
schema.add_field(field_name="id", datatype=DataType.INT64, description="The Primary Key", is_primary=True, auto_id=False)
schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="metadata", datatype=DataType.JSON)
schema.add_field(field_name="document", datatype=DataType.VARCHAR)

index_params = client.prepare_index_params()
index_params.add_index(field_name="embeddings", metric_type="COSINE", index_type="AUTOINDEX")
client.create_collection(collection_name="docstore", schema=schema, index_params=index_params)
"""
