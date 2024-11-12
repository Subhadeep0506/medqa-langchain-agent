import os
import logging
import colorlog
from typing import Union

from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from pymilvus import Collection, connections, MilvusClient, DataType

from ..enums import VectorStoreService


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class VectorStoreFactory:
    @staticmethod
    def get_vectorstore(
        vectorstore_service: str,
        embeddings,
    ) -> Union[PGVector, Milvus]:
        if vectorstore_service == VectorStoreService.PGVECTOR.value:
            logger.log(logging.INFO, "Using PGVector")
            return PGVector(
                embeddings=embeddings,
                collection_name=os.environ["POSTGRES_COLLECTION_NAME"],
                connection=os.environ["POSTGRES_DATABASE_URI"],
                use_jsonb=True,
            )
        elif vectorstore_service == VectorStoreService.MILVUS.value:
            logger.log(logging.INFO, "Using Milvus")
            client = MilvusClient(
                uri=os.environ["MILVUS_DATABASE_URI"],
                token=os.environ["MILVUS_ACCESS_TOKEN"],
            )
            if not client.has_collection(os.environ["MILVUS_COLLECTION_NAME"]):
                logger.log(logging.INFO, "Creating Milvus collection")
                try:
                    schema = client.create_schema(
                        enable_dynamic_field=True, description=""
                    )
                    schema.add_field(
                        field_name="id",
                        datatype=DataType.VARCHAR,
                        max_length=65535,
                        description="The Primary Key",
                        is_primary=True,
                        auto_id=False,
                    )
                    schema.add_field(
                        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024
                    )
                    schema.add_field(field_name="metadata", datatype=DataType.JSON)
                    schema.add_field(
                        field_name="document",
                        datatype=DataType.VARCHAR,
                        max_length=65535,
                    )
                    index_params = client.prepare_index_params()
                    index_params.add_index(
                        field_name="vector",
                        metric_type="COSINE",
                        index_type="AUTOINDEX",
                    )
                    client.create_collection(
                        collection_name="docstore",
                        schema=schema,
                        index_params=index_params,
                    )
                except Exception as e:
                    logger.error("Error creating Milvus collection: %s", e)
            return Milvus(
                embedding_function=embeddings,
                collection_name=os.environ["MILVUS_COLLECTION_NAME"],
                connection_args={
                    "uri": os.environ["MILVUS_DATABASE_URI"],
                    "token": os.environ["MILVUS_ACCESS_TOKEN"],
                },
                auto_id=False,
                primary_field="id",
                text_field="document",
                metadata_field="metadata",
                vector_field="vector",
            )
        else:
            raise ValueError("Unsupported vectorstore service")
