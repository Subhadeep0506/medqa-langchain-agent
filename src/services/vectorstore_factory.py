import os
import logging
from typing import Union

from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from pymilvus import Collection, connections

from ..enums import VectorStoreService


logger = logging.getLogger(__name__)

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
            return Milvus(
                embedding_function=embeddings,
                collection_name=os.environ["MILVUS_COLLECTION_NAME"],
                connection_args={
                    "uri": os.environ["MILVUS_DATABASE_URI"],
                    "token":os.environ["MILVUS_ACCESS_TOKEN"],
                },
            )
        else:
            raise ValueError("Unsupported vectorstore service")
