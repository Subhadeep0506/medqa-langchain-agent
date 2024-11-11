import os
from typing import Union

from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from pymilvus import Collection, connections

from ..enums import VectorStoreService


class VectorStoreFactory:
    @staticmethod
    def get_vectorstore(
        vectorstore_service: str,
        embeddings,
    ) -> Union[PGVector, Milvus]:
        if vectorstore_service == VectorStoreService.PGVECTOR.value:
            return PGVector(
                embeddings=embeddings,
                collection_name=os.environ["POSTGRES_COLLECTION_NAME"],
                connection=os.environ["POSTGRES_DATABASE_URI"],
                use_jsonb=True,
            )
        elif vectorstore_service == VectorStoreService.MILVUS.value:
            connections.connect(
                uri=os.environ["MILVUS_DATABASE_URI"],
                token=os.environ["MILVUS_ACCESS_TOKEN"],
            )
            collection = Collection(name=os.environ["MILVUS_COLLECTION_NAME"])
            return Milvus(
                embedding_function=embeddings,
                collection_name=collection,
                connection_args={
                    "uri": os.environ["MILVUS_DATABASE_URI"],
                },
            )
        else:
            raise ValueError("Unsupported vectorstore service")
