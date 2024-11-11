import os
from typing import Union

from langchain_cohere.embeddings import CohereEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from ..enums import EmbeddingsService


class EmbeddingsFactory:
    @staticmethod
    def get_embeddings(
        embeddings_service: str,
    ) -> Union[CohereEmbeddings, GoogleGenerativeAIEmbeddings]:
        if embeddings_service == EmbeddingsService.COHERE.value:
            return CohereEmbeddings(
                model=os.environ["COHERE_EMBEDDING_MODEL_NAME"],
                cohere_api_key=os.environ["COHERE_API_KEY"],
            )
        elif embeddings_service == EmbeddingsService.GEMINI.value:
            return GoogleGenerativeAIEmbeddings(
                model=os.environ["GEMINI_EMBEDDING_MODEL_NAME"],
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
        else:
            raise ValueError("Unsupported embeddings service")
