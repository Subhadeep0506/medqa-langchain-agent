import os
from typing import Union

from langchain_cohere.chat_models import ChatCohere
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from ..enums import LLMService
from .logger_service import LoggerService

logger = LoggerService.get_logger(__name__)


class LLMFactory:
    @staticmethod
    def get_chat_model(
        llm_service: str,
    ) -> Union[ChatCohere, ChatGoogleGenerativeAI]:
        if llm_service == LLMService.COHERE.value:
            logger.info("Using Cohere chat model.")
            return ChatCohere(
                model=os.environ["COHERE_MODEL_NAME"],
                cohere_api_key=os.environ["COHERE_API_KEY"],
            )
        elif llm_service == LLMService.GEMINI.value:
            logger.info("Using Gemini chat model.")
            return ChatGoogleGenerativeAI(
                model=os.environ["GEMINI_MODEL_NAME"],
                google_api_key=os.environ["GEMINI_API_KEY"],
            )
        else:
            raise ValueError("Unsupported chat service.")
