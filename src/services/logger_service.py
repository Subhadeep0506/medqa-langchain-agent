import logging
import os
from datetime import datetime


class LoggerService:
    _instance = None
    _log_dir = "logs"

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LoggerService, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        # Ensure logs directory exists
        os.makedirs(cls._log_dir, exist_ok=True)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File Handler with daily log rotation
        log_filename = os.path.join(
            cls._log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    @staticmethod
    def get_logger(name=None):
        """
        Get a logger with optional name.
        If no name is provided, returns the root logger.
        """
        return logging.getLogger(name) if name else logging.getLogger()

    @staticmethod
    def set_log_level(level):
        """
        Set the logging level.
        Args:
            level (str or int): Logging level (e.g., 'DEBUG', 'INFO', logging.DEBUG)
        """
        logging.getLogger().setLevel(level)

    @staticmethod
    def log_system_info():
        """
        Log basic system information
        """
        logger = LoggerService.get_logger()
        logger.info("Application Logger Initialized")
        logger.info(f"Log Directory: {LoggerService._log_dir}")


# Create a singleton instance
logger_service = LoggerService()
