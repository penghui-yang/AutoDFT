"""
Logging configuration for AutoDFT.
"""
import logging


def setup_logging(log_file: str = "autodft.log", level: int = logging.INFO):
    """
    Configure logging for the entire application.
    
    Args:
        log_file: Path to the log file
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized")

