import logging
import datetime
import os

logger = None

def initial_logger(logging_path: str = "log", enable_stdout: bool = False) -> None:
    """Initializes the logger for the application."""
    global logger

    now = datetime.datetime.now()
    log_file = os.path.join(
        logging_path, f"deep_research_py_{now.strftime('%Y%m%d_%H%M%S')}.log"
    )
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    # Set up logging to stdout if enabled
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]  # Add UTF-8 encoding
    if enable_stdout:
        console_handler = logging.StreamHandler()
        # Add colors to the console output
        console_format = logging.Formatter(
            '\033[1;36m%(asctime)s\033[0m - '  # Cyan timestamp
            '\033[1;32m%(levelname)s\033[0m - '  # Green level
            '%(message)s'  # Normal message
        )
        console_handler.setFormatter(console_format)
        handlers.append(console_handler)

    # Set up logging to file with detailed format
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handlers[0].setFormatter(file_format)

    # Configure the root logger
    logging.basicConfig(handlers=handlers, force=True)  # Add force=True to ensure configuration
    
    # Configure our specific logger
    logger = logging.getLogger("deep_research_py")
    logger.setLevel(logging.INFO)

def log_event(event_desc: str) -> None:
    """Logs an event with input and output token counts."""
    if not logger:
        return
    logger.info(f"➤ {event_desc}")  # Changed emoji to a simpler one that's widely supported

def log_error(error_desc: str) -> None:
    """Logs an error message."""
    if not logger:
        return
    logger.error(f"✖ Error: {error_desc}")  # Changed emoji to a simpler one

def log_warning(warning_desc: str) -> None:
    """Logs a warning message."""
    if not logger:
        return
    logger.warning(f"⚠ Warning: {warning_desc}")  # Changed emoji to a simpler one
