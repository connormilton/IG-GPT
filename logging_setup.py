# logging_setup.py

import sys
import os
import logging

def setup_logging(config):
    log_level_str = config.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = config.get('LOG_FILE')
    logger_name = "TradingBot"

    bot_logger = logging.getLogger(logger_name)
    if bot_logger.hasHandlers():
        bot_logger.handlers.clear()

    bot_logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    bot_logger.addHandler(ch)

    # File handler with directory creation
    if log_file:
        try:
            directory = os.path.dirname(log_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            bot_logger.addHandler(fh)
        except Exception as e:
            bot_logger.error(f"Error setting up file logging to {log_file}: {e}")

    bot_logger.propagate = False
    bot_logger.info(f"----- Logging configured for {logger_name} (Level: {log_level_str}) -----")

    # Make it accessible outside
    return bot_logger
