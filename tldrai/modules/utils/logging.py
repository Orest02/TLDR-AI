import logging


def configure_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger
