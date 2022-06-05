import logging


def __log():
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    return logger


def PrintLog(message):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger = __log()
    logger.addHandler(console_handler)
    logger.info(message)


def WarnLog(message):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    logger = __log()
    logger.addHandler(console_handler)
    logger.warning(message)
