import logging


def __log():
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    return logger


def PrintLog(message):
    # form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # console_handler_format = '%(acstime)s | %(levelname)s: %(message)s'
    # console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger = __log()
    logger.addHandler(console_handler)
    logger.info(message)



# def WarnLog(message):
#    logging.basicConfig(level=logging.WARNING)
#    logging.warning(me)
