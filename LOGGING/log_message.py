import logging


def PrintLog(message):
    # form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
    # logger = logging.getLogger()
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(form)
    # logger.addHandler(consoleHandler)
    logging.basicConfig(level=logging.INFO)
    logging.info(message)

def WarnLog(message):
    logging.basicConfig(level=logging.WARNING)
    logging.warning(me)