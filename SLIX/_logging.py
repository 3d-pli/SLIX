import logging


def get_logger(name, file=None, level=logging.INFO):
    """
    Returns a logger with the given name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s][%(name)s] - %(message)s')
    if file:
        # Write LOG to file
        fh = logging.FileHandler(file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Write LOG to console
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
