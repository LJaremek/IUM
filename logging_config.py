import logging


def configure_logging() -> None:
    logging.basicConfig(
        filename="logfile.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
        )


configure_logging()
