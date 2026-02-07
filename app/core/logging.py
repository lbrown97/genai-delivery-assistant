import logging


def setup_logging():
    """Configure application-wide logging format and default level."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
