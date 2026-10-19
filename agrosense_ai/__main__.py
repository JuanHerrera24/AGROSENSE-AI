import logging

import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(token=settings.token), host=settings.host, port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
