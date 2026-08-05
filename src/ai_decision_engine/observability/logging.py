import json
import logging
from typing import Any


class RoutingLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ai_decision_engine.routing")

    def event(self, name: str, **fields: Any) -> None:
        self.logger.info(json.dumps({"event": name, **fields}, default=str, sort_keys=True))
