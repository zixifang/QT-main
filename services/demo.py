"""小样本跑通主流程。python -m services.demo"""

from __future__ import annotations

import sys

from services.config_store import ConfigStore
from services.errors import ServiceError
from services.report_service import ReportService


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    config = ConfigStore().load()

    def progress(pct: int, message: str) -> None:
        print(f"[{pct:3d}] {message}", flush=True)

    try:
        outcome = ReportService().export(config, "both", progress=progress)
    except ServiceError as exc:
        print(exc)
        return 1
    print(outcome["summary"])
    if outcome["html"]:
        print(f"HTML {outcome['html']}")
    if outcome["pdf"]:
        print(f"PDF {outcome['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
