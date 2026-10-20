"""长任务。界面用 poll 取进度，用 cancel 停掉计算。"""

from __future__ import annotations

import threading
import traceback

from services.errors import Cancelled


class TaskService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self._seq = 0

    def start(self, name: str, fn) -> str:
        with self._lock:
            self._seq += 1
            task_id = f"t{self._seq}"
            event = threading.Event()
            self._events[task_id] = event
            self._tasks[task_id] = {
                "id": task_id,
                "name": name,
                "status": "running",
                "progress": 0,
                "logs": [],
                "result": None,
                "error": "",
            }
        def run() -> None:
            def progress(pct: int, message: str) -> None:
                with self._lock:
                    task = self._tasks[task_id]
                    task["progress"] = int(pct)
                    task["logs"].append(str(message))
                    if len(task["logs"]) > 400:
                        task["logs"] = task["logs"][-400:]

            def cancel() -> bool:
                return event.is_set()

            try:
                result = fn(progress, cancel)
            except Cancelled as exc:
                with self._lock:
                    task = self._tasks[task_id]
                    task["status"] = "cancelled"
                    task["error"] = str(exc)
                    task["logs"].append(str(exc))
                return
            except Exception as exc:
                text = str(exc) if exc.__class__.__module__ == "services.errors" else traceback.format_exc()
                with self._lock:
                    task = self._tasks[task_id]
                    task["status"] = "error"
                    task["error"] = text
                    task["logs"].append(text)
                return
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "done"
                task["progress"] = 100
                task["result"] = result

        threading.Thread(target=run, name=name, daemon=True).start()
        return task_id

    def cancel(self, task_id: str) -> None:
        event = self._events.get(task_id)
        if event is not None:
            event.set()

    def poll(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return {"id": task_id, "name": "", "status": "missing", "progress": 0, "logs": [], "result": None, "error": "任务不存在"}
            return {
                "id": task["id"],
                "name": task["name"],
                "status": task["status"],
                "progress": task["progress"],
                "logs": list(task["logs"]),
                "result": task["result"],
                "error": task["error"],
            }
