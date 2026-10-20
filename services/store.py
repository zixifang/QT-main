"""行情和因子序列进 parquet（没有 pyarrow 时退回 csv），摘要进 sqlite。"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from services.paths import ROOT

CACHE_VERSION = "1"


class Store:
    def __init__(self, root: Path | None = None) -> None:
        base = Path(root) if root is not None else ROOT
        self.root = base / "cache" / "ndx_enhance"
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            if self.root.exists():
                shutil.rmtree(self.root)

    def forget(self, key: str) -> None:
        with self._lock:
            self._forget_unlocked(key)

    def put_json(self, key: str, kind: str, payload: dict) -> None:
        with self._lock:
            self._write_json(key, kind, payload)

    def get_json(self, key: str) -> dict | None:
        with self._lock:
            return self._read_json(key)

    def write_bundle(self, key: str, kind: str, frame: pd.DataFrame, payload: dict) -> str:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            media = self._write_frame(key, frame)
            body = dict(payload)
            body["media"] = media
            if media == "csv":
                notes = list(body.get("warnings") or [])
                note = "没有 pyarrow，序列改存 csv"
                if note not in notes:
                    notes.append(note)
                body["warnings"] = notes
            self._write_json(key, kind, body)
            return media

    def read_bundle(self, key: str) -> tuple[pd.DataFrame, dict] | None:
        with self._lock:
            frame = self._read_frame(key)
            payload = self._read_json(key)
        if frame is None or payload is None:
            return None
        return frame, payload

    def _forget_unlocked(self, key: str) -> None:
        folder = self.root / "series"
        for suffix in (".parquet", ".csv"):
            path = folder / f"{key}{suffix}"
            if path.exists():
                path.unlink()
        if self.db_path.exists():
            with self._connect() as con:
                con.execute("delete from results where cache_key = ?", (key,))

    def _write_frame(self, key: str, frame: pd.DataFrame) -> str:
        folder = self.root / "series"
        folder.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        out.index.name = "date"
        parquet_path = folder / f"{key}.parquet"
        csv_path = folder / f"{key}.csv"
        if out.shape[0] == 0 or out.shape[1] == 0:
            out.to_csv(csv_path)
            if parquet_path.exists():
                parquet_path.unlink()
            return "empty"
        try:
            out.to_parquet(parquet_path)
            if csv_path.exists():
                csv_path.unlink()
            return "parquet"
        except (ImportError, ValueError, OSError):
            if parquet_path.exists():
                parquet_path.unlink()
            out.to_csv(csv_path)
            return "csv"

    def _read_frame(self, key: str) -> pd.DataFrame | None:
        folder = self.root / "series"
        parquet_path = folder / f"{key}.parquet"
        csv_path = folder / f"{key}.csv"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            return None
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame

    def _write_json(self, key: str, kind: str, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        with self._connect() as con:
            con.execute(
                "insert or replace into results(cache_key, kind, updated, payload) values (?, ?, ?, ?)",
                (key, kind, datetime.now().isoformat(timespec="seconds"), text),
            )
            con.commit()

    def _read_json(self, key: str) -> dict | None:
        if not self.db_path.exists():
            return None
        with self._connect() as con:
            row = con.execute("select payload from results where cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    @property
    def db_path(self) -> Path:
        return self.root / "results.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.execute(
            """
            create table if not exists results (
                cache_key text primary key,
                kind text,
                updated text,
                payload text
            )
            """
        )
        return _Closing(con)


class _Closing:
    """提交并关闭连接。sqlite 自己的上下文管理器不会关闭连接。"""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def __enter__(self) -> sqlite3.Connection:
        return self.con

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.con.commit()
            else:
                self.con.rollback()
        finally:
            self.con.close()
