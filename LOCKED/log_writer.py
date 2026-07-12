"""
append_only_writer —— LOG 区唯一合法写入路径(§0 铁律2)。

本模块刻意不提供任何 update/delete 方法。LOG/ 下的 jsonl 与 tsv
文件只能通过这里的函数追加内容,任何"改历史记录"的需求都必须
通过写入新记录(而不是修改旧记录)来表达。
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Iterable

LOG_ROOT = Path(__file__).resolve().parent.parent / "LOG"


def _to_jsonable(record: Any) -> dict:
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record)
    if isinstance(record, dict):
        return record
    raise TypeError(f"record must be a dict or dataclass instance, got {type(record)!r}")


def append_jsonl(relative_path: str, record: Any, root: Path | None = None) -> Path:
    """向 LOG 区某个 .jsonl 文件追加一条记录(单行 JSON)。"""
    base = root if root is not None else LOG_ROOT
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def append_tsv_row(relative_path: str, columns: Iterable[Any], header: Iterable[str] | None = None,
                    root: Path | None = None) -> Path:
    """向 LOG 区某个 .tsv 文件追加一行。文件不存在且提供了 header 时先写表头。"""
    base = root if root is not None else LOG_ROOT
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = header is not None and not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(str(h) for h in header) + "\n")
        f.write("\t".join(str(c) for c in columns) + "\n")
    return path


def append_text(relative_path: str, text: str, root: Path | None = None) -> Path:
    """向 LOG 区某个文本文件(如反思摘要)追加内容,自动加分隔与换行。"""
    base = root if root is not None else LOG_ROOT
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return path


def read_jsonl(relative_path: str, root: Path | None = None) -> list[dict]:
    """只读辅助函数:读取整份 jsonl(供 scorer/reflector 等消费者使用)。"""
    base = root if root is not None else LOG_ROOT
    path = base / relative_path
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
