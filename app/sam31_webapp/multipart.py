"""Minimal streaming multipart/form-data parser.

Replaces the stdlib ``cgi.FieldStorage`` (deprecated in Python 3.11, removed in
3.13) so the webapp keeps working on newer Python runtimes on Linux.

The public surface intentionally mimics the small subset of ``cgi`` that
``app.py`` relies on: a mapping of field name -> object with ``.name``,
``.filename``, ``.file`` (a seekable binary spool) and ``.value``.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from typing import BinaryIO, Optional

DEFAULT_SPOOL_MAX_BYTES = 4 * 1024 * 1024
READ_CHUNK_BYTES = 512 * 1024

_NAME_RE = re.compile(r"""(?:^|;)\s*name=(?P<q>"?)(?P<value>.*?)(?P=q)(?:;|$)""")
_FILENAME_RE = re.compile(r"""(?:^|;)\s*filename=(?P<q>"?)(?P<value>.*?)(?P=q)(?:;|$)""")
_BOUNDARY_RE = re.compile(r"""(?:^|;)\s*boundary=(?P<q>"?)(?P<value>.*?)(?P=q)(?:;|$)""")


class MultipartParseError(ValueError):
    """Raised when the request body is not valid multipart/form-data."""


@dataclass
class MultipartField:
    name: str
    filename: Optional[str]
    file: BinaryIO
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def value(self) -> str:
        position = self.file.tell()
        self.file.seek(0)
        payload = self.file.read()
        self.file.seek(position)
        return payload.decode("utf-8", errors="replace")

    def __bool__(self) -> bool:
        return True


def extract_boundary(content_type: str) -> str:
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        raise MultipartParseError("Content-Type 中缺少 multipart boundary。")
    boundary = match.group("value").strip()
    if not boundary:
        raise MultipartParseError("Content-Type 中的 multipart boundary 为空。")
    return boundary


def _parse_headers(header_block: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_line in header_block.decode("latin-1").splitlines():
        if not raw_line.strip():
            continue
        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        headers[key.strip().lower()] = value.strip()
    return headers


def _field_from_headers(headers: dict[str, str]) -> tuple[str, Optional[str]]:
    disposition = headers.get("content-disposition", "")
    name_match = _NAME_RE.search(disposition)
    if not name_match:
        raise MultipartParseError("multipart 分部缺少 name 字段。")
    name = name_match.group("value")
    filename_match = _FILENAME_RE.search(disposition)
    filename = filename_match.group("value") if filename_match else None
    if filename is not None:
        filename = filename.replace('\\"', '"').strip()
    return name, filename


def parse_multipart(
    stream: BinaryIO,
    content_type: str,
    content_length: int,
    spool_max_bytes: int = DEFAULT_SPOOL_MAX_BYTES,
) -> dict[str, MultipartField]:
    """Parse a multipart/form-data body from ``stream`` into a field mapping."""

    if content_length <= 0:
        raise MultipartParseError("上传请求没有内容（Content-Length 为 0）。")

    boundary = extract_boundary(content_type)
    first_delimiter = b"--" + boundary.encode("latin-1")
    delimiter = b"\r\n" + first_delimiter

    remaining = int(content_length)
    buffer = bytearray()

    def fill() -> bool:
        nonlocal remaining, buffer
        if remaining <= 0:
            return False
        chunk = stream.read(min(READ_CHUNK_BYTES, remaining))
        if not chunk:
            return False
        remaining -= len(chunk)
        buffer.extend(chunk)
        return True

    def require_bytes(count: int) -> bool:
        while len(buffer) < count:
            if not fill():
                return False
        return True

    # Skip the preamble and position the buffer right after the first boundary.
    while True:
        index = buffer.find(first_delimiter)
        if index != -1:
            del buffer[: index + len(first_delimiter)]
            break
        if not fill():
            raise MultipartParseError("上传数据中没有找到 multipart 起始边界。")

    fields: dict[str, MultipartField] = {}

    while True:
        if not require_bytes(2):
            break
        if buffer[:2] == b"--":
            # Closing delimiter: "--boundary--".
            break
        if buffer[:2] != b"\r\n":
            raise MultipartParseError("multipart 边界后格式不正确。")
        del buffer[:2]

        while b"\r\n\r\n" not in buffer:
            if not fill():
                raise MultipartParseError("multipart 分部头部不完整。")
        header_block, _, rest = bytes(buffer).partition(b"\r\n\r\n")
        buffer = bytearray(rest)

        headers = _parse_headers(header_block)
        name, filename = _field_from_headers(headers)

        spool = tempfile.SpooledTemporaryFile(max_size=spool_max_bytes)
        tail_keep = len(delimiter) - 1
        while True:
            index = buffer.find(delimiter)
            if index != -1:
                spool.write(bytes(buffer[:index]))
                del buffer[: index + len(delimiter)]
                break
            safe = len(buffer) - tail_keep
            if safe > 0:
                spool.write(bytes(buffer[:safe]))
                del buffer[:safe]
            if not fill():
                spool.close()
                raise MultipartParseError("上传数据在分部结束前被截断。")

        spool.seek(0)
        fields[name] = MultipartField(
            name=name,
            filename=filename,
            file=spool,
            headers=headers,
        )

        if not require_bytes(2):
            break

    return fields


def close_fields(fields: dict[str, MultipartField]) -> None:
    for field_obj in fields.values():
        try:
            field_obj.file.close()
        except Exception:
            pass
