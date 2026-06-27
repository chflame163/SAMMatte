from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sam31_webapp.service import Sam31WebService, console_error, console_info
else:
    from .service import Sam31WebService, console_error, console_info


STATIC_ROOT = Path(__file__).resolve().parent / "static"
CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM 3.1 local web app server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--sam-max-inference-pixels",
        default=None,
        help="Max total pixels for SAM 3.1 inference video, e.g. 1920x1080.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="服务启动后自动打开默认浏览器。",
    )
    return parser.parse_args(argv)


def open_browser_async(url: str) -> None:
    def _worker() -> None:
        # Give serve_forever a brief head start before the first browser request lands.
        time.sleep(0.5)
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:  # pragma: no cover - desktop integration varies by host
            console_error(f"自动打开浏览器失败：{exc}")

    threading.Thread(
        target=_worker,
        name="sam31-browser-launcher",
        daemon=True,
    ).start()


def make_handler(service: Sam31WebService):
    class AppHandler(BaseHTTPRequestHandler):
        server_version = "SAM31WebApp/1.0"

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/favicon.ico":
                    self._send_empty(HTTPStatus.NO_CONTENT)
                    return
                if parsed.path == "/":
                    self._serve_file(STATIC_ROOT / "index.html")
                    return
                if parsed.path.startswith("/static/"):
                    relative = parsed.path[len("/static/") :]
                    target = (STATIC_ROOT / relative).resolve()
                    static_root = STATIC_ROOT.resolve()
                    if static_root not in target.parents and target != static_root:
                        self.send_error(HTTPStatus.FORBIDDEN)
                        return
                    self._serve_file(target)
                    return
                if parsed.path.startswith("/media/"):
                    self._serve_media(parsed.path[len("/media/") :])
                    return
                if parsed.path == "/api/status":
                    params = parse_qs(parsed.query)
                    session_id = params.get("session_id", [None])[0]
                    if not session_id:
                        self._send_json({"error": "缺少 session_id。"}, HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json(service.get_status(session_id))
                    return
                if parsed.path == "/api/health":
                    self._send_json({"ok": True})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - defensive server path
                self._handle_exception(exc)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/upload":
                    self._handle_upload()
                    return
                if parsed.path == "/api/prompt":
                    data = self._read_json_body()
                    result = service.apply_prompt(
                        session_id=data["sessionId"],
                        frame_index=int(data["frameIndex"]),
                        mode=data["mode"],
                        points=data.get("points"),
                        box=data.get("box"),
                        text_prompt=data.get("textPrompt"),
                        keyframe_enabled=bool(data.get("keyframeEnabled")),
                    )
                    self._send_json(result)
                    return
                if parsed.path == "/api/start_propagation":
                    data = self._read_json_body()
                    result = service.start_propagation(
                        session_id=data["sessionId"],
                        preview_bitrate=data.get("previewBitrate"),
                        mask_postprocess=data.get("maskPostprocess"),
                        keyframe_enabled=bool(data.get("keyframeEnabled")),
                    )
                    self._send_json(result)
                    return
                if parsed.path == "/api/reset_session":
                    data = self._read_json_body()
                    self._send_json(service.reset_session(data["sessionId"]))
                    return
                if parsed.path == "/api/keyframe/delete":
                    data = self._read_json_body()
                    self._send_json(
                        service.delete_keyframe(
                            session_id=data["sessionId"],
                            frame_index=int(data["frameIndex"]),
                        )
                    )
                    return
                if parsed.path == "/api/export_mask":
                    data = self._read_json_body()
                    self._send_json(
                        service.export_mask_video(
                            session_id=data["sessionId"],
                            bitrate=data.get("bitrate"),
                        )
                    )
                    return
                if parsed.path == "/api/close_session":
                    data = self._read_json_body()
                    self._send_json(service.close_session(data["sessionId"]))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - defensive server path
                self._handle_exception(exc)

        def _handle_upload(self) -> None:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(
                    {"error": "需要使用 multipart/form-data 上传。"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            if "video" not in form:
                self._send_json({"error": "缺少 video 字段。"}, HTTPStatus.BAD_REQUEST)
                return
            video_field = form["video"]
            previous_session_id = (
                form["previous_session_id"].value
                if "previous_session_id" in form and form["previous_session_id"].value
                else None
            )
            sam_max_inference_pixels = (
                form["sam_max_inference_pixels"].value
                if "sam_max_inference_pixels" in form
                and form["sam_max_inference_pixels"].value
                else None
            )
            if not getattr(video_field, "file", None):
                self._send_json({"error": "上传内容中没有文件。"}, HTTPStatus.BAD_REQUEST)
                return
            result = service.create_session_from_upload(
                video_field.file,
                getattr(video_field, "filename", "uploaded.mp4"),
                previous_session_id=previous_session_id,
                sam_max_inference_pixels=sam_max_inference_pixels,
            )
            self._send_json(result)

        def _serve_media(self, relative_path: str) -> None:
            decoded_relative_path = unquote(relative_path)
            target = (service.sessions_root / decoded_relative_path).resolve()
            sessions_root = service.sessions_root.resolve()
            if sessions_root not in target.parents and target != sessions_root:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._serve_file(target)

        def _serve_file(self, file_path: Path) -> None:
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            file_size = file_path.stat().st_size
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            if content_type in {"text/html", "text/css", "text/javascript", "application/javascript"}:
                content_type = f"{content_type}; charset=utf-8"
            range_header = self.headers.get("Range")
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK

            if range_header and range_header.startswith("bytes="):
                range_value = range_header[len("bytes=") :]
                start_text, _, end_text = range_value.partition("-")
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = int(end_text)
                end = min(end, file_size - 1)
                if start >= file_size or start > end:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                status = HTTPStatus.PARTIAL_CONTENT

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with file_path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except CLIENT_DISCONNECT_ERRORS:
                        return
                    remaining -= len(chunk)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except CLIENT_DISCONNECT_ERRORS:
                return

        def _send_empty(self, status: HTTPStatus = HTTPStatus.NO_CONTENT) -> None:
            try:
                self.send_response(status)
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except CLIENT_DISCONNECT_ERRORS:
                return

        def _handle_exception(self, exc: Exception) -> None:
            if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
                return
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            if isinstance(exc, KeyError):
                status = HTTPStatus.NOT_FOUND
            elif isinstance(exc, ValueError):
                status = HTTPStatus.BAD_REQUEST
            elif isinstance(exc, RuntimeError):
                status = HTTPStatus.BAD_REQUEST
            payload = {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
            parsed = urlparse(self.path)
            console_error(f"{self.command} {parsed.path} 失败：{type(exc).__name__}: {exc}")
            self._send_json(payload, status)

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            try:
                status = int(code)
            except (TypeError, ValueError):
                status = None
            if self.command == "GET" and status is not None and status < 400:
                return
            parsed = urlparse(self.path)
            console_info(f"{self.command} {parsed.path} -> {code}")

        def log_error(self, format: str, *args: Any) -> None:
            return

        def log_message(self, format: str, *args: Any) -> None:
            console_info(format % args)

    return AppHandler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = Sam31WebService(
        default_sam_max_inference_pixels=args.sam_max_inference_pixels
    )
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server_url = f"http://{args.host}:{args.port}"
    console_info(f"SAM 3.1 web app listening on {server_url}")
    if args.open_browser:
        open_browser_async(server_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console_info("Shutting down server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
