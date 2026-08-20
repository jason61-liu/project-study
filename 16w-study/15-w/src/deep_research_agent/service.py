from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .agent import ResearchAgent
from .guardrails import PolicyViolation
from .models import Identity, RunRequest
from .store import IdempotencyConflict, SQLiteStore


def make_handler(agent: ResearchAgent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ResearchAgent/0.1"

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise PolicyViolation("request body too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/healthz":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/metrics":
                metrics = agent.store.metrics()
                body = "\n".join(f"research_agent_{key} {value}" for key, value in metrics.items()) + "\n"
                encoded = body.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path.startswith("/v1/runs/"):
                run_id = path.rsplit("/", 1)[-1]
                tenant_id = self.headers.get("X-Tenant-Id", "")
                record = agent.record(tenant_id, run_id)
                self._json(HTTPStatus.OK if record else HTTPStatus.NOT_FOUND, record or {"error": "not_found"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                body = self._body()
                if path == "/v1/runs":
                    state = agent.submit(RunRequest.from_dict(body))
                    self._json(HTTPStatus.OK, state.to_dict())
                    return
                if path.endswith("/approve") and path.startswith("/v1/runs/"):
                    run_id = path.split("/")[-2]
                    identity = body["identity"]
                    approver = Identity(
                        tenant_id=str(identity["tenant_id"]),
                        user_id=str(identity["user_id"]),
                        roles=tuple(identity.get("roles", ["approver"])),
                        scopes=tuple(identity.get("scopes", [])),
                    )
                    state = agent.approve_and_resume(approver.tenant_id, run_id, approver)
                    self._json(HTTPStatus.OK, state.to_dict())
                    return
                if path.endswith("/resume") and path.startswith("/v1/runs/"):
                    run_id = path.split("/")[-2]
                    identity = body["identity"]
                    actor = Identity(
                        tenant_id=str(identity["tenant_id"]),
                        user_id=str(identity["user_id"]),
                        roles=tuple(identity.get("roles", ["researcher"])),
                        scopes=tuple(identity.get("scopes", [])),
                    )
                    state = agent.resume(run_id, actor)
                    self._json(HTTPStatus.OK, state.to_dict())
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except (KeyError, ValueError, PolicyViolation, IdempotencyConflict) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__, "message": str(exc)})
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})

        def log_message(self, format: str, *args: object) -> None:
            # Access log intentionally excludes bodies and identity fields.
            print(json.dumps({"event": "http_access", "message": format % args}))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--db", default=os.getenv("DATABASE_PATH", "data/research.db"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ResearchAgent(SQLiteStore(args.db))))
    server.serve_forever()


if __name__ == "__main__":
    main()
