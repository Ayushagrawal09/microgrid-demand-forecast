"""Tiny web server (standard library only - nothing to install)."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from controller import parse_csv_text, run_forecast, prepare_rows_keep_order
from predict import forecast_rows

HERE = os.path.dirname(os.path.abspath(__file__))


def read_file(*parts):
    with open(os.path.join(HERE, *parts), "rb") as fh:
        return fh.read()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        """Read the request body (works with Content-Length and with chunked uploads from gateways)."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            data = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():  # skip trailers
                        pass
                    return data
                data += self.rfile.read(size)
                self.rfile.readline()  # CRLF after each chunk
        return self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, read_file("static", "index.html"), "text/html")
        elif path == "/api/demo":
            self._send(200, {"csv": read_file("demo_day.csv").decode()})
        elif path == "/api/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/api/forecast", "/api/predict"):
            return self._send(404, {"error": "Not found"})
        try:
            payload = json.loads(self.read_body() or b"{}")
            if path == "/api/predict":
                # Used by predict.py: {"data": [rows]} -> {"predictions": [...]}
                rows = prepare_rows_keep_order(payload.get("data"))
                preds, source = forecast_rows(rows, allow_proxy=False)
                return self._send(200, {"predictions": [round(p, 8) for p in preds], "source": source})
            rows = payload.get("rows") or parse_csv_text(payload.get("csv", ""))
            self._send(200, run_forecast(rows))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # never crash the server
            self._send(500, {"error": f"Something went wrong: {exc}"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", 5000))
    print(f"Dashboard running on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
