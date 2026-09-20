"""
FastAPI backend for the Multi-Agent Orchestrator UI.

    python server.py            # then open http://127.0.0.1:8000
"""

import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import agents

app = FastAPI(title="Multi-Agent Orchestrator")
STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config():
    return {"model": agents.MODEL, "demo": agents.is_demo(), "agents": agents.AGENTS}


@app.post("/api/run")
def run(req: RunRequest):
    order = [a["id"] for a in agents.AGENTS]
    keys = {a["id"]: a["key"] for a in agents.AGENTS}

    def event_stream():
        current = 0
        started = time.time()
        yield sse({"type": "agent", "agent": order[0], "status": "running"})
        try:
            for update in agents.graph.stream({"task": req.task}, stream_mode="updates"):
                for node, output in update.items():
                    now = time.time()
                    yield sse(
                        {
                            "type": "agent",
                            "agent": node,
                            "status": "done",
                            "output": (output or {}).get(keys[node], ""),
                            "seconds": round(now - started, 1),
                        }
                    )
                    started = now
                    current = order.index(node) + 1
                    if current < len(order):
                        yield sse({"type": "agent", "agent": order[current], "status": "running"})
                    if node == order[-1]:
                        yield sse({"type": "final", "report": (output or {}).get(keys[node], "")})
        except Exception as exc:  # noqa: BLE001
            failed = order[min(current, len(order) - 1)]
            yield sse({"type": "error", "agent": failed, "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    mode = "DEMO mode (no API key found)" if agents.is_demo() else f"live · {agents.MODEL}"
    print(f"\n  Multi-Agent Orchestrator -> http://127.0.0.1:8000   [{mode}]\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000)