"""FastAPI app for manual image QA labeling.

Keyboard-driven: y=good, n=bad, s=skip, Backspace=undo.
Annotations saved to JSONL (crash-safe) and DB (on shutdown).
"""

import json
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from db.connection import get_session
from vision.data.manifest import build_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_UPSERT_MANUAL_QA = text("""
    INSERT INTO image_annotations
        (image_id, feature_name, annotation_type, feature_value, confidence, annotator)
    VALUES (:image_id, :feature_name, 'manual_qa', :feature_value, 1.0, 'manual')
    ON CONFLICT (image_id, feature_name, annotation_type)
    DO UPDATE SET feature_value = EXCLUDED.feature_value,
                  created_at    = NOW()
""")


class AnnotateRequest(BaseModel):
    verdict: str  # "good" | "bad" | "skip"


class LabelingState:
    """Holds the image queue, current position, and annotation log."""

    def __init__(self, feature_name: str, manifest_df, jsonl_path: Path):
        self.feature_name = feature_name
        self.jsonl_path = jsonl_path

        # Load existing annotations from JSONL (latest verdict per image wins)
        self.annotations: dict[str, str] = {}
        if jsonl_path.exists():
            for line in jsonl_path.read_text().strip().splitlines():
                entry = json.loads(line)
                self.annotations[entry["image_id"]] = entry["verdict"]

        # Build ordered queue: all manifest images, already-annotated ones go to history
        all_images = []
        for _, row in manifest_df.iterrows():
            all_images.append(
                {
                    "image_id": row["image_id"],
                    "species": row["species"],
                    "feature_value": row["feature_value"],
                }
            )

        # Split into history (already annotated) and queue (remaining)
        self.history: list[dict] = []
        self.queue: list[dict] = []
        for img in all_images:
            if img["image_id"] in self.annotations:
                self.history.append(img)
            else:
                self.queue.append(img)

        self.cursor = 0  # position in self.queue

    @property
    def current(self) -> dict | None:
        if self.cursor < len(self.queue):
            return self.queue[self.cursor]
        return None

    @property
    def total(self) -> int:
        return len(self.history) + len(self.queue)

    @property
    def done_count(self) -> int:
        return len(self.annotations)

    def annotate(self, verdict: str) -> None:
        img = self.current
        if img is None:
            return
        self.annotations[img["image_id"]] = verdict
        self._append_jsonl(img["image_id"], verdict)
        self.history.append(img)
        self.cursor += 1

    def undo(self) -> dict | None:
        if not self.history:
            return None
        img = self.history.pop()
        # If this image is in the current queue position (was just annotated), back up
        if self.cursor > 0 and self.queue[self.cursor - 1]["image_id"] == img["image_id"]:
            self.cursor -= 1
        else:
            # It was from the pre-loaded history, re-insert at cursor
            self.queue.insert(self.cursor, img)
        # Remove annotation
        self.annotations.pop(img["image_id"], None)
        self._append_jsonl(img["image_id"], "__undo__")
        return img

    def _append_jsonl(self, image_id: str, verdict: str) -> None:
        entry = {
            "image_id": image_id,
            "verdict": verdict,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def verdict_counts(self) -> dict[str, int]:
        counts = {"good": 0, "bad": 0, "skip": 0}
        for v in self.annotations.values():
            if v in counts:
                counts[v] += 1
        return counts

    def persist_to_db(self) -> int:
        """Write all annotations to image_annotations table. Returns count."""
        if not self.annotations:
            return 0
        with get_session() as session:
            n = 0
            for image_id, verdict in self.annotations.items():
                if verdict == "__undo__":
                    continue
                session.execute(
                    _UPSERT_MANUAL_QA,
                    {
                        "image_id": image_id,
                        "feature_name": self.feature_name,
                        "feature_value": verdict,
                    },
                )
                n += 1
            session.commit()
        return n


# Global state — initialized by create_app()
_state: LabelingState | None = None


def create_app(feature_name: str, preprocess_version: str = "v1") -> FastAPI:
    """Build the FastAPI app with labeling state for the given feature."""
    global _state

    import yaml

    config = yaml.safe_load(open(PROJECT_ROOT / "vision/config/training.yaml"))
    embedding_dir = (
        PROJECT_ROOT
        / config["paths"]["embeddings"]
        / "facebook-dinov2-base"
        / f"processed-{preprocess_version}"
    )
    processed_dir = PROJECT_ROOT / config["paths"]["processed_images"] / preprocess_version

    # Build manifest (quality-filtered)
    print(f"Building manifest for {feature_name}...")
    with get_session() as session:
        manifest = build_manifest(
            session, feature_name, embedding_dir, preprocess_version, quality_filter=True
        )

    if manifest.empty:
        print("No images in manifest. Nothing to label.")
        sys.exit(1)

    # JSONL path
    jsonl_dir = PROJECT_ROOT / "data" / "manual_qa"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{feature_name}.jsonl"

    _state = LabelingState(feature_name, manifest, jsonl_path)

    counts = _state.verdict_counts()
    print(
        f"Queue: {len(_state.queue)} remaining, {_state.done_count} already annotated "
        f"(good={counts['good']}, bad={counts['bad']}, skip={counts['skip']})"
    )

    app = FastAPI(title=f"Manual QA: {feature_name}")

    # Serve processed images as static files
    app.mount("/images", StaticFiles(directory=str(processed_dir)), name="images")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_html(feature_name)

    @app.get("/api/current")
    async def get_current():
        img = _state.current
        if img is None:
            return {"done": True, "stats": _build_stats()}
        return {
            "done": False,
            "image_id": img["image_id"],
            "species": img["species"],
            "feature_value": img["feature_value"],
            "image_url": f"/images/{img['image_id']}.jpg",
            "stats": _build_stats(),
        }

    @app.post("/api/annotate")
    async def annotate(req: AnnotateRequest):
        if req.verdict not in ("good", "bad", "skip"):
            return {"error": "verdict must be good, bad, or skip"}
        _state.annotate(req.verdict)
        return {"ok": True, "stats": _build_stats()}

    @app.post("/api/undo")
    async def undo():
        img = _state.undo()
        if img is None:
            return {"ok": False, "message": "nothing to undo"}
        return {"ok": True, "image_id": img["image_id"]}

    @app.get("/api/stats")
    async def stats():
        return _build_stats()

    @app.post("/api/shutdown")
    async def shutdown():
        n = _state.persist_to_db()
        return {"persisted": n, "message": f"Saved {n} annotations to DB"}

    # Persist on SIGINT/SIGTERM
    def _on_exit(signum, frame):
        print(f"\nPersisting {_state.done_count} annotations to DB...")
        n = _state.persist_to_db()
        print(f"Saved {n} annotations. JSONL at: {jsonl_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    return app


def _build_stats() -> dict:
    counts = _state.verdict_counts()
    return {
        "total": _state.total,
        "done": _state.done_count,
        "remaining": _state.total - _state.done_count,
        "good": counts["good"],
        "bad": counts["bad"],
        "skip": counts["skip"],
    }


def _build_html(feature_name: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Manual QA — {feature_name}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #111; color: #eee; font-family: -apple-system, sans-serif;
           display: flex; flex-direction: column; height: 100vh; }}
    .header {{ padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333;
               display: flex; justify-content: space-between; align-items: center; }}
    .header h1 {{ font-size: 16px; font-weight: 500; }}
    .stats {{ font-size: 13px; color: #888; }}
    .stats .good {{ color: #4caf50; }}
    .stats .bad {{ color: #f44336; }}
    .stats .skip {{ color: #ff9800; }}
    .progress-bar {{ height: 3px; background: #333; }}
    .progress-fill {{ height: 100%; background: #4caf50; transition: width 0.2s; }}
    .main {{ flex: 1; display: flex; flex-direction: column; align-items: center;
             justify-content: center; padding: 20px; }}
    .meta {{ font-size: 13px; color: #888; margin-bottom: 12px; font-family: monospace; }}
    .image-container {{ flex: 1; display: flex; align-items: center; justify-content: center;
                        max-height: calc(100vh - 160px); }}
    .image-container img {{ max-height: 100%; max-width: 100%; object-fit: contain; }}
    .controls {{ padding: 12px 20px; background: #1a1a1a; border-top: 1px solid #333;
                 text-align: center; font-size: 13px; color: #666; }}
    kbd {{ background: #333; padding: 2px 8px; border-radius: 3px; font-size: 12px;
          border: 1px solid #555; }}
    .done {{ font-size: 24px; color: #4caf50; }}
</style>
</head>
<body>
<div class="header">
    <h1>Manual QA: {feature_name}</h1>
    <div class="stats" id="stats"></div>
</div>
<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
<div class="main">
    <div class="meta" id="meta"></div>
    <div class="image-container" id="img-container">
        <img id="img" src="" />
    </div>
</div>
<div class="controls">
    <kbd>Y</kbd> good &nbsp;&nbsp;
    <kbd>N</kbd> bad &nbsp;&nbsp;
    <kbd>S</kbd> skip &nbsp;&nbsp;
    <kbd>Backspace</kbd> undo
</div>

<script>
let busy = false;

async function loadCurrent() {{
    const res = await fetch('/api/current');
    const data = await res.json();
    if (data.done) {{
        document.getElementById('img-container').innerHTML =
            '<div class="done">All done! Close this tab.</div>';
        document.getElementById('meta').textContent = '';
    }} else {{
        document.getElementById('img').src = data.image_url;
        document.getElementById('meta').textContent =
            data.image_id + ' — ' + data.species + ' (' + data.feature_value + ')';
    }}
    updateStats(data.stats);
}}

function updateStats(s) {{
    const pct = s.total > 0 ? (s.done / s.total * 100).toFixed(1) : 0;
    document.getElementById('stats').innerHTML =
        s.done + ' / ' + s.total +
        ' (<span class="good">' + s.good + ' good</span>, ' +
        '<span class="bad">' + s.bad + ' bad</span>, ' +
        '<span class="skip">' + s.skip + ' skip</span>) — ' + pct + '%';
    document.getElementById('progress').style.width = pct + '%';
}}

async function annotate(verdict) {{
    if (busy) return;
    busy = true;
    await fetch('/api/annotate', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{verdict}})
    }});
    await loadCurrent();
    busy = false;
}}

async function undo() {{
    if (busy) return;
    busy = true;
    await fetch('/api/undo', {{method: 'POST'}});
    await loadCurrent();
    busy = false;
}}

document.addEventListener('keydown', (e) => {{
    if (e.key === 'y' || e.key === 'Y') annotate('good');
    else if (e.key === 'n' || e.key === 'N') annotate('bad');
    else if (e.key === 's' || e.key === 'S') annotate('skip');
    else if (e.key === 'Backspace') {{ e.preventDefault(); undo(); }}
}});

loadCurrent();
</script>
</body>
</html>"""


def run(feature_name: str, port: int = 8002, preprocess_version: str = "v1"):
    """Create app and run uvicorn."""
    app = create_app(feature_name, preprocess_version)
    print(f"\nStarting labeling server at http://localhost:{port}")
    print("Press Ctrl+C to save and quit.\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
