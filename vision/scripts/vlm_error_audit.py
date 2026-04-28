"""Generate an HTML error audit page for VLM feature predictions.

Shows each prediction error as a card with:
- Embedded image thumbnail
- Model prediction, confidence, visual description, reasoning
- GT label (strict + multi-label where applicable)
- Verdict radio buttons (for manual flagging in browser)

Usage:
    python -m vision.scripts.vlm_error_audit --feature cap_color
    python -m vision.scripts.vlm_error_audit --feature hymenium_type
    python -m vision.scripts.vlm_error_audit --feature cap_color --out /tmp/audit.html
    python -m vision.scripts.vlm_error_audit --feature cap_color --include-correct
"""

import argparse
import base64
import json
import sys
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.data.ground_truth import (  # noqa: E402
    GROUND_TRUTH_SPEC,
    load_ground_truth,
    load_ground_truth_multilabel,
)
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY  # noqa: E402

DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "vlm_eval"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed" / "v1"

# Features that support multi-label GT (resolved via canonical palette + corrections).
_MULTILABEL_FEATURES = {"cap_color"}


def _thumb_b64(image_id: str, processed_dir: Path, size: int = 256) -> str:
    """Return base64 data URL for an image thumbnail."""
    from io import BytesIO

    from PIL import Image

    path = processed_dir / f"{image_id}.jpg"
    if not path.exists():
        return ""
    img = Image.open(path)
    img.thumbnail((size, size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_audit_data(
    feature_name: str,
    eval_dir: Path,
    include_correct: bool = False,
) -> list[dict]:
    """Load predictions, GT, and build per-image audit records."""
    manifest = pd.read_csv(eval_dir / "eval_manifest.csv")
    field_name = FEATURE_REGISTRY[feature_name]["field"]

    with get_session() as session:
        gt_single = load_ground_truth(session, feature_name)
        has_multilabel = feature_name in _MULTILABEL_FEATURES
        gt_ml = load_ground_truth_multilabel(session, feature_name) if has_multilabel else {}

    records = []
    feat_dir = eval_dir / feature_name

    for _, row in manifest.iterrows():
        iid = row["image_id"]
        species = row["species"]
        sidecar_path = feat_dir / f"{iid}.json"

        if not sidecar_path.exists():
            continue

        try:
            data = json.loads(sidecar_path.read_text())
        except json.JSONDecodeError:
            continue

        if data.get("error"):
            continue

        pred = data.get(field_name)
        conf = data.get("confidence", "?")

        # cannot_tell → skip for error audit (model correctly abstained)
        if pred is None and conf == "cannot_tell":
            if not include_correct:
                continue

        strict_gt = gt_single.get(species)
        strict_match = pred == strict_gt

        if has_multilabel:
            ml_gt = gt_ml.get(species, set())
            ml_match = pred in ml_gt if pred else False
            is_correct = ml_match
        else:
            ml_gt = set()
            ml_match = strict_match
            is_correct = strict_match

        if not include_correct and is_correct:
            continue

        records.append({
            "image_id": iid,
            "species": species,
            "pred": pred if pred is not None else "(null)",
            "confidence": str(conf),
            "visual_desc": data.get("visual_description") or "",
            "reasoning": data.get("reasoning") or "",
            "strict_gt": strict_gt or "?",
            "ml_gt": sorted(ml_gt) if ml_gt else [],
            "strict_match": strict_match,
            "ml_match": ml_match,
            "has_multilabel": has_multilabel,
        })

    # Sort: errors first, then by species
    records.sort(key=lambda r: (r["ml_match"], r["species"], r["image_id"]))
    return records


def render_html(
    records: list[dict],
    feature_name: str,
    out_path: Path,
    processed_dir: Path,
):
    """Render audit records to a self-contained HTML file."""
    n_errors = sum(1 for r in records if not r["ml_match"])
    n_total = len(records)
    has_multilabel = any(r["has_multilabel"] for r in records)
    feature_label = feature_name.replace("_", " ").title()

    cards_html = []
    for i, r in enumerate(records):
        thumb = _thumb_b64(r["image_id"], processed_dir)
        status = "correct" if r["ml_match"] else ("rescued" if r["strict_match"] else "error")
        border_color = {"error": "#e74c3c", "rescued": "#f39c12", "correct": "#27ae60"}[status]

        # GT display
        gt_rows = f"""<tr><td class="label">Strict GT:</td>
                  <td><span class="gt">{escape(str(r['strict_gt']))}</span></td></tr>"""
        if has_multilabel:
            ml_gt_str = ", ".join(r["ml_gt"]) if r["ml_gt"] else "none"
            gt_rows += f"""<tr><td class="label">Multi-label GT:</td>
                  <td>{escape(ml_gt_str)}</td></tr>"""

        card = f"""
        <div class="card" style="border-left: 5px solid {border_color};" id="card-{i}">
          <div class="card-img">
            <img src="{thumb}" alt="{escape(r['image_id'])}" />
          </div>
          <div class="card-info">
            <div class="species">{escape(r['species'])}</div>
            <div class="id">{escape(r['image_id'])}</div>
            <table class="meta">
              <tr><td class="label">Prediction:</td>
                  <td><span class="pred">{escape(r['pred'])}</span>
                      <span class="conf">({escape(r['confidence'])})</span></td></tr>
              {gt_rows}
            </table>
            <details>
              <summary>VLM reasoning</summary>
              <p class="desc"><b>Visual:</b> {escape(r['visual_desc'])}</p>
              <p class="desc"><b>Reasoning:</b> {escape(r['reasoning'])}</p>
            </details>
            <div class="verdict" data-idx="{i}">
              <label><input type="radio" name="v{i}" value="model_wrong" /> Model wrong</label>
              <label><input type="radio" name="v{i}" value="gt_wrong" /> GT wrong</label>
              <label><input type="radio" name="v{i}" value="ambiguous" /> Ambiguous</label>
              <label><input type="radio" name="v{i}" value="gate_failure" /> Gate failure</label>
            </div>
          </div>
        </div>"""
        cards_html.append(card)

    accuracy_type = "multi-label" if has_multilabel else "strict"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{feature_label} VLM Error Audit ({n_errors} errors / {n_total} shown)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; padding: 20px; }}
  h1 {{ margin-bottom: 10px; }}
  .stats {{ color: #666; margin-bottom: 20px; }}
  .controls {{ margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }}
  .controls button {{ padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
                      background: #fff; cursor: pointer; }}
  .controls button:hover {{ background: #eee; }}
  .filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 15px; }}
  .filter-bar button {{ padding: 4px 10px; border: 1px solid #bbb; border-radius: 3px;
                        background: #fff; cursor: pointer; font-size: 13px; }}
  .filter-bar button.active {{ background: #333; color: #fff; }}
  .card {{ display: flex; background: #fff; border-radius: 6px; margin-bottom: 12px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
  .card.hidden {{ display: none; }}
  .card-img {{ flex: 0 0 256px; background: #eee; }}
  .card-img img {{ width: 256px; height: 256px; object-fit: cover; }}
  .card-info {{ flex: 1; padding: 12px 16px; }}
  .species {{ font-weight: 600; font-size: 15px; font-style: italic; }}
  .id {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
  .meta {{ font-size: 14px; margin-bottom: 8px; }}
  .meta td {{ padding: 1px 8px 1px 0; }}
  .meta .label {{ color: #666; font-weight: 500; }}
  .pred {{ font-weight: 700; color: #2c3e50; }}
  .conf {{ color: #888; font-size: 13px; }}
  .gt {{ font-weight: 700; color: #7f8c8d; }}
  details {{ margin-bottom: 8px; }}
  summary {{ cursor: pointer; color: #3498db; font-size: 13px; }}
  .desc {{ font-size: 13px; color: #555; margin: 4px 0; line-height: 1.4; }}
  .verdict {{ display: flex; gap: 16px; font-size: 13px; padding-top: 4px; flex-wrap: wrap; }}
  .verdict label {{ cursor: pointer; }}
  #export-area {{ margin-top: 20px; }}
  textarea {{ width: 100%; height: 150px; font-family: monospace; font-size: 12px; }}
</style>
</head>
<body>
<h1>{feature_label} Error Audit</h1>
<div class="stats">{n_errors} errors ({accuracy_type}) / {n_total} total shown</div>

<div class="controls">
  <button onclick="expandAll()">Expand all reasoning</button>
  <button onclick="collapseAll()">Collapse all</button>
  <button onclick="exportVerdicts()">Export verdicts (JSON)</button>
</div>

<div class="filter-bar" id="filters">
  <span style="font-size:13px;color:#666;">Filter by prediction:</span>
</div>

{''.join(cards_html)}

<div id="export-area"></div>

<script>
// Collect unique predictions for filter buttons
const cards = document.querySelectorAll('.card');
const predCounts = {{}};
cards.forEach(c => {{
  const pred = c.querySelector('.pred').textContent;
  predCounts[pred] = (predCounts[pred] || 0) + 1;
}});
const filterBar = document.getElementById('filters');
const allBtn = document.createElement('button');
allBtn.textContent = 'All (' + cards.length + ')';
allBtn.className = 'active';
allBtn.onclick = () => filterBy(null);
filterBar.appendChild(allBtn);
Object.entries(predCounts).sort((a,b) => b[1]-a[1]).forEach(([pred, cnt]) => {{
  const btn = document.createElement('button');
  btn.textContent = pred + ' (' + cnt + ')';
  btn.dataset.pred = pred;
  btn.onclick = () => filterBy(pred);
  filterBar.appendChild(btn);
}});

function filterBy(pred) {{
  filterBar.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  if (pred === null) {{
    filterBar.querySelector('button').classList.add('active');
    cards.forEach(c => c.classList.remove('hidden'));
  }} else {{
    filterBar.querySelectorAll('button').forEach(b => {{
      if (b.dataset.pred === pred) b.classList.add('active');
    }});
    cards.forEach(c => {{
      const p = c.querySelector('.pred').textContent;
      c.classList.toggle('hidden', p !== pred);
    }});
  }}
}}

function expandAll() {{
  document.querySelectorAll('details').forEach(d => d.open = true);
}}
function collapseAll() {{
  document.querySelectorAll('details').forEach(d => d.open = false);
}}

function exportVerdicts() {{
  const results = [];
  document.querySelectorAll('.verdict').forEach(v => {{
    const idx = v.dataset.idx;
    const card = document.getElementById('card-' + idx);
    const checked = v.querySelector('input:checked');
    if (checked) {{
      results.push({{
        image_id: card.querySelector('.id').textContent,
        species: card.querySelector('.species').textContent,
        pred: card.querySelector('.pred').textContent,
        verdict: checked.value,
      }});
    }}
  }});
  let area = document.getElementById('export-area');
  area.innerHTML = '<h3>Verdicts (' + results.length + '/' + cards.length + ' flagged)</h3>' +
    '<textarea>' + JSON.stringify(results, null, 2) + '</textarea>' +
    '<p style="font-size:12px;color:#888;">Copy the JSON above and save to a file.</p>';
}}
</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"Audit page: {out_path} ({n_errors} errors, {n_total} total cards)")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML error audit for VLM features")
    parser.add_argument(
        "--feature", type=str, required=True,
        help="Feature to audit (e.g. cap_color, hymenium_type)",
    )
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help=(
            "Directory containing eval_manifest.csv and {feature}/ sidecars. "
            f"Default: {DEFAULT_EVAL_DIR}"
        ),
    )
    parser.add_argument(
        "--processed-dir", type=str, default=None,
        help=(
            "Directory containing the actual images sent to the VLM "
            "(used as thumbnail source). "
            f"Default: {DEFAULT_PROCESSED_DIR}"
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output HTML path (default: {eval_dir}/{feature}_audit.html)",
    )
    parser.add_argument(
        "--include-correct", action="store_true",
        help="Include correct predictions too (not just errors)",
    )
    args = parser.parse_args()

    feature = args.feature
    if feature not in FEATURE_REGISTRY:
        print(f"ERROR: unknown feature '{feature}'. Choose from {list(FEATURE_REGISTRY.keys())}")
        sys.exit(1)
    if feature not in GROUND_TRUTH_SPEC:
        print(f"ERROR: no ground truth spec for '{feature}'.")
        sys.exit(1)

    eval_dir = Path(args.eval_dir) if args.eval_dir else DEFAULT_EVAL_DIR
    processed_dir = Path(args.processed_dir) if args.processed_dir else DEFAULT_PROCESSED_DIR
    out_path = Path(args.out) if args.out else eval_dir / f"{feature}_audit.html"

    print(f"Eval dir:      {eval_dir}")
    print(f"Processed dir: {processed_dir}")
    print(f"Output HTML:   {out_path}")

    records = build_audit_data(feature, eval_dir, include_correct=args.include_correct)
    print(f"Built {len(records)} audit records")
    render_html(records, feature, out_path, processed_dir)


if __name__ == "__main__":
    main()
