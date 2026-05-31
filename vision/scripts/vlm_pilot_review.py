"""CLI: generate a self-contained HTML inspection gallery for a VLM pilot run.

Reads `manifest.parquet` and `annotations/*.json` from a pilot directory and
emits a single HTML file with one card per image: thumbnail (base64-embedded),
species-propagated labels, and the VLM's observations side-by-side. Disagreements
on Tier 1 features are visually highlighted.

The HTML includes a small client-side filter/sort bar (vanilla JS, no external
deps) so the user can scroll through, filter to disagreements only, and judge
the VLM's quality before deciding whether to scale.

Usage:
    python -m vision.scripts.vlm_pilot_review --pilot-dir data/vlm_pilot/run_001 \\
        --out data/vlm_pilot/run_001/review.html
"""

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

THUMB_SIZE = 256

# Mapping: VLM field -> propagated column name (for agreement checks).
COMPARISONS = [
    ("body_form", "propagated_overall_body_form"),
    ("hymenium_type", "propagated_hymenium_type"),
    ("cap_shape", "propagated_cap_shape"),
    ("cap_color_primary", "propagated_cap_color"),
]


def _thumbnail_data_url(image_path: Path) -> str:
    """Resize image to a thumbnail and return as base64 PNG data URL."""
    img = Image.open(image_path)
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _normalize(s: object) -> str | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    return s or None


def _agree(vlm_val: object, propagated_val: object) -> str:
    """Return one of: 'agree', 'disagree', 'na' (one side missing)."""
    v = _normalize(vlm_val)
    p = _normalize(propagated_val)
    if v is None or p is None:
        return "na"
    # color comparison: substring on either side counts as agreement
    if v == p or v in p or p in v:
        return "agree"
    return "disagree"


def _load_annotation(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None


def _render_field(label: str, value: object, *, badge: str | None = None) -> str:
    val_str = "—" if value is None or value == "" else html.escape(str(value))
    badge_html = ""
    if badge == "agree":
        badge_html = '<span class="badge agree">AGREE</span>'
    elif badge == "disagree":
        badge_html = '<span class="badge disagree">DISAGREE</span>'
    elif badge == "new":
        badge_html = '<span class="badge new">NEW</span>'
    return (
        f'<div class="row"><span class="k">{html.escape(label)}</span>'
        f'<span class="v">{val_str}</span>{badge_html}</div>'
    )


def _render_card(row: dict, ann: dict | None, thumb_url: str) -> str:
    image_id = html.escape(str(row["image_id"]))
    species = html.escape(str(row.get("species") or "—"))

    # Disagreement count for filtering
    disagreement_count = 0
    agreement_html_parts = []
    if ann and "error" not in ann:
        for vlm_field, prop_field in COMPARISONS:
            agree = _agree(ann.get(vlm_field), row.get(prop_field))
            if agree == "disagree":
                disagreement_count += 1
            agreement_html_parts.append(
                _render_field(
                    vlm_field, ann.get(vlm_field), badge=agree if agree != "na" else None
                )
            )

    # Propagated labels block
    prop_html = "".join(
        [
            _render_field("hymenium_type", row.get("propagated_hymenium_type")),
            _render_field("overall_body_form", row.get("propagated_overall_body_form")),
            _render_field("cap_shape", row.get("propagated_cap_shape")),
            _render_field("cap_color", row.get("propagated_cap_color")),
        ]
    )

    # VLM block
    if ann is None:
        vlm_html = '<div class="row error">no sidecar JSON found</div>'
        usable = "false"
        view_angle = ""
    elif "error" in ann:
        vlm_html = f'<div class="row error">ERROR: {html.escape(ann["error"])}</div>'
        usable = "false"
        view_angle = ""
    else:
        view_angle = ann.get("view_angle", "")
        usable = "false" if ann.get("image_quality") == "unusable" else "true"
        secondary = ann.get("cap_color_secondary")
        secondary_badge = "new" if secondary else None
        vlm_html = "".join(
            [
                _render_field("view_angle", ann.get("view_angle")),
                _render_field("subject_dominance", ann.get("subject_dominance")),
                _render_field("image_quality", ann.get("image_quality")),
                _render_field("cap_visible", ann.get("cap_visible")),
                _render_field("hymenium_visible", ann.get("hymenium_visible")),
                *agreement_html_parts,
                _render_field("cap_color_secondary", secondary, badge=secondary_badge),
                _render_field("cap_surface_texture", ann.get("cap_surface_texture")),
                _render_field("gill_attachment", ann.get("gill_attachment")),
                _render_field("confidence", ann.get("confidence")),
                _render_field("notes", ann.get("notes")),
            ]
        )

    return f"""
<div class="card" data-species="{species}" data-disagreements="{disagreement_count}"
     data-usable="{usable}" data-view="{html.escape(view_angle)}">
  <img class="thumb" src="{thumb_url}" alt="{image_id}">
  <div class="meta">
    <div class="hdr"><b>{species}</b> <span class="iid">{image_id}</span></div>
    <div class="cols">
      <div class="col">
        <div class="col-title">Propagated (species → image)</div>
        {prop_html}
      </div>
      <div class="col">
        <div class="col-title">VLM observation</div>
        {vlm_html}
      </div>
    </div>
  </div>
</div>
""".strip()


def _summary_stats(rows: list[dict], anns: list[dict | None]) -> dict:
    """Compute headline numbers for the summary header."""
    total = len(rows)
    n_ok = sum(1 for a in anns if a and "error" not in a)
    n_err = sum(1 for a in anns if a and "error" in a)
    n_unusable = sum(1 for a in anns if a and a.get("image_quality") == "unusable")

    view_counts: dict[str, int] = {}
    feature_agree: dict[str, dict[str, int]] = {
        f[0]: {"agree": 0, "disagree": 0, "na": 0} for f in COMPARISONS
    }

    for row, ann in zip(rows, anns):
        if not ann or "error" in ann:
            continue
        view = ann.get("view_angle") or "unknown"
        view_counts[view] = view_counts.get(view, 0) + 1
        for vlm_field, prop_field in COMPARISONS:
            feature_agree[vlm_field][_agree(ann.get(vlm_field), row.get(prop_field))] += 1

    return {
        "total": total,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_unusable": n_unusable,
        "view_counts": view_counts,
        "feature_agree": feature_agree,
    }


def _render_summary(stats: dict) -> str:
    rows = []
    for feat in ("hymenium_type", "body_form", "cap_shape", "cap_color_primary"):
        d = stats["feature_agree"][feat]
        denom = d["agree"] + d["disagree"]
        pct = (100.0 * d["agree"] / denom) if denom else 0.0
        rows.append(
            f"<tr><td>{feat}</td><td>{d['agree']}</td><td>{d['disagree']}</td>"
            f"<td>{d['na']}</td><td>{pct:.1f}%</td></tr>"
        )
    table = (
        "<table class='summary-table'><thead><tr>"
        "<th>feature</th><th>agree</th><th>disagree</th><th>n/a</th><th>%agree</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    view_items = "".join(
        f"<li><b>{html.escape(k)}</b>: {v}</li>"
        for k, v in sorted(stats["view_counts"].items(), key=lambda kv: -kv[1])
    )

    return f"""
<div class="summary">
  <h2>Pilot summary</h2>
  <p>{stats["total"]} images sampled · {stats["n_ok"]} ok · {stats["n_err"]} errors ·
     {stats["n_unusable"]} flagged unusable by VLM</p>
  {table}
  <h3>View-angle distribution</h3>
  <ul class="views">{view_items}</ul>
</div>
""".strip()


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       background: #f5f5f7; color: #1d1d1f; margin: 0; padding: 24px; }
h1 { margin-top: 0; }
.summary { background: #fff; border-radius: 12px; padding: 18px 24px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-bottom: 24px; }
.summary-table { border-collapse: collapse; margin: 12px 0; }
.summary-table th, .summary-table td { padding: 6px 14px; text-align: left;
           border-bottom: 1px solid #eaeaea; font-size: 13px; }
.summary-table th { background: #fafafa; }
.views { list-style: none; padding-left: 0; columns: 3; font-size: 13px; }
.controls { background: #fff; border-radius: 12px; padding: 12px 18px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-bottom: 24px;
           display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.controls label { font-size: 13px; }
.controls input, .controls select { padding: 4px 8px; font-size: 13px; }
.card { background: #fff; border-radius: 12px; padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-bottom: 16px;
        display: flex; gap: 18px; }
.card.has-disagreement { background: #fff8f0; }
.card.unusable { opacity: 0.5; }
.thumb { width: 256px; height: 256px; object-fit: cover; border-radius: 8px;
         flex-shrink: 0; background: #eee; }
.meta { flex: 1; min-width: 0; }
.hdr { font-size: 15px; margin-bottom: 8px; }
.iid { font-family: monospace; font-size: 11px; color: #888; margin-left: 12px; }
.cols { display: flex; gap: 24px; }
.col { flex: 1; min-width: 0; }
.col-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
             color: #888; margin-bottom: 6px; border-bottom: 1px solid #eee;
             padding-bottom: 4px; }
.row { font-size: 13px; padding: 3px 0; display: flex; gap: 8px; align-items: baseline; }
.row .k { color: #666; min-width: 140px; }
.row .v { flex: 1; word-break: break-word; }
.row.error { color: #c00; }
.badge { font-size: 10px; padding: 2px 6px; border-radius: 4px;
         font-weight: 600; letter-spacing: 0.4px; }
.badge.agree { background: #e6f5e6; color: #2a7a2a; }
.badge.disagree { background: #fce4e4; color: #b22; }
.badge.new { background: #e6effc; color: #1556a0; }
.hidden { display: none !important; }
"""

_JS = """
const cards = Array.from(document.querySelectorAll('.card'));

function applyFilters() {
  const onlyDisagree = document.getElementById('only-disagree').checked;
  const onlyUnusable = document.getElementById('only-unusable').checked;
  const speciesFilter = document.getElementById('species-filter').value.trim().toLowerCase();
  const viewFilter = document.getElementById('view-filter').value;

  cards.forEach(card => {
    const dis = parseInt(card.dataset.disagreements || '0', 10);
    const usable = card.dataset.usable === 'true';
    const sp = (card.dataset.species || '').toLowerCase();
    const view = card.dataset.view || '';

    let show = true;
    if (onlyDisagree && dis === 0) show = false;
    if (onlyUnusable && usable) show = false;
    if (speciesFilter && !sp.includes(speciesFilter)) show = false;
    if (viewFilter && view !== viewFilter) show = false;

    card.classList.toggle('hidden', !show);
    card.classList.toggle('has-disagreement', dis > 0);
    card.classList.toggle('unusable', !usable);
  });
}

function applySort() {
  const mode = document.getElementById('sort-mode').value;
  const container = document.getElementById('cards');
  const sorted = cards.slice();
  if (mode === 'disagreements') {
    sorted.sort((a, b) =>
      parseInt(b.dataset.disagreements || '0', 10) - parseInt(a.dataset.disagreements || '0', 10));
  } else if (mode === 'species') {
    sorted.sort((a, b) => (a.dataset.species || '').localeCompare(b.dataset.species || ''));
  }
  sorted.forEach(c => container.appendChild(c));
}

document.getElementById('only-disagree').addEventListener('change', applyFilters);
document.getElementById('only-unusable').addEventListener('change', applyFilters);
document.getElementById('species-filter').addEventListener('input', applyFilters);
document.getElementById('view-filter').addEventListener('change', applyFilters);
document.getElementById('sort-mode').addEventListener('change', applySort);

applyFilters();
"""


def main():
    parser = argparse.ArgumentParser(description="Generate VLM pilot review HTML")
    parser.add_argument(
        "--pilot-dir",
        type=str,
        required=True,
        help="Pilot directory containing manifest.parquet + annotations/",
    )
    parser.add_argument("--out", type=str, required=True, help="Output HTML path")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    manifest_path = pilot_dir / "manifest.parquet"
    annotations_dir = pilot_dir / "annotations"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    rows = df.to_dict("records")

    print(f"Loading {len(rows)} sidecar JSONs and rendering thumbnails...")
    anns: list[dict | None] = []
    cards_html: list[str] = []
    view_angles: set[str] = set()

    for i, row in enumerate(rows, 1):
        ann = _load_annotation(annotations_dir / f"{row['image_id']}.json")
        anns.append(ann)
        try:
            thumb_url = _thumbnail_data_url(Path(row["processed_path"]))
        except Exception as exc:  # noqa: BLE001
            thumb_url = ""
            print(f"  thumbnail failed for {row['image_id']}: {exc}")
        cards_html.append(_render_card(row, ann, thumb_url))
        if ann and "error" not in ann and ann.get("view_angle"):
            view_angles.add(ann["view_angle"])
        if i % 25 == 0:
            print(f"  rendered {i}/{len(rows)}")

    stats = _summary_stats(rows, anns)
    summary_html = _render_summary(stats)

    view_options = "".join(
        f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in sorted(view_angles)
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM pilot review · {html.escape(pilot_dir.name)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>VLM pilot review · {html.escape(pilot_dir.name)}</h1>
{summary_html}
<div class="controls">
  <label><input type="checkbox" id="only-disagree"> only disagreements</label>
  <label><input type="checkbox" id="only-unusable"> only unusable</label>
  <label>species: <input type="text" id="species-filter" placeholder="filter…"></label>
  <label>view:
    <select id="view-filter"><option value="">(any)</option>{view_options}</select>
  </label>
  <label>sort:
    <select id="sort-mode">
      <option value="default">default</option>
      <option value="species">by species</option>
      <option value="disagreements">by # disagreements</option>
    </select>
  </label>
</div>
<div id="cards">
{"".join(cards_html)}
</div>
<script>{_JS}</script>
</body>
</html>
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
