"""VLM-based per-image labeler.

Calls a vision LLM (via `llm.client.structured_vision_completion`) to produce
a `VLMImageAnnotation` per image. Writes one JSON sidecar per image so the
pilot is fully throwaway — no DB writes, easy to re-run with a different model.
"""

import json
from pathlib import Path

from tqdm import tqdm

from llm.client import structured_vision_completion
from vision.labeling.vlm_schema import VLMImageAnnotation

_SYSTEM_PROMPT = (
    "You are a mycology assistant analyzing a single mushroom photograph. "
    "Examine the image and report observable visual features. "
    "Be conservative — if a feature is not clearly visible, leave it null. "
    "Do not infer features from species knowledge; only describe what you can SEE. "
    "Return a structured JSON matching the schema."
)

_USER_PROMPT = (
    "Analyze this mushroom photograph and produce a structured annotation.\n\n"
    "Important rules:\n"
    "- Report only what is visibly determinable in THIS image.\n"
    "- If the cap is not visible, set cap_visible=false and leave cap_* fields null.\n"
    "- If the underside is not visible, set hymenium_visible=false and leave hymenium_* null.\n"
    "- Use the canonical vocabulary for shape/texture/hymenium/body_form fields; "
    "leave them null rather than guessing.\n"
    "- For cap_color_primary, use plain English (e.g. 'red', 'reddish-brown', 'cream').\n"
    "- Set image_quality='unusable' for habitat shots, microscopy, or photos where no "
    "mushroom morphology can be assessed.\n"
)


def label_image(
    image_path: str | Path,
    model_name: str | None = None,
) -> VLMImageAnnotation:
    """Run the VLM on a single image and return a validated annotation.

    Args:
        image_path: Path to the image file (use processed/, not raw/).
        model_name: Optional per-call model override (e.g. 'gemma4:31b-cloud').
            If None, uses LLM_MODEL from .env.
    """
    return structured_vision_completion(
        prompt=_USER_PROMPT,
        image_paths=[image_path],
        response_model=VLMImageAnnotation,
        system=_SYSTEM_PROMPT,
        model=model_name,
    )


def label_images_batch(
    image_paths: list[str | Path],
    out_dir: str | Path,
    model_name: str | None = None,
    image_ids: list[str] | None = None,
) -> list[dict]:
    """Label a batch of images, writing one JSON sidecar per image.

    Per-image exceptions are caught and recorded as `{"error": "..."}` so a
    single bad image does not abort the run.

    Args:
        image_paths: List of image file paths.
        out_dir: Directory where sidecar JSONs are written.
        model_name: Optional model override.
        image_ids: Optional explicit ids (one per image_path). If omitted,
            sidecar files are named by image stem.

    Returns:
        List of dicts: `{"image_id": ..., "image_path": ..., "annotation": ...|None,
        "error": str|None}` in input order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_ids is not None and len(image_ids) != len(image_paths):
        raise ValueError("image_ids length must match image_paths length")

    results: list[dict] = []
    for i, path in enumerate(tqdm(image_paths, desc="VLM labeling", unit="img")):
        path = Path(path)
        img_id = image_ids[i] if image_ids is not None else path.stem
        sidecar = out_dir / f"{img_id}.json"

        try:
            annotation = label_image(path, model_name=model_name)
            payload = annotation.model_dump()
            sidecar.write_text(json.dumps(payload, indent=2))
            results.append(
                {
                    "image_id": img_id,
                    "image_path": str(path),
                    "annotation": payload,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 — pilot wants to keep going
            err = f"{type(exc).__name__}: {exc}"
            sidecar.write_text(json.dumps({"error": err}, indent=2))
            results.append(
                {
                    "image_id": img_id,
                    "image_path": str(path),
                    "annotation": None,
                    "error": err,
                }
            )

    return results
