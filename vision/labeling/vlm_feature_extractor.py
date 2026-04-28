"""Generic per-feature VLM extraction engine.

Feature-agnostic: looks up schema + prompts from the registries in
vlm_feature_schemas and vlm_feature_prompts. Adding a new feature requires
only a new schema class + prompt pair — no changes here.

Output: one JSON sidecar per image at ``{out_dir}/{feature_name}/{image_id}.json``.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from llm.client import _encode_image_to_data_url, structured_vision_completion
from vision.labeling.vlm_feature_prompts import PROMPT_REGISTRY
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY

logger = logging.getLogger(__name__)

REFERENCE_DIR = Path(__file__).parent / "reference_images"

# Per-feature preamble text for few-shot reference images.
_FEWSHOT_PREAMBLE: dict[str, str] = {
    "cap_color": (
        "Reference examples showing what each color looks like on mushroom caps. "
        "Use these to calibrate your classification, especially for borderline cases.\n"
    ),
    "hymenium_type": (
        "Reference examples showing what each hymenium type looks like. "
        "Pay close attention to the structural differences — especially between "
        "gills (sharp blades, stop at stem) and ridges (blunt folds, run down stem). "
        "Use these to calibrate your classification.\n"
    ),
}

_DEFAULT_PREAMBLE = (
    "Reference examples for each class. "
    "Use these to calibrate your classification.\n"
)


def _build_fewshot_content(
    feature_name: str,
    query_image_path: str | Path,
    user_prompt: str,
) -> list[dict] | None:
    """Build interleaved text+image content blocks for few-shot extraction.

    Returns None if no reference images exist for this feature, signaling
    the caller to fall back to the standard single-image path.
    """
    ref_dir = REFERENCE_DIR / feature_name
    if not ref_dir.exists():
        return None

    ref_images = sorted(ref_dir.glob("*.jpg"))
    if not ref_images:
        return None

    preamble = _FEWSHOT_PREAMBLE.get(feature_name, _DEFAULT_PREAMBLE)
    content: list[dict] = [{"type": "text", "text": preamble}]
    for ref_img in ref_images:
        label = ref_img.stem.replace("_ref", "").upper()
        content.append({"type": "text", "text": f"{label}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_to_data_url(ref_img)},
        })

    content.append({"type": "text", "text": f"\n{user_prompt}\n\nNow classify THIS mushroom:"})
    content.append({
        "type": "image_url",
        "image_url": {"url": _encode_image_to_data_url(query_image_path)},
    })
    return content


def extract_feature(
    image_path: str | Path,
    feature_name: str,
    model_name: str | None = None,
    use_few_shot: bool = True,
):
    """Run per-feature VLM extraction on a single image.

    Args:
        image_path: Path to the image file.
        feature_name: Feature to extract (e.g. "hymenium_type", "cap_color").
        model_name: Optional per-call model override.
        use_few_shot: When False, skip few-shot reference images even if available
            on disk. Useful for ablations isolating reference-image effects.

    Returns:
        Validated Pydantic model instance (HymeniumTypeResult, CapColorResult, etc.)

    Raises:
        KeyError: If feature_name is not in the registries.
    """
    schema_info = FEATURE_REGISTRY[feature_name]
    system_prompt, user_prompt = PROMPT_REGISTRY[feature_name]

    fewshot = (
        _build_fewshot_content(feature_name, image_path, user_prompt)
        if use_few_shot
        else None
    )
    if fewshot is not None:
        return structured_vision_completion(
            prompt="",
            image_paths=[],
            response_model=schema_info["schema"],
            system=system_prompt,
            model=model_name,
            content_blocks=fewshot,
        )

    return structured_vision_completion(
        prompt=user_prompt,
        image_paths=[image_path],
        response_model=schema_info["schema"],
        system=system_prompt,
        model=model_name,
    )


def extract_feature_batch(
    image_paths: list[str | Path],
    feature_name: str,
    out_dir: str | Path,
    model_name: str | None = None,
    image_ids: list[str] | None = None,
    skip_existing: bool = True,
    use_few_shot: bool = True,
) -> list[dict]:
    """Extract a feature from a batch of images with JSON sidecar output.

    Per-image exceptions are caught and recorded so a single bad image does not
    abort the run.

    Args:
        image_paths: List of image file paths.
        feature_name: Feature to extract.
        out_dir: Root output directory. Sidecars go to ``{out_dir}/{feature_name}/``.
        model_name: Optional model override.
        image_ids: Optional explicit ids (one per image_path). Falls back to stem.
        skip_existing: Skip images that already have a sidecar on disk.

    Returns:
        List of result dicts in input order:
        ``{"image_id", "image_path", "result": dict|None, "error": str|None, "skipped": bool}``
    """
    if feature_name not in FEATURE_REGISTRY:
        raise KeyError(f"Unknown feature '{feature_name}'. Available: {sorted(FEATURE_REGISTRY)}")

    feature_dir = Path(out_dir) / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    if image_ids is not None and len(image_ids) != len(image_paths):
        raise ValueError("image_ids length must match image_paths length")

    results: list[dict] = []
    skipped = 0

    for i, path in enumerate(tqdm(image_paths, desc=f"VLM {feature_name}", unit="img")):
        path = Path(path)
        img_id = image_ids[i] if image_ids is not None else path.stem
        sidecar = feature_dir / f"{img_id}.json"

        if skip_existing and sidecar.exists():
            skipped += 1
            results.append(
                {
                    "image_id": img_id,
                    "image_path": str(path),
                    "result": None,
                    "error": None,
                    "skipped": True,
                }
            )
            continue

        try:
            result = extract_feature(
                path, feature_name, model_name=model_name, use_few_shot=use_few_shot
            )
            payload = result.model_dump()
            sidecar.write_text(json.dumps(payload, indent=2))
            results.append(
                {
                    "image_id": img_id,
                    "image_path": str(path),
                    "result": payload,
                    "error": None,
                    "skipped": False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("Failed %s on %s: %s", feature_name, img_id, err)
            sidecar.write_text(json.dumps({"error": err}, indent=2))
            results.append(
                {
                    "image_id": img_id,
                    "image_path": str(path),
                    "result": None,
                    "error": err,
                    "skipped": False,
                }
            )

    n_ok = sum(1 for r in results if r["result"] is not None)
    n_err = sum(1 for r in results if r["error"] is not None)
    logger.info(
        "%s batch done: %d ok, %d errors, %d skipped",
        feature_name,
        n_ok,
        n_err,
        skipped,
    )

    return results
