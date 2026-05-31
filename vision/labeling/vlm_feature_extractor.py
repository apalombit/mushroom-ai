"""Generic per-feature VLM extraction engine.

Feature-agnostic: looks up schema + prompts from the registries in
vlm_feature_schemas and vlm_feature_prompts. Adding a new feature requires
only a new schema class + prompt pair — no changes here.

Three extraction modes:
- single-shot (default) — one VLM call, full class enumeration
- few-shot (use_few_shot=True with reference_images/{feature}/) — interleaved
  text+image content blocks for in-context exemplars
- staged (staged=True) — two VLM calls via STAGED_REGISTRY: stage 1 picks a
  coarse family, stage 2 disambiguates within that family

Output: one JSON sidecar per image at ``{out_dir}/{feature_name}/{image_id}.json``.
For staged mode the sidecar additionally includes a ``_stage1`` key with the
stage-1 result for debugging.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from llm.client import _encode_image_to_data_url, structured_vision_completion
from vision.labeling.vlm_feature_prompts import PROMPT_REGISTRY
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY
from vision.labeling.vlm_staged_schemas import STAGED_REGISTRY

logger = logging.getLogger(__name__)

REFERENCE_DIR = Path(__file__).parent / "reference_images"

# Confidence rank for min-confidence merge (lower = more uncertain).
_CONFIDENCE_RANK: dict[str, int] = {"cannot_tell": 0, "low": 1, "high": 2}

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


def extract_feature_staged(
    image_path: str | Path,
    feature_name: str,
    model_name: str | None = None,
):
    """Run two-stage chain-of-inquiry VLM extraction (Path C).

    Stage 1 picks a coarse family (e.g. linear_radial / punctate /
    featureless_or_internal for hymenium). Stage 2 dispatches on the family
    value to a within-family disambiguation prompt and returns the leaf class.

    The final result is the same Pydantic model as single-shot extraction
    (e.g. HymeniumTypeResult), so downstream audit + analysis code works
    unchanged. Confidence is min(stage1.confidence, stage2.confidence).

    Args:
        image_path: Path to the image file.
        feature_name: Feature to extract (must be in STAGED_REGISTRY).
        model_name: Optional per-call model override.

    Returns:
        ``(final_result, stage1_result)`` — the merged final model plus the
        raw stage-1 result for debugging / sidecar metadata.

    Raises:
        KeyError: If *feature_name* is not in STAGED_REGISTRY.
    """
    cfg = STAGED_REGISTRY[feature_name]

    # Stage 1 — coarse family
    s1 = structured_vision_completion(
        prompt=cfg.stage1_user,
        image_paths=[image_path],
        response_model=cfg.stage1_schema,
        system=cfg.stage1_system,
        model=model_name,
    )

    family_value = getattr(s1, cfg.family_field, None)

    # Stage 1 abstain → null final, no stage 2 call
    if family_value is None or s1.confidence == "cannot_tell":
        final = cfg.final_schema(
            visible=bool(s1.visible),
            visual_description=s1.visual_description,
            reasoning=s1.reasoning,
            confidence="cannot_tell",
            **{cfg.leaf_field: None},
        )
        return final, s1

    if family_value not in cfg.families:
        raise ValueError(
            f"Stage 1 returned unknown family {family_value!r} for {feature_name!r}; "
            f"expected one of {sorted(cfg.families)}"
        )

    # Stage 2 — within-family disambiguation
    fam_cfg = cfg.families[family_value]
    s2 = structured_vision_completion(
        prompt=fam_cfg.user,
        image_paths=[image_path],
        response_model=fam_cfg.schema,
        system=fam_cfg.system,
        model=model_name,
    )

    leaf_value = getattr(s2, cfg.leaf_field, None)

    # Min-confidence merge (lower of the two ranks)
    final_conf = min(s1.confidence, s2.confidence, key=_CONFIDENCE_RANK.get)

    # Stage 2 abstain or null leaf → final is null
    if leaf_value is None or s2.confidence == "cannot_tell":
        final = cfg.final_schema(
            visible=True,
            visual_description=s2.visual_description or s1.visual_description,
            reasoning=s2.reasoning or s1.reasoning,
            confidence="cannot_tell",
            **{cfg.leaf_field: None},
        )
        return final, s1

    final = cfg.final_schema(
        visible=True,
        visual_description=s2.visual_description or s1.visual_description,
        reasoning=s2.reasoning or s1.reasoning,
        confidence=final_conf,
        **{cfg.leaf_field: leaf_value},
    )
    return final, s1


def extract_feature_batch(
    image_paths: list[str | Path],
    feature_name: str,
    out_dir: str | Path,
    model_name: str | None = None,
    image_ids: list[str] | None = None,
    skip_existing: bool = True,
    use_few_shot: bool = True,
    staged: bool = False,
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
    if staged:
        if feature_name not in STAGED_REGISTRY:
            raise KeyError(
                f"No staged config for '{feature_name}'. "
                f"Available: {sorted(STAGED_REGISTRY)}"
            )
    elif feature_name not in FEATURE_REGISTRY:
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
            try:
                existing = json.loads(sidecar.read_text())
            except Exception:
                existing = {"error": "unreadable sidecar"}
            if "error" not in existing:
                skipped += 1
                results.append(
                    {
                        "image_id": img_id,
                        "image_path": str(path),
                        "result": existing,
                        "error": None,
                        "skipped": True,
                    }
                )
                continue

        try:
            if staged:
                result, stage1 = extract_feature_staged(
                    path, feature_name, model_name=model_name
                )
                payload = result.model_dump()
                payload["_stage1"] = stage1.model_dump()
            else:
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
            err_lc = err.lower()
            if (
                "session usage limit" in err_lc
                or "upgrade for higher limits" in err_lc
                or "rate limit" in err_lc
                or "too many requests" in err_lc
            ):
                logger.error(
                    "Provider quota/rate limit hit on %s after %d/%d images. "
                    "Stopping cleanly; no error sidecar written. Resume later.",
                    img_id, i, len(image_paths),
                )
                print(
                    f"\n\nQuota/rate limit reached on image {i}/{len(image_paths)}. "
                    f"Stopping. Re-run the same command to resume (valid sidecars "
                    f"will be skipped, this image will be retried).",
                    flush=True,
                )
                break
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
