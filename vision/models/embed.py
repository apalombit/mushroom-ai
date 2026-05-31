"""EmbeddingExtractor: processed images -> cached 768-d DINOv2 vectors.

Applies ImageNet normalization + center crop at extraction time.
Saves one .pt file per image for downstream training.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from vision.models.backbone import FeatureExtractor

# DINOv2 expects ImageNet-normalized 224x224 inputs
EMBEDDING_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmbeddingExtractor:
    """Extracts and caches DINOv2 embeddings for processed images."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "mps",
    ):
        self.backbone = FeatureExtractor(model_name=model_name, device=device)
        self.device = device

    def extract_single(self, image_path: Path) -> torch.Tensor:
        """Extract embedding for a single image. Returns (768,) tensor."""
        img = Image.open(image_path).convert("RGB")
        pixel_values = EMBEDDING_TRANSFORM(img).unsqueeze(0)
        embedding = self.backbone.extract(pixel_values)
        return embedding.squeeze(0).cpu()

    def extract_batch(
        self,
        processed_dir: Path,
        output_dir: Path,
        batch_size: int = 32,
    ) -> dict:
        """Extract embeddings for all images in processed_dir. Idempotent.

        Saves {image_id}.pt in output_dir. Returns stats dict.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(processed_dir.glob("*.jpg"))
        to_process = []
        skipped = 0

        for path in image_paths:
            image_id = path.stem
            out_path = output_dir / f"{image_id}.pt"
            if out_path.exists():
                skipped += 1
            else:
                to_process.append(path)

        if not to_process:
            return {"total": len(image_paths), "new": 0, "skipped": skipped}

        # Process in batches
        new = 0
        for i in range(0, len(to_process), batch_size):
            batch_paths = to_process[i : i + batch_size]
            tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                tensors.append(EMBEDDING_TRANSFORM(img))

            pixel_values = torch.stack(tensors)
            embeddings = self.backbone.extract(pixel_values).cpu()

            for j, path in enumerate(batch_paths):
                image_id = path.stem
                out_path = output_dir / f"{image_id}.pt"
                torch.save(embeddings[j], out_path)
                new += 1

        return {"total": len(image_paths), "new": new, "skipped": skipped}

    def verify_cache(self, processed_dir: Path, output_dir: Path) -> dict:
        """Check embedding cache completeness."""
        processed_ids = {p.stem for p in processed_dir.glob("*.jpg")}
        cached_ids = {p.stem for p in output_dir.glob("*.pt")} if output_dir.exists() else set()

        return {
            "total_processed": len(processed_ids),
            "cached": len(cached_ids & processed_ids),
            "missing": len(processed_ids - cached_ids),
            "orphaned": len(cached_ids - processed_ids),
        }
