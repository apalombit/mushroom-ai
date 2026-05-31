"""Near-duplicate image dedup via DINOv2 CLS embedding cosine similarity.

Default: report-only (writes ``data/duplicates_report.csv``, no deletion).
With ``--apply``: hard-deletes duplicate image_ids from filesystem (every
``<image_id>.*`` or ``<image_id>_*.*`` under ``--data-root``) and from the
three DB tables (``image_annotations``, ``image_quality``, ``image_registry``)
in FK-safe order.

The keeper for each cluster is selected deterministically: most VLM
annotations → most species-propagated annotations → highest resolution →
lowest image_id.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from sqlalchemy import text  # noqa: E402

from db.connection import get_session  # noqa: E402

DEFAULT_EMBEDDINGS_DIR = (
    PROJECT_ROOT / "data" / "embeddings" / "facebook-dinov2-base" / "processed-v1"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "duplicates_report.csv"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def load_embeddings(embeddings_dir: Path) -> tuple[list[str], torch.Tensor]:
    """Load all .pt embeddings into (image_ids, (N,D) tensor), L2-normalized."""
    files = sorted(embeddings_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files under {embeddings_dir}")
    image_ids = [f.stem for f in files]
    vecs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    mat = torch.stack(vecs).to(torch.float32)
    norms = mat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return image_ids, mat / norms


def find_duplicate_edges(
    mat: torch.Tensor,
    threshold: float,
    chunk: int,
) -> list[tuple[int, int, float]]:
    """Return list of (i, j, cos_sim) with i<j and cos_sim >= threshold.

    Chunked to avoid the dense (N, N) similarity matrix.
    """
    n = mat.size(0)
    edges: list[tuple[int, int, float]] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = mat[start:end] @ mat.T  # (chunk, N)
        # Mask lower-triangle and self (i >= j becomes -inf)
        rows = torch.arange(start, end).unsqueeze(1)
        cols = torch.arange(n).unsqueeze(0)
        block.masked_fill_(rows >= cols, -2.0)
        hits = (block >= threshold).nonzero(as_tuple=False)
        for row_local, j in hits.tolist():
            i = start + row_local
            edges.append((i, j, float(block[row_local, j])))
    return edges


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_edges(n: int, edges: list[tuple[int, int, float]]) -> dict[int, list[int]]:
    """Union-find over edges. Returns {root_idx: [member_indices]} for clusters of size >= 2."""
    uf = UnionFind(n)
    for i, j, _ in edges:
        uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    seen_roots: set[int] = set()
    # Only collect members of clusters that have at least one edge
    members = set()
    for i, j, _ in edges:
        members.add(i)
        members.add(j)
    for idx in members:
        root = uf.find(idx)
        groups[root].append(idx)
        seen_roots.add(root)
    return {root: sorted(members_) for root, members_ in groups.items() if len(members_) >= 2}


# ---------------------------------------------------------------------------
# Keeper selection (DB-backed)
# ---------------------------------------------------------------------------

def fetch_keeper_ranks(
    session,
    all_image_ids: list[str],
) -> dict[str, tuple[int, int, int, str]]:
    """Per image_id return (n_vlm_labeled, n_species_propagated, resolution, image_id).

    Used as a sort key (DESC on first 3, ASC on last) to pick the keeper.
    Missing images get zeros — but the algorithm only ranks images that appear
    in the .pt set, which should all be in image_registry; defend anyway.
    """
    ids_in_db = set()
    res_map: dict[str, int] = {}
    rows = session.execute(
        text(
            "SELECT image_id, COALESCE(resolution_w,0) * COALESCE(resolution_h,0) AS res "
            "FROM image_registry WHERE image_id = ANY(:ids)"
        ),
        {"ids": all_image_ids},
    ).fetchall()
    for image_id, res in rows:
        ids_in_db.add(image_id)
        res_map[image_id] = int(res or 0)

    vlm_counts: dict[str, int] = Counter()
    sp_counts: dict[str, int] = Counter()
    rows = session.execute(
        text(
            "SELECT image_id, annotation_type, COUNT(*) "
            "FROM image_annotations WHERE image_id = ANY(:ids) "
            "GROUP BY image_id, annotation_type"
        ),
        {"ids": all_image_ids},
    ).fetchall()
    for image_id, atype, n in rows:
        if atype == "vlm_labeled":
            vlm_counts[image_id] = int(n)
        elif atype == "species_propagated":
            sp_counts[image_id] = int(n)

    ranks: dict[str, tuple[int, int, int, str]] = {}
    for image_id in all_image_ids:
        ranks[image_id] = (
            vlm_counts.get(image_id, 0),
            sp_counts.get(image_id, 0),
            res_map.get(image_id, 0),
            image_id,
        )
    return ranks


def fetch_species(session, all_image_ids: list[str]) -> dict[str, str]:
    rows = session.execute(
        text("SELECT image_id, species FROM image_registry WHERE image_id = ANY(:ids)"),
        {"ids": all_image_ids},
    ).fetchall()
    return {image_id: species for image_id, species in rows}


def fetch_file_paths(session, all_image_ids: list[str]) -> dict[str, str]:
    rows = session.execute(
        text("SELECT image_id, file_path FROM image_registry WHERE image_id = ANY(:ids)"),
        {"ids": all_image_ids},
    ).fetchall()
    return {image_id: fp for image_id, fp in rows}


def pick_keeper(member_image_ids: list[str], ranks: dict[str, tuple]) -> str:
    """Highest (vlm, sp, res) DESC, lowest image_id ASC."""
    def key(image_id: str) -> tuple:
        v, s, r, iid = ranks[image_id]
        return (-v, -s, -r, iid)
    return sorted(member_image_ids, key=key)[0]


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def write_report(
    report_path: Path,
    clusters: dict[int, list[int]],
    image_ids: list[str],
    keepers: dict[int, str],
    sim_to_keeper: dict[str, float],
    species: dict[str, str],
    file_paths: dict[str, str],
    ranks: dict[str, tuple],
    edges: list[tuple[int, int, float]],
    threshold: float,
) -> None:
    cluster_sizes = Counter(len(members) for members in clusters.values())
    n_duplicates = sum((len(m) - 1) for m in clusters.values())
    total_in_clusters = sum(len(m) for m in clusters.values())

    # Edge histogram (0.005 bins from threshold to 1.0)
    bins: list[float] = []
    b = threshold
    while b < 1.0 + 1e-9:
        bins.append(round(b, 4))
        b += 0.005
    bin_counts = [0] * len(bins)
    for _, _, s in edges:
        # Find rightmost bin <= s
        idx = 0
        for k, edge_lo in enumerate(bins):
            if s >= edge_lo:
                idx = k
        bin_counts[idx] += 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as f:
        f.write(f"# threshold: {threshold}\n")
        f.write(f"# total_clusters: {len(clusters)}\n")
        f.write(f"# total_in_clusters: {total_in_clusters}\n")
        f.write(f"# total_duplicates_to_delete: {n_duplicates}\n")
        f.write(f"# cluster_size_distribution: {dict(sorted(cluster_sizes.items()))}\n")
        f.write("# edge_similarity_histogram (lo .. count):\n")
        for lo, cnt in zip(bins, bin_counts, strict=False):
            f.write(f"#   {lo:.4f}: {cnt}\n")
        w = csv.writer(f)
        w.writerow([
            "cluster_id", "image_id", "role", "keeper_image_id",
            "cos_sim_to_keeper", "species", "keeper_species",
            "file_path", "n_vlm_annotations", "cluster_size",
        ])
        for root, members in clusters.items():
            keeper_iid = keepers[root]
            keeper_species = species.get(keeper_iid, "")
            size = len(members)
            for idx in members:
                iid = image_ids[idx]
                role = "keeper" if iid == keeper_iid else "duplicate"
                w.writerow([
                    root,
                    iid,
                    role,
                    keeper_iid,
                    f"{sim_to_keeper.get(iid, 1.0):.6f}",
                    species.get(iid, ""),
                    keeper_species,
                    file_paths.get(iid, ""),
                    ranks[iid][0],
                    size,
                ])


def compute_sim_to_keeper(
    mat: torch.Tensor,
    image_ids: list[str],
    clusters: dict[int, list[int]],
    keepers: dict[int, str],
) -> dict[str, float]:
    iid_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    out: dict[str, float] = {}
    for root, members in clusters.items():
        keeper_iid = keepers[root]
        kv = mat[iid_to_idx[keeper_iid]]
        for idx in members:
            iid = image_ids[idx]
            if iid == keeper_iid:
                out[iid] = 1.0
            else:
                out[iid] = float(mat[idx] @ kv)
    return out


# ---------------------------------------------------------------------------
# Deletion (filesystem + DB)
# ---------------------------------------------------------------------------

def file_matches_image_id(stem: str, image_id: str) -> bool:
    """True if a filename stem belongs to image_id (exact or '<id>_suffix')."""
    return stem == image_id or stem.startswith(f"{image_id}_")


def delete_filesystem_artifacts(
    duplicate_ids: set[str],
    data_root: Path,
    report_path: Path,
) -> dict[str, int]:
    """Walk data_root and unlink every file whose stem matches a duplicate id.

    Returns {top-level-dir: count_deleted} summary.
    """
    deleted_per_dir: Counter = Counter()
    n_total = 0
    for f in data_root.rglob("*"):
        if not f.is_file():
            continue
        if f == report_path:
            continue
        if not file_matches_image_id(f.stem, ""):  # never true; just kept for symmetry
            pass
        # Quick check: stem might be exact match
        if f.stem in duplicate_ids:
            try:
                rel_parent = f.relative_to(data_root).parts[0]
            except ValueError:
                rel_parent = "<root>"
            f.unlink()
            deleted_per_dir[rel_parent] += 1
            n_total += 1
            continue
        # Slower check: '<id>_suffix' form
        for dup_id in duplicate_ids:
            if f.stem.startswith(f"{dup_id}_"):
                try:
                    rel_parent = f.relative_to(data_root).parts[0]
                except ValueError:
                    rel_parent = "<root>"
                f.unlink()
                deleted_per_dir[rel_parent] += 1
                n_total += 1
                break
    print(f"  filesystem: deleted {n_total} files across {len(deleted_per_dir)} dirs")
    for d, c in sorted(deleted_per_dir.items()):
        print(f"    {d}/: {c}")
    return deleted_per_dir


def delete_db_rows(session, duplicate_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Delete from image_annotations, image_quality, image_registry. Returns per-table (before, after).

    Batches IN clauses at 500 ids each. One transaction.
    """
    before = {
        "image_annotations": session.execute(text("SELECT COUNT(*) FROM image_annotations")).scalar(),
        "image_quality": session.execute(text("SELECT COUNT(*) FROM image_quality")).scalar(),
        "image_registry": session.execute(text("SELECT COUNT(*) FROM image_registry")).scalar(),
    }
    for table in ("image_annotations", "image_quality", "image_registry"):
        for start in range(0, len(duplicate_ids), 500):
            batch = duplicate_ids[start : start + 500]
            session.execute(
                text(f"DELETE FROM {table} WHERE image_id = ANY(:ids)"),
                {"ids": batch},
            )
    session.commit()
    after = {
        "image_annotations": session.execute(text("SELECT COUNT(*) FROM image_annotations")).scalar(),
        "image_quality": session.execute(text("SELECT COUNT(*) FROM image_quality")).scalar(),
        "image_registry": session.execute(text("SELECT COUNT(*) FROM image_registry")).scalar(),
    }
    return {k: (before[k], after[k]) for k in before}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DINOv2-embedding near-duplicate dedup")
    p.add_argument("--threshold", type=float, default=0.995)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--apply", action="store_true",
                   help="Perform hard delete from filesystem + DB after writing the report")
    args = p.parse_args()

    print(f"Loading embeddings from {args.embeddings_dir}...")
    image_ids, mat = load_embeddings(args.embeddings_dir)
    print(f"  loaded {len(image_ids)} embeddings, dim={mat.size(1)}")

    print(f"Computing chunked cosine similarity (threshold={args.threshold}, chunk={args.chunk})...")
    edges = find_duplicate_edges(mat, args.threshold, args.chunk)
    print(f"  found {len(edges)} edges >= {args.threshold}")

    clusters = cluster_edges(len(image_ids), edges)
    n_dup = sum(len(m) - 1 for m in clusters.values())
    print(f"  formed {len(clusters)} clusters covering {sum(len(m) for m in clusters.values())} images; {n_dup} would be deleted")

    if not clusters:
        print("No duplicates found. Exiting.")
        return

    cluster_member_iids = [image_ids[idx] for members in clusters.values() for idx in members]
    print("Querying DB for keeper-selection ranks + metadata...")
    with get_session() as session:
        ranks = fetch_keeper_ranks(session, cluster_member_iids)
        species = fetch_species(session, cluster_member_iids)
        file_paths = fetch_file_paths(session, cluster_member_iids)

    keepers: dict[int, str] = {}
    for root, members in clusters.items():
        member_iids = [image_ids[idx] for idx in members]
        keepers[root] = pick_keeper(member_iids, ranks)

    sim_to_keeper = compute_sim_to_keeper(mat, image_ids, clusters, keepers)

    print(f"Writing report to {args.report_path}...")
    write_report(
        args.report_path, clusters, image_ids, keepers, sim_to_keeper,
        species, file_paths, ranks, edges, args.threshold,
    )

    # Sample preview
    print("\nSample duplicates (cluster_id, image_id, cos_sim, species, file_path):")
    samples_shown = 0
    for root, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if samples_shown >= 10:
            break
        keeper_iid = keepers[root]
        print(f"  cluster {root} (size={len(members)}):")
        print(f"    KEEPER    {keeper_iid}  ({species.get(keeper_iid, '')})  {file_paths.get(keeper_iid, '')}")
        for idx in members:
            iid = image_ids[idx]
            if iid == keeper_iid:
                continue
            print(f"    duplicate {iid}  sim={sim_to_keeper[iid]:.4f}  ({species.get(iid, '')})  {file_paths.get(iid, '')}")
        samples_shown += 1

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to delete from filesystem + DB.")
        return

    duplicate_ids = [
        image_ids[idx]
        for root, members in clusters.items()
        for idx in members
        if image_ids[idx] != keepers[root]
    ]
    print(f"\n=== APPLY: deleting {len(duplicate_ids)} duplicates ===")
    print(f"Filesystem walk under {args.data_root}...")
    delete_filesystem_artifacts(set(duplicate_ids), args.data_root, args.report_path)
    print("DB deletion...")
    with get_session() as session:
        deltas = delete_db_rows(session, duplicate_ids)
    for table, (before, after) in deltas.items():
        print(f"  {table}: {before} -> {after} (delta={before - after})")
    print("Done.")


if __name__ == "__main__":
    main()
