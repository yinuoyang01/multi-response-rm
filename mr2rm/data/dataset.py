"""
Dataset and collate for multi-response reward training.

Batch format:
- user turn: text prompt + optional images/videos (content blocks)
- assistant turn: all responses concatenated with a separator
- We record the end token position of each response for scoring.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, BatchSampler
from transformers import PreTrainedTokenizerBase

# Special token used to separate multiple responses in the assistant turn.
# Must be registered via tokenizer.add_special_tokens() so it encodes to a
# single, unique token ID that cannot appear inside normal text.
RESP_SEP_TOKEN = "<|resp_sep|>"


def add_resp_sep_token(tokenizer, model=None):
    """Add RESP_SEP_TOKEN as a special token to the tokenizer.

    Returns the number of tokens added (0 if already present).
    If *model* is provided, resizes its token embeddings accordingly.
    """
    if RESP_SEP_TOKEN in tokenizer.additional_special_tokens:
        return 0  # Already registered — nothing to do
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.additional_special_tokens + [RESP_SEP_TOKEN]}
    )
    if num_added > 0 and model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Handle dict format: combine all list values (e.g., ranking_data, pair_data, single_data)
        combined = []
        for key, value in obj.items():
            if isinstance(value, list):
                combined.extend(value)
        if combined:
            return combined
        # If no lists found, try common keys
        for key in ["ranking_data", "pair_data", "single_data", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        raise ValueError(f"Expected list or dict with list values in {path}, got dict with keys: {list(obj.keys())}")
    raise ValueError(f"Expected list or dict in {path}, got {type(obj)}")


def _maybe_load_image(img_path: str) -> Image.Image:
    """Load image from path. Raises FileNotFoundError if file doesn't exist."""
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    return Image.open(p).convert("RGB")


def _extract_frame_number(filename: str) -> Optional[int]:
    """
    Extract frame number from filename.
    Supports formats like:
    - c01_0001.jpeg -> 1
    - frame_001.jpg -> 1
    - 0001.png -> 1
    """
    # Pattern 1: c{channel}_{frame_number}.{ext}
    match = re.match(r'c\d+_(\d+)\.(jpeg|jpg|png)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 2: frame_{frame_number}.{ext} or {prefix}_{frame_number}.{ext}
    match = re.match(r'.*_(\d+)\.(jpeg|jpg|png)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 3: {frame_number}.{ext} (just numbers)
    match = re.match(r'(\d+)\.(jpeg|jpg|png)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def _generate_timestamps_from_frames(frame_files: List[Path], sampling_interval: float = 4.0) -> List[float]:
    """Generate timestamps for a sequence of video frames.

    Sorts frames by any numeric frame id embedded in the filename (falling back
    to input order) and spaces them ``sampling_interval`` seconds apart.
    """
    ordered = []
    for i, f in enumerate(frame_files):
        frame_num = _extract_frame_number(f.name)
        ordered.append((frame_num if frame_num is not None else i, f))
    ordered.sort(key=lambda x: x[0])
    return [idx * sampling_interval for idx in range(len(ordered))]


@dataclass
class MultiResponseRewardSample:
    prompt: str
    responses: List[str]
    label: int  # Best response index (for backward compatibility)
    rankings: Optional[List[int]] = None  # Full ranking for each response (lower is better, 1=best)
    images: List[Any] = None
    videos: List[str] = None  # Can be video file paths or frame directories
    video_frames: Optional[List[str]] = None  # Frame file paths when video is a directory
    video_timestamps: Optional[List[float]] = None  # Timestamps for frames


class MultiResponseRewardDataset(Dataset):
    """
    Canonical sample format (JSONL, one per line):

        {
            "prompt": "...",
            "responses": ["A", "B", "C", "D"],
            "rankings": [3, 1, 2, 4],      # 1 = best, higher = worse; ties allowed
            "image": "path/to/img.jpg",    # optional; absolute or relative to image_base_dir
            "video": "path/to/vid.mp4"     # optional; file or frame directory
        }
    """

    def __init__(self, data_path: str, image_base_dir: Optional[str] = None, video_base_dir: Optional[str] = None):
        """
        Args:
            data_path: Path to a JSON or JSONL file containing samples.
            image_base_dir: Base directory for resolving relative image paths (optional).
            video_base_dir: Base directory for resolving relative video paths (optional).

        NOTE: This class uses LAZY LOADING - data is only processed when accessed via __getitem__.
        This allows loading large datasets without running out of memory.
        """
        # image_base_dir / video_base_dir: root(s) to resolve relative media paths.
        # Both are optional — absolute paths in samples work without them.
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None
        self.video_base_dir = Path(video_base_dir) if video_base_dir else None
        self.image_fallback_dirs = [self.image_base_dir] if self.image_base_dir else []
        self.video_fallback_dirs = [self.video_base_dir] if self.video_base_dir else []

        # Load raw data (JSONL or JSON file, lazy processing in __getitem__)
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if path.suffix == ".jsonl":
            self.raw_data: List[dict] = _load_jsonl(path)
        elif path.suffix == ".json":
            self.raw_data = _load_json(path)
        else:
            raise ValueError(f"Unsupported data file: {path}")
        print(f"Loaded {len(self.raw_data)} samples from {path}")

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> MultiResponseRewardSample:
        """Process and return a single sample (lazy loading)."""
        item = self.raw_data[idx]
        try:
            return self._process_sample(item)
        except (ValueError, KeyError) as e:
            # Skip bad samples by returning a random valid one
            import warnings
            warnings.warn(f"Skipping bad sample at idx {idx}: {e}")
            alt_idx = (idx + 1) % len(self.raw_data)
            return self.__getitem__(alt_idx)

    def _process_sample(self, item: dict) -> MultiResponseRewardSample:
        """Process a raw data item into a MultiResponseRewardSample."""
        # Get prompt (support both "prompt" and "question")
        prompt = item.get("prompt") or item.get("question", "")
        if not prompt:
            raise ValueError(f"Sample missing 'prompt' or 'question': {list(item.keys())}")

        # Get images (lazy resolution)
        images = self._resolve_images(item)
        
        # Get videos (lazy resolution)
        videos, video_frames, video_timestamps = self._resolve_videos(item)
        
        # Parse responses and label
        responses, label, rankings = self._parse_responses(item)

        # Load images only when accessed (lazy)
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(_maybe_load_image(img))
            else:
                loaded_images.append(img)

        return MultiResponseRewardSample(
            prompt=prompt,
            responses=responses,
            label=label,
            rankings=rankings,
            images=loaded_images,
            videos=videos if isinstance(videos, list) else [videos] if videos else [],
            video_frames=video_frames,
            video_timestamps=video_timestamps,
        )
    
    def _resolve_images(self, item: dict) -> List[str]:
        """Resolve the optional ``image`` field to a list (empty if missing/unresolved)."""
        img_path = item.get("image")
        if not img_path or not isinstance(img_path, str) or not img_path.strip():
            return []
        resolved = self._find_image_path(img_path)
        if resolved is None:
            import warnings
            warnings.warn(
                f"Image not found, treating sample as text-only: {img_path}",
                RuntimeWarning, stacklevel=2,
            )
            return []
        return [resolved]
    
    def _find_image_path(self, img_path: str) -> Optional[str]:
        """Resolve an image path. Accepts absolute paths, or paths relative
        to ``image_base_dir`` (if provided)."""
        p = Path(img_path)
        if p.is_absolute() and p.exists():
            return str(p)
        for base in self.image_fallback_dirs:
            candidate = base / p
            if candidate.exists():
                return str(candidate)
        return None
    
    def _resolve_videos(self, item: dict) -> tuple:
        """Resolve the optional ``video`` field.

        Returns (videos, video_frames, video_timestamps):
          - videos: [path] if a single video file
          - video_frames / video_timestamps: populated if the path is a frame directory
        """
        vid_path = item.get("video")
        if not vid_path or not isinstance(vid_path, str) or not vid_path.strip():
            return [], None, None
        resolved = self._find_video_path(vid_path)
        if resolved is None:
            import warnings
            warnings.warn(
                f"Video not found, treating sample as text-only: {vid_path}",
                RuntimeWarning, stacklevel=2,
            )
            return [], None, None
        if resolved.is_dir():
            frames = sorted(resolved.glob("*.jpeg")) + sorted(resolved.glob("*.jpg")) + sorted(resolved.glob("*.png"))
            if not frames:
                return [], None, None
            sampling_interval = float(item.get("sampling_interval", item.get("fps", 4.0)))
            timestamps = _generate_timestamps_from_frames(frames, sampling_interval=sampling_interval)
            return [], [str(f) for f in frames], timestamps
        return [str(resolved)], None, None
    
    def _find_video_path(self, vid_path: str) -> Optional[Path]:
        """Resolve a video path (file or frame directory). Accepts absolute
        paths, or paths relative to ``video_base_dir`` (if provided)."""
        p = Path(vid_path)
        if p.is_absolute() and p.exists():
            return p
        for base in self.video_fallback_dirs:
            candidate = base / p
            if candidate.exists():
                return candidate
        return None
    
    def _parse_responses(self, item: dict) -> tuple:
        """Parse responses and rankings from a sample.

        Expected schema:
            {"responses": List[str], "rankings": List[int]}
        where ``rankings`` gives each response's rank (1 = best, higher = worse;
        ties allowed). Returns (responses, best_index, rankings).
        """
        if "responses" not in item or "rankings" not in item:
            raise ValueError(
                f"Sample must contain both 'responses' and 'rankings'. "
                f"Got keys: {list(item.keys())}"
            )

        responses = list(item["responses"])
        rankings = list(item["rankings"])
        if len(responses) != len(rankings):
            raise ValueError(
                f"responses and rankings length mismatch: "
                f"{len(responses)} vs {len(rankings)}"
            )

        # Drop empty responses (and corresponding rankings)
        valid = [(r, rk) for r, rk in zip(responses, rankings) if r and r.strip()]
        if not valid:
            raise ValueError("All responses are empty")
        responses = [r for r, _ in valid]
        rankings = [rk for _, rk in valid]

        # Derive best index (lowest rank number). Break ties randomly.
        min_rank = min(rankings)
        best_indices = [i for i, rk in enumerate(rankings) if rk == min_rank]
        label = random.choice(best_indices) if len(best_indices) > 1 else best_indices[0]

        return responses, label, rankings
    
    def get_modality(self, idx: int) -> str:
        """
        Get modality type for a sample: 'text', 'image', or 'video'.
        Priority: video > image > text.

        Actually resolves image/video paths to verify files exist; if all
        referenced media files are missing, the sample is downgraded to 'text'.
        This keeps ModalityBatchSampler batches modality-homogeneous even when
        the underlying dataset has missing media files, which is required for
        Molmo2's vision-token indexing (fails on mixed image/text in a batch).

        Results are cached per idx to avoid repeated filesystem checks.
        """
        if not hasattr(self, "_modality_cache"):
            self._modality_cache = {}
        cached = self._modality_cache.get(idx)
        if cached is not None:
            return cached

        item = self.raw_data[idx]
        modality = "text"

        # Video first: resolve actual path/frames
        if item.get("video"):
            videos, video_frames, _ = self._resolve_videos(item)
            if videos or video_frames:
                modality = "video"

        # Image next: resolve actual path
        if modality == "text" and item.get("image"):
            if self._resolve_images(item):
                modality = "image"

        self._modality_cache[idx] = modality
        return modality

class ModalityBatchSampler(BatchSampler):
    """
    Batch sampler that ensures each batch contains only one modality (text/image/video).
    This prevents mixing modalities in the same batch, avoiding processor errors.
    
    Args:
        dataset: MultiResponseRewardDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle indices within each modality bucket
        modality_weights: Optional dict mapping modality to weight for sampling proportion.
            If None, samples uniformly from available modalities.
            Example: {"image": 0.7, "video": 0.2, "text": 0.1}
    """
    
    def __init__(
        self,
        dataset: MultiResponseRewardDataset,
        batch_size: int,
        shuffle: bool = True,
        modality_weights: Optional[Dict[str, float]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Bucket indices by modality
        self.idx_text = []
        self.idx_image = []
        self.idx_video = []
        
        for idx in range(len(dataset)):
            modality = dataset.get_modality(idx)
            if modality == "text":
                self.idx_text.append(idx)
            elif modality == "image":
                self.idx_image.append(idx)
            elif modality == "video":
                self.idx_video.append(idx)
        
        # Normalize modality weights if provided
        if modality_weights is not None:
            total = sum(modality_weights.values())
            self.modality_weights = {k: v / total for k, v in modality_weights.items()}
        else:
            # Uniform sampling from available modalities
            available = []
            if self.idx_text:
                available.append("text")
            if self.idx_image:
                available.append("image")
            if self.idx_video:
                available.append("video")
            self.modality_weights = {m: 1.0 / len(available) for m in available} if available else {}
        
        # Shuffle buckets initially with deterministic seed
        # This ensures all ranks start with the same shuffle order
        self.epoch = 0
        if self.shuffle:
            rng = random.Random(0)  # Seed 0 for initial shuffle
            rng.shuffle(self.idx_text)
            rng.shuffle(self.idx_image)
            rng.shuffle(self.idx_video)
        
        # Create batch indices for each modality
        self._build_batches()
    
    def _build_batches(self):
        """Build all batches from modality buckets."""
        self.batches = []
        
        # Create batches from each modality bucket
        for modality, indices in [
            ("text", self.idx_text),
            ("image", self.idx_image),
            ("video", self.idx_video),
        ]:
            if not indices:
                continue
            
            # Split indices into batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) == self.batch_size:  # Only full batches
                    self.batches.append((modality, batch_indices))
        
        # Shuffle batches if needed (use epoch-seeded RNG for determinism across ranks)
        if self.shuffle:
            rng = random.Random(self.epoch + 42)
            rng.shuffle(self.batches)
    
    def __iter__(self):
        """Yield batches, sampling from modalities according to weights.
        
        Uses epoch-based seeding to ensure all ranks get the same batch order.
        This is critical for distributed training.
        
        NOTE: Shuffling is done in set_epoch(), not here, to avoid double-shuffling
        and ensure __iter__ has no side effects (important for __len__ calls).
        """
        # Use epoch-based RNG for deterministic batch order across ranks
        epoch = getattr(self, 'epoch', 0)
        rng = random.Random(epoch + 1000)  # Offset to differ from set_epoch seed
        
        # NOTE: Do NOT reshuffle here! set_epoch() already handles shuffling.
        # Reshuffling in __iter__ would cause side effects when __len__ is called.
        
        # Sample batches according to modality weights
        available_batches = {m: [] for m in ["text", "image", "video"]}
        for modality, batch_indices in self.batches:
            available_batches[modality].append(batch_indices)
        
        # Yield batches in proportion to weights
        while any(available_batches.values()):
            # Sample modality according to weights
            modalities = [m for m, batches in available_batches.items() if batches]
            if not modalities:
                break
            
            # Normalize weights for available modalities
            total_weight = sum(self.modality_weights.get(m, 0) for m in modalities)
            if total_weight == 0:
                # Fallback to uniform
                weights = [1.0 / len(modalities)] * len(modalities)
            else:
                weights = [self.modality_weights.get(m, 0) / total_weight for m in modalities]
            
            # Sample modality using deterministic RNG
            modality = rng.choices(modalities, weights=weights, k=1)[0]
            
            # Yield a batch from this modality
            if available_batches[modality]:
                yield available_batches[modality].pop(0)
    
    def __len__(self) -> int:
        """Total number of batches."""
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """Reshuffle batches for a new epoch.
        
        Uses epoch-based seed to ensure all ranks get the same shuffle order.
        This is critical for distributed training where different ranks must
        iterate over the same batch order (though different subsets).
        """
        self.epoch = epoch
        if self.shuffle:
            # Use deterministic seed based on epoch so all ranks get same shuffle
            rng = random.Random(epoch)
            rng.shuffle(self.idx_text)
            rng.shuffle(self.idx_image)
            rng.shuffle(self.idx_video)
            self._build_batches()


# Module-level statistics for tracking response location failures during training
_collate_stats = {
    "total_batches": 0,
    "total_samples": 0,
    "start_idx_failures": 0,  # Could not find assistant_tokens
    "response_not_found": 0,  # Could not find individual response in fallback
    "cursor_out_of_bounds": 0,  # Cursor >= seq_len
    "end_idx_invalid": 0,  # end_idx < cursor
    "fallback_to_last_token": 0,  # Had to use last token as fallback
    "samples_skipped": 0,  # Samples skipped due to truncation
    # Vision budget stats
    "vision_budget_text_too_long": 0,     # Samples where text alone > max_length
    "vision_budget_crops_reduced": 0,     # Batches where max_crops was reduced
    "vision_budget_frames_reduced": 0,    # Batches where video frames were reduced
    "vision_budget_skip_no_fit": 0,       # Samples skipped because even min vision doesn't fit
}


def get_collate_stats() -> Dict[str, int]:
    """Get current collate statistics."""
    return _collate_stats.copy()


def print_collate_stats():
    """Print a compact summary of collate statistics."""
    stats = _collate_stats
    n = stats["total_samples"] or 1
    skipped = stats["samples_skipped"]
    print(
        f"[collate] batches={stats['total_batches']} samples={stats['total_samples']} "
        f"skipped={skipped} ({skipped / n * 100:.1f}%) "
        f"fallback_last_token={stats['fallback_to_last_token']}"
    )


# ---------------------------------------------------------------------------
# Vision token budget helpers
# ---------------------------------------------------------------------------

def _compute_image_token_count(image_grid) -> int:
    """Compute the number of vision tokens from an image_grid.

    image_grid: [resized_h, resized_w, height, width]
    Assumes col_tokens=True (Molmo2 defaults).
    Formula: 4 + resized_h * (resized_w + 1) + height * (width + 1)
    """
    resized_h, resized_w, height, width = image_grid
    return 4 + int(resized_h) * (int(resized_w) + 1) + int(height) * (int(width) + 1)


def _compute_video_token_count(num_frames: int, h: int = 9, w: int = 9) -> int:
    """Approximate number of vision tokens for a video.

    Per frame: 2 (frame_start + frame_end) + h * w patches + ~4 text tokens for timestamp.
    Molmo2 defaults: pool_size=[3,3] -> h=w=9 (ceil(27/3)).
    """
    per_frame = 2 + int(h) * int(w) + 4
    return int(num_frames) * per_frame


def _strip_vision_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip image/video content from messages, keeping only text blocks."""
    stripped = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        if isinstance(msg["content"], list):
            new_content = [
                block for block in msg["content"]
                if block.get("type") == "text"
            ]
            if not new_content:
                new_content = [{"type": "text", "text": ""}]
            new_msg["content"] = new_content
        else:
            new_msg["content"] = msg["content"]
        stripped.append(new_msg)
    return stripped


def _build_processor_kwargs(
    texts,
    images=None,
    videos=None,
    video_kwargs=None,
    max_length: int = 4096,
    max_crops: Optional[int] = None,
) -> dict:
    """Build the kwargs dict for processor() call."""
    kwargs: Dict[str, Any] = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    if images is not None or videos is not None:
        kwargs["return_mm_token_type_ids"] = True
    if images is not None:
        kwargs["images"] = images
    if max_crops is not None:
        kwargs["max_crops"] = max_crops
    if videos is not None:
        videos_list, video_metas_list = zip(*videos)
        kwargs["videos"] = list(videos_list)
        kwargs["video_metadata"] = list(video_metas_list)
    if video_kwargs:
        kwargs.update(video_kwargs)
    return kwargs


def _process_with_vision_budget(
    processor,
    tokenizer,
    texts: List[str],
    images,
    videos,
    video_kwargs: dict,
    messages_list: List[List[Dict]],
    max_length: int,
    vision_budgets: List[int],
    default_max_crops: int = 8,
    min_max_crops: int = 1,
    max_video_frames: int = 16,
    max_vision_tokens: Optional[int] = None,
):
    """Process visual inputs with adaptive vision token budget.

    For images: try decreasing max_crops until vision tokens fit the budget.
    For videos: reduce frame count until total fits the budget.
    vision_budgets are already clamped by max_vision_tokens in Phase 1.

    Returns (batch_inputs, effective_max_crops, effective_num_frames).
    """
    global _collate_stats
    # vision_budgets already incorporates max_vision_tokens clamp from Phase 1
    min_budget = min(vision_budgets) if vision_budgets else 0

    effective_max_crops = default_max_crops
    effective_num_frames = max_video_frames

    if images is not None:
        # Build candidate list of max_crops values to try (descending)
        candidates = sorted(
            {default_max_crops, 4, 2, min_max_crops},
            reverse=True,
        )
        candidates = [c for c in candidates if min_max_crops <= c <= default_max_crops]

        chosen = candidates[-1]  # fallback to minimum
        fits = False
        for try_crops in candidates:
            image_inputs = processor.image_processor.preprocess(
                images, max_crops=try_crops,
            )
            total_vision = sum(
                _compute_image_token_count(g) for g in image_inputs["image_grids"]
            )
            if total_vision <= min_budget:
                chosen = try_crops
                fits = True
                break

        effective_max_crops = chosen
        if effective_max_crops < default_max_crops:
            _collate_stats["vision_budget_crops_reduced"] += 1

        if not fits:
            # Even max_crops=min still doesn't fit -> skip this batch
            _collate_stats["vision_budget_skip_no_fit"] += len(vision_budgets)
            _collate_stats["samples_skipped"] += len(vision_budgets)
            return None, effective_max_crops, effective_num_frames

        batch_inputs = processor(
            **_build_processor_kwargs(
                texts, images=images, max_length=max_length,
                max_crops=effective_max_crops,
            )
        )

        # Post-processing verification: if any sequence was truncated to max_length,
        # the budget estimation was too optimistic. Retry with fewer crops.
        # If still truncated after all retries, skip the sample entirely.
        actual_len = batch_inputs["input_ids"].shape[-1]
        if actual_len >= max_length:
            if effective_max_crops > min_max_crops:
                _collate_stats["vision_budget_post_retry"] = _collate_stats.get("vision_budget_post_retry", 0) + 1
                print(f"[Phase 2.5] Post-processing truncation detected: actual_len={actual_len} >= max_length={max_length}, "
                      f"current max_crops={effective_max_crops}. Retrying with fewer crops...")
                # Try smaller crops until no truncation or we hit minimum
                retry_candidates = [c for c in candidates if c < effective_max_crops]
                for retry_crops in retry_candidates:
                    batch_inputs = processor(
                        **_build_processor_kwargs(
                            texts, images=images, max_length=max_length,
                            max_crops=retry_crops,
                        )
                    )
                    effective_max_crops = retry_crops
                    new_len = batch_inputs["input_ids"].shape[-1]
                    print(f"  [Phase 2.5] Retry max_crops={retry_crops}: actual_len={new_len}")
                    if new_len < max_length:
                        print(f"  [Phase 2.5] OK, fits within max_length.")
                        break
            # After retries (or if already at min_max_crops), check if still truncated
            if batch_inputs["input_ids"].shape[-1] >= max_length:
                print(f"  [Phase 2.5] SKIP: Still truncated after all retries "
                      f"(max_crops={effective_max_crops}, len={batch_inputs['input_ids'].shape[-1]}). "
                      f"Skipping sample.")
                _collate_stats["vision_budget_skip_no_fit"] += len(vision_budgets)
                _collate_stats["samples_skipped"] += len(vision_budgets)
                return None, effective_max_crops, effective_num_frames

    elif videos is not None:
        # Estimate per-frame tokens (Molmo2 default: pool 3×3 -> 9×9 grid)
        per_frame_est = 2 + 9 * 9 + 4  # 87
        max_frames_ok = min_budget // per_frame_est if per_frame_est > 0 else max_video_frames
        if max_frames_ok < 1:
            # Even 1 frame doesn't fit -> skip
            _collate_stats["vision_budget_skip_no_fit"] += len(vision_budgets)
            _collate_stats["samples_skipped"] += len(vision_budgets)
            return None, effective_max_crops, effective_num_frames
        effective_num_frames = min(max_video_frames, max_frames_ok)

        if effective_num_frames < max_video_frames:
            _collate_stats["vision_budget_frames_reduced"] += 1
            # Reduce frames in messages_list and re-extract vision info
            import copy, numpy as np
            from molmo_utils.vision_process import process_vision_info as _pvi
            reduced_msgs = copy.deepcopy(messages_list)
            for msgs in reduced_msgs:
                for msg in msgs:
                    if not isinstance(msg["content"], list):
                        continue
                    for block in msg["content"]:
                        if block.get("type") != "video":
                            continue
                        vdata = block.get("video")
                        if isinstance(vdata, list) and len(vdata) > effective_num_frames:
                            total = len(vdata)
                            idx = np.linspace(0, total - 1, effective_num_frames, dtype=int)
                            block["video"] = [vdata[i] for i in idx]
                            if "timestamps" in block:
                                ts = block["timestamps"]
                                block["timestamps"] = [ts[i] for i in idx]

            texts_r = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in reduced_msgs
            ]
            _, videos_r, vkw_r = _pvi(reduced_msgs)
            batch_inputs = processor(
                **_build_processor_kwargs(
                    texts_r, videos=videos_r, video_kwargs=vkw_r,
                    max_length=max_length,
                )
            )
        else:
            batch_inputs = processor(
                **_build_processor_kwargs(
                    texts, videos=videos, video_kwargs=video_kwargs,
                    max_length=max_length,
                )
            )
    else:
        # No vision at all (shouldn't happen if caller checks, but be safe)
        batch_inputs = processor(
            **_build_processor_kwargs(texts, max_length=max_length)
        )

    return batch_inputs, effective_max_crops, effective_num_frames


def collate_multi_response_reward(
    batch: List[MultiResponseRewardSample],
    tokenizer: PreTrainedTokenizerBase,
    processor,
    max_length: int,
    sep_text: str = "\n\n" + RESP_SEP_TOKEN + "\n\n",
    shuffle_responses: bool = True,
    max_video_frames: int = 16,
    skip_truncated: bool = False,  # Changed to False: allow truncated samples
    # Vision token budget parameters
    default_max_crops: int = 8,
    min_max_crops: int = 1,
    vision_budget_enabled: bool = True,
    vision_budget_safety_margin: int = 256,
    max_vision_tokens: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate that:
    - builds user+assistant messages with concatenated responses
    - tokenizes via processor.apply_chat_template
    - records end token positions for each response
    - optionally shuffles response order to avoid position bias
    """
    global _collate_stats
    _collate_stats["total_batches"] += 1
    _collate_stats["total_samples"] += len(batch)

    # Ensure processor's internal tokenizer also has the resp_sep special token.
    # processor.tokenizer is a separate object from the external tokenizer and
    # may not have it registered, causing <|resp_sep|> to be BPE-split into
    # multiple ordinary tokens during multimodal processing.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not tokenizer:
        add_resp_sep_token(processor.tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    pad_id = tokenizer.pad_token_id

    messages_list: List[List[Dict[str, Any]]] = []
    assistant_texts: List[str] = []
    responses_list: List[List[str]] = []
    labels: List[int] = []
    rankings_list: List[Optional[List[int]]] = []
    orig_response_counts: List[int] = []

    for sample in batch:
        prompt = sample.prompt
        responses = sample.responses
        label = sample.label
        rankings = sample.rankings
        images = sample.images or []
        videos = sample.videos or []
        video_frames = sample.video_frames
        video_timestamps = sample.video_timestamps

        # If both images and videos are present, prefer images and drop videos
        if images and (videos or (video_frames and len(video_frames) > 0)):
            videos = []
            video_frames = None
            video_timestamps = None
        if video_frames and max_video_frames is not None and len(video_frames) > max_video_frames:
            # Evenly sample max_video_frames across the sequence (including endpoints)
            total = len(video_frames)
            indices = []
            for j in range(max_video_frames):
                pos = j * (total - 1) / (max_video_frames - 1)
                idx = int(round(pos))
                if idx >= total:
                    idx = total - 1
                indices.append(idx)
            # Deduplicate while keeping order (in case rounding collides)
            seen = set()
            sampled_indices = []
            for idx in indices:
                if idx not in seen:
                    sampled_indices.append(idx)
                    seen.add(idx)
            # If we lost entries due to dedup, append from the tail to keep count
            i = total - 1
            while len(sampled_indices) < max_video_frames and i >= 0:
                if i not in seen:
                    sampled_indices.append(i)
                    seen.add(i)
                i -= 1
            sampled_indices = sorted(sampled_indices)
            video_frames = [video_frames[i] for i in sampled_indices]
            if video_timestamps:
                video_timestamps = [video_timestamps[i] for i in sampled_indices]

        if shuffle_responses:
            perm = list(range(len(responses)))
            random.shuffle(perm)
            responses = [responses[i] for i in perm]
            label = perm.index(label)
            if rankings is not None:
                rankings = [rankings[i] for i in perm]

        # IMAGE-FIRST ordering to match Molmo2 pre-training distribution.
        user_content: List[Dict[str, Any]] = []
        for img in images:
            if img:  # Only add non-empty images
                user_content.append({"type": "image", "image": img})

        # Handle video frames (directory of frames) with timestamps
        if video_frames and video_timestamps:
            # Pass frames as a list with timestamps for molmo_utils
            user_content.append({
                "type": "video",
                "video": video_frames,  # List of frame file paths
                "timestamps": video_timestamps,  # Required by molmo_utils
            })
        elif videos:
            # Regular video files
            for vid in videos:
                # Only add non-empty video paths
                if vid and (isinstance(vid, str) and vid.strip()):
                    if not vid.startswith(("http://", "https://", "file://")):
                        try:
                            vid = Path(vid).resolve().as_uri()
                        except Exception:
                            pass
                    user_content.append({"type": "video", "video": vid})

        # Text prompt LAST (after image/video) to match Molmo2 default formatter.
        user_content.append({"type": "text", "text": prompt})

        assistant_text = sep_text.join(responses)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        messages_list.append(messages)
        assistant_texts.append(assistant_text)
        responses_list.append(responses)
        labels.append(label)
        rankings_list.append(rankings)
        orig_response_counts.append(len(responses))

    # Check if we have visual inputs (images or videos) based on constructed messages
    # Note: ModalityBatchSampler ensures each batch has only one modality, so we don't need
    # to handle mixed batches here
    has_images = any(
        any(content.get("type") == "image" for content in messages[0]["content"])
        for messages in messages_list
    )
    has_videos = any(
        any(content.get("type") == "video" for content in messages[0]["content"])
        for messages in messages_list
    )
    
    if has_images or has_videos:
        # Use official Molmo2 processing method: process_vision_info + processor
        try:
            from molmo_utils.vision_process import process_vision_info
        except ImportError as e:
            raise ImportError(
                "molmo_utils.vision_process is required for multimodal inputs. "
                "Install via `pip install molmo_utils`."
            ) from e

        # ====================================================================
        # Phase 1: Determine vision token budget.
        # Always estimate text tokens first, then compute remaining budget.
        # If max_vision_tokens is set, clamp the budget to that upper bound.
        # ====================================================================
        vision_budgets: List[int] = []
        text_only_skip_indices: set = set()

        if vision_budget_enabled or max_vision_tokens is not None:
            for idx, messages in enumerate(messages_list):
                text_only_msgs = _strip_vision_from_messages(messages)
                text_only_str = processor.apply_chat_template(
                    text_only_msgs, tokenize=False, add_generation_prompt=False,
                )
                text_token_count = len(tokenizer.encode(text_only_str, add_special_tokens=True))

                if text_token_count > max_length:
                    text_only_skip_indices.add(idx)
                    vision_budgets.append(0)
                    _collate_stats["vision_budget_text_too_long"] += 1
                    _collate_stats["samples_skipped"] += 1
                else:
                    budget = max_length - text_token_count - vision_budget_safety_margin
                    budget = max(budget, 0)
                    if max_vision_tokens is not None:
                        budget = min(budget, max_vision_tokens)
                    vision_budgets.append(budget)
        else:
            vision_budgets = [max_length] * len(messages_list)

        # Filter out samples that are text-too-long
        if text_only_skip_indices:
            keep_mask = [i for i in range(len(messages_list)) if i not in text_only_skip_indices]
            if not keep_mask:
                # All samples skipped – return dummy batch
                import warnings
                warnings.warn(
                    f"Skipping batch: text alone exceeds max_length for all "
                    f"{len(batch)} sample(s).",
                    RuntimeWarning, stacklevel=2,
                )
                dummy_ids = torch.full((1, 1), pad_id, dtype=torch.long)
                return {
                    "input_ids": dummy_ids,
                    "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                    "resp_indices": torch.full((1, 1), -1, dtype=torch.long),
                    "resp_start_indices": torch.full((1, 1), -1, dtype=torch.long),
                    "labels": torch.tensor([0], dtype=torch.long),
                    "rankings": torch.full((1, 1), -1, dtype=torch.long),
                    "_skipped": torch.tensor([1], dtype=torch.long),
                }
            messages_list = [messages_list[i] for i in keep_mask]
            assistant_texts = [assistant_texts[i] for i in keep_mask]
            responses_list = [responses_list[i] for i in keep_mask]
            labels = [labels[i] for i in keep_mask]
            rankings_list = [rankings_list[i] for i in keep_mask]
            orig_response_counts = [orig_response_counts[i] for i in keep_mask]
            vision_budgets = [vision_budgets[i] for i in keep_mask]
            # Re-derive modality flags
            has_images = any(
                any(c.get("type") == "image" for c in msgs[0]["content"])
                for msgs in messages_list
            )
            has_videos = any(
                any(c.get("type") == "video" for c in msgs[0]["content"])
                for msgs in messages_list
            )

        # Step 1: Process text with apply_chat_template (tokenize=False)
        texts = []
        for messages in messages_list:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)

        # Step 2: Process vision info
        images, videos, video_kwargs = process_vision_info(messages_list)
        # Molmo2 does not accept pixel_values and pixel_values_videos together.
        if images is not None:
            videos = None
            video_kwargs = {}

        # ====================================================================
        # Phase 2: Adaptive vision token budget – reduce max_crops / frames
        # until vision tokens fit the remaining budget.
        # ====================================================================
        if vision_budget_enabled and (has_images or has_videos):
            result = _process_with_vision_budget(
                processor=processor,
                tokenizer=tokenizer,
                texts=texts,
                images=images,
                videos=videos,
                video_kwargs=video_kwargs,
                messages_list=messages_list,
                max_length=max_length,
                vision_budgets=vision_budgets,
                default_max_crops=default_max_crops,
                min_max_crops=min_max_crops,
                max_video_frames=max_video_frames,
                max_vision_tokens=max_vision_tokens,
            )
            batch_inputs = result[0]
            if batch_inputs is None:
                # Vision doesn't fit even at minimum — skip entire batch.
                # Return a minimal dummy batch so the DataLoader doesn't crash.
                # The training loop should check for this (batch_size == 0 or a flag).
                import warnings
                warnings.warn(
                    f"Skipping batch: vision tokens exceed budget even at "
                    f"max_crops={min_max_crops}. All {len(batch)} sample(s) dropped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dummy_ids = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long)
                return {
                    "input_ids": dummy_ids,
                    "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                    "resp_indices": torch.full((1, 1), -1, dtype=torch.long),
                    "resp_start_indices": torch.full((1, 1), -1, dtype=torch.long),
                    "labels": torch.tensor([0], dtype=torch.long),
                    "rankings": torch.full((1, 1), -1, dtype=torch.long),
                    "_skipped": torch.tensor([1], dtype=torch.long),
                }
        else:
            # Original path (no budget management or text-only after filtering)
            batch_inputs = processor(
                **_build_processor_kwargs(
                    texts, images=images, videos=videos,
                    video_kwargs=video_kwargs, max_length=max_length,
                )
            )
    else:
        # Text-only: use apply_chat_template to get text, then tokenize
        texts = []
        for messages in messages_list:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        
        # Use tokenizer to tokenize the text
        batch_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]

    # Only search for the special token itself, NOT the full separator text
    # (e.g. "\n\n<|resp_sep|>\n\n").  BPE context-dependent encoding can merge
    # the surrounding "\n\n" with adjacent characters (e.g. ".\n\n" → single token),
    # so the full multi-token pattern may never appear in input_ids.
    # Since <|resp_sep|> is a registered special token it always encodes to
    # exactly one unique token ID that cannot be produced by normal text.
    sep_tokens = tokenizer.encode(RESP_SEP_TOKEN, add_special_tokens=False)
    resp_indices_per_sample: List[List[int]] = []
    resp_start_indices_per_sample: List[List[int]] = []

    keep_indices: List[int] = []
    for idx, (ids, assistant_text, responses, orig_count) in enumerate(
        zip(input_ids, assistant_texts, responses_list, orig_response_counts)
    ):
        ids_list = ids.tolist()

        # Try to encode assistant_text the same way it appears in the full sequence
        # First try without special tokens, then with special tokens if that fails
        assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)

        def find_subseq(seq, subseq):
            """Find subsequence in sequence, with fuzzy matching for truncated cases."""
            if len(subseq) == 0:
                return 0
            if len(subseq) > len(seq):
                return -1
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i : i + len(subseq)] == subseq:
                    return i
            return -1

        # Try to find assistant tokens in the sequence
        start_idx = find_subseq(ids_list, assistant_tokens)
        
        # If not found, try to find a partial match (for truncated sequences)
        if start_idx == -1 and len(assistant_tokens) > 10:
            # Try matching the last part of assistant_tokens (more likely to be preserved)
            for partial_len in [len(assistant_tokens) // 2, len(assistant_tokens) // 4, 20, 10]:
                if partial_len < len(assistant_tokens):
                    partial_tokens = assistant_tokens[-partial_len:]
                    start_idx = find_subseq(ids_list, partial_tokens)
                    if start_idx != -1:
                        # Adjust to account for partial match
                        start_idx = start_idx + partial_len - len(assistant_tokens)
                        break
        if start_idx == -1:
            _collate_stats["start_idx_failures"] += 1
            # Fallback: try to find the first response's start token to anchor,
            # then use sep_tokens to split the rest (same separator-based approach).
            resp_indices: List[int] = []
            resp_start_indices: List[int] = []

            def find_subseq_from(seq, subseq, start_pos):
                for j in range(start_pos, len(seq) - len(subseq) + 1):
                    if seq[j : j + len(subseq)] == subseq:
                        return j
                return -1

            # Try to anchor on the first response's opening tokens
            first_resp_toks = tokenizer.encode(responses[0], add_special_tokens=False)
            anchor = -1
            if len(first_resp_toks) >= 6:
                anchor = find_subseq_from(ids_list, first_resp_toks[:6], 0)
            if anchor == -1 and len(first_resp_toks) >= 3:
                anchor = find_subseq_from(ids_list, first_resp_toks[:3], 0)

            if anchor != -1 and len(sep_tokens) > 0 and len(responses) > 1:
                # Use separator-based splitting from the anchor point
                region = ids_list[anchor:]
                sep_len = len(sep_tokens)
                sep_positions = []
                for j in range(len(region) - sep_len + 1):
                    if region[j : j + sep_len] == sep_tokens:
                        sep_positions.append(j)

                expected_seps = len(responses) - 1
                used_seps = sep_positions[:expected_seps]
                seg_start = 0

                for i in range(len(responses)):
                    abs_start = anchor + seg_start
                    if abs_start >= len(ids_list):
                        _collate_stats["cursor_out_of_bounds"] += 1
                        break
                    if i < len(used_seps):
                        abs_end = anchor + used_seps[i] - 1
                        next_seg_start = used_seps[i] + sep_len
                    else:
                        abs_end = len(ids_list) - 1
                        next_seg_start = len(ids_list)

                    abs_end = min(abs_end, len(ids_list) - 1)
                    if abs_end < abs_start:
                        _collate_stats["end_idx_invalid"] += 1
                        seg_start = next_seg_start
                        continue

                    resp_start_indices.append(abs_start)
                    resp_indices.append(abs_end)
                    seg_start = next_seg_start

            if resp_indices:
                resp_indices_per_sample.append(resp_indices)
                resp_start_indices_per_sample.append(resp_start_indices)
                keep_indices.append(idx)
                continue  # move to next sample

            # Debug: print info then fall back to a safe default instead of crashing
            print(f"Warning: Could not locate assistant responses in input_ids for sample {idx}.")
            print(f"  input_ids length: {len(ids_list)}")
            print(f"  assistant_tokens length: {len(assistant_tokens)}")
            print(f"  First 20 input_ids: {ids_list[:20]}")
            print(f"  First 20 assistant_tokens: {assistant_tokens[:20]}")
            print(f"  Last 20 input_ids: {ids_list[-20:]}")
            print(f"  Last 20 assistant_tokens: {assistant_tokens[-20:]}")
            # Fallback: use the last token as the only response index
            resp_indices_per_sample.append([len(ids_list) - 1])
            resp_start_indices_per_sample.append([len(ids_list) - 1])
            keep_indices.append(idx)  # Don't forget to add to keep_indices!
            continue

        resp_indices: List[int] = []
        resp_start_indices: List[int] = []
        seq_len = len(ids_list)

        # Ensure start_idx is valid
        if start_idx >= seq_len:
            start_idx = seq_len - 1

        # ------------------------------------------------------------------
        # Split the assistant region of input_ids by searching for sep_tokens
        # to find exact boundaries of each response in the actual token stream.
        # This avoids cumulative offset errors from independent re-encoding.
        # ------------------------------------------------------------------
        num_responses = len(responses)
        assistant_region = ids_list[start_idx:]

        if num_responses <= 1 or len(sep_tokens) == 0:
            # Single response (shouldn't happen for reward data) or no separator:
            # the whole assistant region is one response.
            resp_start_indices.append(start_idx)
            resp_indices.append(seq_len - 1)
        else:
            # Find all occurrences of sep_tokens in the assistant region
            sep_positions = []  # positions relative to start_idx
            sep_len = len(sep_tokens)
            for j in range(len(assistant_region) - sep_len + 1):
                if assistant_region[j : j + sep_len] == sep_tokens:
                    sep_positions.append(j)

            # We expect (num_responses - 1) separators.
            # Use the first (num_responses - 1) found positions.
            expected_seps = num_responses - 1

            # Build response boundaries from separator positions
            # response_0: [start_idx, sep_pos_0 - 1]
            # response_1: [sep_pos_0 + sep_len, sep_pos_1 - 1]
            # ...
            # response_N: [sep_pos_{N-1} + sep_len, end_of_assistant_region]
            seg_start = 0  # relative to start_idx
            used_seps = sep_positions[:expected_seps]

            for i in range(num_responses):
                abs_start = start_idx + seg_start
                if abs_start >= seq_len:
                    _collate_stats["cursor_out_of_bounds"] += 1
                    break

                if i < len(used_seps):
                    # End of this response is right before the separator
                    abs_end = start_idx + used_seps[i] - 1
                    next_seg_start = used_seps[i] + sep_len
                else:
                    # Last response (or separators exhausted): extends to end
                    abs_end = seq_len - 1
                    next_seg_start = seq_len  # will stop the loop

                # Clamp to valid range
                abs_end = min(abs_end, seq_len - 1)
                if abs_end < abs_start:
                    _collate_stats["end_idx_invalid"] += 1
                    # This response segment is empty (truncated), skip it
                    seg_start = next_seg_start
                    continue

                resp_start_indices.append(abs_start)
                resp_indices.append(abs_end)
                seg_start = next_seg_start

        # Ensure we have at least one valid index
        if not resp_indices:
            # Fallback: use the last valid token position
            resp_indices = [min(seq_len - 1, start_idx)]
            resp_start_indices = [min(seq_len - 1, start_idx)]
            _collate_stats["fallback_to_last_token"] += 1
        
        # Final validation: ensure all indices are in valid range
        resp_indices = [min(max(idx, 0), seq_len - 1) for idx in resp_indices]
        resp_start_indices = [min(max(idx, 0), seq_len - 1) for idx in resp_start_indices]
        
        # If truncated responses are detected and skipping is enabled, drop the sample
        if skip_truncated and len(resp_indices) < orig_count:
            _collate_stats["samples_skipped"] += 1
            continue

        resp_indices_per_sample.append(resp_indices)
        resp_start_indices_per_sample.append(resp_start_indices)
        keep_indices.append(idx)

    # If we skipped samples, filter all parallel lists accordingly
    if len(keep_indices) != len(messages_list):
        if not keep_indices:
            raise ValueError("All samples were skipped due to truncation.")
        messages_list = [messages_list[i] for i in keep_indices]
        assistant_texts = [assistant_texts[i] for i in keep_indices]
        responses_list = [responses_list[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
        rankings_list = [rankings_list[i] for i in keep_indices]

    # pad resp_indices to the max number of responses with -1
    max_resp = max(len(r) for r in resp_indices_per_sample)
    resp_indices_tensor = torch.full(
        (len(resp_indices_per_sample), max_resp), -1, dtype=torch.long
    )
    for i, resp_idx_list in enumerate(resp_indices_per_sample):
        resp_indices_tensor[i, : len(resp_idx_list)] = torch.tensor(resp_idx_list, dtype=torch.long)
    # pad resp_start_indices to the max number of responses with -1
    resp_start_indices_tensor = torch.full(
        (len(resp_start_indices_per_sample), max_resp), -1, dtype=torch.long
    )
    for i, resp_idx_list in enumerate(resp_start_indices_per_sample):
        resp_start_indices_tensor[i, : len(resp_idx_list)] = torch.tensor(resp_idx_list, dtype=torch.long)
    
    # Fix labels and rankings when some responses were truncated away.
    # If the best response (label) was truncated, we need to re-derive the label
    # from rankings (find the best among surviving responses), not just clamp.
    fixed_labels = []
    fixed_rankings_list = []
    for i, (label, rankings, resp_idx_list) in enumerate(
        zip(labels, rankings_list, resp_indices_per_sample)
    ):
        num_valid = len(resp_idx_list)
        if num_valid == 0:
            fixed_labels.append(0)
            fixed_rankings_list.append(None)
            continue

        # Truncate rankings to the surviving responses
        trunc_rankings = rankings[:num_valid] if rankings is not None and len(rankings) >= num_valid else None

        if label < num_valid:
            # Best response survived — label is still valid
            fixed_labels.append(label)
        elif trunc_rankings is not None:
            # Best response was truncated away — find the best among survivors
            # Best = lowest ranking value (1 = best)
            best_among_valid = int(min(range(num_valid), key=lambda j: trunc_rankings[j]))
            fixed_labels.append(best_among_valid)
            _collate_stats["label_remapped_from_rankings"] = _collate_stats.get("label_remapped_from_rankings", 0) + 1
        else:
            # No rankings available, fall back to 0 (arbitrary but at least valid)
            fixed_labels.append(0)
            _collate_stats["label_clamped_no_rankings"] = _collate_stats.get("label_clamped_no_rankings", 0) + 1

        fixed_rankings_list.append(trunc_rankings)

    batch_inputs["resp_indices"] = resp_indices_tensor
    batch_inputs["resp_start_indices"] = resp_start_indices_tensor
    batch_inputs["labels"] = torch.tensor(fixed_labels, dtype=torch.long)

    # Process rankings: pad to max_resp with -1 (invalid)
    rankings_tensor = torch.full(
        (len(resp_indices_per_sample), max_resp), -1, dtype=torch.long
    )
    for i, (rankings, resp_idx_list) in enumerate(zip(fixed_rankings_list, resp_indices_per_sample)):
        if rankings is not None and len(rankings) == len(resp_idx_list):
            rankings_tensor[i, : len(resp_idx_list)] = torch.tensor(
                rankings[:len(resp_idx_list)], dtype=torch.long
            )
        # If rankings is None, leave as -1 (will be ignored in loss)
    batch_inputs["rankings"] = rankings_tensor

    # If we skipped samples (keep_indices shorter than original), slice batch_inputs tensors accordingly
    original_batch_size = input_ids.shape[0]
    if len(keep_indices) != original_batch_size:
        idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
        # Filter input_ids and attention_mask first
        input_ids = input_ids.index_select(0, idx_tensor)
        attention_mask = attention_mask.index_select(0, idx_tensor)
        # Filter other tensors in batch_inputs
        for k, v in list(batch_inputs.items()):
            if torch.is_tensor(v) and v.shape[0] == original_batch_size:
                batch_inputs[k] = v.index_select(0, idx_tensor)
    
    # Set the (possibly filtered) tensors
    batch_inputs["input_ids"] = input_ids
    batch_inputs["attention_mask"] = attention_mask
    return batch_inputs

