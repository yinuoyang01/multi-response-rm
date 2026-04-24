"""
Simple training loop for multi-response reward:
- Build <image><query><responses> with processor.apply_chat_template
- Take hidden states at each response end token
- Pass through scalar value head for scores
- Train LoRA adapter + value head with cross-entropy over responses
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from mr2rm.data.dataset import (
    MultiResponseRewardDataset,
    ModalityBatchSampler,
    collate_multi_response_reward,
    print_collate_stats,
    add_resp_sep_token,
)
from mr2rm.models.reward_model import MultiResponseRewardModel


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    step_in_epoch: int,
    output_dir: Path,
    is_main_process: bool,
    use_distributed: bool,
    processor=None,
):
    """Save training checkpoint."""
    if not is_main_process:
        return

    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    model_to_save = model.module if use_distributed else model

    # Save LoRA adapter
    model_to_save.base_model.save_pretrained(checkpoint_dir)

    # Save value head separately
    torch.save(model_to_save.value_head.state_dict(), checkpoint_dir / "value_head.pt")

    # Save reward model config for inference/merge
    reward_model_config = {
        "value_head_type": model_to_save.value_head_type,
        "value_head_hidden_dim": model_to_save.value_head_hidden_dim,
        "value_head_activation": model_to_save.value_head_activation,
        "value_head_input_dim": model_to_save.value_head_input_dim,
        "resp_repr_mode": model_to_save.resp_repr_mode,
    }
    with open(checkpoint_dir / "reward_model_config.json", "w") as f:
        json.dump(reward_model_config, f, indent=2)

    # Save optimizer and scheduler state
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Save training state (include step_in_epoch for proper resume)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "step_in_epoch": step_in_epoch,
    }
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(training_state, f)

    # Save RNG states for reproducible resume
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, checkpoint_dir / "rng_state.pt")

    # Save processor if provided
    if processor is not None:
        try:
            if not hasattr(processor, "audio_tokenizer"):
                processor.audio_tokenizer = None
            processor.save_pretrained(checkpoint_dir)
        except Exception as e:
            print(f"⚠️ Failed to save processor: {e}")

    # Save WandB run ID for resume capability
    try:
        import wandb
        if wandb.run is not None:
            wandb_run_id_file = checkpoint_dir / "wandb_run_id.txt"
            wandb_run_id_file.write_text(wandb.run.id)
    except Exception:
        pass  # WandB not available or not initialized

    print(f"💾 Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    model,
    optimizer,
    scheduler,
    device,
    use_distributed: bool,
    is_main_process: bool,
):
    """Load training checkpoint and return epoch, global_step, step_in_epoch."""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load training state
    training_state_path = checkpoint_path / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
        epoch = training_state.get("epoch", 0)
        global_step = training_state.get("global_step", 0)
        step_in_epoch = training_state.get("step_in_epoch", 0)
    else:
        # Fallback: try to infer from checkpoint name
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.startswith("checkpoint-"):
            global_step = int(checkpoint_name.split("-")[1])
            epoch = 0  # Unknown, will need to be set manually
            step_in_epoch = 0
        else:
            epoch = 0
            global_step = 0
            step_in_epoch = 0

    if is_main_process:
        print(f"📂 Loading checkpoint from {checkpoint_dir}")
        print(f"   Resuming from epoch {epoch}, global_step {global_step}, step_in_epoch {step_in_epoch}")

    # Load model (LoRA adapter)
    # Note: base model should already be loaded with PEFT, we need to load the adapter weights
    model_to_load = model.module if use_distributed else model
    try:
        from peft import PeftModel, set_peft_model_state_dict
        # Check if checkpoint contains adapter config
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            # Load adapter weights directly using safetensors/bin files
            adapter_weights_path = checkpoint_path / "adapter_model.safetensors"
            if not adapter_weights_path.exists():
                adapter_weights_path = checkpoint_path / "adapter_model.bin"

            if adapter_weights_path.exists():
                if str(adapter_weights_path).endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapter_state_dict = load_file(str(adapter_weights_path))
                else:
                    adapter_state_dict = torch.load(str(adapter_weights_path), map_location=device)

                # Load adapter weights into model
                set_peft_model_state_dict(model_to_load.base_model, adapter_state_dict)
                if is_main_process:
                    print(f"✅ Loaded adapter weights from {adapter_weights_path}")
            else:
                if is_main_process:
                    print(f"⚠️ No adapter weights found in checkpoint")
        else:
            # Not a PEFT checkpoint
            if is_main_process:
                print(f"⚠️ No adapter_config.json found in checkpoint")
    except Exception as e:
        if is_main_process:
            print(f"⚠️ Failed to load adapter: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Continuing with current model state...")

    # Load value head
    value_head_path = checkpoint_path / "value_head.pt"
    if value_head_path.exists():
        model_to_load.value_head.load_state_dict(
            torch.load(value_head_path, map_location=device)
        )
        if is_main_process:
            print(f"✅ Loaded value head from {value_head_path}")

    # Load optimizer and scheduler
    optimizer_path = checkpoint_path / "optimizer.pt"
    scheduler_path = checkpoint_path / "scheduler.pt"

    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        if is_main_process:
            print(f"✅ Loaded optimizer state")

    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        if is_main_process:
            print(f"✅ Loaded scheduler state")

    # Load RNG states for reproducible resume
    rng_state_path = checkpoint_path / "rng_state.pt"
    if rng_state_path.exists():
        # weights_only=False needed because RNG state contains numpy arrays
        rng_state = torch.load(rng_state_path, map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        if is_main_process:
            print(f"✅ Loaded RNG states for reproducible resume")

    if is_main_process:
        print(f"✅ Checkpoint loaded successfully")

    return epoch, global_step, step_in_epoch


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the output directory."""
    if not output_dir.exists():
        return None

    checkpoints = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Sort by step number (checkpoint-{step})
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0, reverse=True)
    return checkpoints[0]


def find_latest_matching_output_dir(base_dir: Path, pattern: str, verbose: bool = True) -> Optional[Path]:
    """
    Find the latest output directory matching a pattern.

    Args:
        base_dir: The base directory to search in (e.g., reward_models/)
        pattern: A shell-style pattern to match run-name prefixes (e.g., "molmo2-4b_mr2rm_*")
        verbose: Whether to print debug information

    Returns:
        The matching directory, or None if not found.
        If multiple matches, returns the most recently modified one.
    """
    import fnmatch

    if verbose:
        print(f"🔍 [Resume] Searching base_dir={base_dir}, pattern='{pattern}'")

    if not base_dir.exists():
        if verbose:
            print(f"🔍 [Resume] base_dir does not exist!")
        return None

    matching_dirs = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        if fnmatch.fnmatch(d.name, pattern):
            matching_dirs.append(d)

    if verbose:
        print(f"🔍 [Resume] Found {len(matching_dirs)} matching dirs")
        if matching_dirs:
            print(f"🔍 [Resume] Matches: {[d.name for d in matching_dirs]}")

    if not matching_dirs:
        if verbose:
            print(f"🔍 [Resume] No matching directories found")
        return None

    # Sort by modification time, most recent first
    matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if verbose:
        print(f"🔍 [Resume] Selected: {matching_dirs[0]}")
    return matching_dirs[0]


def cleanup_old_checkpoints(output_dir: Path, save_total_limit: int, is_main_process: bool = True):
    """Remove old checkpoints, keeping only the most recent N."""
    if save_total_limit <= 0:
        return
    
    # Only main process should delete checkpoints to avoid race conditions
    if not is_main_process:
        return
    
    checkpoints = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
        reverse=True
    )
    
    if len(checkpoints) > save_total_limit:
        for checkpoint in checkpoints[save_total_limit:]:
            try:
                print(f"🗑️  Removing old checkpoint: {checkpoint.name}")
                shutil.rmtree(checkpoint)
            except FileNotFoundError:
                # File already deleted by another process or doesn't exist - ignore
                pass
            except Exception as e:
                # Log other errors but don't crash
                print(f"⚠️  Failed to remove checkpoint {checkpoint.name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data file (JSON/JSONL), or comma-separated list of paths, or directory containing JSON/JSONL files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, default=None,
                        help="Base directory for resolving relative image paths in samples")
    parser.add_argument("--video_base_dir", type=str, default=None,
                        help="Base directory for resolving relative video paths in samples")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="*", default=None,
                        help="If set, override default LoRA target modules.")
    # Shuffle parameter: default is True (shuffle enabled)
    # Use --no-shuffle to disable shuffling
    shuffle_group = parser.add_mutually_exclusive_group()
    shuffle_group.add_argument("--shuffle", action="store_true", default=True,
                               help="Shuffle training data (default: True)")
    shuffle_group.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                               help="Disable shuffling of training data")
    # Gradient checkpointing to save memory (trades compute for memory)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: True)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", 
                        action="store_false", help="Disable gradient checkpointing")
    # WandB logging (optional)
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Enable WandB logging if WANDB_API_KEY is set")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (fallback to env WANDB_PROJECT)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--skip_truncated", action="store_true", default=False,
                        help="Drop samples that would be truncated to fit max_length")
    # Vision token budget management
    parser.add_argument("--vision_budget", action="store_true", default=True,
                        help="Enable adaptive vision token budget management (default: True)")
    parser.add_argument("--no-vision-budget", dest="vision_budget", action="store_false",
                        help="Disable vision token budget management")
    parser.add_argument("--default_max_crops", type=int, default=8,
                        help="Default max_crops for image processing (default: 8)")
    parser.add_argument("--min_max_crops", type=int, default=1,
                        help="Minimum max_crops to try before skipping (default: 1)")
    parser.add_argument("--vision_budget_safety_margin", type=int, default=256,
                        help="Safety margin (tokens) for vision budget calculation (default: 256)")
    parser.add_argument("--max_vision_tokens", type=int, default=None,
                        help="Hard cap on vision tokens. When set, replaces budget estimation "
                             "with a simple upper limit (e.g., 4096). Recommended to set this "
                             "to prevent truncation of text responses.")
    # Value head config
    parser.add_argument("--value_head_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Value head type: linear (default) or mlp")
    parser.add_argument("--value_head_hidden_dim", type=int, default=None,
                        help="Hidden dimension for MLP value head (default: hidden_size)")
    parser.add_argument("--value_head_activation", type=str, default="selu",
                        choices=["selu", "gelu", "relu", "tanh", "silu"],
                        help="Activation function for MLP value head (default: selu)")
    parser.add_argument("--resp_repr_mode", type=str, default="last",
                        choices=["last", "first", "first_last_concat", "first_last_add", "first_last_sub", "response_mean"],
                        help="How to build response representation: last token (default), combine first+last token, or average all response tokens (response_mean)")
    # Checkpoint and resume
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: save only at end)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep (oldest will be deleted)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Automatically find and resume from the latest checkpoint in output_dir")
    parser.add_argument("--resume_pattern", type=str, default=None,
                        help="Shell-style pattern to find the latest matching run directory for resume. "
                             "E.g., 'molmo2-4b_mr2rm_*' will pick the most recent "
                             "YYYY-MM-DD_molmo2-4b_mr2rm_... directory under --resume_base_dir.")
    parser.add_argument("--resume_base_dir", type=str, default=None,
                        help="Base directory to search for matching output dirs (used with --resume_pattern)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed training if available
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    
    use_distributed = local_rank != -1 and world_size > 1
    
    if use_distributed:
        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = rank == 0
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        is_main_process = True
        local_rank = 0
        rank = 0
        world_size = 1

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    if is_main_process:
        print(f"Loading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Register <|resp_sep|> as a special token so it encodes to a single unique ID.
    # This is done before model loading so resize_token_embeddings can be called.
    num_added = add_resp_sep_token(tokenizer)
    if is_main_process and num_added > 0:
        print(f"✅ Added {num_added} special token(s): <|resp_sep|> -> id {tokenizer.convert_tokens_to_ids('<|resp_sep|>')}")

    # WandB will be initialized later after we determine if resuming
    use_wandb = args.use_wandb and os.environ.get("WANDB_API_KEY")
    wandb_initialized = False  # Track if WandB has been initialized

    if is_main_process:
        print("Loading dataset...")
    train_dataset = MultiResponseRewardDataset(
        args.train_data_path,
        image_base_dir=args.image_base_dir,
        video_base_dir=args.video_base_dir,
    )

    collate_fn = partial(
        collate_multi_response_reward,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        skip_truncated=args.skip_truncated,
        default_max_crops=args.default_max_crops,
        min_max_crops=args.min_max_crops,
        vision_budget_enabled=args.vision_budget,
        vision_budget_safety_margin=args.vision_budget_safety_margin,
        max_vision_tokens=args.max_vision_tokens,
    )

    # Use ModalityBatchSampler to ensure each batch has only one modality
    # This prevents mixing image/video samples in the same batch
    modality_batch_sampler = ModalityBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        modality_weights=None,  # Uniform sampling from available modalities
    )
    
    # For distributed training, we need to filter batches by rank
    if use_distributed:
        # Create a distributed-aware batch sampler
        class DistributedModalityBatchSampler:
            def __init__(self, batch_sampler, num_replicas, rank):
                self.batch_sampler = batch_sampler
                self.num_replicas = num_replicas
                self.rank = rank
                self.epoch = 0
            
            def __iter__(self):
                # Rebuild batches for new epoch
                self.batch_sampler.set_epoch(self.epoch)
                all_batches = list(self.batch_sampler)
                
                # Distribute batches across ranks - ALL RANKS GET SAME NUMBER
                # This is critical for DDP: if ranks have different batch counts,
                # the rank that finishes early will exit the loop while others
                # are still waiting for gradient sync in backward(), causing timeout.
                batches_per_rank = len(all_batches) // self.num_replicas
                start_idx = self.rank * batches_per_rank
                end_idx = start_idx + batches_per_rank  # Same for ALL ranks (drop remainder)
                
                for batch_indices in all_batches[start_idx:end_idx]:
                    yield batch_indices
            
            def __len__(self):
                # Use len() instead of list() to avoid side effects
                # list(batch_sampler) triggers __iter__ which reshuffles batches!
                total_batches = len(self.batch_sampler)
                batches_per_rank = total_batches // self.num_replicas
                return batches_per_rank
            
            def set_epoch(self, epoch):
                self.epoch = epoch
        
        batch_sampler = DistributedModalityBatchSampler(modality_batch_sampler, world_size, rank)
        sampler = None
    else:
        sampler = None
        batch_sampler = modality_batch_sampler

    # DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler
    # Use num_workers=0 for multimodal data to avoid issues with image/video loading in multiprocessing
    # For multimodal data, loading in main process is safer (avoids pickling issues)
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for multimodal data to avoid pickling issues
            pin_memory=True if torch.cuda.is_available() else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for multimodal data to avoid pickling issues
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    if is_main_process:
        print(f"✅ DataLoader created. Dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")

    if is_main_process:
        print("Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )

    # Resize embeddings if we added special tokens (e.g., <|resp_sep|>)
    # Molmo2Embedding doesn't have a .weight attribute, so we can't use the
    # standard resize_token_embeddings. Instead, handle it manually.
    input_embeddings = base_model.get_input_embeddings()
    if hasattr(input_embeddings, 'new_embedding') and hasattr(input_embeddings, 'embedding'):
        current_size = input_embeddings.embedding.shape[0] + input_embeddings.new_embedding.shape[0]
        new_size = len(tokenizer)
        if new_size > current_size:
            extra = new_size - current_size
            pad = torch.zeros(extra, input_embeddings.new_embedding.shape[1],
                              dtype=input_embeddings.new_embedding.dtype,
                              device=input_embeddings.new_embedding.device)
            input_embeddings.new_embedding = torch.nn.Parameter(
                torch.cat([input_embeddings.new_embedding, pad], dim=0)
            )
            if is_main_process:
                print(f"✅ Expanded Molmo2Embedding new_embedding by {extra} tokens (total: {new_size})")
        elif new_size < current_size:
            # Shrink: trim from the end of new_embedding
            trim = current_size - new_size
            input_embeddings.new_embedding = torch.nn.Parameter(
                input_embeddings.new_embedding[:-trim]
            )
            if is_main_process:
                print(f"✅ Shrunk Molmo2Embedding new_embedding by {trim} tokens (total: {new_size})")
        else:
            if is_main_process:
                print(f"✅ Embedding size already matches tokenizer ({current_size}), no resize needed")
    else:
        base_model.resize_token_embeddings(len(tokenizer))

    # Patch Molmo2 build_input_embeddings to avoid in-place op on a leaf tensor
    # Original code does: x.view(...)[is_image_patch] += image_features (in-place)
    # This triggers: "a view of a leaf Variable that requires grad is being used in an in-place operation."
    # We replace it with an out-of-place update.
    try:
        import types

        def patched_build_input_embeddings(self, input_ids, images=None, token_pooling=None):
            # Same logic as upstream, but avoid in-place on view
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            x = self.transformer.wte(input_ids)

            image_features = None
            if images is not None:
                image_features = self.vision_backbone(images, token_pooling).to(x.device)
                is_image_patch = input_ids.view(-1) == self.config.image_patch_id
                assert is_image_patch.sum() == len(image_features)
                x_flat = x.view(-1, x.shape[-1]).clone()
                x_flat[is_image_patch] = x_flat[is_image_patch] + image_features
                x = x_flat.view_as(x)

            x = self.transformer.emb_drop(x)  # type: ignore
            return x, image_features

        if hasattr(base_model, "model") and hasattr(base_model.model, "build_input_embeddings"):
            base_model.model.build_input_embeddings = types.MethodType(
                patched_build_input_embeddings, base_model.model
            )
            if is_main_process:
                print("✅ Patched build_input_embeddings to avoid in-place ops.")
    except Exception as e:
        if is_main_process:
            print(f"⚠️ Failed to patch build_input_embeddings: {e}")

    # Enable gradient checkpointing to reduce memory usage
    if args.gradient_checkpointing:
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
            if is_main_process:
                print("✅ Gradient checkpointing enabled")
        else:
            if is_main_process:
                print("⚠️  Model does not support gradient_checkpointing_enable()")

    # LoRA adaptation
    if is_main_process:
        print("🔧 Using LoRA mode")

    target_modules = args.lora_target_modules
    if target_modules is None:
        # Molmo2 module names: att_proj (q/k/v combined), attn_out (o_proj), ff_proj, ff_out
        target_modules = ["att_proj", "attn_out", "ff_proj", "ff_out"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    base_model = get_peft_model(base_model, lora_config)

    model = MultiResponseRewardModel(
        base_model=base_model,
        value_head_type=args.value_head_type,
        value_head_hidden_dim=args.value_head_hidden_dim,
        value_head_activation=args.value_head_activation,
        resp_repr_mode=args.resp_repr_mode,
    ).to(device)

    # Wrap with DDP if distributed
    if use_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # Get trainable parameters (unwrap DDP if needed)
    model_for_params = model.module if use_distributed else model
    trainable_params = [p for p in model_for_params.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    start_step_in_epoch = 0  # Track which step to resume from within an epoch
    resumed_output_dir = None  # Track if we're resuming from a different output dir

    # Handle --resume_pattern: find latest matching output dir (ignoring date prefix)
    checkpoint_to_resume = args.resume_from_checkpoint
    if args.resume_pattern and not checkpoint_to_resume:
        # Determine base directory for pattern search
        if args.resume_base_dir:
            base_dir = Path(args.resume_base_dir)
        else:
            # Default: parent of output_dir (e.g., reward_models/)
            base_dir = Path(args.output_dir).parent

        matching_output_dir = find_latest_matching_output_dir(base_dir, args.resume_pattern, verbose=is_main_process)
        if matching_output_dir:
            if is_main_process:
                print(f"🔍 Found matching output dir: {matching_output_dir}")
            # Find latest checkpoint in that directory
            latest_checkpoint = find_latest_checkpoint(matching_output_dir)
            if latest_checkpoint:
                checkpoint_to_resume = str(latest_checkpoint)
                resumed_output_dir = matching_output_dir
                if is_main_process:
                    print(f"🔍 Found latest checkpoint: {checkpoint_to_resume}")
            else:
                if is_main_process:
                    print(f"🔍 No checkpoint in {matching_output_dir}, starting from scratch")
        else:
            if is_main_process:
                print(f"🔍 No matching output dir for pattern '{args.resume_pattern}', starting from scratch")

    # Handle --resume flag: automatically find latest checkpoint in current output_dir
    elif args.resume and not checkpoint_to_resume:
        latest_checkpoint = find_latest_checkpoint(Path(args.output_dir))
        if latest_checkpoint:
            checkpoint_to_resume = str(latest_checkpoint)
            if is_main_process:
                print(f"🔍 Found latest checkpoint: {checkpoint_to_resume}")
        else:
            if is_main_process:
                print(f"🔍 No checkpoint found in {args.output_dir}, starting from scratch")

    if checkpoint_to_resume:
        if is_main_process:
            print(f"🔄 Resuming from checkpoint: {checkpoint_to_resume}")
        start_epoch, global_step, start_step_in_epoch = load_checkpoint(
            checkpoint_to_resume,
            model,
            optimizer,
            scheduler,
            device,
            use_distributed,
            is_main_process,
        )
        # Adjust total_steps calculation if resuming
        remaining_steps = total_steps - global_step
        if is_main_process:
            print(f"   Remaining steps: {remaining_steps}")

    # Initialize WandB (after determining if resuming)
    if use_wandb and is_main_process:
        try:
            import wandb
            wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "molmo2-multi-response-reward")
            wandb_run_name = args.wandb_run_name

            # If resuming from checkpoint, try to resume the WandB run
            if checkpoint_to_resume and global_step > 0:
                # Try to find wandb run id from checkpoint
                wandb_id_file = Path(checkpoint_to_resume) / "wandb_run_id.txt"
                if wandb_id_file.exists():
                    wandb_run_id = wandb_id_file.read_text().strip()
                    wandb.init(project=wandb_project, name=wandb_run_name, id=wandb_run_id, resume="must")
                    print(f"✅ WandB resumed | project={wandb_project} run_id={wandb_run_id}")
                else:
                    # No run id saved, create new run but log that we're resuming
                    wandb.init(project=wandb_project, name=wandb_run_name, resume="allow")
                    print(f"✅ WandB initialized (resume mode) | project={wandb_project} run={wandb_run_name}")
            else:
                wandb.init(project=wandb_project, name=wandb_run_name)
                print(f"✅ WandB initialized | project={wandb_project} run={wandb_run_name}")

            wandb.config.update(vars(args))
            wandb_initialized = True
        except Exception as e:
            print(f"⚠️ Failed to initialize WandB: {e}")
            use_wandb = False

    if is_main_process:
        print(f"\n{'='*80}")
        print(f"📋 Training Parameters")
        print(f"{'='*80}")
        print(f"Model Configuration:")
        print(f"  Base model: {args.base_model_path}")
        print(f"  Value head type: {args.value_head_type}")
        if args.value_head_hidden_dim:
            print(f"  Value head hidden dim: {args.value_head_hidden_dim}")
        if args.value_head_type == "mlp":
            print(f"  Value head activation: {args.value_head_activation}")
        print(f"  Response repr mode: {args.resp_repr_mode}")
        print(f"  LoRA r: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  LoRA dropout: {args.lora_dropout}")
        if args.lora_target_modules:
            print(f"  LoRA target modules: {args.lora_target_modules}")
        print(f"\nData Configuration:")
        print(f"  Train data: {args.train_data_path}")
        print(f"  Dataset size: {len(train_dataset)}")
        print(f"  Image base dir: {args.image_base_dir or 'None'}")
        print(f"  Video base dir: {args.video_base_dir or 'None'}")
        print(f"  Max length: {args.max_length}")
        print(f"  Shuffle: {args.shuffle}")
        print(f"  Skip truncated: {args.skip_truncated}")
        print(f"  Vision budget: {args.vision_budget}")
        print(f"  Default max_crops: {args.default_max_crops}")
        print(f"  Min max_crops: {args.min_max_crops}")
        print(f"  Vision budget safety margin: {args.vision_budget_safety_margin}")
        print(f"  Max vision tokens: {args.max_vision_tokens or 'None (use budget estimation)'}")
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Num epochs: {args.num_epochs}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Max grad norm: {args.max_grad_norm}")
        print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"\nLoss: cross-entropy (select best response)")
        print(f"\nOptimization:")
        print(f"  Optimizer: AdamW")
        print(f"  Scheduler: Linear with warmup")
        print(f"  Dtype: {args.dtype}")
        print(f"\nDistributed Training:")
        if use_distributed:
            print(f"  Enabled: Yes")
            print(f"  World size: {world_size}")
            print(f"  Rank: {rank}")
            print(f"  Local rank: {local_rank}")
        else:
            print(f"  Enabled: No")
        print(f"\nCheckpointing:")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Save steps: {args.save_steps or 'Only at end'}")
        print(f"  Save total limit: {args.save_total_limit}")
        print(f"\nResume Configuration:")
        print(f"  --resume flag: {args.resume}")
        print(f"  --resume_pattern: {args.resume_pattern or 'None'}")
        print(f"  --resume_base_dir: {args.resume_base_dir or 'None'}")
        print(f"  --resume_from_checkpoint: {args.resume_from_checkpoint or 'None'}")
        if checkpoint_to_resume:
            print(f"  ✅ Resuming from: {checkpoint_to_resume}")
            print(f"  ✅ Resuming from epoch {start_epoch}, global_step {global_step}, step_in_epoch {start_step_in_epoch}")
        else:
            print(f"  ⚠️ No checkpoint to resume from, starting from scratch")
        print(f"\nWandB:")
        print(f"  Enabled: {use_wandb}")
        if use_wandb:
            print(f"  Project: {args.wandb_project or os.environ.get('WANDB_PROJECT', 'N/A')}")
            print(f"  Run name: {args.wandb_run_name or 'N/A'}")
        print(f"{'='*80}\n")
        
        print(f"Starting training: {total_steps} total steps, {warmup_steps} warmup steps")

    # GPU time tracking
    train_start_time = time.time()
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))

    model.train()
    accum_loss = 0.0  # sum of raw losses over grad_accum steps for logging
    accum_count = 0   # number of non-skipped steps in current accumulation window
    accum_accuracy = 0.0  # sum of accuracies over grad_accum steps
    accum_num_pairs = 0   # sum of num_pairs over grad_accum steps
    micro_step = 0  # counts actual non-skipped forward passes (for gradient accumulation)
    for epoch in range(start_epoch, args.num_epochs):
        if is_main_process:
            print(f"📊 Starting epoch {epoch}/{args.num_epochs}")
        if use_distributed:
            if sampler is not None:
                sampler.set_epoch(epoch)
            elif batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
                batch_sampler.set_epoch(epoch)
        elif batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)

        # NOTE: Removed dist.barrier() here - DDP handles synchronization automatically
        # during backward(). Adding explicit barriers can cause timeouts when ranks
        # have different processing speeds or data loading times.

        # Determine which step to start from in this epoch (for resume)
        epoch_start_step = start_step_in_epoch if epoch == start_epoch else 0
        if epoch_start_step > 0 and is_main_process:
            print(f"⏭️  Skipping first {epoch_start_step} steps (already trained)")

        if is_main_process:
            print("🔄 Starting training loop...")
        step = -1  # Initialize so end-of-epoch check is safe if epoch has no data
        for step, batch in enumerate(train_loader):
            # Skip steps that were already trained (for resume)
            if step < epoch_start_step:
                continue

            # NO barrier here! DDP syncs gradients automatically during backward.
            # Adding barrier causes timeout when ranks have different processing speeds.

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Skip batches dropped by vision budget (all samples too long).
            # IMPORTANT: In DDP we cannot simply `continue` because other ranks
            # may still do backward() on their batch, and DDP requires ALL ranks
            # to participate in every backward pass for gradient sync.  Instead
            # we do a zero-loss forward+backward so the AllReduce still happens.
            if batch.pop("_skipped", None) is not None:
                if use_distributed:
                    # Create a zero-loss that touches model parameters so DDP syncs
                    _zero = sum(p.sum() * 0 for p in model.parameters() if p.requires_grad)
                    _zero.backward()
                # No optimizer step or logging for skipped batches
                continue

            labels = batch.pop("labels")
            resp_indices = batch.pop("resp_indices")
            resp_start_indices = batch.pop("resp_start_indices", None)  # [B, R] start token indices
            rankings = batch.pop("rankings", None)  # [B, R] with -1 for invalid

            if step == 0 and is_main_process:
                print(f"✅ First batch loaded! input_ids shape: {batch.get('input_ids', torch.tensor([])).shape}")
                print(f"   resp_indices shape: {resp_indices.shape}, resp_start_indices: {resp_start_indices.shape if resp_start_indices is not None else 'None'}")
            
            # Pass rankings to model for debugging output
            # resp_start_indices is required for non-"last" resp_repr_modes
            # Pass global_step for debug printing (print first 5 steps + every 100 steps)
            outputs = model(resp_indices=resp_indices, resp_start_indices=resp_start_indices, rankings=rankings, **batch)
            (scores,) = outputs  # [B, R]
            
            if step == 0 and is_main_process:
                print(f"✅ Forward pass completed! Scores shape: {scores.shape}")

            # Cross-entropy loss: select best response
            loss = torch.nn.functional.cross_entropy(scores, labels)
            with torch.no_grad():
                preds = scores.argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
                num_pairs = scores.shape[0]

            # accumulate raw loss for logging, then divide by grad_accum for backward
            accum_loss += loss.item()
            accum_count += 1
            accum_accuracy += accuracy
            accum_num_pairs += num_pairs
            micro_step += 1
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process:
                    # log the average raw loss and accuracy over this accumulation window
                    avg_loss = accum_loss / max(accum_count, 1)
                    avg_accuracy = accum_accuracy / max(accum_count, 1)
                    total_num_pairs = accum_num_pairs
                    # Print progress at every optimizer step
                    print(f"Epoch {epoch} step {step} | global_step {global_step} | loss {avg_loss:.4f} | acc {avg_accuracy:.2%} ({total_num_pairs} pairs)")
                    # Log to WandB at every step (was every 10 steps)
                    if use_wandb:
                        try:
                            import wandb
                            # Get current learning rate from scheduler
                            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                            elapsed_h = (time.time() - train_start_time) / 3600
                            total_gpu_h = elapsed_h * num_gpus
                            wandb.log({
                                "loss": avg_loss,
                                "accuracy": avg_accuracy,
                                "num_pairs": total_num_pairs,
                                "learning_rate": current_lr,
                                "global_step": global_step,
                                "epoch": epoch,
                                "wall_time_hours": elapsed_h,
                                "total_gpu_hours": total_gpu_h,
                            })
                            if global_step <= 3 or global_step % 10 == 0:
                                print(f"✅ Logged to WandB: loss={avg_loss:.4f}, acc={avg_accuracy:.2%}, lr={current_lr:.2e}, step={global_step}, gpu_h={total_gpu_h:.2f}")
                        except Exception as e:
                            print(f"⚠️ Failed to log to WandB: {e}")
                            import traceback
                            traceback.print_exc()

                # reset accumulation buffer after each optimizer step
                accum_loss = 0.0
                accum_count = 0
                accum_accuracy = 0.0
                accum_num_pairs = 0
                
                # Save checkpoint if specified
                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        step + 1,  # step_in_epoch: next step to resume from
                        Path(args.output_dir),
                        is_main_process,
                        use_distributed,
                        processor,
                    )
                    # Cleanup old checkpoints
                    if args.save_total_limit > 0:
                        cleanup_old_checkpoints(Path(args.output_dir), args.save_total_limit, is_main_process)

        # Handle remaining gradients at end of epoch (when micro_steps don't divide evenly by gradient_accumulation_steps)
        if step < 0:
            # Empty epoch (no data) — nothing to do
            if is_main_process:
                print(f"Epoch {epoch} had no data, skipping.")
            continue
        remaining_micro = micro_step % args.gradient_accumulation_steps
        if remaining_micro > 0 and accum_count > 0:
            # There are accumulated gradients that haven't been applied yet
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main_process:
                # Log the average loss and accuracy over the partial accumulation window
                avg_loss = accum_loss / max(accum_count, 1)
                avg_accuracy = accum_accuracy / max(accum_count, 1)
                total_num_pairs = accum_num_pairs
                print(f"Epoch {epoch} END (partial accum: {remaining_micro} micro-steps) | global_step {global_step} | loss {avg_loss:.4f} | acc {avg_accuracy:.2%} ({total_num_pairs} pairs)")
                if use_wandb:
                    try:
                        import wandb
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                        elapsed_h = (time.time() - train_start_time) / 3600
                        total_gpu_h = elapsed_h * num_gpus
                        wandb.log({
                            "loss": avg_loss,
                            "accuracy": avg_accuracy,
                            "num_pairs": total_num_pairs,
                            "learning_rate": current_lr,
                            "global_step": global_step,
                            "epoch": epoch,
                            "wall_time_hours": elapsed_h,
                            "total_gpu_hours": total_gpu_h,
                        })
                    except Exception as e:
                        print(f"⚠️ Failed to log to WandB: {e}")

            # Reset accumulation buffer
            accum_loss = 0.0
            accum_count = 0
            accum_accuracy = 0.0
            accum_num_pairs = 0

    # Print total GPU time summary
    if is_main_process:
        total_wall_h = (time.time() - train_start_time) / 3600
        total_gpu_h = total_wall_h * num_gpus
        print(f"\n{'='*60}")
        print(f"⏱️  Training completed!")
        print(f"   Wall time:      {total_wall_h:.2f} hours")
        print(f"   Num GPUs:       {num_gpus}")
        print(f"   Total GPU time: {total_gpu_h:.2f} GPU-hours")
        print(f"{'='*60}\n")
        if use_wandb:
            try:
                import wandb
                wandb.summary["total_wall_time_hours"] = total_wall_h
                wandb.summary["total_gpu_hours"] = total_gpu_h
                wandb.summary["num_gpus"] = num_gpus
            except Exception:
                pass

    # NOTE: Removed dist.barrier() here. It is unnecessary because:
    # 1. Only rank 0 saves the model, other ranks don't participate.
    # 2. DDP already synchronized gradients in the last backward() call.
    # 3. The barrier was causing NCCL timeouts when ranks finished at different times.
    
    # Save (only on main process)
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Unwrap DDP if needed
        model_to_save = model.module if use_distributed else model
        # Save LoRA adapters
        model_to_save.base_model.save_pretrained(output_dir)
        # Save value head separately
        torch.save(model_to_save.value_head.state_dict(), output_dir / "value_head.pt")
        # Save reward model config so merge_lora.py / inference can reconstruct the value head
        reward_model_config = {
            "value_head_type": model_to_save.value_head_type,
            "value_head_hidden_dim": model_to_save.value_head_hidden_dim,
            "value_head_activation": model_to_save.value_head_activation,
            "value_head_input_dim": model_to_save.value_head_input_dim,
            "resp_repr_mode": model_to_save.resp_repr_mode,
        }
        with open(output_dir / "reward_model_config.json", "w") as f:
            json.dump(reward_model_config, f, indent=2)
        # Save processor for inference convenience
        # Some Molmo2 processors lack audio_tokenizer; add placeholder to avoid AttributeError
        if not hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = None
        processor.save_pretrained(output_dir)
        print(f"✅ Saved model to {output_dir}")
        
        # Print collate statistics
        print_collate_stats()
    
    # Destroy process group - no barrier needed after save
    # Other ranks can exit while rank 0 saves
    if use_distributed:
        try:
            dist.destroy_process_group()
        except Exception as e:
            if is_main_process:
                print(f"⚠️ Error destroying process group (non-fatal): {e}")


if __name__ == "__main__":
    main()

