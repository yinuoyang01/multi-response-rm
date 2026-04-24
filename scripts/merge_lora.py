#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for Molmo2 multi-response reward models.

This script merges a LoRA adapter into the base model for multi-response reward models.

Usage:
    python merge_lora_adapter.py \
        --adapter_path /path/to/adapter \
        --output_dir /path/to/merged/model \
        [--base_model_path BASE_MODEL]  # Optional, auto-detected from adapter_config.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path to import mr2rm
sys.path.insert(0, str(Path(__file__).parent.parent))

from mr2rm.models.reward_model import MultiResponseRewardModel
from mr2rm.data.dataset import add_resp_sep_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_adapter(
    adapter_path,
    output_dir: str,
    base_model_path: str = None,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
):
    """
    Merge LoRA adapter(s) into base model.
    
    Args:
        adapter_path: Path(s) to LoRA adapter directory(ies). Can be a single path (str) or list of paths.
            Each directory should contain adapter_config.json and adapter_model.safetensors
        output_dir: Output directory for merged model
        base_model_path: Base model path (optional, will be read from first adapter_config.json if not provided)
        trust_remote_code: Whether to trust remote code
        device_map: Device map for loading model. For merge, will use single GPU ({"": 0}) to avoid issues.
        torch_dtype: Torch dtype (bfloat16, float16, or float32)
    """
    import torch
    import shutil
    from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
    from peft import PeftModel, PeftConfig
    
    # Support multiple adapters (list) or single adapter (str)
    if isinstance(adapter_path, str):
        adapter_paths = [Path(adapter_path)]
    else:
        adapter_paths = [Path(p) for p in adapter_path]
    
    output_dir = Path(output_dir)
    
    # Auto-detect latest checkpoint if adapter directory contains checkpoints
    def find_latest_checkpoint(adapter_dir: Path) -> Path:
        """Find the best adapter to merge: prefer final model in root dir, fall back to latest checkpoint."""
        if not adapter_dir.exists():
            return adapter_dir

        # Prefer the root directory if it contains adapter_config.json (final trained model)
        # The training script saves the final model directly to output_dir after all epochs,
        # while intermediate checkpoints go to checkpoint-XXXX subdirectories.
        # The final model may have more training steps than the latest checkpoint
        # (e.g., total_steps=4110 but last checkpoint is checkpoint-4000).
        if (adapter_dir / "adapter_config.json").exists():
            logger.info(f"Found final trained model in root directory: {adapter_dir}")
            return adapter_dir

        # No final model in root, fall back to latest checkpoint
        checkpoint_dirs = [
            d for d in adapter_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-") and (d / "adapter_config.json").exists()
        ]

        if not checkpoint_dirs:
            return adapter_dir

        def get_checkpoint_num(path: Path) -> int:
            try:
                return int(path.name.split("-")[1])
            except (ValueError, IndexError):
                return 0

        latest_checkpoint = max(checkpoint_dirs, key=get_checkpoint_num)
        logger.info(f"No final model in root, using latest checkpoint: {latest_checkpoint.name} in {adapter_dir}")
        return latest_checkpoint
    
    # Auto-detect latest checkpoint for each adapter path
    processed_adapter_paths = []
    for adapter_path in adapter_paths:
        if not adapter_path.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_path}")
        
        # Check if this is a checkpoint directory or main adapter directory
        if adapter_path.name.startswith("checkpoint-"):
            # Already a checkpoint directory, use it directly
            processed_path = adapter_path
        else:
            # Main adapter directory, try to find latest checkpoint
            latest_checkpoint = find_latest_checkpoint(adapter_path)
            processed_path = latest_checkpoint
        
        # Validate the processed path
        if not (processed_path / "adapter_config.json").exists():
            raise ValueError(
                f"adapter_config.json not found in {processed_path}. "
                "This doesn't appear to be a LoRA adapter directory or checkpoint."
            )
        
        processed_adapter_paths.append(processed_path)
        if processed_path != adapter_path:
            logger.info(f"Using checkpoint {processed_path.name} from {adapter_path}")
    
    adapter_paths = processed_adapter_paths
    
    # Convert torch_dtype string to torch.dtype
    if torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16  # default
    
    logger.info(f"Loading multi-response reward model from {len(adapter_paths)} adapter(s)")
    logger.info(f"Output directory: {output_dir}")
    
    if device_map == "cpu":
        target_device = "cpu"
        load_device_map = None
    else:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Use device_map={"": 0} to ensure all layers on single GPU
        load_device_map = {"": 0} if torch.cuda.is_available() else None
    
    logger.info(f"Loading model on device: {target_device}")
    
    # Prepare loading kwargs
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if load_device_map is not None:
        load_kwargs["device_map"] = load_device_map
    else:
        load_kwargs["device"] = target_device
    
    # Get base model path
    if base_model_path is None:
        # Read from first adapter's config
        peft_config = PeftConfig.from_pretrained(str(adapter_paths[0]))
        base_model_path = peft_config.base_model_name_or_path
        logger.info(f"Auto-detected base model from adapter_config.json: {base_model_path}")
    else:
        logger.info(f"Using provided base model path: {base_model_path}")
    
    # Load base model
    logger.info(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

    # Load tokenizer early and register <|resp_sep|> so we can resize embeddings
    # before merging LoRA adapters (the adapter was trained with the resized embedding).
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=trust_remote_code,
        )
    except Exception:
        _tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_paths[0]), trust_remote_code=trust_remote_code,
        )
    add_resp_sep_token(_tokenizer)

    # Handle Molmo2Embedding resize manually (it doesn't have standard 'weight' attribute)
    target_vocab_size = len(_tokenizer)
    embed_layer = base_model.get_input_embeddings()
    current_vocab_size = embed_layer.embedding.size(0) + embed_layer.new_embedding.size(0)

    if target_vocab_size != current_vocab_size:
        logger.info(f"Resizing embeddings from {current_vocab_size} to {target_vocab_size}")
        # For Molmo2, we need to manually resize the embedding and new_embedding parameters
        num_new_tokens = target_vocab_size - embed_layer.embedding.size(0)

        # Resize new_embedding parameter
        old_new_embedding = embed_layer.new_embedding.data
        new_new_embedding = torch.zeros(
            num_new_tokens,
            embed_layer.embedding.size(1),
            dtype=embed_layer.embedding.dtype,
            device=embed_layer.embedding.device
        )
        # Copy old new_embedding values if they exist
        copy_size = min(old_new_embedding.size(0), num_new_tokens)
        if copy_size > 0:
            new_new_embedding[:copy_size] = old_new_embedding[:copy_size]

        # Update the parameter
        embed_layer.new_embedding = torch.nn.Parameter(new_new_embedding)

        # NOTE: Do NOT resize lm_head. In Molmo2, the lm_head only covers the base vocab
        # (text_config.vocab_size), not the additional vocab (new_embedding tokens like
        # image special tokens). The additional tokens are input-only embeddings that are
        # never predicted via lm_head.

    # Load and merge adapters one by one (chain merge)
    for i, adapter_path in enumerate(adapter_paths):
        logger.info(f"Loading LoRA adapter {i+1}/{len(adapter_paths)} from: {adapter_path}")
        base_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            trust_remote_code=trust_remote_code,
        )
        
        # Merge immediately (chain merge)
        logger.info(f"Merging adapter {i+1}/{len(adapter_paths)}...")
        base_model = base_model.merge_and_unload()
        logger.info(f"Adapter {i+1}/{len(adapter_paths)} merged successfully")
    
    logger.info(f"Merged {len(adapter_paths)} adapter(s) into base model")
    
    # Convert model to target dtype
    logger.info(f"Converting model to dtype: {torch_dtype}")
    base_model = base_model.to(torch_dtype)
    
    # Read value head config from adapter
    value_head_adapter = adapter_paths[-1]
    value_head_type = "linear"
    value_head_hidden_dim = None
    value_head_activation = "selu"
    resp_repr_mode = "last"

    # Try to load reward_model_config.json first (saved by training script)
    # Search order: adapter root dir, then latest checkpoint subdir
    reward_model_config_path = value_head_adapter / "reward_model_config.json"
    if not reward_model_config_path.exists():
        # Root dir doesn't have config (final save may not have called save_checkpoint).
        # Fall back to latest checkpoint's config.
        checkpoint_dirs = sorted(
            [d for d in value_head_adapter.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-") and (d / "reward_model_config.json").exists()],
            key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
            reverse=True,
        ) if value_head_adapter.is_dir() else []
        if checkpoint_dirs:
            reward_model_config_path = checkpoint_dirs[0] / "reward_model_config.json"
            logger.info(f"reward_model_config.json not in root, using from {checkpoint_dirs[0].name}")

    loaded_from_config = False
    if reward_model_config_path.exists():
        try:
            with open(reward_model_config_path, "r") as f:
                rm_config = json.load(f)
            value_head_type = rm_config.get("value_head_type", value_head_type)
            value_head_hidden_dim = rm_config.get("value_head_hidden_dim", value_head_hidden_dim)
            value_head_activation = rm_config.get("value_head_activation", value_head_activation)
            resp_repr_mode = rm_config.get("resp_repr_mode", resp_repr_mode)
            loaded_from_config = True
            logger.info(f"Loaded reward model config from {reward_model_config_path}: "
                        f"value_head_type={value_head_type}, hidden_dim={value_head_hidden_dim}, "
                        f"activation={value_head_activation}, resp_repr_mode={resp_repr_mode}")
        except Exception as e:
            logger.warning(f"Failed to read reward_model_config.json: {e}")

    if not loaded_from_config:
        logger.info("reward_model_config.json not found, falling back to directory name inference")
        # Try to infer value_head_type and resp_repr_mode from adapter directory name
        # Check for common patterns in directory name
        # Note: If using a checkpoint subdirectory, also check parent directory name
        adapter_name = value_head_adapter.name
        parent_name = value_head_adapter.parent.name if value_head_adapter.parent else ""

        # Check both current name and parent name (in case of checkpoint-XXXX subdirectory)
        names_to_check = [adapter_name, parent_name]
        import re

        found_mlp = False
        for name in names_to_check:
            if "mlp" in name.lower():
                value_head_type = "mlp"
                # Try to extract hidden_dim from name (e.g., mlp1024)
                match = re.search(r"mlp(\d+)", name.lower())
                if match:
                    value_head_hidden_dim = int(match.group(1))
                    logger.info(f"Inferred value_head_type=mlp, hidden_dim={value_head_hidden_dim} from directory name: {name}")
                else:
                    logger.info(f"Inferred value_head_type=mlp from directory name: {name}")
                found_mlp = True
                break

        if not found_mlp:
            value_head_type = "linear"
            logger.info(f"Inferred value_head_type=linear from directory name")

        # Infer activation function from directory name
        for name in names_to_check:
            name_lower = name.lower()
            for act in ["silu", "selu", "gelu", "relu", "tanh"]:
                if f"_{act}_" in name_lower or name_lower.endswith(f"_{act}"):
                    value_head_activation = act
                    logger.info(f"Inferred value_head_activation={act} from directory name: {name}")
                    break
            else:
                continue
            break

        # Infer resp_repr_mode from directory name
        # Patterns: first_last_concat, first_last_add, first_last_sub, response_mean, kwise, first, last (default)
        resp_repr_mode = "last"  # default
        for name in names_to_check:
            name_lower = name.lower()
            if "first_last_concat" in name_lower:
                resp_repr_mode = "first_last_concat"
                logger.info(f"Inferred resp_repr_mode=first_last_concat from directory name: {name}")
                break
            elif "first_last_add" in name_lower:
                resp_repr_mode = "first_last_add"
                logger.info(f"Inferred resp_repr_mode=first_last_add from directory name: {name}")
                break
            elif "first_last_sub" in name_lower:
                resp_repr_mode = "first_last_sub"
                logger.info(f"Inferred resp_repr_mode=first_last_sub from directory name: {name}")
                break
            elif "response_mean" in name_lower:
                resp_repr_mode = "response_mean"
                logger.info(f"Inferred resp_repr_mode=response_mean from directory name: {name}")
                break
            elif "kwise" in name_lower:
                # kwise uses last token by default
                resp_repr_mode = "last"
                logger.info(f"Inferred resp_repr_mode=last (kwise mode) from directory name: {name}")
                break

        if resp_repr_mode == "last":
            logger.info(f"Using default resp_repr_mode=last")

    # Create multi-response reward model
    logger.info(f"Creating MultiResponseRewardModel with value_head_type={value_head_type}, hidden_dim={value_head_hidden_dim}, "
                f"activation={value_head_activation}, resp_repr_mode={resp_repr_mode}")
    reward_model = MultiResponseRewardModel(
        base_model=base_model,
        value_head_type=value_head_type,
        value_head_hidden_dim=value_head_hidden_dim,
        value_head_activation=value_head_activation,
        resp_repr_mode=resp_repr_mode,
    )
    
    # Load value head from adapter (value_head.pt)
    value_head_path = value_head_adapter / "value_head.pt"
    if value_head_path.exists():
        logger.info(f"Loading value head from {value_head_path}")
        reward_model.value_head.load_state_dict(
            torch.load(value_head_path, map_location=target_device)
        )
    else:
        logger.warning(f"Value head not found at {value_head_path}, using randomly initialized value head")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged base model
    logger.info(f"Saving merged base model to {output_dir}")
    reward_model.base_model.save_pretrained(str(output_dir))

    # Fix config.json vocab_size handling:
    # Molmo2Config.vocab_size is a read-only computed property, so we must NOT include it in config.json
    # We need to update text_config.additional_vocab_size to match the actual new_embedding size
    config_path = output_dir / "config.json"

    if config_path.exists():
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Remove top-level vocab_size if present (it's a computed read-only property)
        if 'vocab_size' in config_dict:
            logger.info(f"Removing top-level vocab_size={config_dict['vocab_size']} from config.json (computed property)")
            del config_dict['vocab_size']

        # Update text_config.additional_vocab_size to match actual new_embedding size
        # Get the actual new_embedding size from the resized model
        actual_new_embedding_size = reward_model.base_model.get_input_embeddings().new_embedding.size(0)
        base_embedding_size = reward_model.base_model.get_input_embeddings().embedding.size(0)

        if 'text_config' in config_dict:
            old_additional = config_dict['text_config'].get('additional_vocab_size', 0)
            old_vocab_size = config_dict['text_config'].get('vocab_size', 0)
            if old_additional != actual_new_embedding_size or old_vocab_size != base_embedding_size:
                logger.info(f"Updating text_config.vocab_size from {old_vocab_size} to {base_embedding_size}")
                logger.info(f"Updating text_config.additional_vocab_size from {old_additional} to {actual_new_embedding_size}")
                config_dict['text_config']['vocab_size'] = base_embedding_size
                config_dict['text_config']['additional_vocab_size'] = actual_new_embedding_size
                logger.info(f"Final vocab_size will be: {base_embedding_size} + {actual_new_embedding_size} = {base_embedding_size + actual_new_embedding_size}")

        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Save value head
    logger.info(f"Saving value head to {output_dir}")
    torch.save(reward_model.value_head.state_dict(), output_dir / "value_head.pt")

    # Save reward model config to merged output
    merged_rm_config = {
        "value_head_type": value_head_type,
        "value_head_hidden_dim": value_head_hidden_dim,
        "value_head_activation": value_head_activation,
        "value_head_input_dim": reward_model.value_head_input_dim,
        "resp_repr_mode": resp_repr_mode,
    }
    with open(output_dir / "reward_model_config.json", "w") as f:
        json.dump(merged_rm_config, f, indent=2)
    logger.info(f"Saved reward_model_config.json to {output_dir}")

    # Save tokenizer and processor
    logger.info("Saving tokenizer and processor...")
    try:
        # Try to load tokenizer from base model (preferred) or first adapter
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=trust_remote_code,
            )
            logger.info(f"Loaded tokenizer from base model: {base_model_path}")
        except Exception:
            # Fallback to adapter directory
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_paths[0]),
                trust_remote_code=trust_remote_code,
            )
            logger.info(f"Loaded tokenizer from adapter: {adapter_paths[0]}")
        
        # Ensure <|resp_sep|> is registered as a special token before saving
        add_resp_sep_token(tokenizer)
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Tokenizer saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save tokenizer: {e}")
    
    try:
        # Try to load processor from base model (preferred) or first adapter
        processor_source_path = None
        try:
            processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=trust_remote_code,
            )
            logger.info(f"Loaded processor from base model: {base_model_path}")
            processor_source_path = Path(base_model_path)
        except Exception:
            # Fallback to adapter directory
            processor = AutoProcessor.from_pretrained(
                str(adapter_paths[0]),
                trust_remote_code=trust_remote_code,
            )
            logger.info(f"Loaded processor from adapter: {adapter_paths[0]}")
            processor_source_path = adapter_paths[0]
        
        if processor is not None:
            processor.save_pretrained(str(output_dir))
            logger.info("Processor saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save processor: {e}")
    
    # Copy processor Python files and config files
    logger.info("Copying processor files...")
    processor_python_files = ["processing_molmo2.py", "image_processing_molmo2.py", "video_processing_molmo2.py"]
    processor_config_files = ["preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]
    all_processor_files = processor_python_files + processor_config_files
    
    # Try to find processor files from multiple sources
    processor_source_dirs = []
    
    # 1. From processor_source_path (base model or adapter)
    if processor_source_path is not None and processor_source_path.exists():
        processor_source_dirs.append(processor_source_path)
    
    # 2. From adapter directories
    for adapter_path in adapter_paths:
        if adapter_path.exists() and adapter_path not in processor_source_dirs:
            processor_source_dirs.append(adapter_path)


    # Copy each file from first available source
    copied_files = set()
    for fname in all_processor_files:
        if (output_dir / fname).exists():
            logger.info(f"{fname} already exists in output directory, skipping")
            copied_files.add(fname)
            continue
        
        for source_dir in processor_source_dirs:
            source_file = source_dir / fname
            if source_file.exists():
                dest_file = output_dir / fname
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied {fname} from {source_dir} to {output_dir}")
                copied_files.add(fname)
                break
    
    missing_files = [f for f in all_processor_files if f not in copied_files]
    if missing_files:
        logger.warning(f"Some processor files not found: {missing_files}")
        logger.warning("These files may be needed for AutoProcessor.from_pretrained to work properly")
    else:
        logger.info("All processor files copied successfully")
    
    logger.info(f"✅ Merge complete! Merged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for Molmo2 multi-response reward models"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        nargs="+",  # Support multiple adapters
        required=True,
        help="Path(s) to LoRA adapter directory(ies) (contains adapter_config.json, adapter_model.safetensors). "
             "Can specify multiple adapters for chain merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path (optional, will be read from adapter_config.json if not provided)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code (default: True)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading model. For merge, will use single GPU ({\"\": 0}) to avoid issues. (default: auto)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Merge adapter
    merge_adapter(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )


if __name__ == "__main__":
    main()
