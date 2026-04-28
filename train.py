import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.data_collator import DataCollatorForCSC
from src.encoder import InputHelper
from src.modeling_qwen3_5 import Qwen3_5ForCausalLM
from src.configuration_qwen_3_5 import Qwen3_5Config
from src.adapter import CSCAdapter


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log_distributed_env() -> None:
    if not is_main_process():
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(
        f"[Distributed] rank={rank}, local_rank={local_rank}, world_size={world_size}, "
        f"cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
    )


def maybe_init_distributed(args) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or torch.distributed.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=args.ddp_backend, init_method="env://")


def normalize_layer_indices(indices, num_layers: int):
    normalized = []
    for idx in indices:
        resolved_idx = idx if idx >= 0 else num_layers + idx
        if resolved_idx < 0 or resolved_idx >= num_layers:
            raise ValueError(f"Layer index out of range: {idx} (resolved={resolved_idx}, total={num_layers})")
        if resolved_idx not in normalized:
            normalized.append(resolved_idx)
    return normalized


def configure_csc_adapter(model, adapter_layers, use_cache: bool) -> None:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)

    for target_cfg in (cfg, text_cfg):
        if target_cfg is None:
            continue
        if hasattr(target_cfg, "use_cache"):
            target_cfg.use_cache = use_cache
        if hasattr(target_cfg, "use_csc_adapter"):
            target_cfg.use_csc_adapter = True
        if hasattr(target_cfg, "csc_adapter_layers"):
            target_cfg.csc_adapter_layers = list(adapter_layers)


def ensure_csc_adapters(model, adapter_layers) -> None:
    adapter_layer_set = set(adapter_layers)
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in adapter_layer_set:
            if getattr(layer, "csc_adapter", None) is None:
                layer.csc_adapter = CSCAdapter(
                    config=model.config,
                    num_heads=getattr(model.config, "csc_adapter_num_heads", 4),
                    dropout=getattr(model.config, "csc_adapter_dropout", 0.1),
                )
            layer.use_csc_adapter = True
            layer.csc_adapter_layer_idx = list(adapter_layers)
        else:
            if hasattr(layer, "csc_adapter") and layer.csc_adapter is not None:
                layer.csc_adapter = None
            layer.use_csc_adapter = False
            layer.csc_adapter_layer_idx = list(adapter_layers)


def train(args):
    maybe_init_distributed(args)
    log_distributed_env()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_config = Qwen3_5Config.from_pretrained(args.model)
    config_num_layers = getattr(base_config.text_config, "num_hidden_layers", None)
    if config_num_layers is None:
        raise ValueError("Cannot determine num_hidden_layers from Qwen3.5 config.")
    adapter_layers = normalize_layer_indices(args.plug_idx, config_num_layers)
    if hasattr(base_config, "text_config") and base_config.text_config is not None:
        base_config.text_config.use_csc_adapter = True
        base_config.text_config.csc_adapter_layers = list(adapter_layers)
        base_config.text_config.use_cache = args.use_cache
    base_config.use_cache = args.use_cache
    plugin = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        config=base_config,
        # attn_implementation=args.attn_implementation,
    )
    num_layers = len(plugin.model.layers)
    adapter_layers = normalize_layer_indices(adapter_layers, num_layers)
    configure_csc_adapter(plugin, adapter_layers, args.use_cache)
    ensure_csc_adapters(plugin, adapter_layers)

    input_helper = InputHelper(tokenizer=tokenizer, cache_dir=args.cache)
    dataset = load_dataset("json", data_files={"train": args.dataset})["train"]
    data_collator = DataCollatorForCSC(tokenizer, input_helper, max_length=args.max_length)

    for param in plugin.model.parameters():
        param.requires_grad = False

    # for layer_idx in range(args.unfreeze_first_layers):
    #     for param in plugin.model.layers[layer_idx].parameters():
    #         param.requires_grad = True
    # for layer_idx in range(args.unfreeze_last_layers):
    #     for param in plugin.model.layers[-1*(layer_idx+1)].parameters():
    #         param.requires_grad = True

    for layer_idx in adapter_layers:
        layer = plugin.model.layers[layer_idx]
        if getattr(layer, "csc_adapter", None) is not None:
            for param in layer.csc_adapter.parameters():
                param.requires_grad = True
            # for param in layer.mlp.parameters():
            #     param.requires_grad = True

    use_bf16 = args.bf16 and torch.cuda.is_available()
    use_fp16 = args.fp16 and torch.cuda.is_available()
    if use_bf16 and use_fp16:
        raise ValueError("Cannot enable bf16 and fp16 at the same time.")

    training_args = TrainingArguments(
        output_dir=args.output,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        log_level="info",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        deepspeed=args.deepspeed,
        report_to=[],
    )

    trainer = Trainer(
        model=plugin,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base", help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="data/train.jsonl", help="Training data file")
    parser.add_argument("--cache", type=str, help="Cache directory to the mulitmodal embeddings")
    parser.add_argument("--output", type=str, default="./qwen3-csc-adapter", help="Directory to save Adapter")
    parser.add_argument("--plug_idx", nargs="+", type=int, default=[2,-2], help="Layer indices to insert adapter")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=6, help="Maximum number of checkpoints to save")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--learning_rate", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Enable bf16 mixed precision")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Enable fp16 mixed precision")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use KV cache during training",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer type passed to TrainingArguments",
    )
    parser.add_argument(
        "--unfreeze_first_layers",
        type=int,
        default=0,
        help="Number of first transformer layers to unfreeze",
    )
    parser.add_argument(
        "--unfreeze_last_layers",
        type=int,
        default=2,
        help="Number of last transformer layers to unfreeze",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable when DDP raises unused-parameters errors",
    )
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="DDP backend, e.g., nccl/gloo")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)
