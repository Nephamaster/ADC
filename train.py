import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.data_collator import DataCollatorForCSC
from src.encoder import InputHelper
from src.modeling_qwen3 import Qwen3ForCausalLM


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


def train(args):
    maybe_init_distributed(args)
    log_distributed_env()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    plugin = Qwen3ForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
    )
    if hasattr(plugin, "config"):
        plugin.config.use_cache = args.use_cache
        plugin.config.csc_adapter_layers = args.plug_idx

    input_helper = InputHelper(tokenizer=tokenizer, cache_dir=args.cache)
    dataset = load_dataset("json", data_files={"train": args.dataset})["train"]
    data_collator = DataCollatorForCSC(tokenizer, input_helper, max_length=args.max_length)

    for param in plugin.model.parameters():
        param.requires_grad = False

    num_layers = len(plugin.model.layers)
    for layer_idx in range(args.unfreeze_first_layers):
        for param in plugin.model.layers[layer_idx].parameters():
            param.requires_grad = True
    for layer_idx in range(args.unfreeze_last_layers):
        for param in plugin.model.layers[num_layers-layer_idx].parameters():
            param.requires_grad = True

    for layer_idx in args.plug_idx:
        layer = plugin.model.layers[layer_idx]
        if hasattr(layer, "csc_adapter"):
            for param in layer.csc_adapter.parameters():
                param.requires_grad = True
            for param in layer.mlp.parameters():
                param.requires_grad = True

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
    parser.add_argument("--plug_idx", nargs="+", type=int, default=[28], help="Layer indices to insert adapter")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10)
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
