import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments
from models.modeling_qwen3 import Qwen3ForCausalLM
from models.data_collator import DataCollatorForCSC
from models.encoder import InputHelper
from trainer_csc import CSCTrainer, SaveAdapterCallback
from datasets import load_dataset


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    plugin = Qwen3ForCausalLM.from_pretrained(args.model_name)

    input_helper = InputHelper(tokenizer=tokenizer, cache_dir=args.modal_cache_dir)
    dataset = load_dataset("json", data_files={"train": args.data_file})["train"]
    data_collator = DataCollatorForCSC(tokenizer, input_helper, max_length=args.max_length)
    
    # Freeze LLM backbone
    # for param in plugin.model.parameters():
    #     param.requires_grad = False

    # 冻结整个 Qwen3 主干
    for param in plugin.model.parameters():
        param.requires_grad = False
    
    num_layers = len(plugin.model.layers)
    for layer_idx in range(args.plug_idx[0], num_layers):
        layer = plugin.model.layers[layer_idx]
        for param in layer.self_attn.parameters():
            param.requires_grad = True
        for param in plugin.model.layers[layer_idx].parameters():
            param.requires_grad = True

    # 解冻多模态组件
    # for pidx in plugin.plug_idx:
    #     module = plugin.model.layers[pidx].adapter
    #     for param in module.parameters():
    #         param.requires_grad = True
    # for module in [plugin.phonetic, plugin.glyph]:
    #     for param in module.parameters():
    #         param.requires_grad = True
    for layer_idx in args.plug_idx:
        if hasattr(plugin.model.layers[layer_idx], 'csc_adapter'):
            for param in plugin.model.layers[layer_idx].csc_adapter.parameters():
                param.requires_grad = True
    trainable_params = [name for name, param in plugin.named_parameters() if param.requires_grad]
    print("Trainable parameters:")
    for name in trainable_params:
        print(f"  - {name}")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        bf16=True,
        remove_unused_columns=False,  # 必须！否则会丢弃 pinyins/images
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        log_level='info',
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_pin_memory=True,
        # ddp_find_unused_parameters=False,  # 若报错 "unused parameters" 可设为 True
        # gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        model=plugin,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # callbacks=[SaveAdapterCallback()],
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Base", help="HuggingFace model name or path")
    parser.add_argument("--data_file", type=str, default="data/train.jsonl", help="Training data file")
    parser.add_argument("--modal_cache_dir", type=str, help="Cache directory to the mulitmodal embeddings")
    parser.add_argument("--plug_idx", type=list, default=[16, 28], help="Layer indices to insert adapter")
    parser.add_argument("--gate_type", type=str, default='residual')
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Maximum number of checkpoints to save")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./qwen3-csc-adapter", help="Directory to save Adapter")
    parser.add_argument('--device', type=str, default='cuda', help='The device for training. auto, cpu or cuda')
    args = parser.parse_args()
    train(args)