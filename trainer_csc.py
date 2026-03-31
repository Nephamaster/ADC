import os
import torch
import re
import shutil
from torch.nn import functional as F
from transformers import Trainer
from transformers import TrainerCallback
from loss import FocalLossCSC


class SaveAdapterCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        output_dir = args.output_dir
        step = state.global_step
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")    
        os.makedirs(save_dir, exist_ok=True)
        num_layers = len(model.model.layers)
        for pidx in model.plug_idx:
            layer = model.model.layers[pidx]
            if pidx < 0:
                idx = num_layers + pidx
            else: idx = pidx
            adapter = os.path.join(save_dir, f"adapter_layer_{idx}.bin")
            torch.save(layer.adapter.state_dict(), adapter)
        phon = os.path.join(save_dir, f"phonetic.bin")
        torch.save(model.phonetic.state_dict(), phon)
        glyp = os.path.join(save_dir, f"glyph.bin")
        torch.save(model.glyph.state_dict(), glyp)
        print(f"### Saved encoders and adapter at step {step} to {save_dir}")


class CSCTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = self._custom_nll_loss

    def _custom_nll_loss(self, outputs, labels, num_items_in_batch=None):
        """
        实现公式: L(𝒯) = -Σ log(P(Y'_c | I, X_c))
        """
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len] → [batch_size * seq_len]
        assert labels.min() >= -100, f"非法最小标签: {labels.min()}"
        assert labels.max() < self.model.model.config.vocab_size, f"标签越界: {labels.max()} >= {self.model.model.config.vocab_size}"
        assert not torch.isnan(logits).any(), "模型输出包含 NaN"
        assert not torch.isinf(logits).any(), "模型输出包含 Inf"
        valid_mask = (labels != -100)
        labels_safe = labels.clone()
        labels_safe[~valid_mask] = 0
        target_log_probs = log_probs.gather(
            dim=-1, 
            index=labels_safe.unsqueeze(-1)  # [batch_size, seq_len, 1]
        ).squeeze(-1)  # [batch_size, seq_len]
        # -Σ log(P)
        total_loss = -target_log_probs[valid_mask].sum()
        if self.args.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.args.gradient_accumulation_steps
        return total_loss
    
    def _cleanup_old_checkpoints(self, output_dir: str):
        if self.args.save_total_limit <= 0:
            return
        ckpt_dirs = []
        pattern = re.compile(r"checkpoint-(\d+)")
        for item in os.listdir(output_dir):
            match = pattern.match(item)
            if match:
                step_num = int(match.group(1))
                full_path = os.path.join(output_dir, item)
                if os.path.isdir(full_path):
                    ckpt_dirs.append((step_num, full_path))
        ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
        dirs_to_delete = ckpt_dirs[self.args.save_total_limit:]
        for step_num, dirpath in dirs_to_delete:
            try:
                shutil.rmtree(dirpath)
                print(f"Deleted checkpoint directory: {dirpath} (step {step_num}) due to save_total_limit")
            except OSError as e:
                print(f"Failed to delete checkpoint directory {dirpath}: {e}")

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        step = self.state.global_step
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")    
        os.makedirs(save_dir, exist_ok=True)
        adapter = os.path.join(output_dir, f"adapter.bin")
        torch.save(self.model.scale_adapter.state_dict(), adapter)
        phon = os.path.join(output_dir, f"phonetic.bin")
        torch.save(self.model.phonetic.state_dict(), phon)
        glyp = os.path.join(output_dir, f"glyph.bin")
        torch.save(self.model.glyph.state_dict(), glyp)
        print(f"### Saved encoder and adapter at step {step} to {save_dir}")
        self._cleanup_old_checkpoints(output_dir)