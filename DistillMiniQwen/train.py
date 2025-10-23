import argparse
import os
import time
import math
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from tqdm import tqdm

# Assuming your model is in qwen3_model.py
from DistillMiniQwen.model import Qwen3MoE, Qwen3Config
from DistillMiniQwen.data import * 




class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        args: argparse.Namespace,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.scaler = GradScaler() if args.fp16 else None
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
        
        # Gradient accumulation
        self.accumulation_steps = args.gradient_accumulation_steps
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids = input_ids.to(self.args.device)
            targets = targets.to(self.args.device)
            
            # Forward pass with mixed precision
            if self.args.fp16:
                with autocast():
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100
                    )
                    loss = loss / self.accumulation_steps
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.args.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.args.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.current_step += 1
                
                # Logging
                if self.current_step % self.args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.writer:
                        self.writer.add_scalar('train/loss', loss.item() * self.accumulation_steps, self.current_step)
                        self.writer.add_scalar('train/lr', lr, self.current_step)
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                        'lr': f'{lr:.2e}'
                    })
            
            # Track metrics
            num_tokens = (targets != -100).sum().item()
            total_loss += loss.item() * self.accumulation_steps * num_tokens
            total_tokens += num_tokens
            
            # Save checkpoint
            if self.current_step % self.args.save_interval == 0:
                self.save_checkpoint()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for input_ids, targets in tqdm(self.val_loader, desc="Validating"):
            input_ids = input_ids.to(self.args.device)
            targets = targets.to(self.args.device)
            
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            num_tokens = (targets != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, self.current_step)
            self.writer.add_scalar('val/perplexity', perplexity, self.current_step)
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        checkpoint_dir = Path(self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.__dict__,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{self.current_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Keep only last N checkpoints
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_step_*.pt'))
        if len(checkpoints) > self.args.keep_last_n_checkpoints:
            for old_ckpt in checkpoints[:-self.args.keep_last_n_checkpoints]:
                old_ckpt.unlink()
    
    def train(self):
        print("Starting training...")
        print(f"Total steps: {len(self.train_loader) * self.args.num_epochs // self.accumulation_steps}")
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            
            # Save epoch checkpoint
            self.save_checkpoint()
        
        print("Training completed!")
        if self.writer:
            self.writer.close()


def get_model_config(args):
    """Create model config from arguments"""
    config = Qwen3Config(
        n_embed=args.n_embed,
        n_head=args.n_head,
        n_kv_heads=args.n_kv_heads,
        n_layer=args.n_layer,
        n_mlp=args.n_mlp,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        vocab_size=args.vocab_size,
        tie_word_embeddings=args.tie_word_embeddings,
        head_dim=args.head_dim,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Qwen3 MoE model')
    
    # Model architecture
    parser.add_argument('--n_embed', type=int, default=896, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=14, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=2, help='Number of KV heads (GQA)')
    parser.add_argument('--n_layer', type=int, default=24, help='Number of layers')
    parser.add_argument('--n_mlp', type=int, default=4864, help='MLP hidden dimension')
    parser.add_argument('--rope_theta', type=float, default=1000000.0, help='RoPE theta')
    parser.add_argument('--rms_norm_eps', type=float, default=1e-6, help='RMSNorm epsilon')
    parser.add_argument('--vocab_size', type=int, default=151936, help='Vocabulary size')
    parser.add_argument('--tie_word_embeddings', action='store_true', help='Tie input/output embeddings')
    parser.add_argument('--head_dim', type=int, default=None, help='Head dimension (optional)')
    
    # MoE specific
    parser.add_argument('--num_experts', type=int, default=60, help='Number of experts')
    parser.add_argument('--num_experts_per_tok', type=int, default=4, help='Experts per token')
    parser.add_argument('--moe_intermediate_size', type=int, default=2560, help='MoE intermediate size')
    
    # Training data
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B', help='Tokenizer name or path')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    
    # Optimization
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'constant'])
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=3, help='Keep last N checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N steps')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.vocab_size = len(tokenizer)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TextDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = TextDataset(args.val_data, tokenizer, args.max_length) if args.val_data else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    ) if val_dataset else None
    
    # Create model
    print("Creating model...")
    config = get_model_config(args)
    model = Qwen3MoE(config)
    model = model.to(args.device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Create scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.learning_rate * 0.1
        )
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    else:
        scheduler = None
    
    # Add warmup
    if args.warmup_steps > 0 and scheduler is not None:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=args.warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[args.warmup_steps]
        )
    
    # Resume from checkpoint
    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()