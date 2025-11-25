import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
import csv
from datetime import datetime
from DistillMiniQwen.model import Qwen3MoE, Qwen3Config


def parse_args():
    parser = argparse.ArgumentParser(description='Distill Qwen3 MoE model from teacher')
    
    # Teacher model
    parser.add_argument('--teacher_model', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Teacher model name or path')
    
    # Student architecture
    parser.add_argument('--n_embed', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=4,
                        help='Number of KV heads')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--n_mlp', type=int, default=2048,
                        help='MLP hidden dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0,
                        help='RoPE theta value')
    parser.add_argument('--rms_norm_eps', type=float, default=1e-6,
                        help='RMSNorm epsilon')
    parser.add_argument('--tie_word_embeddings', action='store_true',
                        help='Tie input/output embeddings' , default=True)
    parser.add_argument('--head_dim', type=int, default=None,
                        help='Head dimension (optional)')
    
    # MoE parameters
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts')
    parser.add_argument('--num_experts_per_tok', type=int, default=2,
                        help='Number of experts per token')
    parser.add_argument('--moe_intermediate_size', type=int, default=1024,
                        help='MoE intermediate size')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='jtatman/python-code-dataset-500k',
                        help='Dataset name from HuggingFace')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--instruction_column', type=str, default='instruction',
                        help='Column name for instructions')
    parser.add_argument('--output_column', type=str, default='output',
                        help='Column name for outputs')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to use')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    
    # Distillation parameters
    parser.add_argument('--alpha_ce', type=float, default=0.5,
                        help='Weight for cross-entropy loss')
    parser.add_argument('--alpha_kd', type=float, default=0.5,
                        help='Weight for KL divergence loss')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for distillation')
    
    # Scheduler
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'linear', 'constant'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_t_max', type=int, default=1000,
                        help='T_max for cosine scheduler')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6,
                        help='Min LR for cosine scheduler')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--best_model_name', type=str, default='qwen3_moe_distilled_best.pt',
                        help='Best model checkpoint name')
    parser.add_argument('--final_model_name', type=str, default='qwen3_moe_distilled_final.pt',
                        help='Final model checkpoint name')
    
    # Logging
    parser.add_argument('--log_file', type=str, default='training_logs.csv',
                        help='CSV file to save training logs')
    
    # Testing
    parser.add_argument('--test_prompt', type=str, 
                        default='Instruction: List all files in the current directory\nAnswer:',
                        help='Prompt for testing generation')
    parser.add_argument('--test_max_tokens', type=int, default=50,
                        help='Max tokens for test generation')
    
    # Device and misc
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fp16_teacher', action='store_true', default=True,
                        help='Use FP16 for teacher model')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra checks')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize CSV logging
    log_file = args.log_file
    csv_headers = ['timestamp', 'epoch', 'step', 'total_loss', 'ce_loss', 'kd_loss', 'learning_rate']
    
    # Create log file with headers
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    print(f"Training logs will be saved to: {log_file}")
    
    # ===========================
    # 1. CONFIGURATION
    # ===========================
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    print(f"teacher tokenizer {len(teacher_tokenizer)} ")
    teacher_kwargs = {'trust_remote_code': True}
    if args.fp16_teacher and device == 'cuda':
        teacher_kwargs['torch_dtype'] = torch.float16
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        **teacher_kwargs
    ).to(device)
    teacher_model.eval()
    
    # Add pad token if not present
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # Student config
    config = Qwen3Config(
        n_embed=args.n_embed,
        n_head=args.n_head,
        n_kv_heads=args.n_kv_heads,
        n_layer=args.n_layer,
        n_mlp=args.n_mlp,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        vocab_size=len(teacher_tokenizer),
        tie_word_embeddings=args.tie_word_embeddings,
        head_dim=args.head_dim,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
    )
    
    print("Creating student model...")
    student_model = Qwen3MoE(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Student model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # CRITICAL: Verify model output dimension matches vocab size
    print("\nVerifying model configuration...")
    test_input = torch.randint(0, len(teacher_tokenizer), (1, 10)).to(device)
    with torch.no_grad():
        test_output = student_model(test_input)
    print(f"Model output shape: {test_output.shape}")
    print(f"Expected vocab size: {len(teacher_tokenizer)}")
    print(f"Model config vocab_size: {config.vocab_size}")
    
    if test_output.shape[-1] != len(teacher_tokenizer):
        raise ValueError(
            f"MODEL OUTPUT MISMATCH!\n"
            f"Model outputs {test_output.shape[-1]} logits but tokenizer has {len(teacher_tokenizer)} tokens.\n"
            f"Check your model's lm_head or embedding layer."
        )
    print("✓ Model output dimension verification passed")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_t_max,
            eta_min=args.scheduler_eta_min
        )
    elif args.scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=args.scheduler_t_max
        )
    else:
        scheduler = None
    
    # ===========================
    # 2. DATASET
    # ===========================
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    
    # Take subset
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Using {len(dataset)} samples")
    
    def preprocess(example):
        """Preprocess examples for distillation"""
        prompt = example.get(args.instruction_column, "")
        target = example.get(args.output_column, "")
        
        # Format the text
        text = f"Instruction: {prompt}\nAnswer: {target}"
        
        # Tokenize
        tokens = teacher_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }
    
    print("Preprocessing dataset...")
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Validate dataset
    print("\nValidating dataset...")
    test_batch = next(iter(loader))
    test_ids = test_batch["input_ids"]
    max_id = test_ids.max().item()
    min_id = test_ids.min().item()
    print(f"Token ID range: [{min_id}, {max_id}]")
    print(f"Tokenizer vocab size: {len(teacher_tokenizer)}")
    
    if max_id >= len(teacher_tokenizer):
        raise ValueError(
            f"Dataset contains token IDs ({max_id}) >= vocab size ({len(teacher_tokenizer)}). "
            f"Check your preprocessing."
        )
    print("✓ Dataset validation passed")
    
    # ===========================
    # 3. DISTILLATION LOOP
    # ===========================
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    
    print("\n" + "="*50)
    print("Starting distillation training...")
    print(f"Total steps per epoch: {len(loader)}")
    print(f"Total epochs: {args.num_epochs}")
    print("="*50 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        student_model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Verify input_ids are valid
            if input_ids.max() >= len(teacher_tokenizer):
                print(f"\nWarning: Found invalid token ID {input_ids.max().item()} in batch {batch_idx}")
                print(f"Valid range is [0, {len(teacher_tokenizer)-1}]")
                continue
            
            # Create labels (shift input_ids by 1 for causal LM)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = teacher_tokenizer.pad_token_id
            
            # Mask padding tokens in labels
            labels[labels == teacher_tokenizer.pad_token_id] = -100
            
            # Additional safety: ensure no labels are out of bounds
            valid_mask = (labels >= 0) & (labels < len(teacher_tokenizer))
            labels[~valid_mask & (labels != -100)] = -100
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits[: , : , :len(teacher_tokenizer)]  # (B, T, vocab_size)
                
                # Debug: Check teacher output shape on first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"\nTeacher output shapes:")
                    print(f"  teacher_logits: {teacher_logits.shape}")
                    print(f"  Expected: (batch={input_ids.shape[0]}, seq={input_ids.shape[1]}, vocab={len(teacher_tokenizer)})")
                
                teacher_logits_temp = teacher_logits / args.temperature
                teacher_probs = torch.softmax(teacher_logits_temp, dim=-1)
            
            # Get student predictions
            try:
                student_logits = student_model(input_ids)  # (B, T, vocab_size)
                
                # Debug: Check shapes on first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"\nStudent output shapes:")
                    print(f"  student_logits: {student_logits.shape}")
                    print(f"  labels: {labels.shape}")
                    print(f"  Expected student: (batch={input_ids.shape[0]}, seq={input_ids.shape[1]}, vocab={len(teacher_tokenizer)})")
                    
            except Exception as e:
                print(f"\nError in student forward pass: {e}")
                print(f"Input shape: {input_ids.shape}")
                raise
            
            # Apply temperature to student logits
            student_logits_temp = student_logits / args.temperature
            student_log_probs = torch.log_softmax(student_logits_temp, dim=-1)  # (B, T, vocab_size)
            
            # Verify shapes match
            if batch_idx == 0 and epoch == 0:
                print(f"\nBefore loss computation:")
                print(f"  student_log_probs: {student_log_probs.shape}")
                print(f"  teacher_probs: {teacher_probs.shape}")
                print(f"  Flattened student logits: {student_logits.view(-1, student_logits.size(-1)).shape}")
                print(f"  Flattened labels: {labels.view(-1).shape}")
            # Verify shapes are compatible
            assert student_logits.shape == teacher_logits.shape, \
                f"Shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}"
            
            # Cross entropy loss (student vs. ground truth)
            ce_loss = ce_loss_fn(
                student_logits.view(-1, student_logits.size(-1)),  # (B*T, vocab_size)
                labels.view(-1)    # (B*T,vocab_size)
            )
            

            
            # Compute KL divergence 
            kd_loss = kl_loss_fn(student_log_probs, teacher_probs)
            
            if batch_idx == 0 and epoch == 0:
                print(f"\nLoss computation:")
                print(f"  CE loss: {ce_loss.item():.4f}")
                print(f"  KD loss: {kd_loss.item():.4f}")
                print(f"  Temperature: {args.temperature}")
                print(f"  Temperature^2: {args.temperature ** 2}")

            
            # Combined loss
            loss = args.alpha_ce * ce_loss + args.alpha_kd  * kd_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=args.max_grad_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
            
            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    ce_loss.item(),
                    kd_loss.item(),
                    current_lr
                ])
            
            # Update progress bar
            postfix = {
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'kd': f'{kd_loss.item():.4f}',
            }
            if scheduler is not None:
                postfix['lr'] = f'{scheduler.get_last_lr()[0]:.2e}'
            progress_bar.set_postfix(postfix)
            
            # Save checkpoint
            if (batch_idx + 1) % args.save_interval == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"checkpoint_epoch{epoch+1}_step{batch_idx+1}.pt"
                )
                os.makedirs(args.output_dir, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'step': batch_idx,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config.__dict__,
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                print(f"\n Checkpoint saved to {checkpoint_path}")
        
        # Epoch summary
        avg_loss = total_loss / len(loader)
        avg_ce = total_ce_loss / len(loader)
        avg_kd = total_kd_loss / len(loader)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average CE Loss: {avg_ce:.4f}")
        print(f"  Average KD Loss: {avg_kd:.4f}")
        print(f"{'='*50}\n")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = args.best_model_name
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.__dict__,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, best_model_path)
            print(f" Best model saved to {best_model_path}")
    
    # ===========================
    # 4. SAVE FINAL MODEL
    # ===========================
    final_model_path = args.final_model_name
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'config': config.__dict__,
    }, final_model_path)
    print(f"\n Final distilled model saved to {final_model_path}")
    
    # ===========================
    # 5. TEST GENERATION
    # ===========================
    print("\n" + "="*50)
    print("Testing generation...")
    print("="*50 + "\n")
    
    student_model.eval()
    test_tokens = teacher_tokenizer(args.test_prompt, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        try:
            generated = student_model.generate(
                test_tokens,
                max_new_tokens=args.test_max_tokens,
                stream=False
            )
            generated_text = teacher_tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Prompt: {args.test_prompt}")
            print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"Generation test failed: {e}")
            print("This is expected if generate() method needs adjustment")
    
    print("\n Training completed successfully!")
    print(f"Training logs saved to: {log_file}")


if __name__ == '__main__':
    main()
    
    
    
    
### IN production 


""" python train_distill.py \
  --teacher_model Qwen/Qwen2.5-0.5B \
  --n_embed 512 \
  --n_head 8 \
  --n_layer 6 \
  --num_experts 8 \
  --num_experts_per_tok 2 \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --max_samples 10000 \
  --max_length 256 \
  --output_dir checkpoints \
  --log_file training_logs.csv \
  --device cuda """
  
  
  