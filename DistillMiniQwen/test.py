import torch
import argparse
from transformers import AutoTokenizer
from DistillMiniQwen.model import Qwen3MoE, Qwen3Config
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Test distilled Qwen3 MoE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Tokenizer to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--prompts', nargs='+', default=None,
                        help='Test prompts')
    
    return parser.parse_args()


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = Qwen3Config(**config_dict)
    
    print(f"Model configuration:")
    print(f"  n_embed: {config.n_embed}")
    print(f"  n_head: {config.n_head}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  vocab_size: {config.vocab_size}")
    
    # Create model
    model = Qwen3MoE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'loss' in checkpoint:
        print(f"  Final loss: {checkpoint['loss']:.4f}")
    
    return model, config


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    device='cuda'
):
    """Generate text from prompt with sampling"""
    model.eval()
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    print("Generating", end='', flush=True)
    
    generated_tokens = []
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get logits
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated tokens
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for stop tokens
            if next_token.item() in [151645, 151644, 151643]:  # Qwen stop tokens
                break
            
            # Print progress
            if (i + 1) % 10 == 0:
                print('.', end='', flush=True)
    
    end_time = time.time()
    
    # Decode generated text
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    generated_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "="*70)
    print("Generated:")
    print(generated_only)
    print("="*70)
    
    # Statistics
    tokens_per_sec = len(generated_tokens) / (end_time - start_time)
    print(f"\nStatistics:")
    print(f"  Tokens generated: {len(generated_tokens)}")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  Speed: {tokens_per_sec:.2f} tokens/s")
    
    return full_text


def run_benchmark(model, tokenizer, device='cuda'):
    """Run benchmark on common tasks"""
    print("\n" + "="*70)
    print("RUNNING BENCHMARK")
    print("="*70)
    
    test_prompts = [
        "Instruction: hi, is there a CLI command to roll back any updates/upgrades I made recently?",
        "Instruction:A LiveCD iso can be burned to a DVD-R and run with no problems, right?",
        "Instruction:hello, is there a way to adjust gamma settings in totem? my videos aren't playing with the correct colours",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(test_prompts)}")
        generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.7,
            device=device
        )


def interactive_mode(model, tokenizer, device='cuda'):
    """Run in interactive mode"""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your prompts (type 'quit' or 'exit' to stop)")
    print("Commands:")
    print("  /temp <value>  - Set temperature (0.1-2.0)")
    print("  /tokens <n>    - Set max tokens")
    print("  /clear         - Clear conversation")
    print("="*70 + "\n")
    
    max_tokens = 100
    temperature = 0.7
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == '/temp' and len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        temperature = max(0.1, min(2.0, temperature))
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                
                elif cmd == '/tokens' and len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        print(f"Max tokens set to {max_tokens}")
                    except ValueError:
                        print("Invalid token count")
                    continue
                
                elif cmd == '/clear':
                    print("Conversation cleared")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Format as instruction
            if not user_input.startswith("Instruction:"):
                prompt = f"Instruction: {user_input}\nAnswer:"
            else:
                prompt = user_input
            
            # Generate response
            generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device
            )
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    args = parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    print("\n" + "="*70)
    print("MODEL LOADED SUCCESSFULLY")
    print("="*70)
    
    # Run tests based on mode
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.prompts:
        # Test with provided prompts
        for prompt in args.prompts:
            generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device
            )
    else:
        # Run benchmark
        run_benchmark(model, tokenizer, device)


if __name__ == '__main__':
    main()