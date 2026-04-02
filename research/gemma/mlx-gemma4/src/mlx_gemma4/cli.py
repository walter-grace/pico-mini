"""CLI for mlx-gemma4."""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-gemma4",
        description="Run Google Gemma 4 models on Apple Silicon via MLX",
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Model path or HuggingFace repo ID")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Text prompt (if omitted, enters chat mode)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-chat-format", action="store_true",
                        help="Don't wrap prompt in chat template")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    from .generate import load_model, generate

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)

    if args.prompt:
        # Single prompt mode
        t0 = time.time()
        token_count = 0
        first_token_time = None

        for chunk in generate(model, tokenizer, args.prompt,
                              max_tokens=args.max_tokens,
                              temperature=args.temperature,
                              chat_format=not args.no_chat_format):
            if first_token_time is None:
                first_token_time = time.time()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            token_count += 1

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        tps = token_count / (elapsed - ttft) if elapsed > ttft and token_count > 0 else 0

        if args.verbose:
            print(f"\n\n  {token_count} tokens | {tps:.2f} tok/s | "
                  f"TTFT: {ttft:.2f}s | Total: {elapsed:.2f}s")
        else:
            print()
    else:
        # Chat mode
        print("Type your message. Ctrl+C to exit.\n")
        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue
                if user_input in ("/quit", "/exit"):
                    break

                t0 = time.time()
                token_count = 0
                print()
                for chunk in generate(model, tokenizer, user_input,
                                      max_tokens=args.max_tokens,
                                      temperature=args.temperature):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    token_count += 1

                elapsed = time.time() - t0
                tps = token_count / elapsed if elapsed > 0 else 0
                print(f"\n[{token_count} tok, {tps:.1f} tok/s]\n")

            except (KeyboardInterrupt, EOFError):
                print("\nbye.")
                break


if __name__ == "__main__":
    main()
