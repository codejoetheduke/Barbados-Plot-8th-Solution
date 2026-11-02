"""
Main inference pipeline for VLM extraction
"""
import argparse
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text

from training_config import VLMConfig
from model_setup import load_model_and_tokenizer, print_gpu_stats
from data_preparation import load_datasets
from inference_utils import run_full_inference_pipeline

console = Console()


def print_banner():
    """Print a beautiful banner for the inference pipeline"""
    banner_text = Text()
    banner_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║                                                           ║\n", style="bold cyan")
    banner_text.append("║         ", style="bold cyan")
    banner_text.append("VLM INFERENCE PIPELINE", style="bold yellow")
    banner_text.append("                        ║\n", style="bold cyan")
    banner_text.append("║        ", style="bold cyan")
    banner_text.append("Barbados Survey Plans Extraction", style="bold green")
    banner_text.append("            ║\n", style="bold cyan")
    banner_text.append("║                                                           ║\n", style="bold cyan")
    banner_text.append("╚═══════════════════════════════════════════════════════════╝", style="bold cyan")
    
    console.print("\n")
    console.print(banner_text, justify="center")
    console.print("\n")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="VLM Inference Pipeline")
    
    parser.add_argument("--model_path", type=str, default="./trained_model",
                        help="Path to trained model (or use base model if not trained)")
    parser.add_argument("--test_dataset_path", type=str, default="./model_testdataset/",
                        help="Path to test dataset")
    parser.add_argument("--results_csv", type=str, default="vlm_extraction_results.csv",
                        help="Output path for raw results")
    parser.add_argument("--clean_csv", type=str, default="vlm_parsed_clean.csv",
                        help="Output path for cleaned results")
    parser.add_argument("--use_base_model", action="store_true",
                        help="Use base model instead of fine-tuned model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Disable post-processing (enabled by default)")
    parser.add_argument("--no_install", action="store_true",
                        help="Skip dependency installation")
    
    return parser.parse_args()


def main():
    """Main inference pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Determine if post-processing should be applied
    apply_postprocessing = not args.no_postprocess
    
    # Update configuration
    VLMConfig.update(
        test_dataset_path=args.test_dataset_path,
        results_csv=args.results_csv,
        clean_csv=args.clean_csv,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    config = VLMConfig()
    
    # Step 0: Install dependencies (optional)
    if not args.no_install:
        console.print("[bold cyan]📦 Step 0: Installing Dependencies[/bold cyan]")
        console.rule(style="cyan")
        from model_setup import install_dependencies
        install_dependencies()
    
    # Import unsloth after potential installation
    from unsloth import FastVisionModel
    
    # Step 1: Load model and tokenizer
    console.print("\n[bold cyan]🤖 Step 1: Loading Model[/bold cyan]")
    console.rule(style="cyan")
    
    if args.use_base_model:
        console.print("[yellow]Using base model (not fine-tuned)[/yellow]")
        model, tokenizer = load_model_and_tokenizer(config)
        # Set to inference mode
        FastVisionModel.for_inference(model)
    else:
        console.print(f"[yellow]Loading fine-tuned model from: {args.model_path}[/yellow]")
        try:
            # Import load_trained_model from trainer_setup
            from trainer_setup import load_trained_model
            model, tokenizer = load_trained_model(args.model_path, config)
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load fine-tuned model: {e}[/yellow]")
            console.print("[yellow]Falling back to base model...[/yellow]")
            model, tokenizer = load_model_and_tokenizer(config)
            FastVisionModel.for_inference(model)
    
    # Step 2: Print GPU stats
    console.print("\n[bold cyan]🖥️  Step 2: GPU Statistics[/bold cyan]")
    console.rule(style="cyan")
    print_gpu_stats()
    
    # Step 3: Load test dataset
    console.print("\n[bold cyan]📂 Step 3: Loading Test Dataset[/bold cyan]")
    console.rule(style="cyan")
    _, test_dataset = load_datasets(config, inference_only=True)
    
    console.print(f"[cyan]Test dataset loaded: {len(test_dataset)} samples[/cyan]")
    
    # Step 4: Run inference
    console.print("\n[bold cyan]🔮 Step 4: Running Inference Pipeline[/bold cyan]")
    console.rule(style="cyan")
    
    if apply_postprocessing:
        console.print("[green]Post-processing: ENABLED ✓[/green]")
    else:
        console.print("[yellow]Post-processing: DISABLED[/yellow]")
    
    # Run pipeline with post-processing toggle
    results = run_full_inference_pipeline(
        model, 
        tokenizer, 
        test_dataset, 
        config,
        apply_post_processing=apply_postprocessing
    )
    
    # Unpack results
    if apply_postprocessing:
        raw_df, clean_df, final_df = results
    else:
        raw_df, clean_df = results
        final_df = clean_df
    
    # Final summary
    console.print("\n")
    console.rule("[bold green]Inference Pipeline Complete![/bold green]", style="green")
    console.print("\n")
    
    # Build summary text
    summary_text = (
        f"[bold cyan]Inference Summary[/bold cyan]\n\n"
        f"[yellow]Model:[/yellow] {args.model_path if not args.use_base_model else config.MODEL_NAME}\n"
        f"[yellow]Test Samples:[/yellow] {len(test_dataset)}\n"
        f"[yellow]Successfully Processed:[/yellow] {len(final_df)}\n"
        f"[yellow]Max Tokens:[/yellow] {config.MAX_NEW_TOKENS}\n"
        f"[yellow]Temperature:[/yellow] {config.TEMPERATURE}\n"
        f"[yellow]Post-processing:[/yellow] {'ENABLED ✓' if apply_postprocessing else 'DISABLED'}\n\n"
        f"[green]✨ Raw results saved to:[/green] {config.RESULTS_CSV}\n"
        f"[green]✨ Cleaned results saved to:[/green] {config.CLEAN_CSV}\n"
    )
    
    if apply_postprocessing:
        postprocessed_path = config.CLEAN_CSV.replace('.csv', '_postprocessed.csv')
        summary_text += f"[green]✨ Post-processed results saved to:[/green] {postprocessed_path}\n"
        summary_text += (
            f"\n[bold green]Post-processing applied:[/bold green]\n"
            f"  • LT Num normalized\n"
            f"  • Parish validated\n"
            f"  • Addresses cleaned\n"
            f"  • Names standardized\n"
            f"  • Units normalized\n"
            f"  • Trailing zeros removed"
        )
    
    summary_panel = Panel(
        summary_text,
        title="[bold yellow]🎉 SUCCESS 🎉[/bold yellow]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(summary_panel)
    
    return raw_df, clean_df, final_df


if __name__ == "__main__":
    try:
        raw_df, clean_df, final_df = main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Inference interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Inference pipeline failed:[/bold red]")
        console.print_exception(show_locals=True)