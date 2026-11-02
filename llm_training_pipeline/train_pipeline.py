"""
Main training pipeline for VLM fine-tuning
"""
import argparse
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text

from training_config import VLMConfig
from model_setup import install_dependencies, load_model_and_tokenizer, setup_peft_model, print_gpu_stats
from data_preparation import load_datasets, prepare_training_data, show_sample_conversation
from trainer_setup import create_trainer, train_model, save_model
console = Console()


def print_banner():
    """Print a beautiful banner for the training pipeline"""
    banner_text = Text()
    banner_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║                                                           ║\n", style="bold cyan")
    banner_text.append("║          ", style="bold cyan")
    banner_text.append("VLM TRAINING PIPELINE", style="bold yellow")
    banner_text.append("                         ║\n", style="bold cyan")
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
    parser = argparse.ArgumentParser(description="VLM Training Pipeline")
    
    parser.add_argument("--train_dataset_path", type=str, default="./model_dataset/",
                        help="Path to training dataset")
    parser.add_argument("--test_dataset_path", type=str, default="./model_testdataset/",
                        help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--model_save_path", type=str, default="./trained_model",
                        help="Path to save final model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--no_install", action="store_true",
                        help="Skip dependency installation")
    parser.add_argument("--show_sample", action="store_true",
                        help="Show a sample conversation before training")
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Update configuration
    VLMConfig.update(
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
    )
    
    config = VLMConfig()
    
    # Step 0: Install dependencies (optional)
    if not args.no_install:
        console.print("[bold cyan]📦 Step 0: Installing Dependencies[/bold cyan]")
        console.rule(style="cyan")
        install_dependencies()
    
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback
    
    # Step 1: Load model and tokenizer
    console.print("\n[bold cyan]🤖 Step 1: Loading Model[/bold cyan]")
    console.rule(style="cyan")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Step 2: Setup PEFT
    console.print("\n[bold cyan]⚙️  Step 2: Setting up LoRA Adapters[/bold cyan]")
    console.rule(style="cyan")
    model = setup_peft_model(model, config)
    
    # Step 3: Print GPU stats
    console.print("\n[bold cyan]🖥️  Step 3: GPU Statistics[/bold cyan]")
    console.rule(style="cyan")
    print_gpu_stats()
    
    # Step 4: Load datasets
    console.print("\n[bold cyan]📂 Step 4: Loading Datasets[/bold cyan]")
    console.rule(style="cyan")
    dataset, test_dataset = load_datasets(config)
    
    # Step 5: Prepare training data
    console.print("\n[bold cyan]⚙️  Step 5: Preparing Training Data[/bold cyan]")
    console.rule(style="cyan")
    train_dataset, val_dataset = prepare_training_data(dataset, config)
    
    # Show sample conversation if requested
    if args.show_sample:
        show_sample_conversation(train_dataset, index=0)
    
    # Step 6: Create trainer
    console.print("\n[bold cyan]🎯 Step 6: Setting up Trainer[/bold cyan]")
    console.rule(style="cyan")
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, config)
    
    # Step 7: Train model
    console.print("\n[bold cyan]🚀 Step 7: Training Model[/bold cyan]")
    console.rule(style="cyan")
    trainer_stats = train_model(trainer)
    
    # Step 8: Save model
    if trainer_stats is not None:
        console.print("\n[bold cyan]💾 Step 8: Saving Model[/bold cyan]")
        console.rule(style="cyan")
        save_model(model, tokenizer, args.model_save_path)
    
    # Final summary
    console.print("\n")
    console.rule("[bold green]Training Pipeline Complete![/bold green]", style="green")
    console.print("\n")
    
    summary_panel = Panel(
        f"[bold cyan]Training Summary[/bold cyan]\n\n"
        f"[yellow]Model:[/yellow] {config.MODEL_NAME}\n"
        f"[yellow]Training Samples:[/yellow] {len(train_dataset)}\n"
        f"[yellow]Validation Samples:[/yellow] {len(val_dataset)}\n"
        f"[yellow]Epochs:[/yellow] {config.NUM_TRAIN_EPOCHS}\n"
        f"[yellow]Batch Size:[/yellow] {config.PER_DEVICE_TRAIN_BATCH_SIZE}\n"
        f"[yellow]LoRA Rank:[/yellow] {config.LORA_R}\n\n"
        f"[green]✨ Model saved to:[/green] {args.model_save_path}\n"
        f"[green]✨ Checkpoints saved to:[/green] {config.OUTPUT_DIR}",
        title="[bold yellow]🎉 SUCCESS 🎉[/bold yellow]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(summary_panel)
    
    return model, tokenizer, trainer_stats


if __name__ == "__main__":
    try:
        model, tokenizer, stats = main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Training pipeline failed:[/bold red]")
        console.print_exception(show_locals=True)