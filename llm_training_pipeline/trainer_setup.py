"""
Trainer setup and training utilities
"""
from training_imports import *
from training_config import VLMConfig


def create_trainer(model, tokenizer, train_dataset, val_dataset, config=None):
    """
    Create and configure SFT Trainer
    
    Args:
        model: Model with PEFT adapters
        tokenizer: Model tokenizer
        train_dataset: Training dataset in conversation format
        val_dataset: Validation dataset in conversation format
        config: VLMConfig object or None to use defaults
        
    Returns:
        SFTTrainer: Configured trainer object
    """
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback
    
    console = Console()
    
    if config is None:
        config = VLMConfig()
    
    console.print("\n[cyan]🎯 Setting up trainer...[/cyan]")
    
    # Enable training mode
    FastVisionModel.for_training(model)
    
    # Create training configuration table
    table = Table(title="Training Configuration", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=35)
    table.add_column("Value", style="yellow")
    
    table.add_row("Batch Size (per device)", str(config.PER_DEVICE_TRAIN_BATCH_SIZE))
    table.add_row("Gradient Accumulation Steps", str(config.GRADIENT_ACCUMULATION_STEPS))
    table.add_row("Effective Batch Size", str(config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS))
    table.add_row("Number of Epochs", str(config.NUM_TRAIN_EPOCHS))
    table.add_row("Learning Rate", str(config.LEARNING_RATE))
    table.add_row("Weight Decay", str(config.WEIGHT_DECAY))
    table.add_row("LR Scheduler", config.LR_SCHEDULER_TYPE)
    table.add_row("Optimizer", config.OPTIM)
    table.add_row("Warmup Steps", str(config.WARMUP_STEPS))
    table.add_row("Max Length", str(config.MAX_LENGTH))
    table.add_row("Early Stopping Patience", str(config.EARLY_STOPPING_PATIENCE))
    table.add_row("Eval Strategy", config.EVAL_STRATEGY)
    table.add_row("Eval Steps", str(config.EVAL_STEPS))
    
    console.print(table)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            logging_steps=config.LOGGING_STEPS,
            optim=config.OPTIM,
            weight_decay=config.WEIGHT_DECAY,
            lr_scheduler_type=config.LR_SCHEDULER_TYPE,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            seed=config.SEED,
            output_dir=config.OUTPUT_DIR,
            report_to=config.REPORT_TO,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.MAX_LENGTH,
            eval_strategy=config.EVAL_STRATEGY,
            eval_steps=config.EVAL_STEPS,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
    )
    
    console.print("[green]✅ Trainer configured successfully[/green]")
    
    return trainer


def train_model(trainer):
    """
    Execute training loop
    
    Args:
        trainer: Configured SFTTrainer
        
    Returns:
        trainer_stats: Training statistics
    """
    console = Console()
    
    console.print("\n[bold green]🚀 Starting training...[/bold green]")
    console.print("─" * 60)
    
    # Display training info panel
    info_panel = Panel(
        "[yellow]Training in progress...[/yellow]\n"
        "Monitor the progress bars below for:\n"
        "  • Training loss\n"
        "  • Validation loss\n"
        "  • Learning rate\n"
        "  • GPU utilization\n\n"
        "[dim]Press Ctrl+C to stop training early[/dim]",
        title="[bold cyan]Training Information[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(info_panel)
    
    try:
        trainer_stats = trainer.train()
        
        console.print("\n[bold green]✅ Training completed successfully![/bold green]")
        
        # Display final statistics
        if hasattr(trainer_stats, 'metrics'):
            metrics = trainer_stats.metrics
            
            stats_table = Table(title="Training Statistics", box=box.ROUNDED, show_header=True, header_style="bold green")
            stats_table.add_column("Metric", style="cyan", width=30)
            stats_table.add_column("Value", style="yellow")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    stats_table.add_row(key, f"{value:.6f}")
                else:
                    stats_table.add_row(key, str(value))
            
            console.print(stats_table)
        
        return trainer_stats
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[bold red]❌ Training failed with error:[/bold red]")
        console.print_exception(show_locals=True)
        raise


def save_model(model, tokenizer, save_path="./trained_model"):
    """
    Save trained model and tokenizer
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        save_path: Path to save the model
    """
    console = Console()
    
    console.print(f"\n[cyan]💾 Saving model to {save_path}...[/cyan]")
    
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        console.print(f"[green]✅ Model saved successfully to {save_path}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to save model: {e}[/bold red]")
        raise


def load_trained_model(model_path, config=None):
    """
    Load a trained model from disk for inference
    
    Args:
        model_path: Path to the trained model
        config: VLMConfig object or None to use defaults
        
    Returns:
        tuple: (model, tokenizer)
    """
    from unsloth import FastVisionModel
    
    console = Console()
    
    if config is None:
        config = VLMConfig()
    
    console.print(f"\n[cyan]📂 Loading trained model from {model_path}...[/cyan]")
    
    try:
        # Load model and tokenizer from saved path
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=config.LOAD_IN_4BIT,
            use_gradient_checkpointing="unsloth",
        )
        
        # Set to inference mode
        FastVisionModel.for_inference(model)
        
        console.print(f"[green]✅ Model loaded successfully from {model_path}[/green]")
        
        return model, tokenizer
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model: {e}[/bold red]")
        raise