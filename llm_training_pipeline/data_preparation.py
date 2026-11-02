"""
Data loading and preparation utilities for VLM training and inference
"""
from training_imports import *
from training_config import VLMConfig, EXTRACTION_INSTRUCTION
from datasets import load_from_disk
import os


def load_datasets(config=None, inference_only=False):
    """
    Load training and test datasets.  
    Works in both training and inference modes.
    
    Args:
        config: VLMConfig object or None to use defaults
        inference_only: If True, loads only test dataset
    
    Returns:
        tuple: (train_dataset or None, test_dataset)
    """
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback

    console = Console()
    if config is None:
        config = VLMConfig()

    console.print("\n[cyan]📂 Loading datasets...[/cyan]")

    dataset, test_dataset = None, None

    # --- Try loading training dataset ---
    if not inference_only and os.path.exists(config.TRAIN_DATASET_PATH):
        with console.status("[bold green]Loading training dataset...", spinner="dots"):
            dataset = load_from_disk(config.TRAIN_DATASET_PATH)
        console.print(f"[green]✅ Training dataset loaded: {len(dataset)} samples[/green]")
    else:
        console.print("[yellow]⚠️ Skipping training dataset (inference mode or not found)[/yellow]")

    # --- Load test dataset ---
    if os.path.exists(config.TEST_DATASET_PATH):
        with console.status("[bold green]Loading test dataset...", spinner="dots"):
            test_dataset = load_from_disk(config.TEST_DATASET_PATH)
        console.print(f"[green]✅ Test dataset loaded: {len(test_dataset)} samples[/green]")
    else:
        raise FileNotFoundError(f"[red]❌ Test dataset not found at {config.TEST_DATASET_PATH}[/red]")

    # --- Summary table ---
    table = Table(title="Dataset Information", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan", width=20)
    table.add_column("Samples", justify="right", style="yellow")
    table.add_column("Features", style="green")

    if dataset is not None:
        table.add_row("Training", str(len(dataset)), ", ".join(dataset.features.keys()))
    table.add_row("Test", str(len(test_dataset)), ", ".join(test_dataset.features.keys()))

    console.print(table)
    return dataset, test_dataset


def convert_to_conversation(sample, instruction=None):
    """
    Convert dataset sample to conversation format
    """
    if instruction is None:
        instruction = EXTRACTION_INSTRUCTION

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample.get("text", "")}]},
    ]
    return {"messages": conversation}


def prepare_training_data(dataset, config=None, instruction=None):
    """
    Prepare and split dataset for training
    """
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback

    console = Console()
    if config is None:
        config = VLMConfig()

    console.print("\n[cyan]⚙️  Preparing training data...[/cyan]")
    converted_dataset = []

    for sample in track(dataset, description="Converting samples"):
        converted_dataset.append(convert_to_conversation(sample, instruction))

    console.print(f"[green]✅ Converted {len(converted_dataset)} samples[/green]")

    # Split into train and validation
    console.print(f"[cyan]✂️  Splitting data (validation size: {config.TRAIN_TEST_SPLIT*100:.0f}%)...[/cyan]")

    train_dataset, val_dataset = train_test_split(
        converted_dataset,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_STATE,
    )

    table = Table(title="Train/Validation Split", box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("Split", style="cyan", width=20)
    table.add_column("Samples", justify="right", style="yellow")
    table.add_column("Percentage", justify="right", style="green")

    total = len(train_dataset) + len(val_dataset)
    table.add_row("Training", str(len(train_dataset)), f"{len(train_dataset)/total*100:.1f}%")
    table.add_row("Validation", str(len(val_dataset)), f"{len(val_dataset)/total*100:.1f}%")
    table.add_row("Total", str(total), "100.0%")

    console.print(table)
    return train_dataset, val_dataset


def show_sample_conversation(dataset, index=0):
    """
    Display a sample conversation for inspection
    """
    console = Console()
    if index >= len(dataset):
        console.print(f"[red]❌ Index {index} out of range (dataset size: {len(dataset)})[/red]")
        return

    sample = dataset[index]
    messages = sample.get("messages", [])

    console.print(f"\n[bold cyan]Sample Conversation (Index {index})[/bold cyan]")
    console.print("─" * 60)

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        if role == "user":
            console.print(f"\n[bold yellow]👤 User:[/bold yellow]")
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    display_text = text[:200] + "..." if len(text) > 200 else text
                    console.print(Panel(display_text, border_style="yellow", box=box.ROUNDED))
                elif item.get("type") == "image":
                    console.print("[dim]🖼️  [Image attached][/dim]")
        elif role == "assistant":
            console.print(f"\n[bold green]🤖 Assistant:[/bold green]")
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    console.print(Panel(text, border_style="green", box=box.ROUNDED))

    console.print("─" * 60)
