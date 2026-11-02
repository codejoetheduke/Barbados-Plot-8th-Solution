"""
HuggingFace dataset creation and processing utilities with Rich formatting
"""
from dataset_creation_imports import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def create_hf_dataset(df, image_column='image_path'):
    """
    Convert pandas DataFrame to HuggingFace Dataset
    
    Args:
        df: Pandas DataFrame
        image_column: Name of column containing image paths (will be renamed to 'image')
        
    Returns:
        Dataset: HuggingFace Dataset object
    """
    console.print(f"[cyan]📦 Converting to HuggingFace Dataset format...[/cyan]")
    
    df_copy = df.rename(columns={image_column: "image"}).copy()
    hf_dataset = Dataset.from_pandas(df_copy, preserve_index=False)
    
    console.print("[cyan]🖼️  Casting image column...[/cyan]")
    hf_dataset = hf_dataset.cast_column("image", HFImage(decode=True))
    
    console.print(f"[green]✅ HuggingFace Dataset created: {len(hf_dataset)} examples[/green]")
    return hf_dataset


def validate_images(hf_dataset, num_workers=None):
    """
    Validate and decode images in HuggingFace dataset
    
    Args:
        hf_dataset: HuggingFace Dataset object
        num_workers: Number of worker processes (defaults to CPU count - 1)
        
    Returns:
        Dataset: Validated HuggingFace Dataset
    """
    if num_workers is None:
        num_workers = min(24, max(1, (os.cpu_count() or 4) - 1))
    
    console.print(f"[cyan]🔍 Validating images with {num_workers} workers...[/cyan]")
    
    def _touch_image(example):
        # Accessing the image forces decoding; we return the example unchanged.
        _ = example["image"]
        return example
    
    # The map function itself shows a progress bar
    validated_dataset = hf_dataset.map(
        _touch_image,
        num_proc=num_workers,
        desc=f"Validating {len(hf_dataset)} images",
        batched=False,
        load_from_cache_file=True,
        writer_batch_size=1000
    )
    
    console.print(f"[green]✅ All images validated successfully[/green]")
    return validated_dataset


def save_hf_dataset(hf_dataset, save_path):
    """
    Save HuggingFace dataset to disk
    
    Args:
        hf_dataset: HuggingFace Dataset object
        save_path: Path to save the dataset
    """
    console.print(f"[cyan]💾 Saving dataset to disk...[/cyan]")
    console.print(f"[dim]   Path: {save_path}[/dim]")
    
    try:
        hf_dataset.save_to_disk(save_path)
        console.print(f"[green]✅ Dataset saved successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to save dataset: {e}[/bold red]")
        raise


def print_dataset_info(hf_dataset, dataset_name="Dataset"):
    """
    Print detailed information about the HuggingFace dataset
    
    Args:
        hf_dataset: HuggingFace Dataset object
        dataset_name: Name of the dataset for display
    """
    # Create info table
    table = Table(title=f"{dataset_name} Information", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="yellow")
    
    table.add_row("Number of Examples", str(len(hf_dataset)))
    table.add_row("Number of Features", str(len(hf_dataset.features)))
    table.add_row("Features", ", ".join(hf_dataset.features.keys()))
    
    # Add memory size if available
    try:
        size_mb = hf_dataset.dataset_size / (1024 * 1024)
        table.add_row("Dataset Size", f"{size_mb:.2f} MB")
    except:
        pass
    
    console.print(table)


def prepare_and_save_hf_datasets(train_dataset, test_dataset, 
                                  train_save_path, test_save_path,
                                  num_workers=None):
    """
    Complete pipeline to prepare and save HuggingFace datasets
    
    Args:
        train_dataset: Training DataFrame
        test_dataset: Test DataFrame
        train_save_path: Path to save training dataset
        test_save_path: Path to save test dataset
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_hf_dataset, test_hf_dataset)
    """
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   Creating HuggingFace Datasets[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    # Create training dataset
    console.print("[bold yellow]📚 Training Dataset[/bold yellow]")
    console.print("─" * 50)
    train_hf = create_hf_dataset(train_dataset)
    
    # Show first example
    if len(train_hf) > 0:
        console.print("\n[cyan]First training example preview:[/cyan]")
        first_example = train_hf[0]
        example_panel = Panel(
            f"[yellow]ID:[/yellow] {first_example.get('ID', 'N/A')}\n"
            f"[yellow]Text (truncated):[/yellow] {str(first_example.get('text', 'N/A'))[:100]}...\n"
            f"[yellow]Image:[/yellow] {type(first_example.get('image', 'N/A')).__name__}",
            title="[bold]Sample Data[/bold]",
            border_style="cyan",
            box=box.ROUNDED
        )
        console.print(example_panel)
    
    # Create test dataset
    console.print("\n[bold yellow]📚 Test Dataset[/bold yellow]")
    console.print("─" * 50)
    test_hf = create_hf_dataset(test_dataset)
    
    # Validate images
    console.print("\n[bold yellow]🔍 Image Validation[/bold yellow]")
    console.print("─" * 50)
    
    console.print("[cyan]Validating training images...[/cyan]")
    train_hf = validate_images(train_hf, num_workers)
    
    console.print("\n[cyan]Validating test images...[/cyan]")
    test_hf = validate_images(test_hf, num_workers)
    
    # Save datasets
    console.print("\n[bold yellow]💾 Saving Datasets[/bold yellow]")
    console.print("─" * 50)
    
    console.print("[cyan]Saving training dataset...[/cyan]")
    save_hf_dataset(train_hf, train_save_path)
    
    console.print("\n[cyan]Saving test dataset...[/cyan]")
    save_hf_dataset(test_hf, test_save_path)
    
    # Print final summary
    console.print("\n[bold green]✨ HuggingFace Dataset Summary[/bold green]")
    console.print("─" * 50)
    
    summary_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    summary_table.add_column("Dataset", style="yellow", width=15)
    summary_table.add_column("Examples", justify="right", style="green")
    summary_table.add_column("Save Path", style="cyan")
    
    summary_table.add_row("Training", str(len(train_hf)), train_save_path)
    summary_table.add_row("Test", str(len(test_hf)), test_save_path)
    
    console.print(summary_table)
    
    return train_hf, test_hf