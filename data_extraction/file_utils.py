"""
File management utilities with Rich progress tracking
"""
from dataset_creation_imports import *
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

console = Console()


def clean_directory(directory_path):
    """
    Delete a directory if it exists
    
    Args:
        directory_path: Path to directory to delete
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        console.print(f"[yellow]🗑️  Deleted folder:[/yellow] [cyan]{directory_path}[/cyan]")
    else:
        console.print(f"[dim]ℹ️  Folder not found, nothing to delete: {directory_path}[/dim]")


def copy_images(ids, source_dir, dest_dir, data_type='train'):
    """
    Copy images from source to destination directory with progress bar
    
    Args:
        ids: Set of IDs to copy
        source_dir: Source directory containing images
        dest_dir: Destination directory for copied images
        data_type: Type of data ('train' or 'test')
        
    Returns:
        int: Number of images copied
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied_count = 0
    missing_count = 0
    
    ids_list = list(ids)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Copying {data_type} images...", total=len(ids_list))
        
        for id_value in ids_list:
            image_name = f'anonymised_{id_value}.jpg'
            source_path = os.path.join(source_dir, image_name)
            dest_path = os.path.join(dest_dir, image_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                missing_count += 1
            
            progress.update(task, advance=1)
    
    # Print summary
    if copied_count > 0:
        console.print(f"[green]✅ {data_type.capitalize()}: {copied_count} images copied[/green]")
    if missing_count > 0:
        console.print(f"[yellow]⚠️  {data_type.capitalize()}: {missing_count} images not found[/yellow]")
    
    return copied_count


def organize_images(train_df, test_df, source_path, extract_path):
    """
    Organize images into train and test directories

    Args:
        train_df: Training dataframe (can be None for inference)
        test_df: Test dataframe
        source_path: Source directory containing all images
        extract_path: Base extraction path for organized images

    Returns:
        tuple: (train_copied_count, test_copied_count)
    """
    console.print(f"\n[bold cyan]📁 Organizing images...[/bold cyan]")
    console.print(f"[dim]Source: {source_path}[/dim]")
    console.print(f"[dim]Destination: {extract_path}[/dim]\n")

    # Clean existing directory
    clean_directory(extract_path)

    # ✅ Handle missing train_df gracefully
    if train_df is not None:
        train_ids = set(train_df["ID"]) - set(test_df["ID"])
    else:
        train_ids = set()
        console.print("[yellow]⚠️ Skipping training set (no train_df provided).[/yellow]")

    test_ids = set(test_df["ID"])

    console.print(f"[cyan]📊 IDs to process:[/cyan]")
    console.print(f"  • Training: {len(train_ids)} unique IDs")
    console.print(f"  • Test: {len(test_ids)} unique IDs\n")

    # Copy images
    train_dest = os.path.join(extract_path, "train")
    test_dest = os.path.join(extract_path, "test")

    train_copied = copy_images(train_ids, source_path, train_dest, "train") if train_ids else 0
    test_copied = copy_images(test_ids, source_path, test_dest, "test")

    # Create summary table
    table = Table(
        title="Image Organization Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Count", justify="right", style="yellow")

    table.add_row("Training Images", str(train_copied))
    table.add_row("Test Images", str(test_copied))
    table.add_row("Total Copied", str(train_copied + test_copied))

    console.print("\n")
    console.print(table)

    return train_copied, test_copied
