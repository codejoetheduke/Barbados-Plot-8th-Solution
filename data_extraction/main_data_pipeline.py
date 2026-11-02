"""
Main pipeline for dataset creation with beautiful Rich formatting
"""
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from rich.text import Text
from dataset_creation_utils import DatasetConfig
from dataset_loader import load_datasets, check_and_remove_duplicates
from file_utils import organize_images
from data_preprocessing import prepare_training_dataset, prepare_test_dataset
from hf_dataset_utils import prepare_and_save_hf_datasets

# Initialize Rich console
console = Console()


def parse_args():
    """
    Parse command-line arguments for custom paths.
    """
    parser = argparse.ArgumentParser(description="Dataset creation pipeline")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/survey_plans/",
        help="Path to the main dataset directory"
    )
    parser.add_argument(
        "--extract_path",
        type=str,
        default="./survey_plans/",
        help="Path to store extracted or organized files"
    )
    parser.add_argument(
        "--hf_train_path",
        type=str,
        default="./model_dataset/",
        help="Output path for the Hugging Face training dataset"
    )
    parser.add_argument(
        "--hf_test_path",
        type=str,
        default="./model_testdataset/",
        help="Output path for the Hugging Face test dataset"
    )
    parser.add_argument(
        "--train_csv_path",
        type=str,
        default="data/Train.csv",
        help="Output path for the Hugging Face test dataset"
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default="data/Test.csv",
        help="Output path for the Hugging Face test dataset"
    )
    parser.add_argument(
        "--ss_csv_path",
        type=str,
        default="data/SampleSubmission.csv",
        help="Output path for the Hugging Face test dataset"
    )
    
    return parser.parse_args()


def print_banner():
    """
    Print a beautiful banner for the pipeline
    """
    banner_text = Text()
    banner_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║                                                           ║\n", style="bold cyan")
    banner_text.append("║        ", style="bold cyan")
    banner_text.append("BARBADOS SURVEY PLANS", style="bold yellow")
    banner_text.append("                            ║\n", style="bold cyan")
    banner_text.append("║            ", style="bold cyan")
    banner_text.append("Dataset Creation Pipeline", style="bold green")
    banner_text.append("                    ║\n", style="bold cyan")
    banner_text.append("║                                                           ║\n", style="bold cyan")
    banner_text.append("╚═══════════════════════════════════════════════════════════╝", style="bold cyan")
    
    console.print("\n")
    console.print(banner_text, justify="center")
    console.print("\n")


def print_step_header(step_num, title, emoji="🔹"):
    """
    Print a beautiful step header
    """
    console.print(f"\n[bold cyan]{emoji} Step {step_num}:[/bold cyan] [bold white]{title}[/bold white]")
    console.rule(style="cyan")


def print_dataframe_summary(df, name="Dataset"):
    """
    Print a beautiful summary of a dataframe
    """
    table = Table(title=f"{name} Preview", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    
    # Add columns
    for col in df.columns[:5]:  # Show first 5 columns
        table.add_column(col, style="cyan", no_wrap=False)
    
    # Add rows (first 3)
    for idx, row in df.head(3).iterrows():
        table.add_row(*[str(row[col])[:50] for col in df.columns[:5]])
    
    console.print(table)


def print_config_summary(config):
    """
    Print configuration summary in a nice panel
    """
    config_text = f"""
[cyan]Data Path:[/cyan] {config.DATA_PATH}
[cyan]Extract Path:[/cyan] {config.EXTRACT_PATH}
[cyan]HF Train Path:[/cyan] {config.DRIVE_HF_DATA_PATH}
[cyan]HF Test Path:[/cyan] {config.DRIVE_HF_DATA_PATH_Test}
    """
    
    console.print(Panel(
        config_text.strip(),
        title="[bold yellow]📁 Configuration[/bold yellow]",
        border_style="yellow",
        box=box.DOUBLE
    ))


def print_stats_table(stats_dict):
    """
    Print statistics in a beautiful table
    """
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="yellow", justify="right")
    
    for key, value in stats_dict.items():
        table.add_row(key, str(value))
    
    console.print(table)


def main():
    """
    Main pipeline for creating and saving datasets with beautiful Rich formatting
    """
    start_time = time.time()
    
    # Parse CLI args
    args = parse_args()
    
    # Print beautiful banner
    print_banner()
    
    # Initialize configuration
    config = DatasetConfig(
        data_path=args.data_path,
        extract_path=args.extract_path,
        drive_hf_data_path=args.hf_train_path,
        drive_hf_data_path_test=args.hf_test_path,
    )
    
    print_config_summary(config)
    
    # Step 1: Load datasets
    print_step_header(1, "Loading Datasets", "📂")
    
    with console.status("[bold green]Loading CSV files...", spinner="dots"):
        train_df, test_df, sample_submission_df = load_datasets(train_path=args.train_csv_path,
                                                                test_path=args.test_csv_path,
                                                                submission_path=args.ss_csv_path
                                                                )
    
    stats = {
        "Training Samples": len(train_df),
        "Test Samples": len(test_df),
        "Submission Samples": len(sample_submission_df),
        "Training Features": len(train_df.columns)
    }
    print_stats_table(stats)
    print_dataframe_summary(train_df, "Training Data")
    
    # Step 2: Check and remove duplicates
    print_step_header(2, "Checking for Duplicates", "🔍")
    
    initial_count = len(train_df)
    check_and_remove_duplicates(train_df, subset='ID')
    duplicates_removed = initial_count - len(train_df)
    
    if duplicates_removed > 0:
        console.print(f"[yellow]⚠️  Removed {duplicates_removed} duplicate(s)[/yellow]")
    else:
        console.print("[green]✅ No duplicates found[/green]")
    
    # Step 3: Organize images
    print_step_header(3, "Organizing Images", "🖼️")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Copying images...", total=100)
        train_copied, test_copied = organize_images(
            train_df, test_df, config.DATA_PATH, config.EXTRACT_PATH
        )
        progress.update(task, completed=100)
    
    img_stats = {
        "Training Images Copied": train_copied,
        "Test Images Copied": test_copied,
        "Total Images": train_copied + test_copied
    }
    print_stats_table(img_stats)
    
    # Step 4: Prepare training dataset
    print_step_header(4, "Preparing Training Dataset", "⚙️")
    
    with console.status("[bold green]Processing training data...", spinner="dots"):
        train_dataset = prepare_training_dataset(train_df, config.EXTRACT_PATH)
    
    console.print(f"[green]✅ Training dataset ready: {len(train_dataset)} samples[/green]")
    print_dataframe_summary(train_dataset, "Processed Training Data")
    
    # Step 5: Prepare test dataset
    print_step_header(5, "Preparing Test Dataset", "⚙️")
    
    with console.status("[bold green]Processing test data...", spinner="dots"):
        test_dataset = prepare_test_dataset(test_df, config.EXTRACT_PATH)
    
    console.print(f"[green]✅ Test dataset ready: {len(test_dataset)} samples[/green]")
    print_dataframe_summary(test_dataset, "Processed Test Data")
    
    # Step 6: Validate datasets
    print_step_header(6, "Validating Datasets", "✓")
    
    validations = []
    try:
        assert "image_path" in train_dataset.columns, "Expected 'image_path' in train dataset"
        validations.append(("Train image_path column", "✅ Pass"))
        
        assert "image_path" in test_dataset.columns, "Expected 'image_path' in test dataset"
        validations.append(("Test image_path column", "✅ Pass"))
        
        assert len(train_dataset) > 0, "Train dataset is empty"
        validations.append(("Train dataset not empty", "✅ Pass"))
        
        assert len(test_dataset) > 0, "Test dataset is empty"
        validations.append(("Test dataset not empty", "✅ Pass"))
        
        # Print validation results
        validation_table = Table(box=box.SIMPLE, show_header=False)
        validation_table.add_column("Check", style="cyan")
        validation_table.add_column("Status", style="green")
        
        for check, status in validations:
            validation_table.add_row(check, status)
        
        console.print(validation_table)
        console.print("\n[bold green]✅ All validations passed![/bold green]")
        
    except AssertionError as e:
        console.print(f"[bold red]❌ Validation failed: {e}[/bold red]")
        raise
    
    # Step 7: Create and save HuggingFace datasets
    print_step_header(7, "Creating HuggingFace Datasets", "🤗")
    
    console.print("[cyan]Converting to HuggingFace format and validating images...[/cyan]")
    train_hf, test_hf = prepare_and_save_hf_datasets(
        train_dataset,
        test_dataset,
        config.DRIVE_HF_DATA_PATH,
        config.DRIVE_HF_DATA_PATH_Test
    )
    
    # Final Summary
    elapsed_time = time.time() - start_time
    
    console.print("\n")
    console.rule("[bold green]Pipeline Complete![/bold green]", style="green")
    console.print("\n")
    
    # Create final summary panel
    summary_text = f"""
[bold cyan]📊 Final Statistics[/bold cyan]

[yellow]Training Examples:[/yellow] {len(train_hf):,}
[yellow]Test Examples:[/yellow] {len(test_hf):,}
[yellow]Total Processing Time:[/yellow] {elapsed_time:.2f} seconds

[bold cyan]💾 Output Locations[/bold cyan]

[green]Training Dataset:[/green] {config.DRIVE_HF_DATA_PATH}
[green]Test Dataset:[/green] {config.DRIVE_HF_DATA_PATH_Test}

[bold green]✨ All datasets successfully created and saved![/bold green]
    """
    
    console.print(Panel(
        summary_text.strip(),
        title="[bold yellow]🎉 SUCCESS 🎉[/bold yellow]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    ))
    
    return train_hf, test_hf, train_dataset, test_dataset


if __name__ == "__main__":
    try:
        train_hf, test_hf, train_dataset, test_dataset = main()
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline failed with error:[/bold red]")
        console.print_exception(show_locals=True)