"""
Dataset loading utilities for Barbados Survey Plans with Rich formatting
"""
from dataset_creation_imports import *
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def load_datasets(train_path='data/Train.csv',
                  test_path='data/Test.csv',
                  submission_path='data/SampleSubmission.csv'):
    """
    Load and return train, test, and sample submission datasets
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        submission_path: Path to sample submission CSV
        
    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        sample_submission_df = pd.read_csv(submission_path)
        
        # Create summary table
        table = Table(title="Dataset Loading Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Dataset", style="cyan", width=20)
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Columns", justify="right", style="yellow")
        
        table.add_row("Training", str(len(train_df)), str(len(train_df.columns)))
        table.add_row("Test", str(len(test_df)), str(len(test_df.columns)))
        table.add_row("Submission", str(len(sample_submission_df)), str(len(sample_submission_df.columns)))
        
        console.print(table)
        
        return train_df, test_df, sample_submission_df
        
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ File not found: {e}[/bold red]")
        raise
    except Exception as e:
        console.print(f"[bold red]❌ Error loading datasets: {e}[/bold red]")
        raise


def check_and_remove_duplicates(df, subset='ID', inplace=True):
    """
    Check for and remove duplicates from dataframe
    
    Args:
        df: DataFrame to check
        subset: Column(s) to check for duplicates
        inplace: Whether to modify dataframe in place
        
    Returns:
        DataFrame: Cleaned dataframe (if inplace=False)
    """
    duplicate_count = df.duplicated(subset=subset).sum()
    
    if duplicate_count > 0:
        console.print(f"[yellow]⚠️  Found {duplicate_count} duplicate(s) based on '{subset}'[/yellow]")
    else:
        console.print(f"[green]✅ No duplicates found in '{subset}' column[/green]")
    
    if inplace:
        df.drop_duplicates(subset=subset, inplace=True)
        if duplicate_count > 0:
            console.print(f"[green]✅ Removed {duplicate_count} duplicate(s)[/green]")
        return None
    else:
        return df.drop_duplicates(subset=subset)