"""
Inference-only pipeline: Prepares and saves HuggingFace test dataset
"""
import argparse
import time
import os
import shutil
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from dataset_creation_utils import DatasetConfig
from hf_dataset_utils import create_hf_dataset, validate_images, save_hf_dataset

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only dataset preparation pipeline")

    parser.add_argument(
        "--data_path", type=str,
        default="./data/survey_plans/",
        help="Path to the folder containing anonymised images"
    )
    parser.add_argument(
        "--extract_path", type=str,
        default="./inference_images/",
        help="Path to copy and organize test images"
    )
    parser.add_argument(
        "--hf_test_path", type=str,
        default="./model_testdataset/",
        help="Output path for the Hugging Face test dataset"
    )
    return parser.parse_args()


def print_banner():
    console.rule("[bold cyan]🤗 Inference Dataset Preparation Pipeline[/bold cyan]")


def print_step(title, emoji="🔹"):
    console.print(f"\n[bold cyan]{emoji} {title}[/bold cyan]")
    console.rule(style="cyan")


def print_dataframe_summary(df, name="Dataset"):
    table = Table(title=f"{name} Preview", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(col, style="cyan")
    for _, row in df.head(5).iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)


def copy_all_images(source_path, extract_path):
    """
    Copies all anonymised images from source_path to extract_path.
    """
    os.makedirs(extract_path, exist_ok=True)
    image_files = [f for f in os.listdir(source_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise FileNotFoundError(f"No images found in {source_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Copying images...", total=len(image_files))
        for img in image_files:
            src = os.path.join(source_path, img)
            dst = os.path.join(extract_path, img)
            shutil.copy2(src, dst)
            progress.advance(task)

    return len(image_files)


def create_test_csv_from_images(source_path, output_csv):
    """
    Create a Test.csv file with a single column 'ID' derived from anonymised filenames.
    Example: anonymised_4512-063.jpg → ID = 4512-063
    """
    image_files = [f for f in os.listdir(source_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    data = []
    for img in image_files:
        ID = img.replace("anonymised_", "").split(".")[0]
        data.append({"ID": ID})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df


def main():
    start_time = time.time()
    args = parse_args()
    print_banner()

    config = DatasetConfig(
        data_path=args.data_path,
        extract_path=args.extract_path,
        drive_hf_data_path_test=args.hf_test_path,
    )

    # Step 1: Copy all anonymised images
    print_step("Copying All Test Images", "🖼️")
    copied_count = copy_all_images(config.DATA_PATH, config.EXTRACT_PATH)
    console.print(f"[green]✅ Copied {copied_count} images to:[/green] {config.EXTRACT_PATH}")

    # Step 2: Create Test.csv
    print_step("Creating Test CSV", "📋")
    test_csv_path = os.path.join(config.EXTRACT_PATH, "Test.csv")
    test_df = create_test_csv_from_images(config.EXTRACT_PATH, test_csv_path)
    console.print(f"[green]✅ Test CSV created at:[/green] {test_csv_path}")
    print_dataframe_summary(test_df, "Generated Test CSV")

    # Step 3: Create HuggingFace dataset
    print_step("Creating HuggingFace Test Dataset", "🤗")
    image_paths = [os.path.join(config.EXTRACT_PATH, f"anonymised_{row.ID}.jpg") for _, row in test_df.iterrows()]
    test_dataset = pd.DataFrame({"image_path": image_paths, "ID": test_df["ID"]})
    test_hf = create_hf_dataset(test_dataset)

    # Step 4: Validate and Save
    print_step("Validating and Saving Dataset", "💾")
    test_hf = validate_images(test_hf)
    save_hf_dataset(test_hf, config.DRIVE_HF_DATA_PATH_Test)

    elapsed = time.time() - start_time
    console.print(
        Panel(
            f"[bold cyan]✨ Inference Dataset Ready![/bold cyan]\n\n"
            f"[yellow]Images Processed:[/yellow] {len(test_hf)}\n"
            f"[yellow]Saved to:[/yellow] {config.DRIVE_HF_DATA_PATH_Test}\n"
            f"[yellow]Processing Time:[/yellow] {elapsed:.2f} seconds",
            border_style="green",
            box=box.DOUBLE,
        )
    )

    return test_hf, test_dataset


if __name__ == "__main__":
    try:
        test_hf, test_dataset = main()
    except Exception as e:
        console.print("\n[bold red]❌ Pipeline failed:[/bold red]")
        console.print_exception(show_locals=True)