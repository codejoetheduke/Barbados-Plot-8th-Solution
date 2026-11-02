"""
Inference utilities for VLM model
"""
from training_imports import *
from training_config import VLMConfig, EXTRACTION_INSTRUCTION
import json
import pandas as pd
import re


# ============================================================
# Utility: Clean model output and extract JSON object properly
# ============================================================
def extract_json_from_output(raw_output: str):
    """
    Extract the valid JSON portion from a model output string.
    Works even if output contains 'user'/'assistant' chat logs.
    """
    try:
        json_start = raw_output.rfind('{')
        json_end = raw_output.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = raw_output[json_start:json_end]
            data = json.loads(json_str)
            return data
        else:
            return {"error": "no json found", "raw": raw_output}
    except Exception:
        return {"error": "invalid json", "raw": raw_output}


# ============================================================
# Step 1: Run inference
# ============================================================
def run_inference(model, tokenizer, test_dataset, config=None, instruction=None):
    """
    Run inference on test dataset
    """
    console = Console()

    if config is None:
        config = VLMConfig()

    if instruction is None:
        instruction = EXTRACTION_INSTRUCTION

    console.print("\n[bold cyan]🔮 Running Inference[/bold cyan]")
    console.print("─" * 60)

    # Inference configuration table
    table = Table(title="Inference Configuration", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="yellow")
    table.add_row("Test Samples", str(len(test_dataset)))
    table.add_row("Max New Tokens", str(config.MAX_NEW_TOKENS))
    table.add_row("Temperature", str(config.TEMPERATURE))
    table.add_row("Do Sample", str(config.DO_SAMPLE))
    console.print(table)

    records = []

    console.print("\n[cyan]Processing test samples...[/cyan]")

    for idx in track(range(len(test_dataset)), description="Running inference"):
        sample = test_dataset[idx]
        image = sample["image"]

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        # Convert to model-ready format
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
            padding=False,
        ).to("cuda")

        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                use_cache=True,
                temperature=config.TEMPERATURE,
                do_sample=config.DO_SAMPLE,
            )

        # Decode model output
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract just the JSON part (robust fix)
        parsed = extract_json_from_output(output_text)

        # ID
        image_id = sample.get("ID", idx)
        parsed["ID"] = image_id

        records.append(parsed)

    # Create DataFrame
    df = pd.DataFrame(records)

    console.print(f"\n[green]✅ Inference completed: {len(df)} samples processed[/green]")

    return df


# ============================================================
# Step 2: Save inference results
# ============================================================
def save_inference_results(df, output_path=None, config=None):
    console = Console()

    if config is None:
        config = VLMConfig()
    if output_path is None:
        output_path = config.RESULTS_CSV

    console.print(f"\n[cyan]💾 Saving results to {output_path}...[/cyan]")

    try:
        df.to_csv(output_path, index=False)
        console.print(f"[green]✅ Results saved successfully[/green]")
        console.print("\n[yellow]Preview of results:[/yellow]")
        console.print(df.head())
    except Exception as e:
        console.print(f"[bold red]❌ Failed to save results: {e}[/bold red]")
        raise


# ============================================================
# Step 3: Clean and normalize the parsed results
# ============================================================
def clean_inference_results(df, config=None):
    console = Console()
    if config is None:
        config = VLMConfig()

    console.print("\n[cyan]🧹 Cleaning inference results...[/cyan]")

    clean_df = pd.DataFrame(df.copy())

    # Ensure all expected columns exist
    expected_cols = [
        "ID", "Land Surveyor", "Surveyed For", "Address",
        "Certified date", "Total Area", "Unit of Measurement",
        "Parish", "LT Num"
    ]
    for col in expected_cols:
        if col not in clean_df.columns:
            clean_df[col] = ""

    # Count parsing issues
    errors = clean_df["Land Surveyor"].isna().sum() + clean_df["Land Surveyor"].eq("").sum()

    # Print stats
    table = Table(title="Cleaning Statistics", box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="yellow", justify="right")
    table.add_row("Total Samples", str(len(clean_df)))
    table.add_row("Parsing Errors (est.)", str(errors))
    table.add_row("Success Rate", f"{(len(clean_df) - errors) / len(clean_df) * 100:.2f}%")
    console.print(table)

    return clean_df


# ============================================================
# Step 4: Save cleaned results
# ============================================================
def save_clean_results(clean_df, output_path=None, config=None):
    console = Console()

    if config is None:
        config = VLMConfig()
    if output_path is None:
        output_path = config.CLEAN_CSV

    console.print(f"\n[cyan]💾 Saving cleaned results to {output_path}...[/cyan]")

    try:
        clean_df.to_csv(output_path, index=False)
        console.print(f"[green]✅ Cleaned results saved successfully[/green]")
        console.print("\n[yellow]Preview of cleaned results:[/yellow]")
        console.print(clean_df.head())
    except Exception as e:
        console.print(f"[bold red]❌ Failed to save cleaned results: {e}[/bold red]")
        raise


# ============================================================
# Step 5: Full Pipeline
# ============================================================
def run_full_inference_pipeline(model, tokenizer, test_dataset, config=None, instruction=None, apply_post_processing=False):
    console = Console()
    console.print("\n[bold yellow]═══════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]   Full Inference Pipeline[/bold yellow]")
    console.print("[bold yellow]═══════════════════════════════════════════[/bold yellow]\n")

    # Step 1: Run inference
    raw_df = run_inference(model, tokenizer, test_dataset, config, instruction)

    # Step 2: Save raw results
    save_inference_results(raw_df, config=config)

    # Step 3: Clean results
    clean_df = clean_inference_results(raw_df, config)

    # Step 4: Save cleaned results
    save_clean_results(clean_df, config=config)

    # Step 5: Optional post-processing
    if apply_post_processing:
        console.print("\n[bold cyan]🔧 Step 5: Applying Post-Processing[/bold cyan]")
        console.rule(style="cyan")

        from postprocess import apply_all_cleaning, show_cleaning_comparison
        final_df = apply_all_cleaning(clean_df)

        postprocessed_path = config.CLEAN_CSV.replace('.csv', '_postprocessed.csv')
        console.print(f"\n[cyan]💾 Saving post-processed results to {postprocessed_path}...[/cyan]")
        final_df.to_csv(postprocessed_path, index=False)
        console.print(f"[green]✅ Post-processed results saved[/green]")

        show_cleaning_comparison(clean_df, final_df, sample_size=3)
    else:
        final_df = clean_df  # ✅ ensure consistent return value

    # Summary
    console.print("\n[bold green]═══════════════════════════════════════════[/bold green]")
    console.print("[bold green]   Inference Pipeline Complete! 🎉[/bold green]")
    console.print("[bold green]═══════════════════════════════════════════[/bold green]\n")

    summary = Table(box=box.DOUBLE, show_header=True, header_style="bold cyan")
    summary.add_column("File", style="yellow", width=40)
    summary.add_column("Samples", justify="right", style="green")

    if config is None:
        config = VLMConfig()
    summary.add_row(config.RESULTS_CSV, str(len(raw_df)))
    summary.add_row(config.CLEAN_CSV, str(len(clean_df)))

    if apply_post_processing:
        postprocessed_path = config.CLEAN_CSV.replace('.csv', '_postprocessed.csv')
        summary.add_row(postprocessed_path, str(len(final_df)))

    console.print(summary)

    # ✅ Always return 3 values for consistent unpacking
    return raw_df, clean_df, final_df
