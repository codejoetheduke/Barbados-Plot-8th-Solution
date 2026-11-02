"""
Data preprocessing utilities with Rich formatting
"""
from dataset_creation_imports import *
from rich.console import Console
from rich.progress import track

console = Console()


def create_target_json(row):
    """
    Create JSON string from dataframe row
    
    Args:
        row: DataFrame row containing target fields
        
    Returns:
        str: JSON string of target data
    """
    target_data = {
        "Land Surveyor": row.get("Land Surveyor"),
        "Surveyed For": row.get("Surveyed For"),
        "Certified date": row.get("Certified date"),
        "Total Area": str(row.get("Total Area")) if row.get("Total Area") is not None else None,
        "Unit of Measurement": row.get("Unit of Measurement"),
        "Address": row.get("Address"),
        "Parish": row.get("Parish"),
        "LT Num": row.get("LT Num")
    }
    
    # Convert NaN values to None for JSON null
    for key, value in target_data.items():
        if pd.isna(value) or value == '':
            target_data[key] = None
    
    return json.dumps(target_data)


def create_image_dataframe(image_dir):
    """
    Create dataframe with image paths from directory
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        DataFrame: DataFrame with ID and image_path columns
    """
    if not os.path.exists(image_dir):
        console.print(f"[bold red]❌ Image directory not found: {image_dir}[/bold red]")
        return pd.DataFrame(columns=['ID', 'image_path'])
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    console.print(f"[cyan]📸 Found {len(image_files)} images in {os.path.basename(image_dir)}/[/cyan]")
    
    image_df = pd.DataFrame({
        'ID': [f.replace('anonymised_', '').replace('.jpg', '') for f in image_files],
        'image_path': [os.path.join(image_dir, f) for f in image_files]
    })
    
    return image_df


def prepare_training_dataset(train_df, extract_path):
    """
    Prepare and format training dataset
    
    Args:
        train_df: Training dataframe
        extract_path: Base path where images are extracted
        
    Returns:
        DataFrame: Prepared training dataset with ID, text, and image_path
    """
    console.print("[cyan]⚙️  Processing training data...[/cyan]")
    
    # Clean training data
    train_clean = train_df.copy()
    
    if "geometry" in train_clean.columns:
        train_clean.drop("geometry", axis=1, inplace=True)
        console.print("[dim]  • Dropped 'geometry' column[/dim]")
    
    initial_len = len(train_clean)
    train_clean.drop_duplicates(inplace=True)
    if len(train_clean) < initial_len:
        console.print(f"[yellow]  • Removed {initial_len - len(train_clean)} duplicate rows[/yellow]")
    
    # Create target JSON with progress
    console.print("[cyan]  • Creating JSON targets...[/cyan]")
    train_clean['text'] = train_clean.apply(create_target_json, axis=1)
    train_clean = train_clean[['ID', 'text']]
    
    # Get image paths
    train_image_dir = os.path.join(extract_path, 'train')
    train_image_df = create_image_dataframe(train_image_dir)
    
    # Merge datasets
    console.print("[cyan]  • Merging with image paths...[/cyan]")
    train_dataset = pd.merge(train_clean, train_image_df, on='ID', how='inner')
    
    missing = len(train_clean) - len(train_dataset)
    if missing > 0:
        console.print(f"[yellow]  ⚠️  {missing} training samples have no matching images[/yellow]")
    
    console.print(f"[green]✅ Training dataset prepared: {len(train_dataset)} samples[/green]")
    return train_dataset


def prepare_test_dataset(test_df, extract_path):
    """
    Prepare and format test dataset
    
    Args:
        test_df: Test dataframe
        extract_path: Base path where images are extracted
        
    Returns:
        DataFrame: Prepared test dataset with ID and image_path
    """
    console.print("[cyan]⚙️  Processing test data...[/cyan]")
    
    # Clean test data
    test_clean = test_df.copy()
    test_clean = test_clean[['ID']]
    
    # Get image paths
    test_image_dir = os.path.join(extract_path, 'test')
    test_image_df = create_image_dataframe(test_image_dir)
    
    # Merge datasets
    console.print("[cyan]  • Merging with image paths...[/cyan]")
    test_dataset = pd.merge(test_clean, test_image_df, on='ID', how='inner')
    
    missing = len(test_clean) - len(test_dataset)
    if missing > 0:
        console.print(f"[yellow]  ⚠️  {missing} test samples have no matching images[/yellow]")
    
    console.print(f"[green]✅ Test dataset prepared: {len(test_dataset)} samples[/green]")
    return test_dataset