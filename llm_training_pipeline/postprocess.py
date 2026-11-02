"""
Post-processing utilities for cleaning and normalizing VLM outputs
"""
import re
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# Official Barbados parishes
BARBADOS_PARISHES = [
    "Christ Church", "Saint Andrew", "Saint George", "Saint James",
    "Saint John", "Saint Joseph", "Saint Lucy", "Saint Michael",
    "Saint Peter", "Saint Philip", "Saint Thomas"
]


def clean_lt_num(value):
    """
    Clean and normalize LT Num field
    
    Args:
        value: Raw LT Num value
        
    Returns:
        str: Cleaned LT Num
    """
    if pd.isna(value):
        return value
    
    value = str(value)
    # Remove letters and unwanted symbols (#, *, _, etc.)
    value = re.sub(r"[A-Za-z#_*@]", "", value)
    # Replace '-' and '/' with '.'
    value = re.sub(r"[-/]", ".", value)
    # Remove double dots like '..' and trim whitespace
    value = re.sub(r"\.+", ".", value).strip(". ").strip()
    # Keep only first four numeric blocks (e.g., 00.00.00.000)
    value = re.sub(r'^((?:\d+\.){3}\d+).*', r'\1', value)
    
    return value


def normalize_parish(parish):
    """
    Normalize parish name to official Barbados parishes
    
    Args:
        parish: Raw parish value
        
    Returns:
        str: Normalized parish or "St. Philip" as default
    """
    if pd.isna(parish):
        return "St. Philip"
    
    parish = str(parish).strip()
    
    # Check if it's in the official list
    if parish in BARBADOS_PARISHES:
        return parish
    
    # Default to St. Philip if not found
    return "St. Philip"


def normalize_unit(unit):
    """
    Normalize unit of measurement
    
    Args:
        unit: Raw unit value
        
    Returns:
        str: Normalized unit
    """
    if pd.isna(unit):
        return unit
    
    unit = str(unit).strip().lower()
    
    if unit in ['hectares', 'hectare']:
        return 'ha'
    elif unit in ['sq m', 'square meters', 'sqm', 'm²', 'm2']:
        return 'sq m'
    elif unit in ['acres', 'acre']:
        return 'acres'
    
    return unit


def remove_parish_from_address(address):
    """
    Remove parish names from address field
    
    Args:
        address: Raw address value
        
    Returns:
        str: Address with parish names removed
    """
    if pd.isna(address):
        return address
    
    # Regex patterns for parishes (Saint + St.)
    parish_patterns = [
        r"\bSt\.?\s*Philip\b", r"\bSaint\s*Philip\b",
        r"\bSt\.?\s*Michael\b", r"\bSaint\s*Michael\b",
        r"\bSt\.?\s*Lucy\b", r"\bSaint\s*Lucy\b",
        r"\bSt\.?\s*John\b", r"\bSaint\s*John\b",
        r"\bSt\.?\s*George\b", r"\bSaint\s*George\b",
        r"\bSt\.?\s*James\b", r"\bSaint\s*James\b",
        r"\bSt\.?\s*Thomas\b", r"\bSaint\s*Thomas\b",
        r"\bSt\.?\s*Andrew\b", r"\bSaint\s*Andrew\b",
        r"\bSt\.?\s*Joseph\b", r"\bSaint\s*Joseph\b",
        r"\bSt\.?\s*Peter\b", r"\bSaint\s*Peter\b",
        r"\bChrist\s*Church\b"
    ]
    
    clean_address = address
    for pattern in parish_patterns:
        clean_address = re.sub(pattern, "", clean_address, flags=re.IGNORECASE)
    
    # Remove trailing commas, semicolons, double spaces
    clean_address = re.sub(r"[;,]\s*$", "", clean_address).strip()
    clean_address = re.sub(r"\s{2,}", " ", clean_address)
    
    return clean_address


def clean_address(addr):
    """
    Clean address by keeping everything up to last "Part..." occurrence
    and removing semicolons and quotes
    
    Args:
        addr: Raw address value
        
    Returns:
        str: Cleaned address
    """
    if pd.isna(addr):
        return addr
    
    # Keep everything up to the last ", Part..." occurrence
    match = re.search(r"(.*?,\s*Part[^,]*)", addr, flags=re.IGNORECASE)
    if match:
        addr = match.group(1)
    
    # Remove all semicolons and quotes
    addr = addr.replace(';', '').replace('"', '')
    
    return addr


def remove_trailing_zeros(value):
    """
    Remove trailing zeros from decimal numbers
    
    Args:
        value: Numeric value
        
    Returns:
        str: Number as string without trailing zeros
    """
    if pd.isna(value):
        return value
    
    try:
        # Convert to string, remove trailing zeros
        num_str = str(float(value)).rstrip('0').rstrip('.')
        return num_str
    except:
        return value


def clean_single_name(name):
    """
    Clean a single name by removing middle initials while preserving special cases
    
    Args:
        name: Raw name value
        
    Returns:
        str: Cleaned name
    """
    if pd.isna(name) or not name or str(name).strip() == '':
        return name
    
    name = str(name).strip()
    
    # Skip names that start with initials (like "H.A King" or "H.A. King")
    if re.match(r'^[A-Z]\.?\s*[A-Z]\.?\s+', name):
        return name
    
    # Skip names with quotes (nicknames like D.C "Vallan" Franklin JP)
    if '"' in name:
        return name
    
    # Skip single names (like "Simba")
    if len(name.split()) <= 1:
        return name
    
    # Protect "St." in compound names like "Michelle E. St. Clair"
    name_protected = name.replace(' St. ', ' PROTECTED_ST ')
    
    # Remove middle initials patterns:
    # 1. Single initials: "Lennox J Reid" → "Lennox Reid"
    name_cleaned = re.sub(r'\s+[A-Z]\.?\s+', ' ', name_protected)
    # 2. Multiple initials: "Jamal K.L. Gaskin" → "Jamal Gaskin"
    name_cleaned = re.sub(r'\s+[A-Z]\.[A-Z]\.?\s+', ' ', name_cleaned)
    # 3. Space-separated initials: "Lee B S Brathwaite" → "Lee Brathwaite"
    name_cleaned = re.sub(r'\s+[A-Z]\s+[A-Z]\s+', ' ', name_cleaned)
    # 4. Complex patterns like "Lee B.S Brathwaite" or "Sekani H.C Franklin"
    name_cleaned = re.sub(r'\s+[A-Z]\.[A-Z]\s+', ' ', name_cleaned)
    # 5. Handle remaining single initials that might be left
    name_cleaned = re.sub(r'\s+[A-Z]\.?\s+', ' ', name_cleaned)
    
    # Restore protected "St."
    name_cleaned = name_cleaned.replace(' PROTECTED_ST ', ' St. ')
    
    # Clean up multiple spaces and trim
    name_cleaned = re.sub(r'\s+', ' ', name_cleaned).strip()
    
    return name_cleaned


def clean_land_surveyor_names(names_array):
    """
    Clean land surveyor names by removing middle initials
    
    Args:
        names_array: Array, list, series, or single name
        
    Returns:
        Cleaned names in same format as input
    """
    # Apply cleaning to each name in the array
    if isinstance(names_array, np.ndarray):
        return np.array([clean_single_name(name) for name in names_array])
    elif isinstance(names_array, (list, pd.Series)):
        return [clean_single_name(name) for name in names_array]
    else:
        return clean_single_name(names_array)


def apply_all_cleaning(df):
    """
    Apply all cleaning functions to the dataframe
    
    Args:
        df: DataFrame with raw VLM outputs
        
    Returns:
        pd.DataFrame: Fully cleaned dataframe
    """
    console.print("\n[cyan]🧹 Applying comprehensive cleaning...[/cyan]")
    
    cleaned_df = df.copy()
    
    # Clean LT Num
    if 'LT Num' in cleaned_df.columns:
        console.print("[dim]  • Cleaning LT Num...[/dim]")
        cleaned_df['LT Num'] = cleaned_df['LT Num'].apply(clean_lt_num)
    
    # Normalize Parish
    if 'Parish' in cleaned_df.columns:
        console.print("[dim]  • Normalizing Parish...[/dim]")
        cleaned_df['Parish'] = cleaned_df['Parish'].apply(normalize_parish)
    
    # Normalize Unit of Measurement
    if 'Unit of Measurement' in cleaned_df.columns:
        console.print("[dim]  • Normalizing Unit of Measurement...[/dim]")
        cleaned_df['Unit of Measurement'] = cleaned_df['Unit of Measurement'].apply(normalize_unit)
    
    # Clean Address
    if 'Address' in cleaned_df.columns:
        console.print("[dim]  • Cleaning Address...[/dim]")
        cleaned_df['Address'] = cleaned_df['Address'].apply(remove_parish_from_address)
        cleaned_df['Address'] = cleaned_df['Address'].apply(clean_address)
    
    # Clean Land Surveyor names
    if 'Land Surveyor' in cleaned_df.columns:
        console.print("[dim]  • Cleaning Land Surveyor names...[/dim]")
        cleaned_df['Land Surveyor'] = cleaned_df['Land Surveyor'].apply(clean_land_surveyor_names)
    
    # Clean Surveyed For names
    if 'Surveyed For' in cleaned_df.columns:
        console.print("[dim]  • Cleaning Surveyed For names...[/dim]")
        cleaned_df['Surveyed For'] = cleaned_df['Surveyed For'].apply(clean_land_surveyor_names)
    
    # Remove trailing zeros from Total Area
    if 'Total Area' in cleaned_df.columns:
        console.print("[dim]  • Removing trailing zeros from Total Area...[/dim]")
        cleaned_df['Total Area'] = cleaned_df['Total Area'].apply(remove_trailing_zeros)
    
    console.print("[green]✅ All cleaning operations completed[/green]")
    
    return cleaned_df


def show_cleaning_comparison(original_df, cleaned_df, sample_size=5):
    """
    Display before/after comparison of cleaning
    
    Args:
        original_df: Original dataframe
        cleaned_df: Cleaned dataframe
        sample_size: Number of samples to show
    """
    console.print("\n[bold yellow]📊 Cleaning Comparison[/bold yellow]")
    
    fields_to_compare = ['Land Surveyor', 'Address', 'Parish', 'LT Num', 'Unit of Measurement']
    
    for field in fields_to_compare:
        if field in original_df.columns and field in cleaned_df.columns:
            console.print(f"\n[cyan]{field}:[/cyan]")
            
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
            table.add_column("Before", style="yellow", width=40)
            table.add_column("After", style="green", width=40)
            
            for idx in range(min(sample_size, len(original_df))):
                before = str(original_df.iloc[idx][field])[:40]
                after = str(cleaned_df.iloc[idx][field])[:40]
                table.add_row(before, after)
            
            console.print(table)