"""
Model loading and setup utilities
"""
from training_imports import *
from training_config import VLMConfig


def install_dependencies():
    """
    Install required dependencies based on environment
    """
    console = Console()
    
    console.print("[cyan]📦 Installing dependencies...[/cyan]")
    
    # Check if running in Colab
    is_colab = "COLAB_" in "".join(os.environ.keys())
    
    if not is_colab:
        console.print("[yellow]Non-Colab environment detected[/yellow]")
        os.system("pip install unsloth -q")
    else:
        console.print("[yellow]Colab environment detected[/yellow]")
        import torch
        v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
        xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
        
        os.system(f"pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo -q")
        os.system('pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer -q')
        os.system("pip install --no-deps unsloth -q")
    
    # Install fixed versions
    os.system("pip install transformers==4.57.0 -q")
    os.system("pip install --no-deps trl==0.22.2 -q")
    
    console.print("[green]✅ Dependencies installed successfully[/green]")


def load_model_and_tokenizer(config=None):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback
    """
    Load VLM model and tokenizer
    
    Args:
        config: VLMConfig object or None to use defaults
        
    Returns:
        tuple: (model, tokenizer)
    """
    console = Console()
    
    if config is None:
        config = VLMConfig()
    
    console.print(f"\n[cyan]🤖 Loading model: {config.MODEL_NAME}[/cyan]")
    
    with console.status("[bold green]Loading model...", spinner="dots"):
        model, tokenizer = FastVisionModel.from_pretrained(
            config.MODEL_NAME,
            load_in_4bit=config.LOAD_IN_4BIT,
            use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
            device_map='auto'
        )
    
    console.print("[green]✅ Model and tokenizer loaded successfully[/green]")
    
    return model, tokenizer


def setup_peft_model(model, config=None):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback
    """
    Setup PEFT (LoRA) configuration for the model
    
    Args:
        model: Loaded model
        config: VLMConfig object or None to use defaults
        
    Returns:
        model: Model with PEFT adapters
    """
    console = Console()
    
    if config is None:
        config = VLMConfig()
    
    console.print("\n[cyan]⚙️  Setting up LoRA adapters...[/cyan]")
    
    # Create configuration table
    table = Table(title="LoRA Configuration", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=30)
    table.add_column("Value", style="yellow")
    
    table.add_row("Rank (r)", str(config.LORA_R))
    table.add_row("Alpha", str(config.LORA_ALPHA))
    table.add_row("Dropout", str(config.LORA_DROPOUT))
    table.add_row("Finetune Vision Layers", str(config.FINETUNE_VISION_LAYERS))
    table.add_row("Finetune Language Layers", str(config.FINETUNE_LANGUAGE_LAYERS))
    table.add_row("Finetune Attention Modules", str(config.FINETUNE_ATTENTION_MODULES))
    table.add_row("Finetune MLP Modules", str(config.FINETUNE_MLP_MODULES))
    
    console.print(table)
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.FINETUNE_VISION_LAYERS,
        finetune_language_layers=config.FINETUNE_LANGUAGE_LAYERS,
        finetune_attention_modules=config.FINETUNE_ATTENTION_MODULES,
        finetune_mlp_modules=config.FINETUNE_MLP_MODULES,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        random_state=config.SEED,
        use_rslora=config.USE_RSLORA,
        loftq_config=config.LOFTQ_CONFIG,
    )
    
    console.print("[green]✅ LoRA adapters configured successfully[/green]")
    
    return model


def print_gpu_stats():
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback, TrainerCallback
    """
    Print current GPU memory statistics
    """
    console = Console()
    
    if not torch.cuda.is_available():
        console.print("[yellow]⚠️  No GPU available[/yellow]")
        return
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    table = Table(title="GPU Statistics", box=box.ROUNDED, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="yellow")
    
    table.add_row("GPU Name", gpu_stats.name)
    table.add_row("Total Memory", f"{max_memory} GB")
    table.add_row("Reserved Memory", f"{start_gpu_memory} GB")
    table.add_row("Available Memory", f"{max_memory - start_gpu_memory} GB")
    
    console.print(table)