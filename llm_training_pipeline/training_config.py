"""
Configuration for VLM training and inference
"""

class VLMConfig:
    """Centralized configuration for VLM training"""
    
    # Model Configuration
    MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
    LOAD_IN_4BIT = True
    USE_GRADIENT_CHECKPOINTING = "unsloth"
    
    # LoRA Configuration
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0
    LORA_BIAS = "none"
    USE_RSLORA = False
    LOFTQ_CONFIG = None
    
    # Training Configuration
    FINETUNE_VISION_LAYERS = True
    FINETUNE_LANGUAGE_LAYERS = True
    FINETUNE_ATTENTION_MODULES = True
    FINETUNE_MLP_MODULES = True
    
    # Training Hyperparameters
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    WARMUP_STEPS = 5
    NUM_TRAIN_EPOCHS = 10
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    LR_SCHEDULER_TYPE = "cosine"
    OPTIM = "adamw_8bit"
    MAX_LENGTH = 2048
    
    # Evaluation Configuration
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 10
    METRIC_FOR_BEST_MODEL = "eval_loss"
    LOAD_BEST_MODEL_AT_END = True
    EARLY_STOPPING_PATIENCE = 3
    
    # Data Configuration
    TRAIN_TEST_SPLIT = 0.10  # 10% for validation
    RANDOM_STATE = 42
    SEED = 28
    
    # Paths
    TRAIN_DATASET_PATH = "./model_dataset/"
    TEST_DATASET_PATH = "./model_testdataset/"
    OUTPUT_DIR = "outputs"
    RESULTS_CSV = "vlm_extraction_results.csv"
    CLEAN_CSV = "vlm_parsed_clean.csv"
    
    # Inference Configuration
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    DO_SAMPLE = False
    
    # Logging
    LOGGING_STEPS = 1
    REPORT_TO = "none"
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration with custom values"""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)


# Extraction Instruction Template
EXTRACTION_INSTRUCTION = """
Extract the following information from the survey plan image or text and return it as a valid JSON object.  
Follow these formatting and validation rules strictly:

FIELDS TO EXTRACT:
- Land Surveyor: The exact full name of the licensed surveyor (e.g., "Andrew Brathwaite"). Do not alter capitalization or punctuation.
- Surveyed For: The full name or organization for whom the survey was conducted. Keep exactly as written.
- Certified date: The certification date in ISO format "YYYY-MM-DD".
- Total Area: Numeric only, as a float (e.g., 1317.0). Do not include commas or units.
-Unit of Measurement:
-Normalize the unit to the correct standard abbreviation (SI or commonly accepted format).
-If the area is in square meters, convert all variations (e.g., “square meters”, “sqm”, “m²”) to “sq m”.
-If the area is in hectares, normalize to “ha”.
-If the area is in acres, normalize to “ac”.
-For any other valid area unit, keep the same unit but use its correct standardized abbreviation
- Address: Copy the full address exactly as written on the document (do not abbreviate or modify punctuation).
- Parish: Must match *exactly* one of the following 11 official Barbados parishes:
  ["Christ Church", "St. Andrew", "St. George", "St. James", "St. John", 
   "St. Joseph", "St. Lucy", "St. Michael", "St. Peter", "St. Philip", "St. Thomas"]
  - Match spelling and capitalization exactly.
  - If the parish name is not explicitly written, attempt to infer it from the address or location text 
    (e.g., if the address contains “Oistins”, infer “Christ Church”).
  - When inferring, use your knowledge of Barbados geography to match towns, districts, or landmarks 
    in the address to their corresponding parish.
    For example:
      - “Oistins” → “Christ Church”
      - “Speightstown” → “St. Peter”
      - “Bathsheba” → “St. Joseph”
      - “Bridgetown” → “St. Michael”
  - If no reliable inference can be made, default to "St. Philip".
- LT Num: Extract the lot or valuation reference number, which may appear under labels such as "Valuation No.", "Val Ref No.", "Val Ref #", "Ref No.", "Map Ref No.", "Tax Ref No.", or similar.
  It typically consists of four numeric blocks, separated by periods, slashes, or hyphens (e.g., 00.00.00.000, 00/00/00/000, or 00-00-00-000).
  Normalize it to use periods (.) as separators between the four numeric blocks (e.g., output 00.00.00.000).
  Remove any letters, symbols, or extra spaces. 

RESPONSE FORMAT:
Return only a *valid JSON object* with these exact keys:
{
  "Land Surveyor": "",
  "Surveyed For": "",
  "Certified date": "",
  "Total Area": "",
  "Unit of Measurement": "",
  "Address": "",
  "Parish": "",
  "LT Num": ""
}

Do not include any explanatory text, comments, or markdown — only the JSON object.
"""