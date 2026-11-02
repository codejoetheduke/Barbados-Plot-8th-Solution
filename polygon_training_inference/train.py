import pandas as pd
import random
from pathlib import Path
import yaml, shutil
from glob import glob
from collections import defaultdict
from ultralytics import YOLO
from utils.config import Config

class DatasetOrganizer:
    """Organizes dataset for YOLO training with stratified splitting"""
    
    def __init__(self, image_dir, label_dir, output_dir, df, val_split=0.1):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.df = df
        self.val_split = val_split
        
    def extract_id_prefix(self, original_id):
        """Extract prefix from ID for stratification"""
        if original_id[0].isalpha():
            return original_id.split('-')[0] if '-' in original_id else original_id
        else:
            for char in original_id:
                if char.isdigit() and char != '0':
                    return char
            return original_id[0]
    
    def organize(self):
        """Create train/val split and organize files"""
        # Create directories
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Collect annotated images
        all_labels = glob(str(self.label_dir / "*.txt"))
        all_images = []
        
        for lbl in all_labels:
            name = Path(lbl).stem
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                img_path = self.image_dir / f"{name}{ext}"
                if img_path.exists():
                    all_images.append(str(img_path))
                    break
        
        if not all_images:
            raise ValueError("No matching images found!")
        
        # Create ID mapping
        id_mapping = {}
        for _, row in self.df.iterrows():
            original_id = row['ID']
            anon_name = f"anonymised_{original_id}"
            id_mapping[anon_name] = original_id
        
        # Stratified grouping
        stratified_images = defaultdict(list)
        
        for img_path in all_images:
            base = Path(img_path).stem
            
            if base in id_mapping:
                original_id = id_mapping[base]
                prefix = self.extract_id_prefix(original_id)
                stratified_images[prefix].append(img_path)
            elif base.startswith("anonymised_"):
                original_id = base.replace("anonymised_", "")
                prefix = self.extract_id_prefix(original_id)
                stratified_images[prefix].append(img_path)
            else:
                stratified_images['OTHER'].append(img_path)
        
        # Shuffle within each stratum
        for char in stratified_images:
            random.shuffle(stratified_images[char])
        
        # Train/val split
        train_images, val_images = [], []
        
        print(f"\n📊 Stratification by ID prefix:")
        for prefix in sorted(stratified_images.keys()):
            images_in_stratum = stratified_images[prefix]
            n_total = len(images_in_stratum)
            n_val = max(1, int(self.val_split * n_total))
            
            val_images.extend(images_in_stratum[:n_val])
            train_images.extend(images_in_stratum[n_val:])
            
            print(f"   '{prefix}': {n_total} total → {n_total - n_val} train, {n_val} val")
        
        random.shuffle(train_images)
        random.shuffle(val_images)
        
        # Copy files
        self._move_files(train_images, "train")
        self._move_files(val_images, "val")
        
        print(f"\n✅ Organized {len(train_images)} training and {len(val_images)} validation samples")
        
        # Create dataset.yaml
        self._create_yaml()
        
        return self.output_dir / "dataset.yaml"
    
    def _move_files(self, image_list, split):
        """Copy images and labels to output directory"""
        for img_path in image_list:
            base = Path(img_path).name
            name = Path(img_path).stem
            
            dst_img = self.output_dir / "images" / split / base
            shutil.copy(img_path, dst_img)
            
            src_lbl = self.label_dir / f"{name}.txt"
            if src_lbl.exists():
                dst_lbl = self.output_dir / "labels" / split / f"{name}.txt"
                shutil.copy(src_lbl, dst_lbl)
    
    def _create_yaml(self):
        """Create dataset.yaml file"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['polygon']
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)


class BarbadosTrainer:
    def __init__(self, model_name="yolov8m-seg.pt", config=None):
        self.model = YOLO(model_name)
        self.config = config or Config()

    def train(self, data_yaml, epochs=100, batch=8, device=[0]):
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.config.IMAGE_SIZE,
            batch=batch,
            device=device,
            patience=20,
            optimizer='AdamW',
            lr0=3e-4,
            weight_decay=1e-2,
            seed=self.config.RANDOM_SEED,
        )
        print(f"✅ Training complete. Best model: {self.model.trainer.best}")
        return results
