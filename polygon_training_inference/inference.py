import cv2, numpy as np, pandas as pd
from shapely import wkt
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Polygon
from utils.geometry_utils import GeometryUtils
from utils.model_manager import ModelManager
from utils.config import Config

class BarbadosInference:
    """Handles inference and prediction"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.geom_utils = GeometryUtils()
    
    def predict_image(self, img_path, conf_threshold=None):
        """Run inference on single image and return polygon"""
        if conf_threshold is None:
            conf_threshold = self.config.CONF_THRESHOLD
        
        img = cv2.imread(str(img_path))
        if img is None:
            return Polygon()
        
        # Run YOLO
        results = self.model.predict(img_path, conf=conf_threshold, verbose=False)
        pred_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for result in results:
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for mask, conf in zip(masks, confs):
                if conf < conf_threshold:
                    continue
                mask_resized = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]))
                pred_mask = np.logical_or(pred_mask, mask_resized).astype(np.uint8)
        
        # Convert mask to polygon
        polygon = self.geom_utils.mask_to_polygon(
            pred_mask, 
            simplify_tolerance=self.config.SIMPLIFY_TOLERANCE
        )
        
        return polygon
    
    def validate(self, df, image_dir, num_samples=None):
        """Validate predictions against ground truth"""
        if num_samples:
            df = df.head(num_samples)
        
        ious = []
        
        print("\n🔍 Validating predictions...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row['ID']
            img_path = Path(image_dir) / f"anonymised_{img_name}.jpg"
            
            if not img_path.exists():
                continue
            
            # Ground truth
            gt_polygon = wkt.loads(row['geometry']) if isinstance(row['geometry'], str) else row['geometry']
            
            # Prediction
            pred_polygon = self.predict_image(img_path)
            
            # Compute IoU (Zindi method)
            gt_mask = self.geom_utils.polygon_to_mask(gt_polygon, self.config.MASK_SIZE, self.config.MASK_SIZE)
            pred_mask = self.geom_utils.polygon_to_mask(pred_polygon, self.config.MASK_SIZE, self.config.MASK_SIZE)
            iou = self.geom_utils.compute_iou(gt_mask, pred_mask)
            
            ious.append(iou)
        
        print(f"\n📊 Validation Results:")
        print(f"   Mean IoU: {np.mean(ious):.4f}")
        print(f"   Median IoU: {np.median(ious):.4f}")
        print(f"   Min IoU: {np.min(ious):.4f}")
        print(f"   Max IoU: {np.max(ious):.4f}")
        
        return ious
    
    def generate_submission(self, test_df, image_dir, output_path="submission.csv", format_type="wkt"):
        """
        Generate submission file
    
        Args:
            format_type: "wkt" for WKT format or "list" for [(x,y),...] format
        """
        submission_data = []
    
        print("\n🚀 Generating submission...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_name = row['ID']
            img_path = Path(image_dir) / f"anonymised_{img_name}.jpg"
    
            if not img_path.exists():
                submission_data.append({
                    'ID': img_name,
                    'geometry': "POLYGON EMPTY" if format_type == "wkt" else "[]"
                })
                continue
    
            # Predict polygon
            pred_polygon = self.predict_image(img_path)
    
            # Format output
            if format_type == "list":
                coords = self.geom_utils.polygon_to_list_format(pred_polygon, flip_y=True)
                coords_as_tuples = [(x, y) for x, y in coords]
                geometry = str(coords_as_tuples)
            else:
                geometry = pred_polygon.wkt if not pred_polygon.is_empty else "POLYGON EMPTY"
    
            submission_data.append({
                'ID': img_name,
                'geometry': geometry
            })
    
        # Save submission
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
    
        print(f"\n✅ Submission saved to: {output_path}")
        print(f"   Total predictions: {len(submission_df)}")
    
        empty_count = (
            (submission_df['geometry'] == 'POLYGON EMPTY') if format_type == "wkt"
            else (submission_df['geometry'] == '[]')
        ).sum()
    
        print(f"   Empty predictions: {empty_count}")
    
        return submission_df
