import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely import wkt

class GeometryUtils:
    """Utilities for polygon and mask operations"""

    @staticmethod
    def polygon_to_mask(polygon, width=256, height=256):
        if isinstance(polygon, str):
            polygon = wkt.loads(polygon)
        if polygon is None or polygon.is_empty:
            return np.zeros((height, width), dtype=np.uint8)
        minx, miny, maxx, maxy = polygon.bounds
        coords = np.array(polygon.exterior.coords)
        x_scale = width / (maxx - minx) if maxx != minx else 1
        y_scale = height / (maxy - miny) if maxy != miny else 1
        pixel_coords = np.array([
            [(x - minx) * x_scale, (maxy - y) * y_scale]
            for x, y, *_ in coords
        ], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_coords], 1)
        return mask

    @staticmethod
    def mask_to_polygon(mask, simplify_tolerance=1.0, flip_y=True):
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Polygon()
        largest_contour = max(contours, key=cv2.contourArea)
        height = mask.shape[0]
        coords = [(float(pt[0][0]), float(height - pt[0][1]) if flip_y else float(pt[0][1]))
                  for pt in largest_contour]
        if len(coords) < 3:
            return Polygon()
        polygon = Polygon(coords)
        if simplify_tolerance > 0 and polygon.is_valid:
            polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
        return polygon

    @staticmethod
    def polygon_to_list_format(polygon, flip_y=True):
        if polygon is None or polygon.is_empty:
            return []
        coords = list(polygon.exterior.coords)
        return [[x, -y] if flip_y else [x, y] for x, y in coords]

    @staticmethod
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
