#!/usr/bin/env python3
"""
Fashion and clothing-specific segmentation
Combines object detection with fine-grained segmentation for clothing items
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, Dict, Any, Tuple
import supervision as sv

class FashionSegmenter:
    def __init__(self, sam_model='vit_b', yolo_model='yolov8m-seg.pt', device='auto'):
        """
        Initialize fashion segmentation with YOLO + SAM
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("Using CUDA (GPU)")
            elif torch.backends.mps.is_available():
                # Use CPU for SAM, MPS for YOLO
                self.device = 'cpu'
                self.yolo_device = 'mps'
                print("Using CPU for SAM, MPS for YOLO")
            else:
                self.device = 'cpu'
                self.yolo_device = 'cpu'
                print("Using CPU")
        else:
            self.device = device
            self.yolo_device = device
        
        # Load YOLO for person detection
        print("Loading YOLO for person detection...")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.yolo_device)
        
        # Load SAM for fine segmentation
        sam_path = Path(f'sam_{sam_model}.pth')
        if sam_path.exists():
            print(f"Loading SAM {sam_model} model...")
            self.sam = sam_model_registry[sam_model](checkpoint=str(sam_path))
            self.sam.to(device=self.device)
            
            # Create mask generator for automatic segmentation
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam,
                points_per_side=64,  # Dense points for detailed segmentation
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=2,  # Multiple crop layers for fine details
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            # Also create predictor for targeted segmentation
            self.predictor = SamPredictor(self.sam)
        else:
            print("SAM model not found. Run segment_everything.py first to download it.")
            self.sam = None
            self.mask_generator = None
            self.predictor = None
        
        # Fashion item categories based on typical positions
        self.fashion_categories = {
            'head_region': ['hat', 'cap', 'headband', 'hair_accessory', 'glasses', 'sunglasses'],
            'neck_region': ['necklace', 'scarf', 'tie', 'collar'],
            'upper_body': ['shirt', 'jacket', 'coat', 'sweater', 'vest', 'top'],
            'arms': ['sleeves', 'watch', 'bracelet', 'gloves'],
            'waist_region': ['belt', 'waistband'],
            'lower_body': ['pants', 'jeans', 'skirt', 'shorts', 'dress_bottom'],
            'feet_region': ['shoes', 'socks', 'boots', 'sandals'],
            'accessories': ['bag', 'purse', 'backpack', 'jewelry'],
            'full_outfit': ['dress', 'jumpsuit', 'suit']
        }
    
    def segment_fashion(self, image_path: str) -> Dict[str, Any]:
        """
        Segment fashion items and clothing in an image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        print(f"Processing {Path(image_path).name} for fashion items...")
        
        results = {
            'original_image': image_rgb,
            'image_path': image_path,
            'image_size': (width, height),
            'people': [],
            'fashion_items': [],
            'other_objects': []
        }
        
        # Step 1: Detect people with YOLO
        print("Detecting people...")
        yolo_results = self.yolo(image_rgb, conf=0.3)
        
        person_masks = []
        person_boxes = []
        
        if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            for i, box in enumerate(yolo_results[0].boxes):
                class_id = int(box.cls.cpu().numpy()[0])
                if self.yolo.names[class_id] == 'person':
                    # Get person bounding box
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                    person_boxes.append((x1, y1, x2, y2))
                    
                    # Get person mask if available
                    if yolo_results[0].masks is not None:
                        mask = yolo_results[0].masks.data[i].cpu().numpy()
                        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        person_masks.append(mask_resized > 0.5)
                    
                    results['people'].append({
                        'id': i,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf.cpu().numpy()[0])
                    })
        
        print(f"Found {len(results['people'])} person(s)")
        
        # Step 2: Use SAM to segment everything within person regions
        if self.mask_generator and len(person_boxes) > 0:
            print("Segmenting fashion items within person regions...")
            
            for person_idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                # Crop person region with padding
                pad = 20
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(width, x2 + pad)
                y2_pad = min(height, y2 + pad)
                
                person_crop = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Generate masks for this person
                masks = self.mask_generator.generate(person_crop)
                
                # Process each mask
                for mask_idx, mask_data in enumerate(masks):
                    mask = mask_data['segmentation']
                    
                    # Convert back to full image coordinates
                    full_mask = np.zeros((height, width), dtype=np.uint8)
                    full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask.astype(np.uint8)
                    
                    # Get mask properties
                    bbox = mask_data['bbox']  # In crop coordinates
                    mask_x, mask_y, mask_w, mask_h = bbox
                    
                    # Adjust to full image coordinates
                    mask_x += x1_pad
                    mask_y += y1_pad
                    
                    # Classify the fashion item based on position
                    item_type = self._classify_fashion_item(
                        (mask_x, mask_y, mask_w, mask_h),
                        (x1, y1, x2, y2),
                        mask_data['area'],
                        person_crop.shape
                    )
                    
                    # Extract the fashion item
                    isolated = image_rgb.copy()
                    isolated[full_mask == 0] = 0
                    
                    # Crop to bounding box
                    crop_x1 = int(mask_x)
                    crop_y1 = int(mask_y)
                    crop_x2 = int(mask_x + mask_w)
                    crop_y2 = int(mask_y + mask_h)
                    
                    isolated_cropped = isolated[crop_y1:crop_y2, crop_x1:crop_x2]
                    mask_cropped = full_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    fashion_item = {
                        'id': len(results['fashion_items']),
                        'person_id': person_idx,
                        'type': item_type,
                        'bbox': (crop_x1, crop_y1, crop_x2, crop_y2),
                        'area': mask_data['area'],
                        'mask': full_mask,
                        'isolated_full': isolated,
                        'isolated_cropped': isolated_cropped,
                        'mask_cropped': mask_cropped
                    }
                    
                    results['fashion_items'].append(fashion_item)
            
            print(f"Found {len(results['fashion_items'])} fashion items")
        
        # Step 3: Also detect other objects in the scene
        print("Detecting other objects in scene...")
        for i, box in enumerate(yolo_results[0].boxes if len(yolo_results) > 0 and yolo_results[0].boxes is not None else []):
            class_id = int(box.cls.cpu().numpy()[0])
            class_name = self.yolo.names[class_id]
            
            if class_name != 'person':
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                
                # Get mask if available
                mask = None
                if yolo_results[0].masks is not None and i < len(yolo_results[0].masks.data):
                    mask = yolo_results[0].masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)
                
                results['other_objects'].append({
                    'id': len(results['other_objects']),
                    'type': class_name,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(box.conf.cpu().numpy()[0]),
                    'mask': mask
                })
        
        print(f"Found {len(results['other_objects'])} other objects")
        
        return results
    
    def _classify_fashion_item(self, mask_bbox: Tuple, person_bbox: Tuple, area: int, person_shape: tuple) -> str:
        """
        Classify a fashion item based on its position relative to the person
        """
        mask_x, mask_y, mask_w, mask_h = mask_bbox
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        person_height = person_y2 - person_y1
        person_width = person_x2 - person_x1
        
        # Calculate relative position
        rel_y = (mask_y + mask_h/2 - person_y1) / person_height
        rel_x = (mask_x + mask_w/2 - person_x1) / person_width
        aspect_ratio = mask_w / mask_h if mask_h > 0 else 1
        area_ratio = area / (person_shape[0] * person_shape[1])
        
        # Classify based on position
        if rel_y < 0.15:  # Top 15% - head region
            if aspect_ratio > 0.8 and aspect_ratio < 1.3:
                return "head/hair"
            return "hat/cap"
        elif rel_y < 0.25:  # Neck region
            if mask_w < person_width * 0.3:
                return "necklace/tie"
            return "collar/scarf"
        elif rel_y < 0.5:  # Upper body
            if area_ratio > 0.15:
                return "shirt/jacket"
            elif rel_x < 0.3 or rel_x > 0.7:
                return "arm/sleeve"
            else:
                return "upper_garment"
        elif rel_y < 0.6:  # Waist region
            if mask_w > person_width * 0.6:
                return "belt/waistband"
            return "mid_accessory"
        elif rel_y < 0.85:  # Lower body
            if area_ratio > 0.1:
                return "pants/skirt"
            return "lower_garment"
        else:  # Feet region
            return "shoes/socks"
    
    def save_fashion_items(self, results: Dict, output_dir: str = 'fashion_output'):
        """
        Save segmented fashion items organized by type
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        base_name = Path(results['image_path']).stem
        
        # Create organized directories
        dirs = {
            'all_items': output_path / 'all_items',
            'by_person': output_path / 'by_person',
            'by_type': output_path / 'by_type',
            'visualization': output_path / 'visualization'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print(f"\nSaving fashion segmentation results...")
        
        # Group items by person and type
        items_by_person = {}
        items_by_type = {}
        
        for item in results['fashion_items']:
            # By person
            person_id = item['person_id']
            if person_id not in items_by_person:
                items_by_person[person_id] = []
            items_by_person[person_id].append(item)
            
            # By type
            item_type = item['type']
            if item_type not in items_by_type:
                items_by_type[item_type] = []
            items_by_type[item_type].append(item)
        
        # Save fashion items
        for item in results['fashion_items']:
            item_id = item['id']
            item_type = item['type'].replace('/', '_')
            person_id = item['person_id']
            
            # Save to all_items
            if item['isolated_cropped'].size > 0:
                # Create transparent PNG
                cropped_rgba = np.zeros((*item['isolated_cropped'].shape[:2], 4), dtype=np.uint8)
                cropped_rgba[:, :, :3] = item['isolated_cropped']
                cropped_rgba[:, :, 3] = item['mask_cropped'] * 255
                
                # Save to all_items
                all_path = dirs['all_items'] / f"{base_name}_person{person_id}_{item_id:03d}_{item_type}.png"
                Image.fromarray(cropped_rgba).save(str(all_path))
                
                # Save to by_type
                type_dir = dirs['by_type'] / item_type
                type_dir.mkdir(exist_ok=True)
                type_path = type_dir / f"{base_name}_p{person_id}_{item_id:03d}.png"
                Image.fromarray(cropped_rgba).save(str(type_path))
                
                # Save to by_person
                person_dir = dirs['by_person'] / f"person_{person_id}"
                person_dir.mkdir(exist_ok=True)
                person_path = person_dir / f"{item_type}_{item_id:03d}.png"
                Image.fromarray(cropped_rgba).save(str(person_path))
        
        # Save other objects
        other_dir = output_path / 'other_objects'
        other_dir.mkdir(exist_ok=True)
        
        for obj in results['other_objects']:
            obj_type = obj['type']
            obj_id = obj['id']
            
            if obj['mask'] is not None:
                x1, y1, x2, y2 = obj['bbox']
                
                # Extract object
                isolated = results['original_image'].copy()
                isolated[obj['mask'] == 0] = 0
                isolated_cropped = isolated[y1:y2, x1:x2]
                mask_cropped = obj['mask'][y1:y2, x1:x2]
                
                # Save with transparency
                if isolated_cropped.size > 0:
                    cropped_rgba = np.zeros((*isolated_cropped.shape[:2], 4), dtype=np.uint8)
                    cropped_rgba[:, :, :3] = isolated_cropped
                    cropped_rgba[:, :, 3] = mask_cropped * 255
                    
                    obj_path = other_dir / f"{base_name}_obj_{obj_type}_{obj_id:02d}.png"
                    Image.fromarray(cropped_rgba).save(str(obj_path))
        
        # Create visualization
        self._create_fashion_visualization(results, dirs['visualization'] / f"{base_name}_fashion.jpg")
        
        # Save summary
        summary_path = output_path / f"{base_name}_fashion_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Fashion Segmentation Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Size: {results['image_size'][0]}x{results['image_size'][1]}\n\n")
            
            f.write(f"People detected: {len(results['people'])}\n")
            f.write(f"Fashion items found: {len(results['fashion_items'])}\n")
            f.write(f"Other objects: {len(results['other_objects'])}\n\n")
            
            if items_by_person:
                f.write("Fashion items by person:\n")
                for person_id, items in items_by_person.items():
                    f.write(f"  Person {person_id}: {len(items)} items\n")
                    for item in items:
                        f.write(f"    - {item['type']}\n")
            
            if items_by_type:
                f.write("\nFashion items by type:\n")
                for item_type, items in items_by_type.items():
                    f.write(f"  {item_type}: {len(items)} items\n")
            
            if results['other_objects']:
                f.write("\nOther objects in scene:\n")
                for obj in results['other_objects']:
                    f.write(f"  - {obj['type']} ({obj['confidence']:.1%})\n")
        
        print(f"Results saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  People: {len(results['people'])}")
        print(f"  Fashion items: {len(results['fashion_items'])}")
        print(f"  Other objects: {len(results['other_objects'])}")
    
    def _create_fashion_visualization(self, results: Dict, output_path: Path):
        """Create visualization of fashion segmentation"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(results['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Fashion items overlay
        axes[0, 1].imshow(results['original_image'])
        axes[0, 1].set_title(f'{len(results["fashion_items"])} Fashion Items Detected')
        
        # Color fashion items by type
        type_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        unique_types = list(set(item['type'] for item in results['fashion_items']))
        type_color_map = {t: type_colors[i % 20] for i, t in enumerate(unique_types)}
        
        for item in results['fashion_items']:
            x1, y1, x2, y2 = item['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor=type_color_map[item['type']],
                                    facecolor='none')
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(x1, y1-2, item['type'], fontsize=8,
                          color=type_color_map[item['type']],
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        axes[0, 1].axis('off')
        
        # All masks colored
        mask_overlay = np.zeros((*results['original_image'].shape[:2], 3))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results['fashion_items'])))
        
        for i, item in enumerate(results['fashion_items']):
            mask = item['mask']
            color = colors[i][:3]
            mask_overlay[mask == 1] = color
        
        axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title('Fashion Item Masks')
        axes[1, 0].axis('off')
        
        # Overlay on original
        axes[1, 1].imshow(results['original_image'])
        axes[1, 1].imshow(mask_overlay, alpha=0.5)
        axes[1, 1].set_title('Fashion Items Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fashion and clothing segmentation')
    parser.add_argument('images', nargs='+', help='Image file(s) to process')
    parser.add_argument('-o', '--output', default='fashion_output', help='Output directory')
    parser.add_argument('--sam-model', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model size')
    parser.add_argument('--yolo-model', default='yolov8m-seg.pt', help='YOLO model')
    parser.add_argument('-d', '--device', default='auto', help='Device (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = FashionSegmenter(
        sam_model=args.sam_model,
        yolo_model=args.yolo_model,
        device=args.device
    )
    
    # Process images
    for image_path in args.images:
        if Path(image_path).is_file():
            try:
                results = segmenter.segment_fashion(image_path)
                segmenter.save_fashion_items(results, args.output)
                print()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("Done!")

if __name__ == '__main__':
    main()
