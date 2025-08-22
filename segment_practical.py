#!/usr/bin/env python3
"""
Practical segmentation for useful items only
Segments: hair, head/body, individual clothes, accessories, and scene objects
NO tiny body parts like ears, eyes, etc.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Dict, Any, Tuple
import supervision as sv
from scipy import ndimage

class PracticalSegmenter:
    def __init__(self, sam_model='vit_b', yolo_model='yolov8m.pt', device='auto'):
        """
        Initialize practical segmentation
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.yolo_device = 'cuda'
                print("Using CUDA (GPU)")
            elif torch.backends.mps.is_available():
                self.device = 'cpu'  # SAM on CPU
                self.yolo_device = 'mps'  # YOLO on MPS
                print("Using CPU for SAM, MPS for YOLO")
            else:
                self.device = 'cpu'
                self.yolo_device = 'cpu'
                print("Using CPU")
        else:
            self.device = device
            self.yolo_device = device
        
        # Load YOLO for object detection
        print("Loading YOLO for object detection...")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.yolo_device)
        
        # Load SAM
        sam_path = Path(f'sam_{sam_model}.pth')
        if sam_path.exists():
            print(f"Loading SAM {sam_model} model...")
            self.sam = sam_model_registry[sam_model](checkpoint=str(sam_path))
            self.sam.to(device=self.device)
            
            # Configure for practical segmentation
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam,
                points_per_side=32,  # Balanced for speed and quality
                pred_iou_thresh=0.86,  # Good threshold
                stability_score_thresh=0.92,
                crop_n_layers=1,  # Single crop layer for speed
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Small minimum to catch details
            )
        else:
            print("SAM model not found. Run segment_everything.py first to download.")
            sys.exit(1)
        
        # Categories we actually want
        self.wanted_categories = {
            'person_parts': ['hair', 'head', 'face', 'body', 'person'],
            'clothing': ['shirt', 'jacket', 'coat', 'pants', 'dress', 'skirt', 'shoes', 'hat'],
            'accessories': ['bag', 'backpack', 'glasses', 'watch', 'jewelry', 'belt'],
            'scene_objects': ['furniture', 'vase', 'plant', 'book', 'device', 'decoration']
        }
    
    def segment_practical(self, image_path: str, merge_body_parts: bool = True) -> Dict[str, Any]:
        """
        Segment image into practical, useful segments
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        print(f"\nProcessing {Path(image_path).name} for practical segmentation...")
        
        results = {
            'original_image': image_rgb,
            'image_path': image_path,
            'image_size': (width, height),
            'segments': {
                'hair': [],
                'head_face': [],
                'body': [],
                'clothing': [],
                'accessories': [],
                'scene_objects': []
            }
        }
        
        # Step 1: Detect objects with YOLO
        print("Detecting main objects...")
        yolo_results = self.yolo(image_rgb, conf=0.25)
        
        person_boxes = []
        other_objects = []
        
        if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            for i, box in enumerate(yolo_results[0].boxes):
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = self.yolo.names[class_id]
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                conf = float(box.conf.cpu().numpy()[0])
                
                if class_name == 'person':
                    person_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                else:
                    # Other scene objects
                    other_objects.append({
                        'type': class_name,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        print(f"Found {len(person_boxes)} person(s) and {len(other_objects)} other objects")
        
        # Step 2: Generate SAM masks
        print("Generating detailed segments...")
        all_masks = self.mask_generator.generate(image_rgb)
        
        # Sort by area (largest first)
        all_masks = sorted(all_masks, key=lambda x: x['area'], reverse=True)
        
        print(f"Generated {len(all_masks)} initial segments")
        
        # Step 3: Classify and merge segments
        processed_masks = []
        used_pixels = np.zeros((height, width), dtype=bool)
        
        print(f"Processing {len(all_masks)} segments...")
        kept_count = 0
        skipped_small = 0
        
        for mask_data in all_masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h
            x, y, w, h = [int(v) for v in bbox]
            area = mask_data['area']
            
            # Don't skip based on overlap - we want all segments!
            
            # Classify the segment
            segment_type = self._classify_segment(
                mask, bbox, person_boxes, image_rgb, area
            )
            
            # Skip only very tiny segments unless they're important
            if area < 100 and segment_type not in ['hair', 'head_face', 'accessory']:
                skipped_small += 1
                continue
            
            kept_count += 1
            
            # Process based on type
            if segment_type in ['hair', 'head_face'] and merge_body_parts:
                # Try to merge with nearby similar segments
                mask = self._merge_nearby_segments(mask, all_masks, segment_type, bbox)
            
            # Create segment info
            isolated = image_rgb.copy()
            isolated[mask == 0] = 0
            
            # Crop to bounding box
            y_min, y_max = np.where(mask.any(axis=1))[0][[0, -1]]
            x_min, x_max = np.where(mask.any(axis=0))[0][[0, -1]]
            
            isolated_cropped = isolated[y_min:y_max+1, x_min:x_max+1]
            mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]
            
            segment_info = {
                'type': segment_type,
                'bbox': (x_min, y_min, x_max, y_max),
                'area': np.sum(mask),
                'mask': mask.astype(np.uint8),
                'isolated_full': isolated,
                'isolated_cropped': isolated_cropped,
                'mask_cropped': mask_cropped.astype(np.uint8)
            }
            
            # Add to appropriate category
            if segment_type == 'hair':
                results['segments']['hair'].append(segment_info)
            elif segment_type == 'head_face':
                results['segments']['head_face'].append(segment_info)
            elif segment_type == 'body':
                results['segments']['body'].append(segment_info)
            elif segment_type in ['shirt', 'jacket', 'pants', 'dress', 'skirt', 'shoes']:
                segment_info['clothing_type'] = segment_type
                results['segments']['clothing'].append(segment_info)
            elif segment_type == 'accessory':
                results['segments']['accessories'].append(segment_info)
            elif segment_type == 'object':
                results['segments']['scene_objects'].append(segment_info)
            
            # Don't mark pixels as used since we want overlapping segments
            # used_pixels |= mask
        
        # Step 4: Add YOLO-detected scene objects
        for obj in other_objects:
            x1, y1, x2, y2 = obj['bbox']
            
            # Create mask for object region
            obj_mask = np.zeros((height, width), dtype=np.uint8)
            obj_mask[y1:y2, x1:x2] = 1
            
            # Extract object
            isolated = image_rgb.copy()
            isolated[obj_mask == 0] = 0
            isolated_cropped = isolated[y1:y2, x1:x2]
            
            segment_info = {
                'type': obj['type'],
                'bbox': obj['bbox'],
                'area': (x2-x1) * (y2-y1),
                'mask': obj_mask,
                'isolated_full': isolated,
                'isolated_cropped': isolated_cropped,
                'mask_cropped': obj_mask[y1:y2, x1:x2],
                'confidence': obj['confidence']
            }
            
            results['segments']['scene_objects'].append(segment_info)
        
        # Print debug info
        print(f"\nDebug: Kept {kept_count} segments, Skipped {skipped_small} small segments")
        
        # Print summary
        total_segments = sum(len(segs) for segs in results['segments'].values())
        print(f"\nSegmentation complete! Found {total_segments} practical segments:")
        for category, segments in results['segments'].items():
            if segments:
                print(f"  {category}: {len(segments)} items")
        
        return results
    
    def _classify_segment(self, mask: np.ndarray, bbox: tuple, person_boxes: list, 
                         image: np.ndarray, area: int) -> str:
        """
        Classify a segment into practical categories
        """
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        # Check if inside person box
        in_person = False
        person_bbox = None
        for person in person_boxes:
            px1, py1, px2, py2 = person['bbox']
            # Check if segment center is in person box
            center_x = x + w/2
            center_y = y + h/2
            if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                in_person = True
                person_bbox = person['bbox']
                break
        
        if in_person and person_bbox:
            px1, py1, px2, py2 = person_bbox
            person_height = py2 - py1
            
            # Relative position in person
            rel_y = (y + h/2 - py1) / person_height
            aspect_ratio = w / h if h > 0 else 1
            area_ratio = area / ((px2-px1) * person_height)
            
            # Analyze color for hair detection
            segment_pixels = image[mask]
            if len(segment_pixels) > 0:
                mean_color = np.mean(segment_pixels, axis=0)
                # Check for hair-like colors (browns, blacks, blondes)
                is_hair_color = self._is_hair_color(mean_color)
            else:
                is_hair_color = False
            
            # Classification logic
            if rel_y < 0.2:  # Top 20% of person
                if is_hair_color or y < py1 + person_height * 0.15:
                    return 'hair'
                elif aspect_ratio > 0.6 and aspect_ratio < 1.6:
                    return 'head_face'
                else:
                    return 'hair'  # Default top to hair
            
            elif rel_y < 0.3:  # Upper head/neck region (expanded)
                if area_ratio > 0.03:  # Lower threshold
                    return 'head_face'
                else:
                    return 'accessory'  # Could be glasses, earrings
            
            elif rel_y < 0.55:  # Upper body
                if area_ratio > 0.15:
                    return 'shirt'  # or jacket
                elif area_ratio > 0.08:
                    return 'jacket'
                else:
                    return 'accessory'
            
            elif rel_y < 0.85:  # Lower body
                if area_ratio > 0.15:
                    return 'pants'  # or skirt
                else:
                    return 'accessory'
            
            else:  # Feet region
                return 'shoes'
        
        else:
            # Not in person - it's a scene object
            return 'object'
    
    def _is_hair_color(self, color: np.ndarray) -> bool:
        """
        Check if color is likely hair color
        """
        r, g, b = color
        
        # Brown hair (most common)
        if 20 < r < 150 and 10 < g < 100 and 5 < b < 80:
            return True
        
        # Black hair
        if r < 60 and g < 60 and b < 60:
            return True
        
        # Blonde hair
        if 150 < r < 250 and 120 < g < 220 and 60 < b < 180:
            return True
        
        # Red/auburn hair
        if 100 < r < 200 and 30 < g < 120 and 10 < b < 80 and r > g * 1.3:
            return True
        
        return False
    
    def _merge_nearby_segments(self, mask: np.ndarray, all_masks: list, 
                              segment_type: str, bbox: tuple) -> np.ndarray:
        """
        Merge nearby segments of similar type
        """
        x, y, w, h = bbox
        merged_mask = mask.copy()
        
        # Look for nearby segments to merge
        for other_mask_data in all_masks:
            other_mask = other_mask_data['segmentation']
            other_bbox = other_mask_data['bbox']
            ox, oy, ow, oh = other_bbox
            
            # Check if nearby (within 20 pixels)
            x_dist = min(abs(x - (ox + ow)), abs((x + w) - ox))
            y_dist = min(abs(y - (oy + oh)), abs((y + h) - oy))
            
            if x_dist < 20 and y_dist < 20:
                # Check if similar type (would be classified the same)
                if segment_type in ['hair', 'head_face']:
                    # Merge if touching or very close
                    dilated_mask = ndimage.binary_dilation(merged_mask, iterations=10)
                    if np.any(dilated_mask & other_mask):
                        merged_mask |= other_mask
        
        return merged_mask
    
    def save_practical_segments(self, results: Dict, output_dir: str = 'practical_output'):
        """
        Save practical segments organized by category
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        base_name = Path(results['image_path']).stem
        
        # Create directories
        dirs = {}
        for category in results['segments'].keys():
            if results['segments'][category]:  # Only create if has items
                dirs[category] = output_path / category
                dirs[category].mkdir(exist_ok=True)
        
        # Also create combined directory
        all_dir = output_path / 'all_segments'
        all_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving practical segments...")
        
        # Save each category
        segment_id = 0
        for category, segments in results['segments'].items():
            for i, segment in enumerate(segments):
                if segment['isolated_cropped'].size == 0:
                    continue
                
                # Create transparent PNG
                cropped_rgba = np.zeros((*segment['isolated_cropped'].shape[:2], 4), dtype=np.uint8)
                cropped_rgba[:, :, :3] = segment['isolated_cropped']
                cropped_rgba[:, :, 3] = segment['mask_cropped'] * 255
                
                # Determine filename
                if category == 'clothing' and 'clothing_type' in segment:
                    item_name = segment['clothing_type']
                elif category == 'scene_objects' and 'type' in segment:
                    item_name = segment['type']
                else:
                    item_name = category.rstrip('s')  # Remove plural
                
                # Save to category folder
                if category in dirs:
                    cat_path = dirs[category] / f"{base_name}_{item_name}_{i:02d}.png"
                    Image.fromarray(cropped_rgba).save(str(cat_path))
                
                # Save to all folder
                all_path = all_dir / f"{base_name}_{segment_id:03d}_{item_name}.png"
                Image.fromarray(cropped_rgba).save(str(all_path))
                
                segment_id += 1
        
        # Create visualization
        self._create_visualization(results, output_path / f"{base_name}_practical.jpg")
        
        # Save summary
        summary_path = output_path / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Practical Segmentation Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Size: {results['image_size'][0]}x{results['image_size'][1]}\n\n")
            
            total = sum(len(segs) for segs in results['segments'].values())
            f.write(f"Total segments: {total}\n\n")
            
            f.write("Segments by category:\n")
            for category, segments in results['segments'].items():
                if segments:
                    f.write(f"\n{category.upper()}:\n")
                    for i, seg in enumerate(segments):
                        if 'clothing_type' in seg:
                            name = seg['clothing_type']
                        elif 'type' in seg:
                            name = seg['type']
                        else:
                            name = category.rstrip('s')
                        f.write(f"  [{i}] {name} - {seg['area']} pixels\n")
        
        print(f"Results saved to: {output_path}")
        
        # Print summary
        print("\nSummary:")
        for category, segments in results['segments'].items():
            if segments:
                print(f"  {category}: {len(segments)} items")
    
    def _create_visualization(self, results: Dict, output_path: Path):
        """
        Create clean visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original
        axes[0, 0].imshow(results['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # All segments with labels
        axes[0, 1].imshow(results['original_image'])
        axes[0, 1].set_title('Practical Segments')
        
        # Define colors for each category
        category_colors = {
            'hair': 'purple',
            'head_face': 'yellow',
            'body': 'cyan',
            'clothing': 'green',
            'accessories': 'orange',
            'scene_objects': 'red'
        }
        
        for category, segments in results['segments'].items():
            color = category_colors.get(category, 'blue')
            for seg in segments:
                x1, y1, x2, y2 = seg['bbox']
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=2, edgecolor=color,
                                        facecolor='none')
                axes[0, 1].add_patch(rect)
                
                # Add label
                label = seg.get('clothing_type', seg.get('type', category.rstrip('s')))
                axes[0, 1].text(x1, y1-2, label, fontsize=8, color=color,
                              bbox=dict(boxstyle='round,pad=0.2', 
                                      facecolor='white', alpha=0.8))
        axes[0, 1].axis('off')
        
        # Mask overlay
        mask_overlay = np.zeros((*results['original_image'].shape[:2], 3))
        
        segment_idx = 0
        for category, segments in results['segments'].items():
            for seg in segments:
                color = plt.cm.rainbow(segment_idx / 20)[:3]
                mask_overlay[seg['mask'] == 1] = color
                segment_idx += 1
        
        axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title('Segment Masks')
        axes[1, 0].axis('off')
        
        # Overlay
        axes[1, 1].imshow(results['original_image'])
        axes[1, 1].imshow(mask_overlay, alpha=0.4)
        axes[1, 1].set_title('Segments Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Practical segmentation - no tiny body parts!')
    parser.add_argument('images', nargs='+', help='Image file(s) to process')
    parser.add_argument('-o', '--output', default='practical_output', help='Output directory')
    parser.add_argument('--no-merge', action='store_true', help="Don't merge body parts")
    parser.add_argument('-d', '--device', default='auto', help='Device (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Initialize
    segmenter = PracticalSegmenter(device=args.device)
    
    # Process images
    for image_path in args.images:
        if Path(image_path).is_file():
            try:
                results = segmenter.segment_practical(
                    image_path, 
                    merge_body_parts=not args.no_merge
                )
                segmenter.save_practical_segments(results, args.output)
                print()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("Done!")

if __name__ == '__main__':
    main()
