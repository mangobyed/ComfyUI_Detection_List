#!/usr/bin/env python3
"""
Segment EVERYTHING in an image - every object, clothing item, and element
Uses SAM (Segment Anything Model) for comprehensive segmentation
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Dict, Any
import requests
from tqdm import tqdm

class EverythingSegmenter:
    def __init__(self, model_type='vit_b', device='auto'):
        """
        Initialize SAM for segmenting everything
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to use
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("Using CUDA (GPU)")
            elif torch.backends.mps.is_available():
                self.device = 'cpu'  # SAM doesn't work well with MPS yet
                print("Using CPU (SAM doesn't support MPS yet)")
            else:
                self.device = 'cpu'
                print("Using CPU")
        else:
            self.device = device
        
        # Model URLs
        model_urls = {
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        }
        
        # Download model if needed
        model_path = Path(f'sam_{model_type}.pth')
        if not model_path.exists():
            print(f"Downloading SAM {model_type} model...")
            self._download_model(model_urls[model_type], model_path)
        
        # Load SAM model
        print(f"Loading SAM {model_type} model...")
        self.sam = sam_model_registry[model_type](checkpoint=str(model_path))
        self.sam.to(device=self.device)
        
        # Create mask generator with optimized parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,  # More points = more masks
            pred_iou_thresh=0.86,  # Quality threshold
            stability_score_thresh=0.92,  # Stability threshold
            crop_n_layers=1,  # Use crops for better small object detection
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Minimum mask size in pixels
        )
        
        print("Model loaded successfully!")
    
    def _download_model(self, url: str, path: Path):
        """Download SAM model weights"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def segment_everything(self, image_path: str, min_area: int = 100) -> Dict[str, Any]:
        """
        Segment everything in an image
        
        Args:
            image_path: Path to input image
            min_area: Minimum area for a mask to be kept
            
        Returns:
            Dictionary containing all masks and metadata
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        print(f"Processing {Path(image_path).name} ({width}x{height})...")
        print("Generating masks for everything in the image...")
        
        # Generate masks
        masks = self.mask_generator.generate(image_rgb)
        
        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Filter small masks
        masks = [m for m in masks if m['area'] >= min_area]
        
        print(f"Found {len(masks)} distinct segments")
        
        # Process masks
        results = {
            'original_image': image_rgb,
            'image_path': image_path,
            'image_size': (width, height),
            'segments': []
        }
        
        for i, mask_data in enumerate(masks):
            # Get mask
            mask = mask_data['segmentation'].astype(np.uint8)
            
            # Get bounding box
            bbox = mask_data['bbox']  # x, y, w, h format
            x, y, w, h = [int(v) for v in bbox]
            
            # Extract isolated object
            isolated_full = image_rgb.copy()
            isolated_full[mask == 0] = 0
            
            # Crop to bounding box
            isolated_cropped = isolated_full[y:y+h, x:x+w]
            mask_cropped = mask[y:y+h, x:x+w]
            
            # Estimate what this might be based on position and size
            segment_type = self._estimate_segment_type(mask_data, image.shape, i)
            
            segment_info = {
                'id': i,
                'type': segment_type,
                'area': mask_data['area'],
                'bbox': (x, y, x+w, y+h),
                'mask': mask,
                'isolated_full': isolated_full,
                'isolated_cropped': isolated_cropped,
                'mask_cropped': mask_cropped,
                'stability_score': mask_data['stability_score'],
                'predicted_iou': mask_data['predicted_iou']
            }
            results['segments'].append(segment_info)
        
        return results
    
    def _estimate_segment_type(self, mask_data: Dict, image_shape: tuple, index: int) -> str:
        """Estimate what type of segment this might be based on position and size"""
        bbox = mask_data['bbox']
        x, y, w, h = bbox
        area_ratio = mask_data['area'] / (image_shape[0] * image_shape[1])
        aspect_ratio = w / h if h > 0 else 1
        
        # Position in image (normalized)
        center_x = (x + w/2) / image_shape[1]
        center_y = (y + h/2) / image_shape[0]
        
        # Simple heuristics to guess segment type
        if area_ratio > 0.4:
            return "background"
        elif area_ratio > 0.15:
            if center_y < 0.5 and aspect_ratio > 0.5 and aspect_ratio < 2:
                return "person_upper"
            elif center_y > 0.5:
                return "person_lower"
            else:
                return "large_object"
        elif y < image_shape[0] * 0.3:  # Top third
            if aspect_ratio > 0.8 and aspect_ratio < 1.5:
                return "head_item"  # Could be head, hat, hair
            else:
                return "upper_item"
        elif y > image_shape[0] * 0.7:  # Bottom third
            return "lower_item"  # Could be shoes, pants bottom
        else:  # Middle
            if w > h:
                return "horizontal_item"
            else:
                return "vertical_item"
    
    def save_everything(self, results: Dict, output_dir: str = 'everything_output'):
        """
        Save all segmented items
        
        Args:
            results: Results from segment_everything
            output_dir: Output directory
        """
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        base_name = Path(results['image_path']).stem
        
        # Create subdirectories by type
        dirs = {
            'all_masks': output_path / 'all_masks',
            'isolated': output_path / 'isolated',
            'by_type': output_path / 'by_type',
            'transparent': output_path / 'transparent',
            'visualization': output_path / 'visualization'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print(f"\nSaving {len(results['segments'])} segments...")
        
        # Group segments by type
        segments_by_type = {}
        for segment in results['segments']:
            seg_type = segment['type']
            if seg_type not in segments_by_type:
                segments_by_type[seg_type] = []
            segments_by_type[seg_type].append(segment)
        
        # Save individual segments
        for segment in results['segments']:
            seg_id = segment['id']
            seg_type = segment['type']
            
            # Create type directory
            type_dir = dirs['by_type'] / seg_type
            type_dir.mkdir(exist_ok=True)
            
            # Save mask
            mask_path = dirs['all_masks'] / f"{base_name}_{seg_id:03d}_{seg_type}_mask.png"
            cv2.imwrite(str(mask_path), segment['mask'] * 255)
            
            # Save isolated (full size)
            isolated_path = dirs['isolated'] / f"{base_name}_{seg_id:03d}_{seg_type}.jpg"
            cv2.imwrite(str(isolated_path), cv2.cvtColor(segment['isolated_full'], cv2.COLOR_RGB2BGR))
            
            # Save transparent cropped
            if segment['isolated_cropped'].size > 0:
                cropped_rgba = np.zeros((*segment['isolated_cropped'].shape[:2], 4), dtype=np.uint8)
                cropped_rgba[:, :, :3] = segment['isolated_cropped']
                cropped_rgba[:, :, 3] = segment['mask_cropped'] * 255
                
                trans_path = dirs['transparent'] / f"{base_name}_{seg_id:03d}_{seg_type}.png"
                Image.fromarray(cropped_rgba).save(str(trans_path))
                
                # Also save in type directory
                type_trans_path = type_dir / f"{base_name}_{seg_id:03d}.png"
                Image.fromarray(cropped_rgba).save(str(type_trans_path))
        
        # Create visualization
        self._create_visualization(results, dirs['visualization'] / f"{base_name}_overview.jpg")
        
        # Save summary
        summary_path = output_path / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Size: {results['image_size'][0]}x{results['image_size'][1]}\n")
            f.write(f"Total segments: {len(results['segments'])}\n\n")
            
            f.write("Segments by type:\n")
            for seg_type, segments in segments_by_type.items():
                f.write(f"  {seg_type}: {len(segments)} segments\n")
            
            f.write("\nAll segments:\n")
            for segment in results['segments']:
                f.write(f"  [{segment['id']:3d}] {segment['type']:15s} - Area: {segment['area']:6d} pixels\n")
        
        print(f"Results saved to: {output_path}")
        print(f"\nSegments by type:")
        for seg_type, segments in segments_by_type.items():
            print(f"  {seg_type}: {len(segments)} segments")
    
    def _create_visualization(self, results: Dict, output_path: Path):
        """Create visualization showing all segments"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(results['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # All masks with different colors
        mask_overlay = np.zeros((*results['original_image'].shape[:2], 3))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results['segments'])))
        
        for i, segment in enumerate(results['segments']):
            mask = segment['mask']
            color = colors[i][:3]
            mask_overlay[mask == 1] = color
        
        axes[0, 1].imshow(mask_overlay)
        axes[0, 1].set_title(f'All {len(results["segments"])} Segments')
        axes[0, 1].axis('off')
        
        # Overlay on original
        axes[1, 0].imshow(results['original_image'])
        axes[1, 0].imshow(mask_overlay, alpha=0.5)
        axes[1, 0].set_title('Segments Overlay')
        axes[1, 0].axis('off')
        
        # Bounding boxes with labels
        axes[1, 1].imshow(results['original_image'])
        
        # Group by type for coloring
        type_colors = {}
        unique_types = list(set(s['type'] for s in results['segments']))
        type_color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        for i, t in enumerate(unique_types):
            type_colors[t] = type_color_map[i]
        
        for segment in results['segments'][:20]:  # Show first 20 to avoid clutter
            x1, y1, x2, y2 = segment['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=1, edgecolor=type_colors[segment['type']], 
                                    facecolor='none')
            axes[1, 1].add_patch(rect)
            
            # Add label
            axes[1, 1].text(x1, y1-2, f"{segment['id']}:{segment['type'][:8]}", 
                          fontsize=6, color=type_colors[segment['type']], 
                          bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        axes[1, 1].set_title('Segment Types (first 20)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment EVERYTHING in an image')
    parser.add_argument('images', nargs='+', help='Image file(s) to process')
    parser.add_argument('-o', '--output', default='everything_output', help='Output directory')
    parser.add_argument('-m', '--model', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model size (vit_b=base, vit_l=large, vit_h=huge)')
    parser.add_argument('--min-area', type=int, default=100, 
                       help='Minimum segment area in pixels')
    parser.add_argument('-d', '--device', default='auto', help='Device (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = EverythingSegmenter(model_type=args.model, device=args.device)
    
    # Process images
    for image_path in args.images:
        if Path(image_path).is_file():
            try:
                results = segmenter.segment_everything(image_path, min_area=args.min_area)
                segmenter.save_everything(results, args.output)
                print()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    print("Done!")

if __name__ == '__main__':
    main()
