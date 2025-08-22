#!/usr/bin/env python3
"""
Object Segmentation Script using YOLOv8
Segments all objects in an image and outputs masks or isolated objects
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class ObjectSegmenter:
    def __init__(self, model_name='yolov8m-seg.pt', device='auto'):
        """
        Initialize the segmentation model
        
        Args:
            model_name: YOLOv8 segmentation model to use
                       Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
            device: Device to use ('cpu', 'mps' for Mac, 'cuda' for GPU, or 'auto')
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("Using CUDA (GPU)")
            elif torch.backends.mps.is_available():
                self.device = 'mps'
                print("Using MPS (Apple Silicon)")
            else:
                self.device = 'cpu'
                print("Using CPU")
        else:
            self.device = device
            
        # Load YOLO model
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Get class names
        self.class_names = self.model.names
        
    def segment_image(self, image_path: str, conf_threshold: float = 0.25) -> dict:
        """
        Segment all objects in an image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Run inference
        results = self.model(img_rgb, conf=conf_threshold)
        
        # Process results
        output = {
            'original_image': img_rgb,
            'image_path': image_path,
            'image_size': (width, height),
            'objects': []
        }
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for i in range(len(masks)):
                # Get mask
                mask = masks[i]
                
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Extract object using mask
                isolated_object = img_rgb.copy()
                isolated_object[mask_binary == 0] = 0  # Black background
                
                # Get bounding box
                x1, y1, x2, y2 = boxes[i].astype(int)
                cropped_object = isolated_object[y1:y2, x1:x2]
                cropped_mask = mask_binary[y1:y2, x1:x2]
                
                # Store object info
                obj_info = {
                    'id': i,
                    'class_name': self.class_names[classes[i]],
                    'class_id': classes[i],
                    'confidence': confidences[i],
                    'bbox': (x1, y1, x2, y2),
                    'mask': mask_binary,
                    'mask_resized': mask_resized,
                    'isolated_full': isolated_object,
                    'isolated_cropped': cropped_object,
                    'cropped_mask': cropped_mask
                }
                output['objects'].append(obj_info)
                
        return output
    
    def save_results(self, results: dict, output_dir: str = 'output'):
        """
        Save segmentation results to files
        
        Args:
            results: Results dictionary from segment_image
            output_dir: Directory to save outputs
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get base name from input image
        base_name = Path(results['image_path']).stem
        
        # Create subdirectories
        masks_dir = output_path / 'masks'
        isolated_dir = output_path / 'isolated'
        cropped_dir = output_path / 'cropped'
        composite_dir = output_path / 'composite'
        
        for dir_path in [masks_dir, isolated_dir, cropped_dir, composite_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Save individual objects
        print(f"\nFound {len(results['objects'])} objects in {base_name}")
        for obj in results['objects']:
            obj_id = obj['id']
            class_name = obj['class_name'].replace(' ', '_')
            confidence = obj['confidence']
            
            # File names
            prefix = f"{base_name}_obj{obj_id:02d}_{class_name}_{confidence:.2f}"
            
            # Save mask
            mask_path = masks_dir / f"{prefix}_mask.png"
            cv2.imwrite(str(mask_path), obj['mask'] * 255)
            
            # Save isolated object (full image with background removed)
            isolated_path = isolated_dir / f"{prefix}_isolated.png"
            cv2.imwrite(str(isolated_path), cv2.cvtColor(obj['isolated_full'], cv2.COLOR_RGB2BGR))
            
            # Save cropped object
            if obj['isolated_cropped'].size > 0:
                cropped_path = cropped_dir / f"{prefix}_cropped.png"
                cv2.imwrite(str(cropped_path), cv2.cvtColor(obj['isolated_cropped'], cv2.COLOR_RGB2BGR))
                
                # Save cropped with transparency
                cropped_rgba = np.zeros((*obj['isolated_cropped'].shape[:2], 4), dtype=np.uint8)
                cropped_rgba[:, :, :3] = obj['isolated_cropped']
                cropped_rgba[:, :, 3] = obj['cropped_mask'] * 255
                
                cropped_alpha_path = cropped_dir / f"{prefix}_cropped_alpha.png"
                Image.fromarray(cropped_rgba).save(str(cropped_alpha_path))
            
            print(f"  [{obj_id:2d}] {class_name}: {confidence:.2%} - Saved")
        
        # Create and save composite visualization
        self._save_composite(results, composite_dir / f"{base_name}_composite.jpg")
        
        # Save summary
        summary_path = output_path / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Size: {results['image_size'][0]}x{results['image_size'][1]}\n")
            f.write(f"Objects found: {len(results['objects'])}\n\n")
            
            for obj in results['objects']:
                f.write(f"Object {obj['id']:2d}: {obj['class_name']}\n")
                f.write(f"  Confidence: {obj['confidence']:.2%}\n")
                f.write(f"  Bounding box: {obj['bbox']}\n\n")
        
        print(f"\nResults saved to: {output_path}")
        
    def _save_composite(self, results: dict, output_path: Path):
        """Create and save a composite visualization"""
        n_objects = len(results['objects'])
        if n_objects == 0:
            print("No objects detected, skipping composite")
            return
            
        # Calculate grid size
        cols = min(4, n_objects)
        rows = (n_objects + cols - 1) // cols
        
        # Create figure
        fig_height = max(8, rows * 3)
        fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 3, fig_height))
        
        # Ensure axes is always 2D
        if (rows + 1) == 1 and cols == 1:
            axes = np.array([[axes]])
        elif (rows + 1) == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Show original with all masks
        ax = axes[0, 0]
        ax.imshow(results['original_image'])
        ax.set_title('Original + All Masks')
        ax.axis('off')
        
        # Overlay all masks with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, n_objects))
        for i, obj in enumerate(results['objects']):
            mask = obj['mask']
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask == 1] = colors[i]
            colored_mask[mask == 1, 3] = 0.4  # Alpha
            ax.imshow(colored_mask)
        
        # Hide other cells in first row
        for j in range(1, cols):
            axes[0, j].axis('off')
        
        # Show individual objects
        for i, obj in enumerate(results['objects']):
            row = (i // cols) + 1
            col = i % cols
            
            ax = axes[row, col]
            
            # Show cropped object
            if obj['isolated_cropped'].size > 0:
                ax.imshow(obj['isolated_cropped'])
                title = f"{obj['class_name']}\n{obj['confidence']:.1%}"
                ax.set_title(title, fontsize=9)
            ax.axis('off')
        
        # Hide empty cells
        for i in range(n_objects, rows * cols):
            row = (i // cols) + 1
            col = i % cols
            ax = axes[row, col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def process_batch(self, image_paths: List[str], output_dir: str = 'output', 
                     conf_threshold: float = 0.25):
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            conf_threshold: Confidence threshold
        """
        print(f"\nProcessing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")
            try:
                results = self.segment_image(image_path, conf_threshold)
                self.save_results(results, output_dir)
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Segment objects in images using YOLOv8')
    parser.add_argument('input', nargs='+', help='Input image(s) or directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--model', '-m', default='yolov8m-seg.pt', 
                      help='Model to use (yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, 
                      help='Confidence threshold (0-1)')
    parser.add_argument('--device', '-d', default='auto', 
                      help='Device to use (cpu, mps, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    for input_path in args.input:
        path = Path(input_path)
        if path.is_file():
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_paths.append(str(path))
        elif path.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend([str(p) for p in path.glob(ext)])
    
    if not image_paths:
        print("No valid images found!")
        return
    
    # Initialize segmenter
    segmenter = ObjectSegmenter(model_name=args.model, device=args.device)
    
    # Process images
    if len(image_paths) == 1:
        results = segmenter.segment_image(image_paths[0], args.conf)
        segmenter.save_results(results, args.output)
    else:
        segmenter.process_batch(image_paths, args.output, args.conf)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
