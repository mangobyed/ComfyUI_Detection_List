#!/usr/bin/env python3
"""
Quick segmentation script for extracting isolated objects
Simple interface for production use
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from typing import List, Optional

def quick_segment(
    image_path: str,
    output_dir: str = 'quick_output',
    model_name: str = 'yolov8m-seg.pt',
    conf_threshold: float = 0.25,
    save_transparent: bool = True,
    save_masks: bool = True
) -> int:
    """
    Quick segmentation of an image
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        model_name: YOLO model to use
        conf_threshold: Confidence threshold
        save_transparent: Save objects with transparent background
        save_masks: Save binary masks
        
    Returns:
        Number of objects detected
    """
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using {device.upper()}")
    
    # Load model
    print("Loading model...")
    model = YOLO(model_name)
    model.to(device)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return 0
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Run inference
    print(f"Processing {Path(image_path).name}...")
    results = model(img_rgb, conf=conf_threshold)
    
    if len(results) == 0 or results[0].masks is None:
        print("No objects detected")
        return 0
        
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    base_name = Path(image_path).stem
    
    # Process detections
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    
    n_objects = len(masks)
    print(f"Found {n_objects} object(s)")
    
    for i in range(n_objects):
        # Get class name
        class_name = model.names[classes[i]].replace(' ', '_')
        conf = confidences[i]
        
        # Get mask
        mask = masks[i]
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Get bounding box
        x1, y1, x2, y2 = boxes[i].astype(int)
        
        # Save mask if requested
        if save_masks:
            mask_path = output_path / f"{base_name}_{i:02d}_{class_name}_mask.png"
            cv2.imwrite(str(mask_path), mask_binary * 255)
        
        # Extract cropped object
        cropped_img = img_rgb[y1:y2, x1:x2]
        cropped_mask = mask_binary[y1:y2, x1:x2]
        
        # Save with transparent background if requested
        if save_transparent:
            cropped_rgba = np.zeros((*cropped_img.shape[:2], 4), dtype=np.uint8)
            cropped_rgba[:, :, :3] = cropped_img
            cropped_rgba[:, :, 3] = cropped_mask * 255
            
            alpha_path = output_path / f"{base_name}_{i:02d}_{class_name}.png"
            Image.fromarray(cropped_rgba).save(str(alpha_path))
            print(f"  [{i:2d}] {class_name} ({conf:.1%}) -> {alpha_path.name}")
        else:
            # Save with black background
            isolated = cropped_img.copy()
            isolated[cropped_mask == 0] = 0
            
            isolated_path = output_path / f"{base_name}_{i:02d}_{class_name}.jpg"
            cv2.imwrite(str(isolated_path), cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))
            print(f"  [{i:2d}] {class_name} ({conf:.1%}) -> {isolated_path.name}")
    
    return n_objects

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick object segmentation')
    parser.add_argument('images', nargs='+', help='Image file(s) to process')
    parser.add_argument('-o', '--output', default='quick_output', help='Output directory')
    parser.add_argument('-m', '--model', default='yolov8m-seg.pt', help='Model to use')
    parser.add_argument('-c', '--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no-transparent', action='store_true', help='Save without transparency')
    parser.add_argument('--no-masks', action='store_true', help='Skip saving masks')
    
    args = parser.parse_args()
    
    # Process each image
    total_objects = 0
    for image_path in args.images:
        if Path(image_path).is_file():
            n = quick_segment(
                image_path,
                args.output,
                args.model,
                args.conf,
                save_transparent=not args.no_transparent,
                save_masks=not args.no_masks
            )
            total_objects += n
            print()
    
    print(f"Total: {total_objects} objects extracted from {len(args.images)} image(s)")
    print(f"Output saved to: {args.output}/")

if __name__ == '__main__':
    main()
