#!/usr/bin/env python3
"""
Simple script to help reload the YOLOv8 detection node in ComfyUI
Run this script after making changes to the node code.
"""

import importlib
import sys
import os

def reload_node():
    """Attempt to reload the YOLOv8 detection node"""
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Try to reload the module
        module_name = 'yolov8_object_detection_node'
        if module_name in sys.modules:
            print(f"Reloading {module_name}...")
            importlib.reload(sys.modules[module_name])
            print("✅ Module reloaded successfully!")
        else:
            print(f"Module {module_name} not found in sys.modules")
            print("Available modules:", [m for m in sys.modules.keys() if 'yolo' in m.lower()])
        
        # Also try reloading the main init module
        init_module = '__init__'
        if init_module in sys.modules:
            importlib.reload(sys.modules[init_module])
            print("✅ Init module reloaded!")
            
    except Exception as e:
        print(f"❌ Error reloading module: {e}")
        print("You may need to restart ComfyUI to load the updated node.")

if __name__ == "__main__":
    print("🔄 Attempting to reload YOLOv8 detection node...")
    reload_node()
    print("\n📝 If this doesn't work, please restart ComfyUI completely.")
