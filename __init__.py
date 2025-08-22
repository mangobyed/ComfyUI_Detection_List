"""
YOLOv8 Object Detection Node for ComfyUI
Automatic Mask Generation using YOLOv8
"""

from .yolov8_object_detection_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
