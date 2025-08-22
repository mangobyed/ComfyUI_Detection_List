import torch
import cv2
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
from ultralytics import YOLO

class YOLOv8ObjectDetectionNode:
    """
    ComfyUI Node for YOLOv8 Object Detection
    Takes an image input and outputs multiple images for each detected object
    """
    
    def __init__(self):
        self.model = None
        self.device = model_management.get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], 
                             {"default": "yolov8n.pt"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.25, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.45, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "padding": ("INT", {
                    "default": 10, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1
                }),
                "max_size": ("INT", {
                    "default": 512, 
                    "min": 128, 
                    "max": 1024, 
                    "step": 64
                }),
            },
            "optional": {
                "custom_model_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("detected_objects", "object_count", "class_names", "detection_info")
    
    FUNCTION = "detect_objects"
    CATEGORY = "image/object_detection"
    
    def load_model(self, model_name, custom_model_path=""):
        """Load YOLOv8 model"""
        try:
            if custom_model_path and custom_model_path.strip():
                model_path = custom_model_path.strip()
            else:
                # Use default models
                model_path = model_name
            
            if self.model is None or self.current_model != model_path:
                print(f"Loading YOLOv8 model: {model_path}")
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.current_model = model_path
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to nano model
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
            self.current_model = "yolov8n.pt"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI tensors are in format [batch, height, width, channels]
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first batch
        
        # Convert from [0,1] to [0,255] if needed
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        # Convert to numpy and ensure uint8
        np_image = tensor.cpu().numpy().astype(np.uint8)
        
        # Convert to PIL
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image, target_size=None):
        """Convert PIL Image to ComfyUI tensor format"""
        # Resize if target size is provided
        if target_size is not None:
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert PIL to numpy
        np_image = np.array(pil_image)
        
        # Convert to float and normalize to [0,1]
        tensor = torch.from_numpy(np_image).float() / 255.0
        
        # Add batch dimension: [batch, height, width, channels]
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def crop_object(self, pil_image, bbox, padding=10):
        """Crop object from image with optional padding"""
        width, height = pil_image.size
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        
        # Crop the image
        cropped = pil_image.crop((x1, y1, x2, y2))
        return cropped
    
    def detect_objects(self, image, model_name, confidence_threshold, iou_threshold, padding, max_size, custom_model_path=""):
        """Main detection function"""
        try:
            # Load model
            self.load_model(model_name, custom_model_path)
            
            # Convert tensor to PIL
            pil_image = self.tensor_to_pil(image)
            
            # Run inference
            results = self.model(pil_image, 
                               conf=confidence_threshold, 
                               iou=iou_threshold,
                               device=self.device)
            
            # Extract detection results
            detections = results[0]
            boxes = detections.boxes
            
            print(f"YOLOv8 Detection: Model inference complete")
            
            if boxes is None or len(boxes) == 0:
                # No objects detected, return original image
                print(f"YOLOv8 Detection: No objects detected with confidence > {confidence_threshold}")
                return (image, 0, "", "No objects detected")
            
            # Process detections
            detected_images = []
            class_names = []
            detection_info_list = []
            cropped_objects = []
            
            # First pass: crop all objects
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Crop object from image
                cropped_object = self.crop_object(pil_image, bbox, padding)
                cropped_objects.append(cropped_object)
                
                # Store class name and info
                class_names.append(class_name)
                detection_info_list.append(f"Object {i+1}: {class_name} (conf: {confidence:.3f})")
            
            # Determine target size for standardization
            if cropped_objects:
                # Find the maximum dimensions among all cropped objects
                max_width = max(obj.size[0] for obj in cropped_objects)
                max_height = max(obj.size[1] for obj in cropped_objects)
                
                # Use a reasonable maximum size to avoid memory issues
                target_size = (min(max_width, max_size), min(max_height, max_size))
                
                print(f"YOLOv8 Detection: Found {len(cropped_objects)} objects, resizing to {target_size}")
                
                # Second pass: resize all objects to the same size and convert to tensors
                for cropped_object in cropped_objects:
                    object_tensor = self.pil_to_tensor(cropped_object, target_size)
                    detected_images.append(object_tensor)
                
                # Concatenate all detected objects into a batch
                batch_tensor = torch.cat(detected_images, dim=0)
                class_names_str = ", ".join(class_names)
                detection_info_str = "; ".join(detection_info_list)
                object_count = len(detected_images)
            else:
                batch_tensor = image
                class_names_str = ""
                detection_info_str = "No objects detected"
                object_count = 0
            
            return (batch_tensor, object_count, class_names_str, detection_info_str)
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return (image, 0, "", f"Error: {str(e)}")

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "YOLOv8ObjectDetectionNode": YOLOv8ObjectDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOv8ObjectDetectionNode": "YOLOv8 Object Detection"
}
