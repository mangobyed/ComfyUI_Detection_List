import torch
import cv2
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
from ultralytics import YOLO

class YOLOv8ObjectDetectionNode:
    """
    ComfyUI Node for YOLOv8 Object Extraction
    Extracts inanimate objects from images, excluding people and animals
    Outputs multiple cropped images for each detected object
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
            },
            "optional": {
                "custom_model_path": ("STRING", {"default": ""}),
                "max_size": ("INT", {
                    "default": 512, 
                    "min": 128, 
                    "max": 1024, 
                    "step": 64
                }),
                "preserve_aspect_ratio": ("BOOLEAN", {"default": True}),
                "exclude_person": ("BOOLEAN", {"default": True}),
                "objects_only": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("detected_objects", "object_count", "class_names", "detection_info")
    
    FUNCTION = "detect_objects"
    CATEGORY = "image/object_extraction"
    
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
    
    def pil_to_tensor(self, pil_image, target_size=None, preserve_aspect_ratio=True):
        """Convert PIL Image to ComfyUI tensor format"""
        # Resize if target size is provided
        if target_size is not None:
            if preserve_aspect_ratio:
                pil_image = self.resize_with_padding(pil_image, target_size)
            else:
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert PIL to numpy
        np_image = np.array(pil_image)
        
        # Convert to float and normalize to [0,1]
        tensor = torch.from_numpy(np_image).float() / 255.0
        
        # Add batch dimension: [batch, height, width, channels]
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def resize_with_padding(self, pil_image, target_size):
        """Resize image while maintaining aspect ratio using padding"""
        target_width, target_height = target_size
        original_width, original_height = pil_image.size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image maintaining aspect ratio
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with target size and black background
        padded_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calculate position to center the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image onto the padded background
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        return padded_image
    
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
    
    def get_excluded_classes(self, exclude_person=True, objects_only=True):
        """Get list of class IDs to exclude from detection"""
        excluded_classes = set()
        
        # COCO dataset class IDs to exclude
        if exclude_person:
            excluded_classes.add(0)  # person
        
        if objects_only:
            # Exclude living beings and focus on inanimate objects
            living_beings = {
                0,   # person
                14,  # bird
                15,  # cat
                16,  # dog
                17,  # horse
                18,  # sheep
                19,  # cow
                20,  # elephant
                21,  # bear
                22,  # zebra
                23,  # giraffe
            }
            excluded_classes.update(living_beings)
        
        return excluded_classes
    
    def detect_objects(self, image, model_name, confidence_threshold, iou_threshold, padding, custom_model_path="", max_size=512, preserve_aspect_ratio=True, exclude_person=True, objects_only=True):
        """Main detection function"""
        try:
            # Load model
            self.load_model(model_name, custom_model_path)
            
            # Get classes to exclude
            excluded_classes = self.get_excluded_classes(exclude_person, objects_only)
            
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
            print(f"YOLOv8 Detection: Excluding classes: {sorted(excluded_classes)} {'(objects only mode)' if objects_only else '(custom exclusions)'}")
            
            if boxes is None or len(boxes) == 0:
                # No objects detected, return original image
                print(f"YOLOv8 Detection: No objects detected with confidence > {confidence_threshold}")
                return (image, 0, "", "No objects detected")
            
            # Process detections
            detected_images = []
            class_names = []
            detection_info_list = []
            cropped_objects = []
            
            # First pass: crop all objects (excluding filtered classes)
            object_count = 0
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Skip excluded classes
                if class_id in excluded_classes:
                    print(f"Skipping {class_name} (excluded class)")
                    continue
                
                object_count += 1
                
                # Crop object from image
                cropped_object = self.crop_object(pil_image, bbox, padding)
                cropped_objects.append(cropped_object)
                
                # Store class name and info
                class_names.append(class_name)
                detection_info_list.append(f"Object {object_count}: {class_name} (conf: {confidence:.3f})")
            
            # Determine target size for standardization
            if cropped_objects:
                # Find the maximum dimensions among all cropped objects
                max_width = max(obj.size[0] for obj in cropped_objects)
                max_height = max(obj.size[1] for obj in cropped_objects)
                
                # Use a reasonable maximum size to avoid memory issues
                target_size = (min(max_width, max_size), min(max_height, max_size))
                
                resize_method = "with aspect ratio preservation (padding)" if preserve_aspect_ratio else "by stretching"
                print(f"YOLOv8 Detection: Found {len(cropped_objects)} objects, standardizing to {target_size} {resize_method}")
                
                # Second pass: resize all objects to the same size and convert to tensors
                for i, cropped_object in enumerate(cropped_objects):
                    object_tensor = self.pil_to_tensor(cropped_object, target_size, preserve_aspect_ratio)
                    print(f"Object {i+1} tensor shape: {object_tensor.shape}")
                    detected_images.append(object_tensor)
                
                # Verify all tensors have the same shape before concatenating
                if len(detected_images) > 1:
                    first_shape = detected_images[0].shape
                    for i, tensor in enumerate(detected_images):
                        if tensor.shape != first_shape:
                            print(f"WARNING: Tensor {i} shape mismatch: {tensor.shape} vs expected {first_shape}")
                            # Force resize the tensor to match
                            detected_images[i] = torch.nn.functional.interpolate(
                                tensor.permute(0, 3, 1, 2), 
                                size=(first_shape[1], first_shape[2]), 
                                mode='bilinear', 
                                align_corners=False
                            ).permute(0, 2, 3, 1)
                
                # Concatenate all detected objects into a batch
                try:
                    batch_tensor = torch.cat(detected_images, dim=0)
                    print(f"Successfully created batch tensor with shape: {batch_tensor.shape}")
                except Exception as cat_error:
                    print(f"Error concatenating tensors: {cat_error}")
                    # Fallback: return first detected object only
                    batch_tensor = detected_images[0]
                    print(f"Fallback: returning single object with shape: {batch_tensor.shape}")
                class_names_str = ", ".join(class_names)
                detection_info_str = "; ".join(detection_info_list)
                object_count = len(detected_images)
                print(f"YOLOv8 Detection: Final output - {object_count} objects after filtering: [{class_names_str}]")
            else:
                batch_tensor = image
                class_names_str = ""
                detection_info_str = "No objects detected after filtering" if len(boxes) > 0 else "No objects detected"
                object_count = 0
                print(f"YOLOv8 Detection: No valid objects after filtering (detected {len(boxes)} total)")
            
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
