# Troubleshooting Guide

## ❌ "Sizes of tensors must match" Error

If you're getting this error:
```
Error in object detection: Sizes of tensors must match except in dimension 0. Expected size 732 but got size 450 for tensor number 1 in the list.
```

### 🔧 Solution Steps:

#### 1. **Restart ComfyUI** (Most Important)
ComfyUI needs to be completely restarted to load updated custom nodes:
- Close ComfyUI completely
- Restart ComfyUI 
- Load your workflow again

#### 2. **Update from GitHub**
Make sure you have the latest version:
```bash
cd ComfyUI/custom_nodes/ComfyUI_Detection_List
git pull origin main
```

#### 3. **Check Debug Output**
After restarting, you should see these debug messages in the console:
```
YOLOv8 Detection: Model inference complete
YOLOv8 Detection: Found X objects, resizing to (width, height)
Object 1 tensor shape: torch.Size([1, height, width, 3])
Object 2 tensor shape: torch.Size([1, height, width, 3])
Successfully created batch tensor with shape: torch.Size([X, height, width, 3])
```

If you **don't see these messages**, ComfyUI is still using the old version.

#### 4. **Manual Reload** (Alternative)
If restarting doesn't work, try:
```bash
cd ComfyUI/custom_nodes/ComfyUI_Detection_List
python reload_node.py
```

### 🎯 What Was Fixed:

- **Tensor Standardization**: All cropped objects are now resized to the same dimensions
- **Shape Verification**: Added checks to ensure all tensors match before concatenating  
- **Fallback Handling**: If concatenation still fails, returns single object instead of crashing
- **Debug Logging**: Added detailed console output to track the process

### 🔍 Verification:

The node is working correctly when you see:
1. ✅ Objects detected (e.g., "1 person, 1 tie")
2. ✅ Debug messages showing tensor shapes
3. ✅ "Successfully created batch tensor" message
4. ✅ Multiple images output in the batch

### 🚫 Common Issues:

1. **Old Node Version**: ComfyUI not restarted after update
2. **Installation Issues**: Node not properly installed in custom_nodes folder
3. **Dependencies**: Missing ultralytics or torch packages

### 📞 Still Having Issues?

1. Check the [GitHub Issues](https://github.com/mangobyed/ComfyUI_Detection_List/issues)
2. Create a new issue with:
   - ComfyUI console output
   - Your workflow JSON
   - Python/ComfyUI versions
   - Full error message

### 💡 Pro Tips:

- Always restart ComfyUI after updating custom nodes
- Use lower confidence thresholds (0.1-0.3) to detect more objects
- Increase max_size parameter for higher quality crops
- Check console output for debug information
