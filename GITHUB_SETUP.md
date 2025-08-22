# GitHub Repository Setup Guide

Follow these steps to upload your ComfyUI YOLOv8 Detection Node to the GitHub repository.

## 📁 Repository Structure

Your repository should contain these files:
```
ComfyUI_Detection_List/
├── README.md                          # Main documentation
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore rules
├── __init__.py                      # ComfyUI node registration
├── yolov8_object_detection_node.py  # Main node implementation
├── install.sh                       # Linux/Mac installation script
├── install.bat                      # Windows installation script
├── example_workflow.json            # Example ComfyUI workflow
└── GITHUB_SETUP.md                  # This setup guide
```

## 🚀 Upload to GitHub

### Method 1: Using GitHub Web Interface (Recommended for beginners)

1. **Navigate to your repository**: Go to [https://github.com/mangobyed/ComfyUI_Detection_List](https://github.com/mangobyed/ComfyUI_Detection_List)

2. **Upload files**:
   - Click "uploading an existing file" or "Add file" → "Upload files"
   - Drag and drop all the files from `/Users/edoardoottone/Documents/automatic mask/` into the upload area
   - OR click "choose your files" and select all files

3. **Commit the files**:
   - Add a commit message: `Initial release - YOLOv8 object detection node for ComfyUI`
   - Add description: `Adds YOLOv8-based object detection node with batch output functionality`
   - Click "Commit new files"

### Method 2: Using Git Command Line

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mangobyed/ComfyUI_Detection_List.git
   cd ComfyUI_Detection_List
   ```

2. **Copy files**:
   ```bash
   # Copy all files from your local directory to the git repository
   cp -r "/Users/edoardoottone/Documents/automatic mask/"* .
   ```

3. **Add and commit**:
   ```bash
   git add .
   git commit -m "Initial release - YOLOv8 object detection node for ComfyUI"
   git push origin main
   ```

### Method 3: Using GitHub Desktop

1. **Clone repository** in GitHub Desktop
2. **Copy files** to the local repository folder
3. **Commit** with message: "Initial release - YOLOv8 object detection node for ComfyUI"
4. **Push** to origin

## 📝 Post-Upload Steps

After uploading the files:

1. **Verify the repository** looks correct
2. **Create a release** (optional):
   - Go to "Releases" tab
   - Click "Create a new release"
   - Tag version: `v1.0.0`
   - Release title: `Initial Release`
   - Add release notes describing features

3. **Update repository settings**:
   - Add topics/tags: `comfyui`, `yolo`, `object-detection`, `computer-vision`, `machine-learning`
   - Add a description: "YOLOv8 object detection node for ComfyUI with batch output"
   - Add the repository website (if any)

4. **Test the installation**:
   - Try cloning and installing from the GitHub repository
   - Test the installation scripts
   - Verify the ComfyUI node works correctly

## 🛠 Making Changes

When you need to update the code:

1. **Clone the repository** (if not already done)
2. **Make your changes**
3. **Test thoroughly**
4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

## 📚 Additional Resources

- [GitHub Documentation](https://docs.github.com/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [ComfyUI Custom Nodes Guide](https://github.com/comfyanonymous/ComfyUI)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## 🔍 File Descriptions

- **`yolov8_object_detection_node.py`**: Main node implementation with YOLOv8 integration
- **`__init__.py`**: Registers the node with ComfyUI
- **`requirements.txt`**: Python package dependencies
- **`install.sh/.bat`**: Automated installation scripts
- **`example_workflow.json`**: Sample ComfyUI workflow demonstrating usage
- **`README.md`**: Comprehensive documentation for users
- **`LICENSE`**: MIT license for open source distribution
- **`.gitignore`**: Excludes unnecessary files from version control

## ✅ Verification Checklist

Before considering the repository complete:

- [ ] All files uploaded successfully
- [ ] README displays correctly with proper formatting
- [ ] Installation scripts are executable
- [ ] Requirements.txt contains all necessary dependencies
- [ ] Example workflow loads in ComfyUI
- [ ] Node appears in ComfyUI node menu
- [ ] Basic functionality tested
- [ ] Repository has proper description and tags
- [ ] License is appropriate (MIT recommended)

## 🎉 You're Done!

Your ComfyUI YOLOv8 Detection Node repository is now ready for users to clone, install, and use!
