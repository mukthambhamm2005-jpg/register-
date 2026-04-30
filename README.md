# 🚗 Edge-Compatible Computer Vision for Road Damage Detection

A complete deep learning project for automated detection and classification of road damages using a lightweight YOLO-based model suitable for edge devices.

---

## 📋 Project Overview

This project builds a production-ready computer vision system that detects **4 types of road damage**:
- **Longitudinal Crack** 🔻
- **Transverse Crack** ↔️
- **Alligator Crack** 🕸️
- **Pothole** ⭕

### Key Features:
✅ **Lightweight & Edge-Optimized** - Uses MobileNetV3-Small backbone  
✅ **YOLO-based Detection** - Fast real-time predictions  
✅ **Production-Ready Code** - Clean, modular, well-documented  
✅ **Comprehensive Pipeline** - From preprocessing to deployment  
✅ **Evaluation Metrics** - mAP@0.5, Precision, Recall, F1-Score  
✅ **Flask Web App** - Easy-to-use inference interface  

---

## 📁 Project Structure

```
copilot road/
├── preprocessing/          # Data preprocessing & augmentation
├── models/                 # Neural network architectures
├── training/               # Training pipeline & checkpoints
├── evaluation/             # Metrics & validation
├── inference/              # Prediction & visualization
├── app/                    # Flask web app (optional)
├── checkpoints/            # Saved model weights
├── outputs/                # Generated predictions & results
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
└── main.py                # Entry point (will be created)
```

---

## 🛠️ Setup Instructions

### Step 1: Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Note:** This may take 5-10 minutes depending on your internet speed and whether CUDA drivers are installed.

### Step 3: Verify Installation

```bash
python config.py
```

You should see the configuration printed successfully.

---

## 📊 Dataset Information

Your dataset is already prepared at: `C:\Users\hp\Desktop\archive`

**Structure:**
- **Train**: ~3,500+ labeled images
- **Validation**: ~1,000+ images
- **Test**: ~1,000+ images
- **Format**: YOLO (.txt label files with normalized bbox coordinates)
- **Classes**: 4 (as defined in `data.yaml`)

---

## 🚀 Quick Start (Execution Order)

Follow these steps in order. After each step, ask if you're ready to move to the next one.

### **Step 1**: ✅ **COMPLETED** - Project Setup
- ✓ Created directory structure
- ✓ Created config.py with all hyperparameters
- ✓ Created requirements.txt with dependencies
- ✓ Ready to install dependencies

### **Step 2**: Data Preprocessing (NEXT)
- Load YOLO dataset
- Apply preprocessing (Median filtering, CLAHE, brightness correction)
- Implement augmentation (flip, rotation, brightness)
- Visualize samples

### **Step 3**: Model Architecture
- Implement YOLO detector with MobileNetV3-Small backbone
- Define loss functions
- Create training utilities

### **Step 4**: Training
- Data loaders
- Training loop with validation
- Checkpoint saving
- Learning curve visualization

### **Step 5**: Evaluation
- Compute mAP@0.5
- Calculate Precision, Recall, F1-Score
- Generate confusion matrix
- Performance analysis

### **Step 6**: Inference
- Load trained model
- Perform predictions
- Draw bounding boxes with confidence scores
- Save results

### **Step 7**: Flask Web App (Optional)
- Create web interface
- Upload image functionality
- Real-time detection display
- REST API endpoints

---

## 🎯 Expected Outcomes

After completing this project, you'll have:

1. **Trained Model** - Saved in `checkpoints/`
2. **Evaluation Report** - Metrics in `outputs/`
3. **Inference Results** - Predictions with visualizations
4. **Web Application** - Interactive Flask interface
5. **Documentation** - Complete project documentation

---

## 💻 System Requirements

- **Python**: 3.10 or 3.11 (3.12 is the latest but may have compatibility issues)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA capability (optional but recommended)
- **Disk Space**: ~5GB for dataset + models

---

## 📝 Configuration

All settings are in `config.py`. You can customize:
- Image size, batch size, learning rate
- Number of epochs, augmentation strategies
- Model architecture, device (CPU/GPU)
- Confidence thresholds, evaluation metrics

---

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Make sure virtual environment is activated and run `pip install -r requirements.txt`

**Issue**: Out of memory errors
- **Solution**: Reduce `BATCH_SIZE` in config.py

**Issue**: CUDA not found
- **Solution**: Either install CUDA (advanced) or set `DEVICE = "cpu"` in config.py

---

## 📚 Resources

- [PyTorch Official Docs](https://pytorch.org/)
- [YOLO Concept](https://docs.ultralytics.com/)
- [OpenCV Guide](https://docs.opencv.org/)

---

## 📬 Questions?

I'll guide you through each step with detailed explanations and code comments. Just let me know when you're ready to proceed!

---

**Next Step**: Install dependencies and verify setup. Ready? 🚀
