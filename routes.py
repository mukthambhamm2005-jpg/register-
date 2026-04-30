"""
Flask Web Application
=====================
Web interface for road damage detection.

Features:
- Upload image via browser
- Run detection automatically
- Display results with bounding boxes
- Show detection statistics

Usage:
    python -m app.routes
    python main.py --mode app
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import os
import uuid
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

import config
from inference.predictor import DamagePredictor

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE
app.config['UPLOAD_FOLDER'] = config.OUTPUT_DIR / "uploads"
app.config['RESULTS_FOLDER'] = config.OUTPUT_DIR / "web_results"
app.config['SECRET_KEY'] = 'road-damage-detection-secret-key'

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Lazy initialization of predictor."""
    global predictor
    if predictor is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
        
        # Fallback to any checkpoint if best not found
        if not checkpoint_path.exists():
            checkpoints = list(config.CHECKPOINT_DIR.glob("*.pth"))
            if checkpoints:
                checkpoint_path = checkpoints[0]
        
        if checkpoint_path.exists():
            predictor = DamagePredictor(
                checkpoint_path=checkpoint_path,
                device=config.INFERENCE_DEVICE,
                conf_threshold=0.3
            )
        else:
            print("Warning: No checkpoint found for web app")
            predictor = None
    
    return predictor


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and detection."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        saved_name = f"{unique_id}_{filename}"
        upload_path = app.config['UPLOAD_FOLDER'] / saved_name
        file.save(upload_path)
        
        # Run detection
        pred = get_predictor()
        
        if pred is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Load image and run inference
        image = cv2.imread(str(upload_path))
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 500
        
        detections = pred.predict(image)
        
        # Draw detections
        output_image = pred.draw_detections(image, detections)
        
        # Save result
        result_name = f"result_{saved_name}"
        result_path = app.config['RESULTS_FOLDER'] / result_name
        cv2.imwrite(str(result_path), output_image)
        
        # Convert images to base64 for display
        original_b64 = image_to_base64(upload_path)
        result_b64 = image_to_base64(result_path)
        
        # Prepare detection info
        detection_info = []
        for det in detections:
            detection_info.append({
                'class': det['class_name'],
                'confidence': f"{det['confidence']:.3f}",
                'bbox': [f"{x:.1f}" for x in det['bbox']]
            })
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'result_image': result_b64,
            'detections': detection_info,
            'num_detections': len(detections),
            'result_filename': result_name
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result image."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """REST API endpoint for detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        # Read image from request
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Run detection
        pred = get_predictor()
        if pred is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        detections = pred.predict(image)
        
        # Format response
        results = []
        for det in detections:
            results.append({
                'class': det['class_name'],
                'class_id': det['class_id'],
                'confidence': float(det['confidence']),
                'bbox': {
                    'x1': float(det['bbox'][0]),
                    'y1': float(det['bbox'][1]),
                    'x2': float(det['bbox'][2]),
                    'y2': float(det['bbox'][3])
                }
            })
        
        return jsonify({
            'success': True,
            'detections': results,
            'num_detections': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def image_to_base64(image_path):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        data = f.read()
    
    # Determine MIME type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    b64_string = base64.b64encode(data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_string}"


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': get_predictor() is not None,
        'device': config.INFERENCE_DEVICE
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" FLASK WEB APP")
    print("=" * 60)
    print(f"\nStarting server on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

