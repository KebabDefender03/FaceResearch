import os
import warnings
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/tmp/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def extract_face_with_landmarks(img_path):
    """Extract face with landmarks for visualization"""
    try:
        # Use RetinaFace directly to get proper landmarks
        from retinaface import RetinaFace as RF
        
        # Detect faces and get landmarks
        detections = RF.detect_faces(img_path)
        
        if detections and len(detections) > 0:
            # Get first detected face
            first_face_key = list(detections.keys())[0]
            face_info = detections[first_face_key]
            
            # Get facial area (x1, y1, x2, y2)
            facial_area = face_info.get('facial_area', [0, 0, 100, 100])
            x1, y1, x2, y2 = facial_area
            
            # Get landmarks - RetinaFace returns dict with 'right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left'
            # Each is a tuple (x, y)
            raw_landmarks = face_info.get('landmarks', {})
            
            # Load and crop the face
            import cv2
            img = cv2.imread(img_path)
            
            # Add padding around face
            padding = 20
            h, w = img.shape[:2]
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            face_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Resize to standard size
            face_crop = cv2.resize(face_crop, (160, 160))
            
            # Calculate scale factors for landmark adjustment
            crop_w = x2_pad - x1_pad
            crop_h = y2_pad - y1_pad
            scale_x = 160 / crop_w
            scale_y = 160 / crop_h
            
            # Convert landmarks to be relative to cropped/resized image
            # Use int() to convert numpy int64 to regular Python int
            landmarks = {}
            for name, coords in raw_landmarks.items():
                if coords:
                    # coords is (x, y) tuple - convert to cropped image coordinates
                    rel_x = (float(coords[0]) - x1_pad) * scale_x
                    rel_y = (float(coords[1]) - y1_pad) * scale_y
                    landmarks[name] = {'x': int(rel_x), 'y': int(rel_y)}
            
            # Encode face image to base64
            _, buffer = cv2.imencode('.jpg', face_crop)
            base64_img = base64.b64encode(buffer).decode('utf-8')
            
            print(f"Detected landmarks: {landmarks}")  # Debug
            
            # Convert all numpy types to Python types for JSON serialization
            score = face_info.get('score', 0)
            confidence = float(score) * 100 if score else 0
            
            return {
                'image': f"data:image/jpeg;base64,{base64_img}",
                'facial_area': {'x': int(x1), 'y': int(y1), 'w': int(x2-x1), 'h': int(y2-y1)},
                'confidence': round(confidence, 1),
                'landmarks': landmarks
            }
            
    except Exception as e:
        print(f"Face extraction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: try DeepFace with opencv and estimate landmarks
        try:
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend='opencv',
                align=True,
                enforce_detection=True
            )
            if faces and len(faces) > 0:
                face_img = faces[0]['face']
                face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                face_img = cv2.resize(face_img, (160, 160))
                _, buffer = cv2.imencode('.jpg', face_img)
                base64_img = base64.b64encode(buffer).decode('utf-8')
                
                # Estimate landmarks for 160x160 face
                estimated_landmarks = {
                    'left_eye': {'x': 48, 'y': 56},
                    'right_eye': {'x': 112, 'y': 56},
                    'nose': {'x': 80, 'y': 88},
                    'mouth_left': {'x': 56, 'y': 120},
                    'mouth_right': {'x': 104, 'y': 120}
                }
                
                return {
                    'image': f"data:image/jpeg;base64,{base64_img}",
                    'facial_area': faces[0].get('facial_area', {}),
                    'confidence': None,
                    'landmarks': estimated_landmarks
                }
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
    
    return None

def get_embedding(img_path, model_name):
    """Get face embedding vector"""
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=True
        )
        if embedding and len(embedding) > 0:
            return embedding[0]['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Selecteer beide afbeeldingen'})

    file1 = request.files['file1']
    file2 = request.files['file2']
    
    # 1. READ THE CHOSEN MODEL FROM THE FRONTEND
    # Default to 'Facenet512' if nothing is sent
    selected_model = request.form.get('model', 'facenet') 
    
    # Map the tab name to the actual DeepFace model name
    if selected_model == 'arcface':
        model_backend = "ArcFace"
    else:
        model_backend = "Facenet512"

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], "img1.jpg")
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], "img2.jpg")
    file1.save(path1)
    file2.save(path2)

    try:
        # Extract aligned faces with landmarks for visualization
        face1_data = extract_face_with_landmarks(path1)
        face2_data = extract_face_with_landmarks(path2)
        
        # Get embeddings for visualization
        embedding1 = get_embedding(path1, model_backend)
        embedding2 = get_embedding(path2, model_backend)
        
        # Downsample embeddings for visualization (512 -> 64 points)
        embedding1_viz = None
        embedding2_viz = None
        if embedding1 and embedding2:
            # Take every 8th value to get 64 points
            embedding1_viz = [round(embedding1[i], 4) for i in range(0, 512, 8)]
            embedding2_viz = [round(embedding2[i], 4) for i in range(0, 512, 8)]
        
        # Calculate cosine distance manually to show the math
        math_details = None
        if embedding1 and embedding2:
            import math
            # Convert to numpy for easier calculation
            e1 = np.array(embedding1)
            e2 = np.array(embedding2)
            
            # Dot product: sum of element-wise multiplication
            dot_product = float(np.dot(e1, e2))
            
            # Magnitudes (L2 norms): sqrt of sum of squares
            magnitude1 = float(np.linalg.norm(e1))
            magnitude2 = float(np.linalg.norm(e2))
            
            # Cosine similarity = dot_product / (magnitude1 * magnitude2)
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            
            # Cosine distance = 1 - cosine_similarity
            cosine_distance = 1 - cosine_similarity
            
            math_details = {
                'dot_product': round(dot_product, 6),
                'magnitude1': round(magnitude1, 6),
                'magnitude2': round(magnitude2, 6),
                'cosine_similarity': round(cosine_similarity, 6),
                'cosine_distance': round(cosine_distance, 6),
                'embedding_size': len(embedding1)
            }
        
        # 2. RUN DEEPFACE WITH THE SELECTED MODEL
        result = DeepFace.verify(
            img1_path=path1, 
            img2_path=path2, 
            model_name=model_backend
        )
        
        return jsonify({
            'verified': result['verified'],
            'distance': round(result['distance'], 4),
            'threshold': result['threshold'],
            'model': model_backend,
            'filename1': file1.filename,
            'filename2': file2.filename,
            'face1': face1_data,
            'face2': face2_data,
            'embedding1': embedding1_viz,
            'embedding2': embedding2_viz,
            'math': math_details
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)