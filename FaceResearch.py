import os
import warnings
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
            'model': model_backend, # Send back which model was used
            'filename1': file1.filename,
            'filename2': file2.filename
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)