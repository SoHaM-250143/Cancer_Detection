from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import json
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cancer-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ==========================
# SIMPLE USER STORAGE
# ==========================
users = {}
patient_history = []

# ==========================
# LOAD CANCER INFORMATION
# ==========================
try:
    with open('cancer_info.json', 'r') as f:
        cancer_info = json.load(f)
    print("Cancer info loaded successfully!")
except Exception as e:
    print(f"Error loading cancer_info.json: {e}")
    cancer_info = {}

# ==========================
# LOAD AI MODELS
# ==========================
print(" Loading AI Models...")

try:
    # CNN Model
    cnn_model_path = "model_cnn123.keras"
    cnn_model = load_model(cnn_model_path)
    print(" CNN model loaded!")

    # Dummy call to initialize
    _ = cnn_model(np.zeros((1,128,128,3), dtype=np.float32))
    print("CNN model initialized!")

    # Feature extractor
    input_tensor = Input(shape=(128,128,3))
    x = cnn_model.layers[0](input_tensor)
    for layer in cnn_model.layers[1:-1]:
        x = layer(x)
    feature_model = Model(inputs=input_tensor, outputs=x)
    print("Feature extractor ready!")

    # Scaler
    scaler_path = "scaler_model.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    print(f"Scaler loaded: {scaler is not None}")

    # ML Models
    ml_models_dir = "part_b_outputs/models"
    ml_models = {}
    if os.path.exists(ml_models_dir):
        for f in os.listdir(ml_models_dir):
            if f.endswith(".joblib"):
                model_name = f.replace(".joblib","")
                ml_models[model_name] = joblib.load(os.path.join(ml_models_dir,f))
        print(f"{len(ml_models)} ML models loaded")
    else:
        print(" ML models directory not found")

    # Ensemble Models
    ensemble_dir = "part_c_outputs/models"
    ensemble_models = {}
    if os.path.exists(ensemble_dir):
        for f in os.listdir(ensemble_dir):
            if f.endswith(".joblib"):
                model_name = f.replace(".joblib","")
                ensemble_models[model_name] = joblib.load(os.path.join(ensemble_dir,f))
        print(f"{len(ensemble_models)} Ensemble models loaded")
    else:
        print("Ensemble models directory not found")

    # Label names
    label_names = ['BloodCancer benign', 'BloodCancer malignant', 'BrainCancer', 
                   'BreastCancer benign', 'BreastCancer malignant', 'KidneyCancer', 
                   'LungCancer benign', 'LungCancer malignant', 'Normal', 'SkinCancer']

except Exception as e:
    print(f"Error loading AI models: {e}")
    cnn_model = None
    feature_model = None
    ml_models = {}
    ensemble_models = {}
    label_names = []

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_cancer_stage(probability, cancer_type):
    """
    Determine cancer stage based on probability and cancer type
    """
    if cancer_type == 'Normal':
        return "No Cancer Detected"
    
    if 'benign' in cancer_type.lower():
        return "Benign - Early Stage"
    
    if probability < 0.25:
        return "Stage I (Early Stage)"
    elif probability < 0.50:
        return "Stage II (Localized)"
    elif probability < 0.75:
        return "Stage III (Regional Spread)"
    else:
        return "Stage IV (Advanced Stage)"

def predict_image(img_path):
    """
    Predict cancer type from image using all models
    """
    if not cnn_model:
        raise Exception("AI models not loaded properly")
    
    # Load and preprocess image
    img = load_img(img_path, target_size=(128,128))
    img_arr = img_to_array(img)/255.0
    img_arr_exp = np.expand_dims(img_arr, axis=0)

    # CNN Prediction
    cnn_probs = cnn_model.predict(img_arr_exp, verbose=0)[0]
    cnn_index = np.argmax(cnn_probs)
    cnn_class = label_names[cnn_index]
    cnn_conf = cnn_probs[cnn_index]
    cnn_stage = get_cancer_stage(cnn_conf, cnn_class)

    # Feature Extraction for ML models
    features = feature_model.predict(img_arr_exp, verbose=0)
    features_flat = features.reshape(1,-1)
    features_scaled = scaler.transform(features_flat) if scaler else features_flat

    # ML Predictions
    ml_results = {}
    for name, model in ml_models.items():
        try:
            if name in ['SVM_linear','KNN','LogisticRegression']:
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features_flat)[0]
            ml_results[name] = label_names[pred]
        except Exception as e:
            print(f"Error with ML model {name}: {e}")
            ml_results[name] = "Error"

    # Ensemble Predictions
    ens_results = {}
    for name, model in ensemble_models.items():
        try:
            if name in ['Voting_Hard','Voting_Soft','Stacking']:
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(features_flat)[0]
            ens_results[name] = label_names[pred]
        except Exception as e:
            print(f"Error with ensemble model {name}: {e}")
            ens_results[name] = "Error"

    return {
        'cnn_class': cnn_class,
        'cnn_conf': cnn_conf,
        'cnn_stage': cnn_stage,
        'ml_results': ml_results,
        'ens_results': ens_results
    }

# ==========================
# ROUTES
# ==========================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            session['is_doctor'] = user.get('is_doctor', False)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        is_doctor = 'is_doctor' in request.form
        
        if username in users:
            flash('Username already exists!', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return redirect(url_for('register'))
        
        # Store user
        users[username] = {
            'email': email,
            'password': generate_password_hash(password),
            'is_doctor': is_doctor,
            'created_at': datetime.now()
        }
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    # Get user's recent history
    user_history = [h for h in patient_history if h.get('user') == session['user']]
    recent_history = user_history[-5:] if user_history else []
    
    return render_template('dashboard.html', 
                         username=session['user'],
                         is_doctor=session.get('is_doctor', False),
                         recent_history=recent_history)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze medical image"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    if 'image' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    # Get patient information
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_age = request.form.get('patient_age', 0)
    patient_gender = request.form.get('patient_gender', 'Not specified')
    notes = request.form.get('notes', '')
    
    # Save uploaded file
    filename = f"{session['user']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    
    try:
        # Get prediction
        result = predict_image(image_path)
        
        # Store in history
        history_record = {
            'id': len(patient_history) + 1,
            'user': session['user'],
            'patient_name': patient_name,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'image_filename': filename,
            'cnn_prediction': result['cnn_class'],
            'cnn_confidence': result['cnn_conf'],
            'cancer_stage': result['cnn_stage'],
            'ml_predictions': result['ml_results'],
            'ensemble_predictions': result['ens_results'],
            'created_at': datetime.now(),
            'notes': notes
        }
        patient_history.append(history_record)
        
        flash('Analysis completed successfully!', 'success')
        
        # Get updated recent history
        user_history = [h for h in patient_history if h.get('user') == session['user']]
        recent_history = user_history[-5:] if user_history else []
        
        return render_template('dashboard.html', 
                             username=session['user'],
                             is_doctor=session.get('is_doctor', False),
                             result=result, 
                             image_file=filename,
                             cancer_info=cancer_info,
                             recent_history=recent_history)
    
    except Exception as e:
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/history')
def history():
    """View analysis history"""
    if 'user' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    # Get user's history
    user_history = [h for h in patient_history if h.get('user') == session['user']]
    user_history.reverse()  # Show latest first
    
    return render_template('history.html', 
                         username=session['user'],
                         is_doctor=session.get('is_doctor', False),
                         histories=user_history)

@app.route('/cancer_info/<cancer_type>')
def get_cancer_info(cancer_type):
    """Get cancer information API"""
    info = cancer_info.get(cancer_type, {})
    return jsonify(info)

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/test-css')
def test_css():
    """Test CSS loading"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1 style="color: red;">If this is red, CSS is NOT loading</h1>
        <h1 class="test-class">If this is styled, CSS IS loading</h1>
        <div class="alert alert-success">This is a success alert</div>
        <div class="alert alert-error">This is an error alert</div>
        <button class="btn primary">Primary Button</button>
        <button class="btn secondary">Secondary Button</button>
    </body>
    </html>
    """

# ==========================
# ERROR HANDLERS
# ==========================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ==========================
# APPLICATION STARTUP
# ==========================

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("=" * 50)
    print(" CANCER DETECTION SYSTEM STARTING...")
    print("=" * 50)
    print(f" Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f" Secret key: {app.config['SECRET_KEY'][:10]}...")
    print(f"AI Models: {len(ml_models) + len(ensemble_models) + 1} loaded")
    print(f" Cancer types: {len(cancer_info)} loaded")
    print("=" * 50)
    print("Server running at: http://127.0.0.1:5000")
    print(" Available routes:")
    print("   /              - Home page")
    print("   /login         - User login")
    print("   /register      - User registration")
    print("   /dashboard     - User dashboard")
    print("   /history       - Analysis history")
    print("   /test-css      - Test CSS loading")
    print("=" * 50)
    
    # Start the application
    app.run(debug=True, host='127.0.0.1', port=5000)