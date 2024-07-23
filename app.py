from flask import Flask, request, jsonify
# from prediction import *
from utils import check_rating, store_feedback
from aws_utils import download_song_from_awss3, remove_temp_file
from gpt_utils import llm_agent
import librosa
from inference import extract_features, get_prediction, retrain_model
from flask_cors import CORS


app = Flask(__name__)

# Setup CORS for frontend
CORS(app, origins=["*"])

# UPLOAD_FOLDER = 'D:\Corey Stanford\Music Critique Corey Sanford\Music Critique Corey Sanford'  # Specify the folder where you want to store uploaded files
ALLOWED_EXTENSIONS = ['mp3']  # Specify the allowed file extensions

@app.route('/api/predict',methods=['POST'])
def prediction_api():
    song_name = request.form.get('filepath')
    bucket_name = request.form.get('bucket_name')
    bucket_region = request.form.get('bucket_region')
    
    if song_name == None or bucket_name == None or bucket_region == None:
        return jsonify({"error": "Invalid Inputs"})
    
    if song_name.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Song must be in .mp3 format."})
    
    try:
        local_file_path = download_song_from_awss3(song_name, bucket_name, bucket_region)
        print("File loaded Successfully")
    except Exception as e:
        if e.split(':').strip()[-1] == 'Access Denied':
            print(f"error: {e}")
            return jsonify({"error": "{e}"})
    
    # Load the song
    y, sr = librosa.load(local_file_path)
    
    # Extract features
    features = extract_features(y,sr)
    
    # Deleting the temporary stored song file from server
    remove_temp_file(local_file_path)
    
    beta = check_rating(features)
    features_dict = features.to_dict('records')[0]
    
    if beta:
        analysis = llm_agent(features_dict, beta)
        return jsonify({"popularity": beta, "analysis": analysis})
    else:
        popularity = get_prediction(features)
        
        analysis = llm_agent(features_dict, popularity)
        return jsonify({"popularity": popularity, "analysis": analysis})


@app.route('/api/feedback',methods=['POST'])
def feedback_api():
    feedback = request.form.get('feedback')
    feedback=float(feedback)
    song_path = request.form.get('filepath')
    bucket_name = request.form.get('bucket_name')
    bucket_region = request.form.get('bucket_region')
    
    if song_path == None or bucket_name == None or bucket_region == None or feedback == None:
        return jsonify({"error": "Invalid Inputs"})
    
    if song_path.split('.')[-1] != 'mp3':
        return jsonify({"error": "Song must be in .mp3 format."})
    
    try:
        local_file_path = download_song_from_awss3(song_path, bucket_name, bucket_region)
        print("File loaded Successfully")
    except Exception as e:
        if e.split(':').strip()[-1] == 'Access Denied':
            print(f"error: {e}")
            return jsonify({"error": "{e}"})
    
    # Load the song
    y, sr = librosa.load(local_file_path)
    
    # Extract features
    features = extract_features(y,sr)
    
    # Deleting the temporary stored song file from server
    remove_temp_file(local_file_path)
    
    store_feedback(features, feedback)
    
    return jsonify({"feedback": feedback})


# Tested. Running on local. Takes data X from big_data table and data Y from retrain_db
# If len(data Y ) < thresh then model is not trained.
# Else, model retrained and stored. retrain_db is truncated and data Y is appended to big_data
@app.route('/api/retrain',methods=['GET'])
def retrain_api():
    is_retrained = retrain_model()
    
    if is_retrained:
        return jsonify({"retraining": "Model retrained successfully"})
    else:
        return jsonify({"retraining": "Not enough data to retrain"})


if __name__ == "__main__":
    app.run(host = '0.0.0.0',
            port=8080)
