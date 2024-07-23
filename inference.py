import librosa
import numpy as np
import pickle
import pandas as pd
from utils import concat_tables_to_retrain, get_big_data_as_dataframe
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import os


def extract_features(y,sr):
    features = [
        float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        float(np.mean(librosa.feature.zero_crossing_rate(y))),
        float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
        float(np.mean(librosa.feature.mfcc(y=y, sr=sr))),
        float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        ]
    
    data = {
        'spectral_rolloff': features[0],
        'zero_crossing_rate': features[1],
        'spectral_bandwidth': features[2],
        'spectral_centroid': features[3],
        'spectral_contrast': features[4],
        'mfcc': features[5],
        'chroma': features[6]
        }
    
    data = pd.DataFrame(data, index=[0])
    
    return data


# Function to get features in the form of a dataframe and output the predicted popularity by model
def get_prediction(features):
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    score = loaded_model.predict(features)
    score = round(score[0])
    #print('model predicted popularity : ',result)
    return score


def random_forest(X_train, y_train):
    # Create the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)
    
    
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    
    return model


def retrain_model():
    is_concated = concat_tables_to_retrain()
    is_retrained = False
    
    if is_concated:
        dataframe = get_big_data_as_dataframe()
        X = dataframe.drop(columns=['popularity'])
        y = dataframe['popularity']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Delete the existing model.pkl file if it exists
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')

        # Save the trained model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        is_retrained = True
        
        return is_retrained
    
    return is_retrained
