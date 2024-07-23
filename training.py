import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle


def combine_data(dataset, top100_bot36):
    data = dataset.drop(columns=['tempo', 'rms', 'track_id'])
    data = pd.concat([data, top100_bot36], axis=0)
    
    return data


def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2


def XG_Boost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                               param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters found by GridSearchCV
    print("Best parameters found: ", grid_search.best_params_)
    
    # Use the best estimator to predict
    best_model = grid_search.best_estimator_
    
    return best_model


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


def train_model(dataset_path, top100_bot36_path):
    dataset = pd.read_csv(dataset_path)
    top100_bot36 = pd.read_csv(top100_bot36_path)
    
    data = combine_data(dataset, top100_bot36)
    
    X = data.drop(columns=['popularity'])
    y = data['popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = random_forest(X_train, y_train)
    
    tr_mae, tr_mse, tr_rmse, tr_r2 = test_model(X_train, y_train, model)
    print(f'Train Results:\nmae: {tr_mae}\nmse: {tr_mse}\nrmse: {tr_rmse}\nr2: {tr_r2}')
    
    mae, mse, rmse, r2 = test_model(X_test, y_test, model)
    print(f'Test Results:\nmae: {mae}\nmse: {mse}\nrmse: {rmse}\nr2: {r2}')
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


# Load data
dataset_path = 'dataset.csv'
top100_bot36_path = 'features_top100_bot37.csv'

train_model(dataset_path, top100_bot36_path)
