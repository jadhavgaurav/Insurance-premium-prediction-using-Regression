import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load Data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Preprocessing
def preprocess_data(df):
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object']
    
    # Define transformations
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # PowerTransform y to reduce skewness
    y_transformer = PowerTransformer()
    y = y_transformer.fit_transform(y.values.reshape(-1, 1))
    
    return X, y, preprocessor, y_transformer

# Train Model
def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train.ravel())
    y_pred = model_pipeline.predict(X_test)
    
    print('R2 Score:', r2_score(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return model_pipeline

# Save Model
def save_model(pipeline, y_transformer, model_path='model.joblib', y_path='y_transformer.joblib'):
    joblib.dump(pipeline, model_path)
    joblib.dump(y_transformer, y_path)

if __name__ == '__main__':
    df = load_data('insurance.csv')
    X, y, preprocessor, y_transformer = preprocess_data(df)
    model_pipeline = train_model(X, y, preprocessor)
    save_model(model_pipeline, y_transformer)
