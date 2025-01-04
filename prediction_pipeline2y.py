from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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

    # Define transformations for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[('num', numerical_transformer, numerical_cols),
                      ('cat', categorical_transformer, categorical_cols)])

    # PowerTransform y to reduce skewness
    y_transformer = PowerTransformer()
    y = y_transformer.fit_transform(y.values.reshape(-1, 1))

    return X, y, preprocessor, y_transformer

# Hyperparameter tuning function for SVR, Random Forest, and CatBoost models
def tune_hyperparameters(model, X, y):
    if isinstance(model.named_steps['regressor'], SVR):
        param_grid = {
            'regressor__C': [0.1, 1, 10],
            'regressor__gamma': ['scale', 'auto', 0.1, 1],
            'regressor__kernel': ['linear', 'rbf']
        }

    elif isinstance(model.named_steps['regressor'], RandomForestRegressor):
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10]
        }

    elif isinstance(model.named_steps['regressor'], CatBoostRegressor):
        param_grid = {
            'regressor__iterations': [500, 1000],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__depth': [6, 8, 10]
        }

    else:
        param_grid = {}

    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5, n_jobs=-1)
    grid_search.fit(X, y.ravel())
    print(f"Best Parameters for {model.named_steps['regressor']}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Train Multiple Models and Select the Best Based on Cross-Validation
def train_model(X, y, preprocessor):
    # Cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        'Support Vector Machine': SVR()
    }

    best_model = None
    best_cv_score = -np.inf

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])

        # Perform Cross-Validation
        cv_scores = cross_val_score(pipeline, X, y.ravel(), scoring='r2', cv=kf, n_jobs=-1)
        mean_cv_r2 = np.mean(cv_scores)

        print(f'{name} - Cross-Validation Mean R2 Score: {mean_cv_r2:.4f}')

        if mean_cv_r2 > best_cv_score:
            best_cv_score = mean_cv_r2
            best_model = pipeline

    print(f'Best Model: {best_model.named_steps["regressor"]} with Cross-Validation R2 Score: {best_cv_score:.4f}')
    
    # Hyperparameter tuning for the best model
    best_model = tune_hyperparameters(best_model, X, y)

    # Print R2 Score after tuning
    best_model_score = best_model.score(X, y)  # Calculate R2 score on the whole dataset (you can also split into train/test if needed)
    print(f'R2 Score after Hyperparameter Tuning: {best_model_score:.4f}')
    
    return best_model

# Save Model
def save_model(pipeline, y_transformer, model_path='model.joblib', y_path='y_transformer.joblib'):
    joblib.dump(pipeline, model_path)
    joblib.dump(y_transformer, y_path)

if __name__ == '__main__':
    df = load_data('insurance.csv')
    X, y, preprocessor, y_transformer = preprocess_data(df)
    best_pipeline = train_model(X, y, preprocessor)
    save_model(best_pipeline, y_transformer)
