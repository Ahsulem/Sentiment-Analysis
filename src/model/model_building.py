import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Optional dependency: Optuna. If not available, fall back to sklearn's RandomizedSearchCV
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame, 
    max_features: int, 
    ngram_range: tuple,
    sublinear_tf: bool = True,
    min_df: int = 3,
    max_df: float = 0.90
) -> tuple:
    """Apply TF-IDF with ngrams and SMOTE oversampling to the data."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, 
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df
        )

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Apply SMOTE to handle class imbalance across high-dimensional features
        smote = SMOTE(random_state=42)
        X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

        logger.debug(f"TF-IDF & SMOTE complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and vectorizer saved successfully.')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def optimize_lgbm(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 20) -> dict:
    """Run Optuna to find the best LightGBM hyperparameters optimizing for Macro F1.

    If Optuna is not installed, fall back to RandomizedSearchCV from scikit-learn.
    """
    logger.debug(f'Starting hyperparameter optimization (Optuna installed={HAS_OPTUNA}) with {n_trials} trials...')

    if HAS_OPTUNA:
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': "multi_logloss",
                'class_weight': "balanced",
                'n_jobs': -1,
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'num_leaves': trial.suggest_int('num_leaves', 31, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)

                preds = model.predict(X_fold_val)
                # Optimize for Macro F1 to balance evaluation across minority classes
                score = f1_score(y_fold_val, preds, average='macro')
                cv_scores.append(score)

            return np.mean(cv_scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.debug(f'Optuna optimization finished. Best CV Macro F1: {study.best_value:.4f}')
        return study.best_params
    else:
        logger.warning('Optuna is not installed — falling back to RandomizedSearchCV.')

        param_dist = {
            'n_estimators': [100, 150, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.02, 0.05, 0.1],
            'max_depth': [10, 15, 20, 30, 40, 50],
            'num_leaves': [31, 50, 80, 120, 150],
            'min_child_samples': [10, 20, 30, 50, 80, 100],
            'reg_alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            'reg_lambda': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        }

        base_estimator = lgb.LGBMClassifier(
            objective='multiclass', 
            num_class=3,
            metric='multi_logloss', 
            class_weight='balanced',
            n_jobs=-1, 
            random_state=42
        )

        rnd_search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=n_trials,
            scoring='f1_macro',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        rnd_search.fit(X_train, y_train)
        logger.debug(f'RandomizedSearchCV finished. Best CV Macro F1: {rnd_search.best_score_:.4f}')

        return rnd_search.best_params_


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, best_params: dict) -> lgb.LGBMClassifier:
    """Train the final LightGBM model using Optuna's best parameters."""
    try:
        final_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': "multi_logloss",
            'class_weight': "balanced",
            'n_jobs': -1,
            'random_state': 42,
            **best_params
        }
        
        best_model = lgb.LGBMClassifier(**final_params)
        best_model.fit(X_train, y_train)
        
        logger.debug('Final LightGBM model training completed with optimized parameters.')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        mb_params = params['model_building']
        max_features = mb_params['max_features']
        ngram_range = tuple(mb_params['ngram_range'])
        sublinear_tf = mb_params.get('sublinear_tf', True)
        min_df = mb_params.get('min_df', 3)
        max_df = mb_params.get('max_df', 0.90)

        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Step 1: Apply TF-IDF feature engineering & SMOTE oversampling
        X_train_tfidf, y_train = apply_tfidf(
            train_data, 
            max_features, 
            ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df
        )

        # Step 2: Run hyperparameter search optimizing for Macro F1
        best_params = optimize_lgbm(X_train_tfidf, y_train, n_trials=20)

        # Step 3: Train final model
        final_model = train_lgbm(X_train_tfidf, y_train, best_params)

        # Step 4: Save artifact
        save_model(final_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()