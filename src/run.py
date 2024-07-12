import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from data import Data
from model import get_model
from importance import Importance
from train import train_model
import utils
import shap
import json

def compute_shap_values(model, X):
    """Compute SHAP values for the model and dataset X."""
    explainer = shap.DeepExplainer(model, torch.tensor(X, dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32), check_additivity=False)
    return np.abs(shap_values).mean(axis=0)

def analyze_results(scores, feature_importances, results_path):
    mean_score = -1 * scores.mean()
    std_score = scores.std()

    results = {
        'mean_squared_error': mean_score,
        'std_deviation': std_score,
        'feature_importances': feature_importances
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Results saved to", results_path)

def main(args):
    data = Data(args.data_path)
    data.load_data()
    data.scale_and_transform()
    feature_names = data.data.drop(columns=['MEDV', 'B']).columns
    mse_scores_list = []
    r2_scores_list = []
    feature_importance_list = []

    if args.model == 'mlp':
        for seed in range(10):
            utils.seed_everything(seed)
            kf = KFold(n_splits=args.cv, shuffle=True, random_state=seed)
            mse_scores = []
            r2_scores = []
            all_shap_values = np.zeros((data.X.shape[1],))

            for train_index, val_index in kf.split(data.X):
                X_train, X_val = data.X[train_index], data.X[val_index]
                y_train, y_val = data.y[train_index], data.y[val_index]

                model = train_model(X_train, y_train, X_val, y_val, X_train.shape[1])
                model.eval()  # Switch to evaluation mode

                y_pred = model(torch.tensor(X_val, dtype=torch.float32)).detach().numpy()
                mse_score = mean_squared_error(y_val, y_pred.squeeze())
                r2_score_value = r2_score(y_val, y_pred.squeeze())
                
                mse_scores.append(mse_score)
                r2_scores.append(r2_score_value)

                # Compute SHAP values for the validation set
                shap_values = compute_shap_values(model, X_val)
                if shap_values.ndim == 2 and shap_values.shape[1] == 1:
                    shap_values = shap_values.reshape(-1)  # Reshape (12, 1) to (12,)
                all_shap_values += shap_values

            mse_scores_list.append(-1 * mse_scores)
            r2_scores_list.append(r2_scores)
            feature_importance_list.append(all_shap_values / len(data.X))  # Average over total examples

    else:
        for seed in range(10):
            utils.seed_everything(seed)
            model = get_model(args.model)
            model.fit(data.X, data.y)
            mse_scores = cross_val_score(model, data.X, data.y, cv=args.cv, scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, data.X, data.y, cv=args.cv, scoring=make_scorer(r2_score))
            mse_scores_list.append(mse_scores)
            r2_scores_list.append(r2_scores)

            # Collect feature importances
            importance = Importance(model, args.model, feature_names)
            feature_importances = importance.get_feature_importance()
            feature_importance_list.append(feature_importances)

    # Convert lists to numpy arrays for easier averaging
    mse_scores_array = np.array(mse_scores_list)
    r2_scores_array = np.array(r2_scores_list)
    
    # Calculate mean and standard deviation across all seeds
    mean_mse_scores = np.mean(mse_scores_array, axis=0)
    mean_r2_scores = np.mean(r2_scores_array, axis=0)

    # Calculate average feature importances
    avg_feature_importance = {}
    if args.model == 'mlp':
        avg_shap_values = np.mean(np.array(feature_importance_list), axis=0)
        avg_feature_importance = dict(zip(feature_names, avg_shap_values))
    else:
        if feature_importance_list:
            for key in feature_importance_list[0]:
                avg_feature_importance[key] = np.mean([fi[key] for fi in feature_importance_list])

    filename = f'/{args.model}_cv{args.cv}.json'
    
    analyze_results(mean_mse_scores, avg_feature_importance, args.results_path + filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['svm', 'rf', 'mlp'], help='Model to use')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()
    
    main(args)
