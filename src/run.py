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
from result import analyze_results, plot_shap_summary
import utils
import shap
import json
import os

def main(args):
    if args.task == 'train':
        train_models(args)
    elif args.task == 'shap':
        make_shap_graphs(args)

def train_models(args):
    data = Data(args.data_path)
    data.load_data()
    data.scale_and_transform()
    feature_names = data.data.drop(columns=['MEDV', 'B']).columns
    mse_scores_list = []
    r2_scores_list = []
    shap_feature_importance_list = []
    other_feature_importance_list = []

    if args.model == 'mlp':
        for seed in range(10):
            utils.seed_everything(seed)
            kf = KFold(n_splits=args.cv, shuffle=True, random_state=seed)
            mse_scores = []
            r2_scores = []
            all_shap_values = np.zeros((data.X.shape[1],))
            all_other_values = np.zeros((data.X.shape[1],))

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
                importance = Importance(model, args.model, feature_names, X=data.X, y=data.y, use_shap=True)
                shap_values = importance.get_feature_importance()
                if isinstance(shap_values, dict):
                    shap_values = np.array(list(shap_values.values()))
                if shap_values.ndim == 2 and shap_values.shape[1] == 1:
                    shap_values = shap_values.reshape(-1)  # Reshape (12, 1) to (12,)
                all_shap_values += shap_values

                # Compute other feature importances
                importance = Importance(model, args.model, feature_names, X=data.X, y=data.y, use_shap=False)
                other_values = importance.get_feature_importance()
                if isinstance(other_values, dict):
                    other_values = np.array(list(other_values.values()))
                if other_values.ndim == 2 and other_values.shape[1] == 1:
                    other_values = other_values.reshape(-1)  # Reshape (12, 1) to (12,)
                all_other_values += other_values

            mse_scores_list.append(mse_scores)
            r2_scores_list.append(r2_scores)
            shap_feature_importance_list.append(all_shap_values / len(data.X))  # Average over total examples
            other_feature_importance_list.append(all_other_values / len(data.X))

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
            importance = Importance(model, args.model, feature_names, X=data.X, use_shap=True)
            shap_values = importance.get_feature_importance()
            shap_feature_importance_list.append(shap_values)

            importance = Importance(model, args.model, feature_names, X=data.X, use_shap=False)
            other_values = importance.get_feature_importance()
            other_feature_importance_list.append(other_values)

    # Convert lists to numpy arrays for easier averaging
    mse_scores_array = np.array(mse_scores_list)
    r2_scores_array = np.array(r2_scores_list)
    
    # Calculate mean and standard deviation across all seeds
    mean_mse_scores = np.mean(mse_scores_array, axis=0)
    mean_r2_scores = np.mean(r2_scores_array, axis=0)

    # Calculate average feature importances
    avg_shap_feature_importance = {}
    avg_other_feature_importance = {}
    if args.model == 'mlp':
        avg_shap_values = np.mean(np.array(shap_feature_importance_list), axis=0)
        avg_shap_feature_importance = dict(zip(feature_names, avg_shap_values))
        avg_other_values = np.mean(np.array(other_feature_importance_list), axis=0)
        avg_other_feature_importance = dict(zip(feature_names, avg_other_values))
    else:
        if shap_feature_importance_list:
            for key in shap_feature_importance_list[0]:
                avg_shap_feature_importance[key] = np.mean([fi[key] for fi in shap_feature_importance_list])
        if other_feature_importance_list:
            for key in other_feature_importance_list[0]:
                avg_other_feature_importance[key] = np.mean([fi[key] for fi in other_feature_importance_list])

    filename = f'/{args.model}_cv{args.cv}.json'
    
    analyze_results(mean_mse_scores, mean_r2_scores, 
                    avg_shap_feature_importance, avg_other_feature_importance, args.results_path + filename)

def make_shap_graphs(args):
    data = Data(args.data_path)
    data.load_data()
    data.scale_and_transform()

    model = get_model(args.model)
    model.fit(data.X, data.y)

    os.makedirs(args.results_path, exist_ok=True)
    plot_shap_summary(model, data.X, output_dir=args.results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['train', 'shap'], help='Task to perform: train models or generate SHAP graphs')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, required=True, choices=['svm', 'rf', 'mlp', 'linear', 'ridge', 'lasso'], help='Model to use')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()
    
    main(args)
