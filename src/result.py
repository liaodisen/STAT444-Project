from sklearn.metrics import mean_squared_error, r2_score
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import shap

def analyze_results(mse_scores, r2_scores, shap_feature_importances, other_feature_importances, results_path):
    mse_mean_score = -1 * mse_scores.mean()
    mse_std_score = mse_scores.std()
    r2_mean_score = r2_scores.mean()
    r2_std_score = r2_scores.std()

    results = {
        'mean_squared_error': mse_mean_score,
        'r2': r2_mean_score,
        'mse_std': mse_std_score,
        'r2_std': r2_std_score,
        'feature_importances': {
            'shap': shap_feature_importances,
            'other': other_feature_importances
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Results saved to", results_path)

def plot_CV_scores(results_path, output_dir='plots'):
    """
    Plot the scores of the models.
    """
    models = ['linear', 'mlp', 'rf', 'ridge', 'svm', 'lasso']
    result_data = []

    for model in models:
        # Read the results from the JSON file
        with open(results_path + f'/{model}_cv5.json', 'r') as f:
            results = json.load(f)

        result_data.append({
            'Model': model,
            'Mean Squared Error': results['mean_squared_error'],
            'MSE Standard Deviation': results['mse_std'],
            'R-squared': results['r2'],
            'R2 Standard Deviation': results['r2_std']
        })

    df = pd.DataFrame(result_data)

    # Plot MSE with error bars
    plt.figure(figsize=(10, 6))
    barplot_mse = sns.barplot(x='Model', y='Mean Squared Error', yerr=df['MSE Standard Deviation'], data=df, capsize=0.2, palette='Blues')
    
    # Annotate the bars above the error bars for MSE
    for p in barplot_mse.patches:
        height = p.get_height()
        barplot_mse.annotate(format(height, '.2f'), 
                             (p.get_x() + p.get_width() / 2., height), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 10), 
                             textcoords = 'offset points')
    
    plt.title('Mean Squared Error on Different Models')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Model')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path_mse = os.path.join(output_dir, 'mse_barplot.png')
    plt.savefig(output_path_mse)
    plt.close()

    # Plot R2 with error bars
    plt.figure(figsize=(10, 6))
    barplot_r2 = sns.barplot(x='Model', y='R-squared', yerr=df['R2 Standard Deviation'], data=df, capsize=0.2, palette='Greens')
    
    # Annotate the bars above the error bars for R2
    for p in barplot_r2.patches:
        height = p.get_height()
        barplot_r2.annotate(format(height, '.2f'), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 10), 
                            textcoords = 'offset points')
    
    plt.title('R-squared on Different Models')
    plt.ylabel('R-squared')
    plt.xlabel('Model')

    output_path_r2 = os.path.join(output_dir, 'r2_barplot.png')
    plt.savefig(output_path_r2)
    plt.close()


def normalize_importances(importance_dict):
    total = abs(sum(importance_dict.values()))
    return {k: abs(v) / total for k, v in importance_dict.items()}

def plot_feature_importance(results_path, output_dir='../plots'):
    """
    Plot the feature importance for SHAP and other others for each model.
    """
    models = ['linear', 'mlp', 'rf', 'ridge', 'svm', 'lasso']
    other_names = {
        'linear': 'Coefficients',
        'mlp': 'Permutation',
        'rf': 'Gini Importance',
        'ridge': 'Coefficients',
        'svm': 'Coefficients',
        'lasso': 'Coefficients'
    }

    for model in models:
        # Read the results from the JSON file
        with open(results_path + f'/{model}_cv5.json', 'r') as f:
            results = json.load(f)
        
        shap_importances = normalize_importances(results['feature_importances']['shap'])
        other_importances = normalize_importances(results['feature_importances']['other'])

        # Convert dictionaries to DataFrames
        shap_df = pd.DataFrame(list(shap_importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        other_df = pd.DataFrame(list(other_importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

        # Plot SHAP importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=shap_df, palette='viridis')
        plt.title(f'SHAP Feature Importances for {model.upper()} Model')
        plt.xlabel('SHAP Importance')
        plt.ylabel('Feature')
        output_path_shap = os.path.join(output_dir, f'{model}_shap_importance.png')
        plt.savefig(output_path_shap)
        plt.close()

        # Plot other-specific importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=other_df, palette='magma')
        plt.title(f'{other_names[model]} Feature Importances for {model.upper()} Model')
        plt.xlabel(f'{other_names[model]} Importance')
        plt.ylabel('Feature')
        output_path_other = os.path.join(output_dir, f'{model}_other_importance.png')
        plt.savefig(output_path_other)
        plt.close()

def plot_shap_summary(model, model_name, feature_names, X, output_dir='../plots'):
    """
    Plot SHAP summary plots and dependence plots for the given model.
    """
    if isinstance(model, torch.nn.Module):
        # Convert X into a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Use the DeepExplainer for PyTorch models
        explainer = shap.DeepExplainer(model, X_tensor)
        shap_values = explainer.shap_values(X_tensor, check_additivity=False)
    else:
        # Use the TreeExplainer or KernelExplainer for other models
        if model_name == 'rf':
            explainer = shap.TreeExplainer(model, X)
            shap_values = explainer.shap_values(X)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # SHAP summary plot
    plt.figure(figsize=(12, 8))
    if isinstance(model, torch.nn.Module):
        shap.summary_plot(shap_values, X_tensor.numpy(), plot_type='bar')
    else:
        shap.summary_plot(shap_values, X, plot_type='bar')
    output_path_summary = os.path.join(output_dir, f'{model_name}_shap_summary.png')
    plt.savefig(output_path_summary)

    # SHAP force plot for the first instance
    shap.initjs()
    if isinstance(model, torch.nn.Module):
        shap.force_plot(explainer.expected_value, shap_values[0, :], X_tensor[0, :].numpy())
    else:
        shap.force_plot(explainer.expected_value, shap_values[0,:], X[0,:], feature_names = feature_names, matplotlib=True, figsize=(50, 10))
    output_path_force = os.path.join(output_dir, f'{model_name}_shap_force_plot.png')
    plt.savefig(output_path_force)
    plt.close()
    # SHAP dependence plots for each feature
    # for feature in self.data.columns:
    #     plt.figure(figsize=(12, 8))
    #     shap.dependence_plot(feature, shap_values, X)
    #     output_path_dependence = os.path.join(output_dir, f'{model}_shap_dependence_{feature}.png')
    #     plt.savefig(output_path_dependence)
    #     plt.close()


# Example usage:
plot_CV_scores(results_path='results')
plot_feature_importance(results_path='results')
