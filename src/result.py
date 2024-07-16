from sklearn.metrics import mean_squared_error, r2_score
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        
        shap_importances = results['feature_importances']['shap']
        other_importances = results['feature_importances']['other']

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

# Example usage:
plot_CV_scores(results_path='results')
plot_feature_importance(results_path='results')
