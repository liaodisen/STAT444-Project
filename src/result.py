from sklearn.metrics import mean_squared_error, r2_score
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def plot_CV(results_path, output_dir='plots'):
    """
    plot the scores of the model
    """
    models = ['linear', 'mlp', 'rf', 'ridge', 'svm']
    result_data = []
    for model in models:
        # Read the results from the JSON file
        with open(results_path + f'/{model}_cv5.json', 'r') as f:
            results = json.load(f)

        result_data.append({
            'Model': model,
            'Mean Squared Error': results['mean_squared_error'],
            'Standard Deviation': results['std_deviation']
        })

        df = pd.DataFrame(result_data)

    # Plot the bar plot with error bars
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Model', y='Mean Squared Error', yerr=df['Standard Deviation'], data=df, capsize=0.2)
    
    # Annotate the bars above the error bars
    for p in barplot.patches:
        height = p.get_height()
        barplot.annotate(format(height, '.2f'), 
                         (p.get_x() + p.get_width() / 2. + 0.2, height), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 10), 
                         textcoords = 'offset points')
    sns.barplot(x='Model', y='Mean Squared Error', yerr=df['Standard Deviation'], data=df, capsize=0.2, 
                palette='Blues')
    plt.title('Mean Squared Error on Different Models')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Model')
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mse_barplot.png')
    plt.savefig(output_path)

plot_CV(results_path='results')