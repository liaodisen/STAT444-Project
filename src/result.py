from sklearn.metrics import mean_squared_error, r2_score
import json

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
