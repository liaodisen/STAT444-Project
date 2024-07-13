import numpy as np
import shap
import torch

class Importance:
    def __init__(self, model, model_name, feature_names, X, use_shap=False):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.X = X
        self.use_shap = use_shap

    def get_feature_importance(self):
        """Return a list of feature importances."""
        if hasattr(self.model, 'feature_importances_') and not self.use_shap:
            # For random forest models (using Gini importance)
            importance = self.model.feature_importances_
        elif self.model_name == 'rf' and self.use_shap:
            return self._get_shap_values()
        elif self.model_name == 'mlp':
            return self._get_shap_values()
        elif self.model_name in ['linear', 'ridge'] and self.use_shap:
            return self._get_shap_values()
        elif hasattr(self.model, 'coef_'):
            # For linear models (using coefficients)
            if self.model.coef_.ndim == 1:
                importance = np.abs(self.model.coef_)
            else:
                importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            raise ValueError("The provided model does not support feature importance retrieval.")
        
        return dict(zip(self.feature_names, importance))

    def _get_shap_values(self):
        """Compute SHAP values for the model and dataset X."""
        if self.model_name == 'mlp':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                explainer = shap.DeepExplainer(self.model, torch.tensor(self.X, dtype=torch.float32))
                shap_values = explainer.shap_values(torch.tensor(self.X, dtype=torch.float32), check_additivity=False)
        elif self.model_name in ['linear', 'ridge']:
            explainer = shap.LinearExplainer(self.model, self.X)
            shap_values = explainer.shap_values(self.X)
        else:  # Assume tree-based model for now
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X)
        
        shap_values = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, shap_values))
