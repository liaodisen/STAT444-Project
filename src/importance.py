import numpy as np
import shap
import torch
from captum.attr import FeaturePermutation

class Importance:
    def __init__(self, model, model_name, feature_names, X, y=None, use_shap=False):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.X = X
        self.y = y
        self.use_shap = use_shap

    def get_feature_importance(self):
        """Return a list of feature importances."""
        if hasattr(self.model, 'feature_importances_') and not self.use_shap:
            # For random forest models (using Gini importance)
            importance = self.model.feature_importances_
        elif self.model_name == 'rf' and self.use_shap:
            return self._get_shap_values()
        elif self.model_name == 'mlp' and self.use_shap:
            return self._get_shap_values()
        elif self.model_name == 'mlp' and not self.use_shap:
            return self._get_permutation_importance()
        elif self.model_name in ['linear', 'ridge', 'lasso', 'svm'] and self.use_shap:
            return self._get_shap_values()
        elif hasattr(self.model, 'coef_'):
            # For linear models (using coefficients)
            if self.model.coef_.ndim == 1:
                importance = np.abs(self.model.coef_)
            else:
                importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            raise ValueError("The provided model does not support feature importance retrieval.")
        
        return dict(zip(self.feature_names, self._normalize(importance)))

    def _get_shap_values(self):
        """Compute SHAP values for the model and dataset X."""
        if self.model_name == 'mlp':
            explainer = shap.DeepExplainer(self.model, torch.tensor(self.X, dtype=torch.float32))
            shap_values = explainer.shap_values(torch.tensor(self.X, dtype=torch.float32), check_additivity=False)
        elif self.model_name in ['linear', 'ridge', 'lasso']:
            explainer = shap.LinearExplainer(self.model, self.X)
            shap_values = explainer.shap_values(self.X)
        elif self.model_name == 'svm':
            explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X, 10))
            shap_values = explainer.shap_values(shap.sample(self.X, 10))
        else:  # Assume tree-based model for now
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X)
        
        shap_values = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, self._normalize(shap_values)))

    def _get_permutation_importance(self):
        """Compute feature importance using Captum's permutation importance."""
        if self.y is None:
            raise ValueError("Target values (y) must be provided for permutation importance.")
        

        attr = FeaturePermutation(self.model)
        perm_importance = attr.attribute(torch.tensor(self.X, dtype=torch.float32))
        
        return dict(zip(self.feature_names, self._normalize(perm_importance.detach().numpy().mean(axis=0))))
    
    def _normalize(self, values):
        """Normalize values to a range between 0 and 1, and sum to 1"""
        total = np.sum(values)
        return values / total


