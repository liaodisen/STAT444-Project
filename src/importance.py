import numpy as np

class Importance:
    def __init__(self, model, model_name, feature_names):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names

    def get_feature_importance(self):
        """Return a list of feature importances."""
        if hasattr(self.model, 'feature_importances_'):
            # For random forest models (using Gini importance)
            importance = self.model.feature_importances_
        elif self.model_name == 'rf':
            importance =  self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models (using coefficients)
            if self.model.coef_.ndim == 1:
                importance = np.abs(self.model.coef_)
            else:
                importance =  np.mean(np.abs(self.model.coef_), axis=0)
        else:
            raise ValueError("The provided model does not support feature importance retrieval.")
        
        return dict(zip(self.feature_names, importance))