import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

class Data:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load data from the CSV file."""
        self.data = pd.read_csv(self.filepath).dropna()
        self.X = self.data.drop(columns=['MEDV', 'B']).to_numpy()
        self.y = self.data['MEDV'].to_numpy()
        print(f"Data loaded from {self.filepath}. Shape: {self.data.shape}")

    def get_summary(self):
        """Print a summary of the data."""
        if self.data is not None:
            print("Summary statistics of the data:")
            print(self.data.describe())
        else:
            print("Data is not loaded. Please load the data first.")

    def get_missing_values(self):
        """Print the number of missing values for each column."""
        if self.data is not None:
            print("Missing values in each column:")
            print(self.data.isnull().sum())
        else:
            print("Data is not loaded. Please load the data first.")

    def get_data(self):
        """Return the dataframe."""
        if self.data is not None:
            return self.data
        else:
            print("Data is not loaded. Please load the data first.")
            return None

    def scale_and_transform(self):
        """Scale the data and apply log transformation on skewed features."""
        if self.data is not None:
            # Log transformation on skewed features
            skewed_features = self.data.drop(columns=['MEDV', 'B']).apply(lambda x: skew(x.dropna()))
            skewed_features = skewed_features[abs(skewed_features) > 0.75].index
            for feature in skewed_features:
                self.data[feature] = np.log1p(self.data[feature])
            
            # Standardize the dataset
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.data.drop(columns=['MEDV', 'B']))
            print("Data scaling and log transformation complete.")
        else:
            print("Data is not loaded. Please load the data first.")

data = Data('../HousingData.csv')
data.load_data()
data.get_summary()
data.get_missing_values()
data.scale_and_transform()
df = data.get_data()
print(data.X)
