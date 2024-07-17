# Boston Housing Dataset Analysis

We are analyzing the Boston Housing dataset for our STAT444 final project.

## Project Parts

### 1. Model Fitting
We will fit the dataset using the following regression methods:
- **Linear Regression methods covered in the course**:
  - Ridge Regression
  - Lasso Regression
  - Linear Regression
- **Two widely used machine learning regression methods in industry and academia**:
  - Random Forest
  - Support Vector Machine (SVM)
- **Neural Network**

### 2. Model Interpretation
For each regression method, we will interpret the model using feature importance methods:
- **Linear-based methods** (Linear Regression, Ridge Regression, Lasso Regression and SVM with linear kernel): We will use the coefficients of the parameters as indicators of feature importance.
- **Random Forest**: We will use the impurity decrease as the measure of feature importance.
- **Neural Network**: We will use permutation of features as a measure of feature importance.
- **All methods**: We will also use SHAP values to assess feature importance and compare the results across different methods.

## SHAP

- Machine Learning algorithms only give the relationship between input and the output, it is a black-box. We want to open the black-box and make the algorithm explain their prediction.
- Shapley Values can be defined as: a prediction can be explained by assuming that each feature value of the instance is a "player" in a game where the prediction is payout. Shapley values - a method from coalitional game theory - tells us how to fairly distribute the "payout" among the features

### Coalition Game
- $n$ is the number of "players"
- $u : 2^{[n]} \rightarrow \mathbb{R}$, the "payoff" function
- Valuation: what is the value of each player $i$
- (Symmetric) probabilistic value: $\phi_i = \phi(\{i\}) = \sum_{S \not\ni i} p_S \cdot [u(S \cup \{i\}) - u(S)]$

### SHAP Explanation Force Plots
- We can use SHAP to explain individual predicions.