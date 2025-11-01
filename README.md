# BigMart Sales Prediction Analysis Pipeline

A comprehensive machine learning pipeline for predicting retail sales using the BigMart dataset.

## Overview

This project implements a complete end-to-end machine learning workflow including:
- Data preprocessing and feature engineering
- Exploratory data analysis
- Multiple machine learning models (Linear Regression, Random Forest, XGBoost, Gradient Boosting, Neural Networks)
- Ensemble methods (Voting and Stacking)
- Comprehensive model validation and performance assessment
- Sensitivity and robustness analysis
- SHAP-based model interpretation

## Setup Instructions

### 1. Virtual Environment

The virtual environment has been created. Activate it:

**Windows PowerShell:**
```powershell
.\venv\Scripts\activate.ps1
```

**Windows CMD:**
```cmd
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies

Packages should already be installed. If not, run:
```bash
pip install -r requirements.txt
```

### 3. Data Requirements

Ensure `BigMart.csv` is in the same directory as `bigmart_analysis.py`.

## Running the Analysis

### Complete Pipeline

Run the entire analysis pipeline:
```bash
python bigmart_analysis.py
```

**Expected Runtime:** 20-40 minutes (depending on hardware)

**Note:** Grid search for hyperparameter tuning is computationally intensive. The script will print progress updates.

### What Gets Generated

The pipeline will create:

**CSV Files (in current directory):**
- `model_performance_summary.csv` - Performance metrics for all models
- `statistical_comparison.csv` - Statistical tests comparing models
- `performance_by_outlet_type.csv` - Performance breakdown by outlet type
- `performance_by_category.csv` - Performance breakdown by product category

**Visualization Files (in `Visualizations/` folder):**
- `eda_distributions.png` - Sales distributions and patterns
- `correlation_matrix.png` - Feature correlations
- `eda_relationships.png` - Relationship plots between features and sales
- `shap_summary.png` - SHAP value summary plot
- `shap_importance.png` - Feature importance from SHAP
- `cross_validation_results.png` - Cross-validation performance
- `model_performance_comparison.png` - Comprehensive model comparison
- `prediction_vs_actual.png` - Prediction quality visualization
- `sensitivity_split_sizes.png` - Sensitivity to train-test split
- `robustness_by_outlet_type.png` - Performance across outlet types
- `robustness_by_category.png` - Performance across product categories

**Model File (in current directory):**
- `best_model.pkl` - Saved best performing model (can be loaded with joblib)

## Project Structure

```
.
├── BigMart.csv                  # Input dataset
├── bigmart_analysis.py          # Main analysis script
├── requirements.txt             # Python dependencies
├── process.txt                  # Detailed process documentation
├── README.md                    # This file
├── venv/                        # Virtual environment
├── Visualizations/              # Generated PNG visualizations
│   ├── eda_distributions.png
│   ├── correlation_matrix.png
│   ├── shap_summary.png
│   └── ... (11 total PNG files)
└── [Generated CSV files]        # Performance metrics and model
    ├── model_performance_summary.csv
    ├── statistical_comparison.csv
    └── best_model.pkl
```

## Pipeline Stages

### Stage 1: Data Preprocessing
- Cleaning and standardizing categorical variables
- Imputing missing values (Item_Weight, Outlet_Size)
- Feature engineering (Outlet_Age, Item_MRP_Bins, Item_Type_Grouped, etc.)
- Encoding categorical variables
- Outlier analysis
- Train-test split (80-20)

### Stage 2: Exploratory Data Analysis
- Descriptive statistics
- Distribution visualizations
- Correlation analysis
- Relationship plots

### Stage 3: Model Development
Seven models trained and evaluated:
1. **Linear Regression** - Baseline model
2. **Random Forest** - With grid search hyperparameter tuning
3. **XGBoost** - Gradient boosting with extensive tuning
4. **Gradient Boosting** - Alternative boosting implementation
5. **Neural Network** - Multi-layer perceptron with scaling
6. **Voting Ensemble** - Average of RF, XGBoost, and GB
7. **Stacking Ensemble** - Meta-learner combining base models

### Stage 4: Model Validation
- Multiple metrics: RMSE, MAE, R², MAPE
- 5-fold cross-validation
- Statistical comparison (paired t-tests, effect sizes)
- Performance summary and visualization

### Stage 5: Sensitivity & Robustness
- Performance across different train-test splits
- Generalization across outlet types
- Generalization across product categories
- Best model saved for deployment

## Understanding the Results

### Key Metrics

- **RMSE (Root Mean Squared Error):** Average prediction error in sales units. Lower is better.
- **MAE (Mean Absolute Error):** Average absolute error. More interpretable than RMSE.
- **R² (R-squared):** Proportion of variance explained. Range 0-1, higher is better.
- **MAPE (Mean Absolute Percentage Error):** Average percentage error. Easy to communicate.

### Model Selection

The best model is automatically saved as `best_model.pkl` based on test set RMSE.

To load and use the saved model:
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_model.pkl')

# Prepare your data (must match training preprocessing)
new_data = pd.DataFrame({...})  # Your preprocessed features

# Make predictions
predictions = model.predict(new_data)
```

## Process Documentation

See `process.txt` for comprehensive documentation including:
- Detailed reasoning behind each preprocessing step
- Alternative approaches considered
- Ensemble method recommendations
- Statistical methodology explanations
- Advanced techniques and future directions

## Customization

### Modify Hyperparameter Search

Edit the `param_grid` dictionaries in `bigmart_analysis.py`:

```python
# Example: Modify Random Forest tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Add more options
    'max_depth': [10, 20, 30, None],
    # ... other parameters
}
```

### Change Train-Test Split

Modify the split ratio:
```python
analysis.prepare_train_test_split(test_size=0.25)  # 75-25 split instead of 80-20
```

### Adjust Cross-Validation Folds

Change k in cross-validation:
```python
analysis.cross_validation_analysis(k=10)  # Use 10 folds instead of 5
```

## Troubleshooting

### Memory Issues

If you encounter memory errors during grid search:
1. Reduce parameter grid size
2. Reduce n_estimators in tree models
3. Use smaller train-test split (more data in test set)

### Slow Performance

To speed up analysis:
1. Reduce grid search combinations
2. Lower n_estimators (e.g., 100 instead of 300)
3. Reduce cross-validation folds (k=3 instead of k=5)
4. Comment out neural network training (typically slowest)

### Import Errors

Ensure all packages are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Advanced Usage

### Using Specific Models Only

Modify the `main()` function to run only desired models:
```python
# Only train XGBoost and Random Forest
analysis.train_xgboost() \
       .train_random_forest() \
       .generate_performance_summary()
```

### Adding Custom Models

Extend the `BigMartAnalysis` class:
```python
def train_custom_model(self):
    from sklearn.svm import SVR
    
    model = SVR(kernel='rbf', C=1.0)
    model.fit(self.X_train, self.y_train)
    
    # ... compute metrics and store results
    
    return self
```

## Performance Expectations

Based on similar retail datasets:

| Model | Expected R² | Expected RMSE |
|-------|-------------|---------------|
| Linear Regression | 0.50-0.60 | ~1200-1300 |
| Random Forest | 0.60-0.70 | ~1100-1200 |
| XGBoost | 0.65-0.75 | ~1050-1150 |
| Gradient Boosting | 0.60-0.70 | ~1100-1200 |
| Neural Network | 0.55-0.65 | ~1150-1250 |
| Voting Ensemble | 0.68-0.75 | ~1050-1150 |
| Stacking Ensemble | 0.68-0.76 | ~1040-1140 |

*Actual performance depends on data quality and preprocessing choices.*

## Citation & References

### Algorithms
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS.

### Dataset
BigMart Sales Prediction dataset from analytics competitions.

## License

This analysis pipeline is provided for educational and research purposes.

## Support

For questions about the methodology, refer to `process.txt`.

For technical issues, check the troubleshooting section above.

## Future Enhancements

Potential improvements documented in `process.txt`:
- LightGBM and CatBoost implementations
- Quantile regression for prediction intervals
- AutoML exploration with H2O or Auto-sklearn
- Time-series components if temporal data available
- Geographic features if location data available
- Deep learning with TabNet

---

**Author:** BigMart Analysis Pipeline  
**Version:** 1.0  
**Last Updated:** October 2025

