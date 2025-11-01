"""
BigMart Sales Prediction Pipeline
Comprehensive machine learning analysis following structured methodology
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues on Windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import xgboost as xgb
import shap
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create Visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = 'Visualizations'
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


class BigMartAnalysis:
    """Complete pipeline for BigMart sales prediction and analysis"""
    
    def __init__(self, data_path='BigMart.csv'):
        """Initialize the analysis pipeline"""
        self.data_path = data_path
        self.df = None
        self.df_preprocessed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.feature_names = None
        
    # ==================== SECTION 1: DATA PREPROCESSING ====================
    
    def load_data(self):
        """Load the BigMart dataset"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        return self
    
    def clean_categorical_variables(self):
        """Clean and standardize categorical variables"""
        print("\n" + "="*80)
        print("CLEANING CATEGORICAL VARIABLES")
        print("="*80)
        
        # Standardize Item_Fat_Content
        print("\nOriginal Item_Fat_Content values:", self.df['Item_Fat_Content'].value_counts())
        fat_content_map = {
            'Low Fat': 'Low Fat',
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'Regular': 'Regular',
            'reg': 'Regular'
        }
        self.df['Item_Fat_Content'] = self.df['Item_Fat_Content'].replace(fat_content_map)
        print("Standardized Item_Fat_Content values:", self.df['Item_Fat_Content'].value_counts())
        
        # Handle non-consumable items - they shouldn't have fat content
        non_consumable = ['Health and Hygiene', 'Household', 'Others']
        self.df.loc[self.df['Item_Type'].isin(non_consumable), 'Item_Fat_Content'] = 'Non-Edible'
        print("\nFinal Item_Fat_Content distribution:")
        print(self.df['Item_Fat_Content'].value_counts())
        
        return self
    
    def impute_missing_values(self):
        """Impute missing values using statistical and predictive methods"""
        print("\n" + "="*80)
        print("IMPUTING MISSING VALUES")
        print("="*80)
        
        # Item_Weight: Impute by mean weight per Item_Identifier (same product should have same weight)
        print("\nImputing Item_Weight...")
        item_weight_mean = self.df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
        self.df['Item_Weight'].fillna(item_weight_mean, inplace=True)
        
        # If still missing, use overall mean
        overall_mean_weight = self.df['Item_Weight'].mean()
        self.df['Item_Weight'].fillna(overall_mean_weight, inplace=True)
        print(f"Item_Weight missing values remaining: {self.df['Item_Weight'].isnull().sum()}")
        
        # Outlet_Size: Impute using mode per Outlet_Type (similar outlet types likely have similar sizes)
        print("\nImputing Outlet_Size...")
        outlet_size_mode = self.df.groupby('Outlet_Type')['Outlet_Size'].apply(
            lambda x: x.mode()[0] if not x.mode().empty else 'Medium'
        )
        self.df['Outlet_Size'] = self.df.apply(
            lambda row: outlet_size_mode[row['Outlet_Type']] if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'],
            axis=1
        )
        print(f"Outlet_Size missing values remaining: {self.df['Outlet_Size'].isnull().sum()}")
        
        # Handle zero visibility (anomaly)
        print("\nHandling zero Item_Visibility...")
        zero_visibility_count = (self.df['Item_Visibility'] == 0).sum()
        print(f"Records with zero visibility: {zero_visibility_count}")
        mean_visibility = self.df.groupby('Item_Type')['Item_Visibility'].transform('mean')
        self.df.loc[self.df['Item_Visibility'] == 0, 'Item_Visibility'] = mean_visibility
        
        return self
    
    def feature_engineering(self):
        """Create derived features to enhance predictive power"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # Outlet_Age: Years since establishment (more established stores may perform differently)
        current_year = 2013  # Assuming data is from 2013
        self.df['Outlet_Age'] = current_year - self.df['Outlet_Establishment_Year']
        print(f"\nCreated Outlet_Age: min={self.df['Outlet_Age'].min()}, max={self.df['Outlet_Age'].max()}")
        
        # Item_MRP_Bins: Price categories (Low, Medium, High, Premium)
        self.df['Item_MRP_Bins'] = pd.cut(
            self.df['Item_MRP'],
            bins=[0, 69, 136, 203, 270],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        print(f"\nCreated Item_MRP_Bins:")
        print(self.df['Item_MRP_Bins'].value_counts())
        
        # Item_Type_Grouped: Aggregate similar item types
        item_type_map = {
            'Dairy': 'Food',
            'Soft Drinks': 'Drinks',
            'Meat': 'Food',
            'Fruits and Vegetables': 'Food',
            'Household': 'Non-Consumable',
            'Baking Goods': 'Food',
            'Snack Foods': 'Food',
            'Frozen Foods': 'Food',
            'Breakfast': 'Food',
            'Health and Hygiene': 'Non-Consumable',
            'Hard Drinks': 'Drinks',
            'Canned': 'Food',
            'Breads': 'Food',
            'Starchy Foods': 'Food',
            'Others': 'Non-Consumable',
            'Seafood': 'Food'
        }
        self.df['Item_Type_Grouped'] = self.df['Item_Type'].map(item_type_map)
        print(f"\nCreated Item_Type_Grouped:")
        print(self.df['Item_Type_Grouped'].value_counts())
        
        # Item_Visibility_Ratio: Normalized visibility per item type
        avg_visibility_by_type = self.df.groupby('Item_Type')['Item_Visibility'].transform('mean')
        self.df['Item_Visibility_Ratio'] = self.df['Item_Visibility'] / avg_visibility_by_type
        print(f"\nCreated Item_Visibility_Ratio: mean={self.df['Item_Visibility_Ratio'].mean():.3f}")
        
        # Outlet_Item_Combination: Interaction feature
        self.df['Outlet_Item_Combo'] = self.df['Outlet_Type'] + '_' + self.df['Item_Type_Grouped']
        
        # Extract Item Category from Item_Identifier (FD=Food, DR=Drinks, NC=Non-Consumable)
        self.df['Item_Category'] = self.df['Item_Identifier'].str[:2]
        print(f"\nCreated Item_Category:")
        print(self.df['Item_Category'].value_counts())
        
        return self
    
    def encode_variables(self):
        """Encode categorical variables for machine learning"""
        print("\n" + "="*80)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*80)
        
        # Create a copy for preprocessing
        self.df_preprocessed = self.df.copy()
        
        # Label Encoding for ordinal variables
        label_encoders = {}
        ordinal_cols = ['Outlet_Size', 'Outlet_Location_Type', 'Item_MRP_Bins']
        
        # Define ordinal mappings
        size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        location_map = {'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2}
        mrp_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Premium': 3}
        
        self.df_preprocessed['Outlet_Size'] = self.df_preprocessed['Outlet_Size'].map(size_map).astype('int64')
        self.df_preprocessed['Outlet_Location_Type'] = self.df_preprocessed['Outlet_Location_Type'].map(location_map).astype('int64')
        self.df_preprocessed['Item_MRP_Bins'] = self.df_preprocessed['Item_MRP_Bins'].map(mrp_map).astype('int64')
        
        print(f"Label encoded: {ordinal_cols}")
        
        # One-Hot Encoding for nominal variables
        nominal_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type', 
                       'Item_Type_Grouped', 'Item_Category']
        
        self.df_preprocessed = pd.get_dummies(
            self.df_preprocessed,
            columns=nominal_cols,
            drop_first=True  # Avoid multicollinearity
        )
        
        print(f"\nOne-hot encoded: {nominal_cols}")
        print(f"Total features after encoding: {self.df_preprocessed.shape[1]}")
        
        # Drop unnecessary columns
        cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year', 
                       'Outlet_Item_Combo']
        self.df_preprocessed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        return self
    
    def outlier_analysis(self):
        """Analyze and handle outliers appropriately"""
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)
        
        # Analyze Item_Outlet_Sales for outliers
        Q1 = self.df['Item_Outlet_Sales'].quantile(0.25)
        Q3 = self.df['Item_Outlet_Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df['Item_Outlet_Sales'] < lower_bound) | 
                          (self.df['Item_Outlet_Sales'] > upper_bound)]
        
        print(f"\nItem_Outlet_Sales statistics:")
        print(self.df['Item_Outlet_Sales'].describe())
        print(f"\nOutliers detected (IQR method): {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
        # In retail, high sales are legitimate and valuable for prediction
        print("\nDecision: Retaining outliers as they represent legitimate high-sales scenarios")
        
        # Check other numerical features
        numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
        for feature in numerical_features:
            z_scores = np.abs(stats.zscore(self.df[feature]))
            outlier_count = (z_scores > 3).sum()
            print(f"{feature}: {outlier_count} outliers (|z| > 3)")
        
        return self
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "="*80)
        print("PREPARING TRAIN-TEST SPLIT")
        print("="*80)
        
        # Separate features and target
        X = self.df_preprocessed.drop('Item_Outlet_Sales', axis=1)
        y = self.df_preprocessed['Item_Outlet_Sales']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"\nTarget variable statistics:")
        print(f"Training - Mean: {self.y_train.mean():.2f}, Std: {self.y_train.std():.2f}")
        print(f"Testing - Mean: {self.y_test.mean():.2f}, Std: {self.y_test.std():.2f}")
        
        return self
    
    # ==================== SECTION 2: EXPLORATORY DATA ANALYSIS ====================
    
    def perform_eda(self):
        """Comprehensive exploratory data analysis"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Descriptive statistics
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        # Target variable distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.df['Item_Outlet_Sales'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Item Outlet Sales')
        plt.ylabel('Frequency')
        plt.title('Distribution of Item Outlet Sales')
        
        plt.subplot(1, 3, 2)
        self.df.boxplot(column='Item_Outlet_Sales', by='Outlet_Type', ax=plt.gca())
        plt.xlabel('Outlet Type')
        plt.ylabel('Sales')
        plt.title('Sales by Outlet Type')
        plt.suptitle('')
        
        plt.subplot(1, 3, 3)
        self.df.groupby('Item_Type_Grouped')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh')
        plt.xlabel('Average Sales')
        plt.title('Average Sales by Item Type Group')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'eda_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: {os.path.join(VISUALIZATIONS_DIR, 'eda_distributions.png')}")
        plt.close()
        
        # Correlation analysis
        numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 
                         'Item_Visibility_Ratio', 'Item_Outlet_Sales']
        correlation_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Correlation Matrix - Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'correlation_matrix.png')}")
        plt.close()
        
        print("\nKey Correlations with Item_Outlet_Sales:")
        sales_corr = correlation_matrix['Item_Outlet_Sales'].sort_values(ascending=False)
        print(sales_corr)
        
        # Additional visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Item MRP vs Sales
        axes[0, 0].scatter(self.df['Item_MRP'], self.df['Item_Outlet_Sales'], alpha=0.3)
        axes[0, 0].set_xlabel('Item MRP')
        axes[0, 0].set_ylabel('Item Outlet Sales')
        axes[0, 0].set_title('Item MRP vs Sales')
        
        # Outlet Age vs Sales
        self.df.groupby('Outlet_Age')['Item_Outlet_Sales'].mean().plot(ax=axes[0, 1], marker='o')
        axes[0, 1].set_xlabel('Outlet Age')
        axes[0, 1].set_ylabel('Average Sales')
        axes[0, 1].set_title('Outlet Age vs Average Sales')
        
        # Item Visibility vs Sales
        axes[1, 0].scatter(self.df['Item_Visibility'], self.df['Item_Outlet_Sales'], alpha=0.3)
        axes[1, 0].set_xlabel('Item Visibility')
        axes[1, 0].set_ylabel('Item Outlet Sales')
        axes[1, 0].set_title('Item Visibility vs Sales')
        
        # Sales by Outlet Location Type
        self.df.boxplot(column='Item_Outlet_Sales', by='Outlet_Location_Type', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Outlet Location Type')
        axes[1, 1].set_ylabel('Sales')
        axes[1, 1].set_title('Sales by Location Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'eda_relationships.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'eda_relationships.png')}")
        plt.close()
        
        return self
    
    # ==================== SECTION 3: MODEL DEVELOPMENT ====================
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def train_linear_regression(self):
        """Train baseline Linear Regression model"""
        print("\n" + "-"*80)
        print("Training Linear Regression (Baseline)")
        print("-"*80)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['Linear Regression'] = model
        self.model_results['Linear Regression'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test
        }
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\n" + "-"*80)
        print("Training Random Forest Regressor")
        print("-"*80)
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Performing Grid Search (this may take a few minutes)...")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', 
                                   n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        model = grid_search.best_estimator_
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['Random Forest'] = model
        self.model_results['Random Forest'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return self
    
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("\n" + "-"*80)
        print("Training XGBoost Regressor")
        print("-"*80)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        print("Performing Grid Search (this may take several minutes)...")
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        model = grid_search.best_estimator_
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['XGBoost'] = model
        self.model_results['XGBoost'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def train_neural_network(self):
        """Train Multi-layer Perceptron Neural Network"""
        print("\n" + "-"*80)
        print("Training Multi-layer Perceptron Neural Network")
        print("-"*80)
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        print("Performing Grid Search (this may take several minutes)...")
        mlp = MLPRegressor(max_iter=1000, random_state=42, early_stopping=True)
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        model = grid_search.best_estimator_
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['MLP Neural Network'] = {'model': model, 'scaler': scaler}
        self.model_results['MLP Neural Network'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting Regressor"""
        print("\n" + "-"*80)
        print("Training Gradient Boosting Regressor")
        print("-"*80)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        print("Performing Grid Search...")
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        model = grid_search.best_estimator_
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['Gradient Boosting'] = model
        self.model_results['Gradient Boosting'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def train_ensemble_voting(self):
        """Train Voting Regressor ensemble"""
        print("\n" + "-"*80)
        print("Training Voting Regressor Ensemble")
        print("-"*80)
        
        # Use best models from previous training
        estimators = [
            ('rf', self.models['Random Forest']),
            ('xgb', self.models['XGBoost']),
            ('gb', self.models['Gradient Boosting'])
        ]
        
        model = VotingRegressor(estimators=estimators)
        model.fit(self.X_train, self.y_train)
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['Voting Ensemble'] = model
        self.model_results['Voting Ensemble'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test
        }
        
        return self
    
    def train_ensemble_stacking(self):
        """Train Stacking Regressor ensemble"""
        print("\n" + "-"*80)
        print("Training Stacking Regressor Ensemble")
        print("-"*80)
        
        # Base learners
        estimators = [
            ('rf', self.models['Random Forest']),
            ('xgb', self.models['XGBoost']),
            ('gb', self.models['Gradient Boosting'])
        ]
        
        # Meta-learner
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=5
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, "
              f"R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, "
              f"R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        self.models['Stacking Ensemble'] = model
        self.model_results['Stacking Ensemble'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test
        }
        
        return self
    
    def analyze_feature_importance_shap(self):
        """Analyze feature importance using SHAP values"""
        print("\n" + "-"*80)
        print("SHAP Feature Importance Analysis")
        print("-"*80)
        
        try:
            # Use XGBoost model for SHAP analysis
            model = self.models['XGBoost']
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                             show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'shap_summary.png')}")
            plt.close()
            
            # Feature importance bar plot
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Features by SHAP Importance:")
            print(shap_importance.head(15))
            
            plt.figure(figsize=(10, 8))
            shap_importance.head(15).plot(x='feature', y='importance', kind='barh', legend=False)
            plt.xlabel('Mean |SHAP Value|')
            plt.ylabel('Feature')
            plt.title('Top 15 Features by SHAP Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'shap_importance.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'shap_importance.png')}")
            plt.close()
            
        except Exception as e:
            print(f"\nWarning: SHAP analysis failed due to compatibility issue: {str(e)}")
            print("Using alternative feature importance from Random Forest instead...")
            
            # Fallback to Random Forest feature importance
            model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Important Features (from Random Forest):")
            print(feature_importance.head(15))
            
            plt.figure(figsize=(10, 8))
            feature_importance.head(15).plot(x='feature', y='importance', kind='barh', legend=False)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Top 15 Features by Importance (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png')}")
            plt.close()
        
        return self
    
    # ==================== SECTION 4: MODEL VALIDATION ====================
    
    def cross_validation_analysis(self, k=5):
        """Perform k-fold cross-validation on all models"""
        print("\n" + "="*80)
        print(f"K-FOLD CROSS-VALIDATION (k={k})")
        print("="*80)
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = {}
        
        # Prepare full dataset
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        
        models_to_validate = {
            'Linear Regression': LinearRegression(),
            'Random Forest': self.models['Random Forest'],
            'XGBoost': self.models['XGBoost'],
            'Gradient Boosting': self.models['Gradient Boosting']
        }
        
        for model_name, model in models_to_validate.items():
            print(f"\nCross-validating {model_name}...")
            
            # RMSE scores (negative MSE converted to RMSE)
            cv_scores_mse = cross_val_score(model, X, y, cv=kfold, 
                                           scoring='neg_mean_squared_error', n_jobs=-1)
            cv_scores_rmse = np.sqrt(-cv_scores_mse)
            
            # R2 scores
            cv_scores_r2 = cross_val_score(model, X, y, cv=kfold, 
                                          scoring='r2', n_jobs=-1)
            
            cv_results[model_name] = {
                'RMSE_mean': cv_scores_rmse.mean(),
                'RMSE_std': cv_scores_rmse.std(),
                'R2_mean': cv_scores_r2.mean(),
                'R2_std': cv_scores_r2.std(),
                'RMSE_scores': cv_scores_rmse,
                'R2_scores': cv_scores_r2
            }
            
            print(f"  RMSE: {cv_scores_rmse.mean():.2f} (+/- {cv_scores_rmse.std():.2f})")
            print(f"  R²: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std():.4f})")
        
        # Visualize CV results
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        model_names = list(cv_results.keys())
        rmse_means = [cv_results[m]['RMSE_mean'] for m in model_names]
        rmse_stds = [cv_results[m]['RMSE_std'] for m in model_names]
        r2_means = [cv_results[m]['R2_mean'] for m in model_names]
        r2_stds = [cv_results[m]['R2_std'] for m in model_names]
        
        axes[0].bar(model_names, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Cross-Validation RMSE Scores')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(model_names, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Cross-Validation R² Scores')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: {os.path.join(VISUALIZATIONS_DIR, 'cross_validation_results.png')}")
        plt.close()
        
        self.cv_results = cv_results
        return self
    
    def statistical_comparison(self):
        """Statistical comparison of model performance using paired t-tests"""
        print("\n" + "="*80)
        print("STATISTICAL MODEL COMPARISON")
        print("="*80)
        
        # Compare models using paired t-test on CV scores
        models_list = list(self.cv_results.keys())
        
        print("\nPaired t-test results (RMSE - lower is better):")
        print("-" * 60)
        
        comparison_results = []
        
        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                model1 = models_list[i]
                model2 = models_list[j]
                
                scores1 = self.cv_results[model1]['RMSE_scores']
                scores2 = self.cv_results[model2]['RMSE_scores']
                
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                # Calculate Cohen's d (effect size)
                mean_diff = scores1.mean() - scores2.mean()
                pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                cohens_d = mean_diff / pooled_std
                
                comparison_results.append({
                    'Model 1': model1,
                    'Model 2': model2,
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Cohens d': cohens_d,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
                
                print(f"{model1} vs {model2}:")
                print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                print(f"  Cohen's d: {cohens_d:.4f}")
                print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                print()
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv('statistical_comparison.csv', index=False)
        print("Saved: statistical_comparison.csv")
        
        return self
    
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        summary_data = []
        
        for model_name, results in self.model_results.items():
            summary_data.append({
                'Model': model_name,
                'Train_RMSE': results['train']['RMSE'],
                'Test_RMSE': results['test']['RMSE'],
                'Train_MAE': results['train']['MAE'],
                'Test_MAE': results['test']['MAE'],
                'Train_R2': results['train']['R2'],
                'Test_R2': results['test']['R2'],
                'Train_MAPE': results['train']['MAPE'],
                'Test_MAPE': results['test']['MAPE']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_RMSE')
        
        print("\n" + summary_df.to_string(index=False))
        
        summary_df.to_csv('model_performance_summary.csv', index=False)
        print("\nSaved: model_performance_summary.csv")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = summary_df['Model']
        
        # RMSE comparison
        x = np.arange(len(models))
        width = 0.35
        axes[0, 0].bar(x - width/2, summary_df['Train_RMSE'], width, label='Train', alpha=0.8)
        axes[0, 0].bar(x + width/2, summary_df['Test_RMSE'], width, label='Test', alpha=0.8)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # R2 comparison
        axes[0, 1].bar(x - width/2, summary_df['Train_R2'], width, label='Train', alpha=0.8)
        axes[0, 1].bar(x + width/2, summary_df['Test_R2'], width, label='Test', alpha=0.8)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend()
        
        # MAE comparison
        axes[1, 0].bar(x - width/2, summary_df['Train_MAE'], width, label='Train', alpha=0.8)
        axes[1, 0].bar(x + width/2, summary_df['Test_MAE'], width, label='Test', alpha=0.8)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # MAPE comparison
        axes[1, 1].bar(x - width/2, summary_df['Train_MAPE'], width, label='Train', alpha=0.8)
        axes[1, 1].bar(x + width/2, summary_df['Test_MAPE'], width, label='Test', alpha=0.8)
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'model_performance_comparison.png')}")
        plt.close()
        
        # Prediction vs Actual plots for best models
        best_models = summary_df.head(3)['Model'].tolist()
        
        fig, axes = plt.subplots(1, len(best_models), figsize=(15, 5))
        if len(best_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(best_models):
            y_pred = self.model_results[model_name]['predictions']
            
            axes[idx].scatter(self.y_test, y_pred, alpha=0.5)
            axes[idx].plot([self.y_test.min(), self.y_test.max()], 
                          [self.y_test.min(), self.y_test.max()], 
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Sales')
            axes[idx].set_ylabel('Predicted Sales')
            axes[idx].set_title(f'{model_name}\nR²={self.model_results[model_name]["test"]["R2"]:.4f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'prediction_vs_actual.png')}")
        plt.close()
        
        return self
    
    # ==================== SECTION 5: SENSITIVITY & ROBUSTNESS ====================
    
    def sensitivity_different_splits(self):
        """Test model performance with different train-test splits"""
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS - Different Train-Test Splits")
        print("="*80)
        
        test_sizes = [0.15, 0.20, 0.25, 0.30]
        split_results = {model_name: [] for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']}
        
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        
        for test_size in test_sizes:
            print(f"\nTesting with {int(test_size*100)}% test split...")
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
            
            for model_name in split_results.keys():
                if model_name == 'Random Forest':
                    model = RandomForestRegressor(**self.model_results['Random Forest']['best_params'], 
                                                 random_state=42)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(**self.model_results['XGBoost']['best_params'], 
                                            random_state=42)
                else:
                    model = GradientBoostingRegressor(**self.model_results['Gradient Boosting']['best_params'],
                                                      random_state=42)
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                metrics = self.calculate_metrics(y_te, y_pred)
                
                split_results[model_name].append({
                    'test_size': test_size,
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2']
                })
        
        # Visualize results
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_name, results in split_results.items():
            test_sizes_plot = [r['test_size'] * 100 for r in results]
            rmse_values = [r['RMSE'] for r in results]
            r2_values = [r['R2'] for r in results]
            
            axes[0].plot(test_sizes_plot, rmse_values, marker='o', label=model_name)
            axes[1].plot(test_sizes_plot, r2_values, marker='o', label=model_name)
        
        axes[0].set_xlabel('Test Set Size (%)')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('RMSE vs Test Set Size')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_xlabel('Test Set Size (%)')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('R² Score vs Test Set Size')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'sensitivity_split_sizes.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: {os.path.join(VISUALIZATIONS_DIR, 'sensitivity_split_sizes.png')}")
        plt.close()
        
        return self
    
    def robustness_by_outlet_type(self):
        """Assess model generalizability across outlet types"""
        print("\n" + "="*80)
        print("ROBUSTNESS ANALYSIS - Performance by Outlet Type")
        print("="*80)
        
        # Get best performing model
        best_model_name = min(self.model_results.items(), 
                             key=lambda x: x[1]['test']['RMSE'])[0]
        print(f"\nUsing best model: {best_model_name}")
        
        if best_model_name == 'MLP Neural Network':
            model = self.models[best_model_name]['model']
            scaler = self.models[best_model_name]['scaler']
            X_test_eval = scaler.transform(self.X_test)
        else:
            model = self.models[best_model_name]
            X_test_eval = self.X_test
        
        # Get outlet type from original test data
        test_indices = self.X_test.index
        outlet_types = self.df.loc[test_indices, 'Outlet_Type']
        
        predictions = model.predict(X_test_eval)
        
        outlet_performance = []
        
        for outlet_type in outlet_types.unique():
            mask = outlet_types == outlet_type
            y_true_subset = self.y_test[mask]
            y_pred_subset = predictions[mask]
            
            metrics = self.calculate_metrics(y_true_subset, y_pred_subset)
            
            outlet_performance.append({
                'Outlet_Type': outlet_type,
                'Count': mask.sum(),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE']
            })
            
            print(f"\n{outlet_type}:")
            print(f"  Samples: {mask.sum()}")
            print(f"  RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}")
        
        outlet_df = pd.DataFrame(outlet_performance)
        outlet_df.to_csv('performance_by_outlet_type.csv', index=False)
        print("\nSaved: performance_by_outlet_type.csv")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].bar(outlet_df['Outlet_Type'], outlet_df['RMSE'])
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE by Outlet Type')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(outlet_df['Outlet_Type'], outlet_df['R2'], color='green')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² by Outlet Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(outlet_df['Outlet_Type'], outlet_df['MAE'], color='orange')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE by Outlet Type')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(outlet_df['Outlet_Type'], outlet_df['Count'], color='purple')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Sample Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'robustness_by_outlet_type.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'robustness_by_outlet_type.png')}")
        plt.close()
        
        return self
    
    def robustness_by_product_category(self):
        """Assess model generalizability across product categories"""
        print("\n" + "="*80)
        print("ROBUSTNESS ANALYSIS - Performance by Product Category")
        print("="*80)
        
        best_model_name = min(self.model_results.items(), 
                             key=lambda x: x[1]['test']['RMSE'])[0]
        print(f"\nUsing best model: {best_model_name}")
        
        if best_model_name == 'MLP Neural Network':
            model = self.models[best_model_name]['model']
            scaler = self.models[best_model_name]['scaler']
            X_test_eval = scaler.transform(self.X_test)
        else:
            model = self.models[best_model_name]
            X_test_eval = self.X_test
        
        test_indices = self.X_test.index
        item_type_grouped = self.df.loc[test_indices, 'Item_Type_Grouped']
        
        predictions = model.predict(X_test_eval)
        
        category_performance = []
        
        for category in item_type_grouped.unique():
            mask = item_type_grouped == category
            y_true_subset = self.y_test[mask]
            y_pred_subset = predictions[mask]
            
            metrics = self.calculate_metrics(y_true_subset, y_pred_subset)
            
            category_performance.append({
                'Category': category,
                'Count': mask.sum(),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE']
            })
            
            print(f"\n{category}:")
            print(f"  Samples: {mask.sum()}")
            print(f"  RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}")
        
        category_df = pd.DataFrame(category_performance)
        category_df.to_csv('performance_by_category.csv', index=False)
        print("\nSaved: performance_by_category.csv")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].bar(category_df['Category'], category_df['RMSE'])
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE by Product Category')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(category_df['Category'], category_df['R2'], color='green')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² by Product Category')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(category_df['Category'], category_df['MAE'], color='orange')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE by Product Category')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(category_df['Category'], category_df['Count'], color='purple')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Sample Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'robustness_by_category.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'robustness_by_category.png')}")
        plt.close()
        
        return self
    
    def save_best_model(self):
        """Save the best performing model"""
        print("\n" + "="*80)
        print("SAVING BEST MODEL")
        print("="*80)
        
        best_model_name = min(self.model_results.items(), 
                             key=lambda x: x[1]['test']['RMSE'])[0]
        best_model = self.models[best_model_name]
        
        joblib.dump(best_model, 'best_model.pkl')
        print(f"\nBest model ({best_model_name}) saved as: best_model.pkl")
        print(f"Test RMSE: {self.model_results[best_model_name]['test']['RMSE']:.2f}")
        print(f"Test R²: {self.model_results[best_model_name]['test']['R2']:.4f}")
        
        return self


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BIGMART SALES PREDICTION - COMPREHENSIVE ANALYSIS PIPELINE")
    print("="*80)
    
    # Initialize analysis
    analysis = BigMartAnalysis('BigMart.csv')
    
    # SECTION 1: Data Preprocessing
    analysis.load_data() \
           .clean_categorical_variables() \
           .impute_missing_values() \
           .feature_engineering() \
           .encode_variables() \
           .outlier_analysis() \
           .prepare_train_test_split()
    
    # SECTION 2: Exploratory Data Analysis
    analysis.perform_eda()
    
    # SECTION 3: Model Development
    print("\n" + "="*80)
    print("SECTION 3: MODEL DEVELOPMENT")
    print("="*80)
    
    analysis.train_linear_regression() \
           .train_random_forest() \
           .train_xgboost() \
           .train_gradient_boosting() \
           .train_neural_network() \
           .train_ensemble_voting() \
           .train_ensemble_stacking() \
           .analyze_feature_importance_shap()
    
    # SECTION 4: Model Validation
    analysis.cross_validation_analysis(k=5) \
           .statistical_comparison() \
           .generate_performance_summary()
    
    # SECTION 5: Sensitivity & Robustness
    analysis.sensitivity_different_splits() \
           .robustness_by_outlet_type() \
           .robustness_by_product_category() \
           .save_best_model()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"\nVisualization Files (in {VISUALIZATIONS_DIR}/):")
    print("  - eda_distributions.png")
    print("  - correlation_matrix.png")
    print("  - eda_relationships.png")
    print("  - shap_summary.png")
    print("  - shap_importance.png")
    print("  - cross_validation_results.png")
    print("  - model_performance_comparison.png")
    print("  - prediction_vs_actual.png")
    print("  - sensitivity_split_sizes.png")
    print("  - robustness_by_outlet_type.png")
    print("  - robustness_by_category.png")
    print("\nPerformance Metrics (in current directory):")
    print("  - model_performance_summary.csv")
    print("  - statistical_comparison.csv")
    print("  - performance_by_outlet_type.csv")
    print("  - performance_by_category.csv")
    print("  - best_model.pkl")


if __name__ == "__main__":
    main()

