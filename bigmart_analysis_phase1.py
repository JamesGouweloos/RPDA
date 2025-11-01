"""
BigMart Sales Prediction Pipeline - PHASE 1 IMPROVEMENTS
Enhanced version implementing:
1. Hierarchical modeling (separate Grocery Store model)
2. Fixed Random Forest regularization
3. Interaction features
4. CatBoost algorithm
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import xgboost as xgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Will skip CatBoost model.")
    print("Install with: pip install catboost")

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

VISUALIZATIONS_DIR = 'Visualizations_Phase1'
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


class BigMartAnalysisPhase1:
    """Phase 1 Enhanced Pipeline with Hierarchical Modeling"""
    
    def __init__(self, data_path='BigMart.csv'):
        self.data_path = data_path
        self.df = None
        self.df_preprocessed = None
        
        # Separate datasets for hierarchical modeling
        self.df_grocery = None
        self.df_supermarket = None
        
        # Training/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Grocery-specific splits
        self.X_train_grocery = None
        self.X_test_grocery = None
        self.y_train_grocery = None
        self.y_test_grocery = None
        
        # Supermarket-specific splits
        self.X_train_supermarket = None
        self.X_test_supermarket = None
        self.y_train_supermarket = None
        self.y_test_supermarket = None
        
        self.models = {}
        self.model_results = {}
        self.feature_names = None
        
    def load_and_preprocess(self):
        """Complete preprocessing pipeline"""
        print("="*80)
        print("PHASE 1: ENHANCED PREPROCESSING")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\nLoaded {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Standardize Item_Fat_Content
        fat_content_map = {
            'Low Fat': 'Low Fat', 'low fat': 'Low Fat', 'LF': 'Low Fat',
            'Regular': 'Regular', 'reg': 'Regular'
        }
        self.df['Item_Fat_Content'] = self.df['Item_Fat_Content'].replace(fat_content_map)
        
        # Non-edible items
        non_consumable = ['Health and Hygiene', 'Household', 'Others']
        self.df.loc[self.df['Item_Type'].isin(non_consumable), 'Item_Fat_Content'] = 'Non-Edible'
        
        # Impute missing values
        item_weight_mean = self.df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
        self.df['Item_Weight'].fillna(item_weight_mean, inplace=True)
        self.df['Item_Weight'].fillna(self.df['Item_Weight'].mean(), inplace=True)
        
        outlet_size_mode = self.df.groupby('Outlet_Type')['Outlet_Size'].apply(
            lambda x: x.mode()[0] if not x.mode().empty else 'Medium'
        )
        self.df['Outlet_Size'] = self.df.apply(
            lambda row: outlet_size_mode[row['Outlet_Type']] if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'],
            axis=1
        )
        
        # Handle zero visibility
        mean_visibility = self.df.groupby('Item_Type')['Item_Visibility'].transform('mean')
        self.df.loc[self.df['Item_Visibility'] == 0, 'Item_Visibility'] = mean_visibility
        
        # Feature engineering - Basic features
        current_year = 2013
        self.df['Outlet_Age'] = current_year - self.df['Outlet_Establishment_Year']
        
        self.df['Item_MRP_Bins'] = pd.cut(
            self.df['Item_MRP'],
            bins=[0, 69, 136, 203, 270],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        item_type_map = {
            'Dairy': 'Food', 'Soft Drinks': 'Drinks', 'Meat': 'Food',
            'Fruits and Vegetables': 'Food', 'Household': 'Non-Consumable',
            'Baking Goods': 'Food', 'Snack Foods': 'Food', 'Frozen Foods': 'Food',
            'Breakfast': 'Food', 'Health and Hygiene': 'Non-Consumable',
            'Hard Drinks': 'Drinks', 'Canned': 'Food', 'Breads': 'Food',
            'Starchy Foods': 'Food', 'Others': 'Non-Consumable', 'Seafood': 'Food'
        }
        self.df['Item_Type_Grouped'] = self.df['Item_Type'].map(item_type_map)
        
        avg_visibility_by_type = self.df.groupby('Item_Type')['Item_Visibility'].transform('mean')
        self.df['Item_Visibility_Ratio'] = self.df['Item_Visibility'] / avg_visibility_by_type
        
        self.df['Item_Category'] = self.df['Item_Identifier'].str[:2]
        
        # *** PHASE 1 ENHANCEMENT: INTERACTION FEATURES ***
        print("\n*** Creating Interaction Features ***")
        self.df['MRP_x_Visibility'] = self.df['Item_MRP'] * self.df['Item_Visibility']
        self.df['MRP_x_Weight'] = self.df['Item_MRP'] * self.df['Item_Weight']
        self.df['MRP_x_Age'] = self.df['Item_MRP'] * self.df['Outlet_Age']
        self.df['Weight_x_Visibility'] = self.df['Item_Weight'] * self.df['Item_Visibility']
        self.df['Age_x_Size_Numeric'] = self.df['Outlet_Age'] * self.df['Outlet_Size'].map(
            {'Small': 0, 'Medium': 1, 'High': 2}
        )
        
        print(f"Added 5 interaction features:")
        print("  - MRP_x_Visibility")
        print("  - MRP_x_Weight")
        print("  - MRP_x_Age")
        print("  - Weight_x_Visibility")
        print("  - Age_x_Size_Numeric")
        
        # Create preprocessing copy
        self.df_preprocessed = self.df.copy()
        
        # Ordinal encoding
        size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        location_map = {'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2}
        mrp_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Premium': 3}
        
        self.df_preprocessed['Outlet_Size'] = self.df_preprocessed['Outlet_Size'].map(size_map).astype('int64')
        self.df_preprocessed['Outlet_Location_Type'] = self.df_preprocessed['Outlet_Location_Type'].map(location_map).astype('int64')
        self.df_preprocessed['Item_MRP_Bins'] = self.df_preprocessed['Item_MRP_Bins'].map(mrp_map).astype('int64')
        
        # One-hot encoding
        nominal_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type', 
                       'Item_Type_Grouped', 'Item_Category']
        
        self.df_preprocessed = pd.get_dummies(
            self.df_preprocessed,
            columns=nominal_cols,
            drop_first=True
        )
        
        # Drop unnecessary columns
        cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year']
        self.df_preprocessed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        print(f"\nTotal features after preprocessing: {self.df_preprocessed.shape[1]}")
        
        return self
    
    def create_hierarchical_splits(self, test_size=0.2):
        """Create separate train-test splits for grocery stores and supermarkets"""
        print("\n" + "="*80)
        print("*** PHASE 1: HIERARCHICAL MODELING SETUP ***")
        print("="*80)
        
        # Identify grocery stores in original data
        grocery_mask = self.df['Outlet_Type'] == 'Grocery Store'
        
        print(f"\nTotal samples: {len(self.df)}")
        print(f"Grocery Store samples: {grocery_mask.sum()} ({grocery_mask.sum()/len(self.df)*100:.1f}%)")
        print(f"Supermarket samples: {(~grocery_mask).sum()} ({(~grocery_mask).sum()/len(self.df)*100:.1f}%)")
        
        # Split preprocessed data
        X = self.df_preprocessed.drop('Item_Outlet_Sales', axis=1)
        y = self.df_preprocessed['Item_Outlet_Sales']
        
        self.feature_names = X.columns.tolist()
        
        # Get indices for grocery and supermarket
        grocery_indices = self.df_preprocessed.index[grocery_mask]
        supermarket_indices = self.df_preprocessed.index[~grocery_mask]
        
        # Create main split (all data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Create grocery-specific split
        X_grocery = X.loc[grocery_indices]
        y_grocery = y.loc[grocery_indices]
        
        if len(X_grocery) > 50:  # Only if enough samples
            self.X_train_grocery, self.X_test_grocery, self.y_train_grocery, self.y_test_grocery = train_test_split(
                X_grocery, y_grocery, test_size=test_size, random_state=42
            )
            print(f"\nGrocery Store split:")
            print(f"  Training: {len(self.X_train_grocery)} samples")
            print(f"  Testing: {len(self.X_test_grocery)} samples")
        else:
            print("\n⚠️ Warning: Insufficient grocery store samples for separate model")
            self.X_train_grocery = None
        
        # Create supermarket-specific split
        X_supermarket = X.loc[supermarket_indices]
        y_supermarket = y.loc[supermarket_indices]
        
        self.X_train_supermarket, self.X_test_supermarket, self.y_train_supermarket, self.y_test_supermarket = train_test_split(
            X_supermarket, y_supermarket, test_size=test_size, random_state=42
        )
        
        print(f"\nSupermarket split:")
        print(f"  Training: {len(self.X_train_supermarket)} samples")
        print(f"  Testing: {len(self.X_test_supermarket)} samples")
        
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # Avoid division by zero in MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def train_improved_random_forest(self):
        """Train Random Forest with enhanced regularization"""
        print("\n" + "-"*80)
        print("Training Improved Random Forest (Reduced Overfitting)")
        print("-"*80)
        
        # *** PHASE 1: IMPROVED HYPERPARAMETERS ***
        param_grid = {
            'n_estimators': [50, 100, 150],  # Reduced from 300
            'max_depth': [8, 10, 12],  # More conservative
            'min_samples_split': [15, 20, 25],  # Increased from 2,5,10
            'min_samples_leaf': [4, 6, 8],  # Increased from 1,2,4
            'max_features': ['sqrt', 0.7]  # Added feature sampling
        }
        
        print("Using regularized hyperparameters to reduce overfitting...")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', 
                                   n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Out-of-Bag R²: {model.oob_score_:.4f}")
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        overfitting_gap = train_metrics['R2'] - test_metrics['R2']
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, R2: {train_metrics['R2']:.4f}")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, R2: {test_metrics['R2']:.4f}")
        print(f"Overfitting Gap: {overfitting_gap:.4f} (Target: <0.05)")
        
        if overfitting_gap < 0.05:
            print("✅ Overfitting successfully reduced!")
        else:
            print(f"⚠️ Still showing overfitting: {overfitting_gap:.4f}")
        
        self.models['Random Forest (Improved)'] = model
        self.model_results['Random Forest (Improved)'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_,
            'oob_score': model.oob_score_
        }
        
        return self
    
    def train_catboost(self):
        """Train CatBoost model"""
        if not CATBOOST_AVAILABLE:
            print("\n⚠️ CatBoost not available, skipping...")
            return self
        
        print("\n" + "-"*80)
        print("*** PHASE 1: Training CatBoost Regressor ***")
        print("-"*80)
        
        # CatBoost hyperparameters
        param_grid = {
            'iterations': [200, 300, 500],
            'learning_rate': [0.03, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        print("Performing Grid Search (this may take several minutes)...")
        catboost_model = CatBoostRegressor(
            random_state=42,
            verbose=0,
            loss_function='RMSE'
        )
        
        grid_search = GridSearchCV(
            catboost_model, param_grid, cv=3, 
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)
        
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_pred_train)
        test_metrics = self.calculate_metrics(self.y_test, y_pred_test)
        
        print(f"Training - RMSE: {train_metrics['RMSE']:.2f}, R2: {train_metrics['R2']:.4f}")
        print(f"Testing  - RMSE: {test_metrics['RMSE']:.2f}, R2: {test_metrics['R2']:.4f}")
        
        self.models['CatBoost'] = model
        self.model_results['CatBoost'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def train_hierarchical_models(self):
        """Train separate models for grocery stores and supermarkets"""
        print("\n" + "="*80)
        print("*** PHASE 1: HIERARCHICAL MODELING ***")
        print("="*80)
        
        if self.X_train_grocery is not None and len(self.X_train_grocery) > 30:
            print("\nTraining Grocery Store Specialized Model...")
            print("-"*80)
            
            # Try multiple algorithms for grocery stores
            grocery_models = {}
            
            # Linear model (might work better for smaller grocery sales)
            ridge_grocery = Ridge(alpha=10)
            ridge_grocery.fit(self.X_train_grocery, self.y_train_grocery)
            y_pred_grocery = ridge_grocery.predict(self.X_test_grocery)
            ridge_metrics = self.calculate_metrics(self.y_test_grocery, y_pred_grocery)
            grocery_models['Ridge'] = (ridge_grocery, ridge_metrics)
            
            # XGBoost for grocery
            xgb_grocery = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            xgb_grocery.fit(self.X_train_grocery, self.y_train_grocery)
            y_pred_grocery = xgb_grocery.predict(self.X_test_grocery)
            xgb_metrics = self.calculate_metrics(self.y_test_grocery, y_pred_grocery)
            grocery_models['XGBoost'] = (xgb_grocery, xgb_metrics)
            
            # Select best grocery model
            best_grocery_model_name = max(grocery_models.items(), key=lambda x: x[1][1]['R2'])[0]
            best_grocery_model, best_grocery_metrics = grocery_models[best_grocery_model_name]
            
            print(f"\nBest Grocery Model: {best_grocery_model_name}")
            print(f"  RMSE: {best_grocery_metrics['RMSE']:.2f}")
            print(f"  R²: {best_grocery_metrics['R2']:.4f}")
            print(f"  MAPE: {best_grocery_metrics['MAPE']:.2f}%")
            
            self.models['Grocery_Specialized'] = best_grocery_model
            self.model_results['Grocery_Specialized'] = {
                'test': best_grocery_metrics,
                'predictions': y_pred_grocery,
                'algorithm': best_grocery_model_name
            }
        else:
            print("\n⚠️ Insufficient grocery samples, using fallback strategy")
            self.models['Grocery_Specialized'] = None
        
        # Train supermarket model
        print("\nTraining Supermarket Specialized Model...")
        print("-"*80)
        
        # Use XGBoost for supermarkets (proven to work well)
        xgb_supermarket = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1
        )
        xgb_supermarket.fit(self.X_train_supermarket, self.y_train_supermarket)
        y_pred_supermarket = xgb_supermarket.predict(self.X_test_supermarket)
        supermarket_metrics = self.calculate_metrics(self.y_test_supermarket, y_pred_supermarket)
        
        print(f"Supermarket Model Performance:")
        print(f"  RMSE: {supermarket_metrics['RMSE']:.2f}")
        print(f"  R²: {supermarket_metrics['R2']:.4f}")
        print(f"  MAPE: {supermarket_metrics['MAPE']:.2f}%")
        
        self.models['Supermarket_Specialized'] = xgb_supermarket
        self.model_results['Supermarket_Specialized'] = {
            'test': supermarket_metrics,
            'predictions': y_pred_supermarket
        }
        
        return self
    
    def evaluate_hierarchical_model(self):
        """Evaluate the complete hierarchical model on full test set"""
        print("\n" + "="*80)
        print("HIERARCHICAL MODEL - FULL EVALUATION")
        print("="*80)
        
        # Identify grocery vs supermarket in test set
        test_outlet_types = self.df.loc[self.X_test.index, 'Outlet_Type']
        grocery_mask_test = (test_outlet_types == 'Grocery Store').values
        
        # Initialize predictions array
        hierarchical_predictions = np.zeros(len(self.X_test))
        
        # Predict grocery stores
        if self.models['Grocery_Specialized'] is not None:
            grocery_indices = np.where(grocery_mask_test)[0]
            if len(grocery_indices) > 0:
                hierarchical_predictions[grocery_indices] = self.models['Grocery_Specialized'].predict(
                    self.X_test.iloc[grocery_indices]
                )
                print(f"Predicted {len(grocery_indices)} grocery store samples with specialized model")
        
        # Predict supermarkets
        supermarket_indices = np.where(~grocery_mask_test)[0]
        if len(supermarket_indices) > 0:
            hierarchical_predictions[supermarket_indices] = self.models['Supermarket_Specialized'].predict(
                self.X_test.iloc[supermarket_indices]
            )
            print(f"Predicted {len(supermarket_indices)} supermarket samples with specialized model")
        
        # Calculate overall metrics
        hierarchical_metrics = self.calculate_metrics(self.y_test, hierarchical_predictions)
        
        print(f"\n*** HIERARCHICAL MODEL PERFORMANCE ***")
        print(f"Overall Test RMSE: {hierarchical_metrics['RMSE']:.2f}")
        print(f"Overall Test R²: {hierarchical_metrics['R2']:.4f}")
        print(f"Overall Test MAE: {hierarchical_metrics['MAE']:.2f}")
        print(f"Overall Test MAPE: {hierarchical_metrics['MAPE']:.2f}%")
        
        # Calculate performance by segment
        print(f"\nSegment Performance:")
        if np.sum(grocery_mask_test) > 0:
            grocery_metrics = self.calculate_metrics(
                self.y_test.values[grocery_mask_test],
                hierarchical_predictions[grocery_mask_test]
            )
            print(f"  Grocery Stores - R²: {grocery_metrics['R2']:.4f}, RMSE: {grocery_metrics['RMSE']:.2f}")
        
        supermarket_metrics = self.calculate_metrics(
            self.y_test.values[~grocery_mask_test],
            hierarchical_predictions[~grocery_mask_test]
        )
        print(f"  Supermarkets - R²: {supermarket_metrics['R2']:.4f}, RMSE: {supermarket_metrics['RMSE']:.2f}")
        
        self.model_results['Hierarchical_Combined'] = {
            'test': hierarchical_metrics,
            'predictions': hierarchical_predictions,
            'grocery_metrics': grocery_metrics if np.sum(grocery_mask_test) > 0 else None,
            'supermarket_metrics': supermarket_metrics
        }
        
        return self
    
    def generate_phase1_comparison(self):
        """Compare Phase 1 models against baseline"""
        print("\n" + "="*80)
        print("PHASE 1 vs BASELINE COMPARISON")
        print("="*80)
        
        # Read baseline results
        try:
            baseline_df = pd.read_csv('model_performance_summary.csv')
            baseline_best_r2 = baseline_df['Test_R2'].max()
            baseline_best_rmse = baseline_df['Test_RMSE'].min()
            baseline_best_model = baseline_df.loc[baseline_df['Test_R2'].idxmax(), 'Model']
            
            print(f"\nBaseline Best Model: {baseline_best_model}")
            print(f"  R²: {baseline_best_r2:.4f}")
            print(f"  RMSE: {baseline_best_rmse:.2f}")
        except:
            baseline_best_r2 = 0.6134
            baseline_best_rmse = 1025.13
            baseline_best_model = "Stacking Ensemble"
            print(f"\nUsing reported baseline: R² = {baseline_best_r2:.4f}, RMSE = {baseline_best_rmse:.2f}")
        
        # Phase 1 results
        phase1_models = []
        for model_name, results in self.model_results.items():
            if 'test' in results:
                phase1_models.append({
                    'Model': model_name,
                    'Test_R2': results['test']['R2'],
                    'Test_RMSE': results['test']['RMSE'],
                    'Test_MAE': results['test']['MAE'],
                    'Test_MAPE': results['test']['MAPE']
                })
        
        phase1_df = pd.DataFrame(phase1_models).sort_values('Test_R2', ascending=False)
        
        print("\n" + "="*80)
        print("PHASE 1 MODEL RESULTS")
        print("="*80)
        print(phase1_df.to_string(index=False))
        
        # Best Phase 1 model
        best_phase1_r2 = phase1_df['Test_R2'].max()
        best_phase1_rmse = phase1_df['Test_RMSE'].min()
        best_phase1_model = phase1_df.loc[phase1_df['Test_R2'].idxmax(), 'Model']
        
        print(f"\n*** IMPROVEMENT ANALYSIS ***")
        print(f"Best Phase 1 Model: {best_phase1_model}")
        print(f"  R²: {best_phase1_r2:.4f}")
        print(f"  RMSE: {best_phase1_rmse:.2f}")
        
        r2_improvement = (best_phase1_r2 - baseline_best_r2) * 100
        rmse_improvement = ((baseline_best_rmse - best_phase1_rmse) / baseline_best_rmse) * 100
        
        print(f"\nImprovement over Baseline:")
        print(f"  R² gain: {r2_improvement:+.2f} percentage points ({best_phase1_r2:.4f} vs {baseline_best_r2:.4f})")
        print(f"  RMSE improvement: {rmse_improvement:+.2f}% ({best_phase1_rmse:.2f} vs {baseline_best_rmse:.2f})")
        
        if r2_improvement >= 8:
            print("\n✅ SUCCESS! Achieved Phase 1 target (+8-12 percentage points)")
        elif r2_improvement >= 5:
            print("\n✅ GOOD PROGRESS! Getting close to Phase 1 target")
        else:
            print("\n⚠️ More optimization needed to reach Phase 1 target")
        
        # Save results
        phase1_df.to_csv('phase1_performance_summary.csv', index=False)
        print("\nSaved: phase1_performance_summary.csv")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        models = ['Baseline\n' + baseline_best_model] + phase1_df['Model'].tolist()
        r2_values = [baseline_best_r2] + phase1_df['Test_R2'].tolist()
        colors = ['lightcoral'] + ['lightgreen' if r2 > baseline_best_r2 else 'lightyellow' 
                                    for r2 in phase1_df['Test_R2']]
        
        axes[0].barh(models, r2_values, color=colors, edgecolor='black')
        axes[0].set_xlabel('Test R²')
        axes[0].set_title('R² Comparison: Baseline vs Phase 1')
        axes[0].axvline(baseline_best_r2, color='red', linestyle='--', label='Baseline')
        axes[0].legend()
        
        # RMSE comparison
        rmse_values = [baseline_best_rmse] + phase1_df['Test_RMSE'].tolist()
        colors = ['lightcoral'] + ['lightgreen' if rmse < baseline_best_rmse else 'lightyellow' 
                                    for rmse in phase1_df['Test_RMSE']]
        
        axes[1].barh(models, rmse_values, color=colors, edgecolor='black')
        axes[1].set_xlabel('Test RMSE (lower is better)')
        axes[1].set_title('RMSE Comparison: Baseline vs Phase 1')
        axes[1].axvline(baseline_best_rmse, color='red', linestyle='--', label='Baseline')
        axes[1].legend()
        axes[1].invert_xaxis()  # Lower is better
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'phase1_vs_baseline.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(VISUALIZATIONS_DIR, 'phase1_vs_baseline.png')}")
        plt.close()
        
        return self
    
    def save_best_phase1_model(self):
        """Save the best Phase 1 model"""
        print("\n" + "="*80)
        print("SAVING BEST PHASE 1 MODEL")
        print("="*80)
        
        # Find best model
        best_model_name = max(
            self.model_results.items(),
            key=lambda x: x[1]['test']['R2'] if 'test' in x[1] else -999
        )[0]
        
        best_r2 = self.model_results[best_model_name]['test']['R2']
        best_rmse = self.model_results[best_model_name]['test']['RMSE']
        
        # Save model
        if best_model_name == 'Hierarchical_Combined':
            # Save both grocery and supermarket models
            joblib.dump({
                'grocery_model': self.models['Grocery_Specialized'],
                'supermarket_model': self.models['Supermarket_Specialized'],
                'feature_names': self.feature_names,
                'type': 'hierarchical'
            }, 'best_model_phase1.pkl')
            print(f"\nSaved hierarchical model system:")
            print(f"  - Grocery Store specialized model")
            print(f"  - Supermarket specialized model")
            print(f"  - Feature names for deployment")
        else:
            best_model = self.models[best_model_name]
            joblib.dump(best_model, 'best_model_phase1.pkl')
            print(f"\nSaved: {best_model_name}")
        
        print(f"\nPerformance:")
        print(f"  Test R²: {best_r2:.4f}")
        print(f"  Test RMSE: {best_rmse:.2f}")
        print(f"\nModel saved as: best_model_phase1.pkl")
        
        return self


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BIGMART SALES PREDICTION - PHASE 1 IMPROVEMENTS")
    print("="*80)
    print("\nPhase 1 Enhancements:")
    print("  1. ✅ Hierarchical modeling (Grocery + Supermarket models)")
    print("  2. ✅ Improved Random Forest regularization")
    print("  3. ✅ Interaction features (5 new features)")
    print("  4. ✅ CatBoost algorithm")
    print("\nTarget: R² = 0.70-0.73 (+9-12 percentage points)")
    print("="*80)
    
    # Initialize analysis
    analysis = BigMartAnalysisPhase1('BigMart.csv')
    
    # Run Phase 1 pipeline
    analysis.load_and_preprocess() \
           .create_hierarchical_splits() \
           .train_hierarchical_models() \
           .evaluate_hierarchical_model() \
           .train_improved_random_forest() \
           .train_catboost() \
           .generate_phase1_comparison() \
           .save_best_phase1_model()
    
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - phase1_performance_summary.csv")
    print("  - best_model_phase1.pkl")
    print(f"  - Visualizations in {VISUALIZATIONS_DIR}/")
    print("\nNext Steps:")
    print("  - Review phase1_performance_summary.csv")
    print("  - Compare against baseline results")
    print("  - If target achieved, proceed to Phase 2")
    print("  - If not, iterate on Phase 1 improvements")


if __name__ == "__main__":
    main()

