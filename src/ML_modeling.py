import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust DATA_PATH as per your project structure, same as in hypothesis_testing.py
DATA_PATH = os.path.join('data', 'MachineLearningRating_v3.txt')

# --- Helper Functions (Re-using/Adapting from previous tasks) ---

def load_data(filepath, delimiter='|'):
    """
    Loads the insurance claims data from a text file.
    Handles potential initial loading errors and allows specifying a delimiter.
    """
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False) # low_memory=False to handle mixed types in large files
        print(f"Data loaded successfully from {filepath} with delimiter '{delimiter}'. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the data TXT is in the 'data' directory and named 'insurance_claims.txt'.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}\n"
              "Please check if the delimiter is correct and the file format is consistent.")
        return None

def preprocess_data_for_ml(df):
    """
    Performs comprehensive preprocessing for ML models:
    Type conversions, NaN handling, feature engineering, and selection.
    """
    df_processed = df.copy()

    # --- Type Conversions and Initial Cleaning ---
    if 'TransactionMonth' in df_processed.columns:
        df_processed['TransactionDate'] = pd.to_datetime(df_processed['TransactionMonth'], errors='coerce')
        # Extract features from TransactionDate
        df_processed['TransactionMonth_num'] = df_processed['TransactionDate'].dt.month
        df_processed['TransactionYear'] = df_processed['TransactionDate'].dt.year
        df_processed['DayOfWeek'] = df_processed['TransactionDate'].dt.dayofweek # Monday=0, Sunday=6
    else:
        print("Warning: 'TransactionDate' column not found. Skipping date feature engineering.")

    # Convert 'TransactionMonth' (YYMM format) if 'TransactionDate' is missing
    if 'TransactionMonth' in df_processed.columns and 'TransactionDate' not in df_processed.columns:
        df_processed['TransactionMonth_dt'] = df_processed['TransactionMonth'].astype(str).apply(
            lambda x: pd.to_datetime(f'20{x[:2]}-{x[2:]}-01', errors='coerce') if len(x) == 4 else np.nan
        )
        df_processed['TransactionMonth_num'] = df_processed['TransactionMonth_dt'].dt.month
        df_processed['TransactionYear'] = df_processed['TransactionMonth_dt'].dt.year


    numerical_cols_to_convert = [
        'TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'Cylinders',
        'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CapitalOutstanding',
        'NumberOfVehiclesInFleet', 'SumInsured', 'CalculatedPremiumPerTerm',
        'ExcessSelected', 'RegistrationYear'
    ]
    for col in numerical_cols_to_convert:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            print(f"Warning: Numerical column '{col}' not found. Skipping conversion.")

    # Fill numerical NaNs for modeling purposes (e.g., with median or mean)
    # Consider using IterativeImputer or more sophisticated methods for production
    for col in ['Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CapitalOutstanding',
                'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm', 'ExcessSelected', 'RegistrationYear']:
        if col in df_processed.columns:
            # For simplicity, fill with median. For real project, analyze missingness patterns.
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                # print(f"Filled NaNs in '{col}' with median: {median_val}")

    # Fill NaNs for TotalPremium and TotalClaims before creating derived features
    # If these are primary targets, rows with NaNs might be dropped, or imputed with sophisticated method.
    if 'TotalPremium' in df_processed.columns:
        df_processed['TotalPremium'].fillna(df_processed['TotalPremium'].median(), inplace=True)
    if 'TotalClaims' in df_processed.columns:
        df_processed['TotalClaims'].fillna(0, inplace=True) # Assume no claim if TotalClaims is NaN, or drop. 0 is safer.

    # --- Feature Engineering ---
    # Create 'HadClaim' binary variable: 1 if TotalClaims > 0, else 0
    df_processed['HadClaim'] = (df_processed['TotalClaims'] > 0).astype(int)

    # Calculate Loss Ratio (TotalClaims / TotalPremium)
    # Handle cases where TotalPremium is 0 or NaN to avoid division errors
    df_processed['LossRatio'] = np.where(
        (df_processed['TotalPremium'].notnull()) & (df_processed['TotalPremium'] > 0),
        df_processed['TotalClaims'] / df_processed['TotalPremium'],
        0
    )

    # Calculate Margin (TotalPremium - TotalClaims)
    df_processed['Margin'] = df_processed['TotalPremium'] - df_processed['TotalClaims']

    # Age of Vehicle
    current_year = 2015 # Based on max data date August 2015
    if 'RegistrationYear' in df_processed.columns:
        df_processed['VehicleAge'] = current_year - df_processed['RegistrationYear']
        df_processed['VehicleAge'] = df_processed['VehicleAge'].apply(lambda x: max(0, x)) # Ensure no negative age
        df_processed['VehicleAge'].fillna(df_processed['VehicleAge'].median(), inplace=True) # Fill any NaNs after calculation
    else:
        print("Warning: 'RegistrationYear' not found. Skipping 'VehicleAge' feature engineering.")

    # Ratio of CustomValueEstimate to SumInsured (could indicate over/under insurance)
    if 'CustomValueEstimate' in df_processed.columns and 'SumInsured' in df_processed.columns:
        df_processed['ValueToSumInsuredRatio'] = np.where(
            (df_processed['SumInsured'].notnull()) & (df_processed['SumInsured'] > 0),
            df_processed['CustomValueEstimate'] / df_processed['SumInsured'],
            0 # Or use NaN, but 0 might be safer for models
        )
        df_processed['ValueToSumInsuredRatio'].fillna(0, inplace=True) # Fill NaNs created by invalid ratios
    else:
        print("Warning: 'CustomValueEstimate' or 'SumInsured' not found. Skipping 'ValueToSumInsuredRatio'.")


    # --- Categorical Feature Handling ---
    categorical_cols_to_process = [
        'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language',
        'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 'Province',
        'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType', 'Make',
        'Model', 'Bodytype', 'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle',
        'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory',
        'CoverType', 'CoverGroup', 'Section', 'Product', 'StatutoryClass',
        'StatutoryRiskType', 'PostalCode' # PostalCode can be treated as categorical for OHE
    ]

    for col in categorical_cols_to_process:
        if col in df_processed.columns:
            # Fill NaNs for categorical columns with a placeholder
            df_processed[col].fillna('Unknown', inplace=True)
            df_processed[col] = df_processed[col].astype(str) # Ensure it's string for OHE

        else:
            print(f"Warning: Categorical column '{col}' not found. Skipping processing.")

    # Drop original TransactionMonth and TransactionDate if new derived features are sufficient
    df_processed = df_processed.drop(columns=['TransactionMonth', 'TransactionDate', 'TransactionYearMonth_dt'], errors='ignore')
    # Drop IDs that won't be features
    df_processed = df_processed.drop(columns=['UnderwrittenCoverID', 'PolicyID'], errors='ignore')

    print("Data preprocessing for ML complete. Features engineered and NaNs handled.")
    return df_processed

# --- Model Training and Evaluation Functions ---

def train_and_evaluate_regression(X, y, model_name, model, target_name):
    """Trains and evaluates a regression model."""
    print(f"\n--- Training and Evaluating {model_name} for {target_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # 75:25 split

    # Create a column transformer for one-hot encoding and scaling
    categorical_features = X_train.select_dtypes(include='object').columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Ensure non-negative predictions for claims
    if target_name == 'Claim Severity':
        y_pred[y_pred < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")

    return pipeline, rmse, r2

def train_and_evaluate_classification(X, y, model_name, model, target_name):
    """Trains and evaluates a classification model."""
    print(f"\n--- Training and Evaluating {model_name} for {target_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # Stratify for imbalanced classes

    # Create a column transformer for one-hot encoding and scaling
    categorical_features = X_train.select_dtypes(include='object').columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  ROC-AUC: {roc_auc:.2f}")

    return pipeline, accuracy, precision, recall, f1, roc_auc

def plot_shap_feature_importance(model_pipeline, feature_names, title="SHAP Feature Importance"):
    """
    Plots SHAP feature importance for a given trained pipeline.
    Handles pipelines with preprocessors.
    """
    # Extract the trained model from the pipeline
    if hasattr(model_pipeline, 'named_steps') and 'regressor' in model_pipeline.named_steps:
        model = model_pipeline.named_steps['regressor']
        preprocessor = model_pipeline.named_steps['preprocessor']
    elif hasattr(model_pipeline, 'named_steps') and 'classifier' in model_pipeline.named_steps:
        model = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']
    else:
        print("Model is not part of a scikit-learn Pipeline with a preprocessor, or missing expected step names.")
        return

    # Create a dummy DataFrame to get preprocessed feature names
    dummy_X = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    preprocessed_dummy_X = preprocessor.transform(dummy_X)

    # Get feature names after one-hot encoding
    new_feature_names = preprocessor.get_feature_names_out()

    # Use a small subset of the training data for SHAP, especially for large datasets
    # For a full implementation, you'd pass X_train or a sample from it
    # Here, we need to create a dummy X for SHAP based on the transformed features
    # For tree models, we can use TreeExplainer
    if "tree" in str(type(model)).lower() or "xgboost" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    else:
        # For linear models, use LinearExplainer, otherwise KernelExplainer (slower)
        explainer = shap.LinearExplainer(model, preprocessed_dummy_X[0:1]) # Use a single background sample
        print("Using LinearExplainer for SHAP. For non-tree models, this might be computationally intensive or require careful background data selection.")

    # Create a small, preprocessed test set to explain
    # For a proper SHAP plot, you'd use a representative sample of your X_test after preprocessing
    # This is a conceptual placeholder
    if isinstance(preprocessed_dummy_X, np.ndarray):
        # Convert to DataFrame if it's a numpy array to assign column names
        preprocessed_dummy_X_df = pd.DataFrame(preprocessed_dummy_X, columns=new_feature_names)
    else: # If it's a sparse matrix or DataFrame
        preprocessed_dummy_X_df = preprocessed_dummy_X.toarray() if hasattr(preprocessed_dummy_X, 'toarray') else preprocessed_dummy_X
        preprocessed_dummy_X_df = pd.DataFrame(preprocessed_dummy_X_df, columns=new_feature_names)


    # Calculate SHAP values for a small sample
    # Note: For real applications, pass a sensible sample of preprocessed X_test
    # This is a placeholder for demonstration.
    try:
        shap_values = explainer.shap_values(preprocessed_dummy_X_df.head(10)) # Explain first 10 samples
    except Exception as e:
        print(f"Error calculating SHAP values: {e}. SHAP plotting may require more specific setup based on model type and data format.")
        print("Continuing without SHAP plot.")
        return

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list): # For multi-output models or multi-class classification
        shap.summary_plot(shap_values[0], preprocessed_dummy_X_df.head(10), feature_names=new_feature_names, show=False)
    else:
        shap.summary_plot(shap_values, preprocessed_dummy_X_df.head(10), feature_names=new_feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Special Task: Linear Regression per Zipcode ---
def fit_linear_regression_per_zipcode(df):
    """
    Fits a linear regression model that predicts TotalClaims for each zipcode.
    Due to the potential for many zipcodes, this will print results for a sample
    of zipcodes with sufficient data.
    """
    print("\n--- Fitting Linear Regression Model Per Zipcode (for TotalClaims) ---")

    if 'PostalCode' not in df.columns or 'TotalClaims' not in df.columns:
        print("Skipping per-zipcode regression: 'PostalCode' or 'TotalClaims' column missing.")
        return

    # Filter for policies with claims as TotalClaims is the target
    df_claims = df[df['TotalClaims'] > 0].copy()
    if df_claims.empty:
        print("No claims data available to fit models per zipcode.")
        return

    # Select numerical features relevant for TotalClaims prediction per zipcode
    # Avoid highly correlated features or those already used for claim prediction
    numerical_features_for_zip_lr = [
        'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm',
        'ExcessSelected', 'Kilowatts', 'Cubiccapacity', 'VehicleAge' # Engineered feature
    ]
    # Ensure selected features are in the dataframe and numeric
    numerical_features_for_zip_lr = [
        f for f in numerical_features_for_zip_lr if f in df_claims.columns and pd.api.types.is_numeric_dtype(df_claims[f])
    ]

    if not numerical_features_for_zip_lr:
        print("No suitable numerical features found for per-zipcode linear regression after filtering.")
        return

    results_per_zip = {}
    unique_zipcodes = df_claims['PostalCode'].unique()

    # To avoid iterating through millions of zipcodes, let's pick a sample or those with high policy counts
    # Get zipcodes with at least 50 policies with claims
    zipcode_counts = df_claims['PostalCode'].value_counts()
    eligible_zipcodes = zipcode_counts[zipcode_counts >= 50].index.tolist() # Adjust threshold as needed

    if not eligible_zipcodes:
        print("No zipcodes with sufficient claims data (>=50 policies) to fit individual models.")
        return

    print(f"Fitting models for {len(eligible_zipcodes)} eligible zip codes (min 50 claims per zip)...")

    for zip_code in eligible_zipcodes:
        zip_df = df_claims[df_claims['PostalCode'] == zip_code]
        if len(zip_df) < 50: # Ensure enough samples for regression
            continue

        X_zip = zip_df[numerical_features_for_zip_lr]
        y_zip = zip_df['TotalClaims']

        # Drop rows with any NaN in features or target for this specific zip_df
        zip_df_clean = zip_df.dropna(subset=numerical_features_for_zip_lr + ['TotalClaims'])
        X_zip = zip_df_clean[numerical_features_for_zip_lr]
        y_zip = zip_df_clean['TotalClaims']

        if X_zip.empty or len(X_zip) < 2: # Need at least two data points to fit a line
            continue

        try:
            model = LinearRegression()
            model.fit(X_zip, y_zip)
            y_pred_zip = model.predict(X_zip)
            rmse_zip = np.sqrt(mean_squared_error(y_zip, y_pred_zip))
            r2_zip = r2_score(y_zip, y_pred_zip)
            results_per_zip[zip_code] = {'model': model, 'rmse': rmse_zip, 'r2': r2_zip}
            # print(f"  Zip {zip_code}: RMSE={rmse_zip:.2f}, R2={r2_zip:.2f}")
        except Exception as e:
            # print(f"  Could not fit model for Zip {zip_code}: {e}")
            pass # Skip zips where model fitting fails

    print(f"Successfully fitted models for {len(results_per_zip)} zip codes.")
    # Print results for a few top performing zips or random sample
    if results_per_zip:
        print("\n--- Sample of Per-Zipcode Model Results (Top 5 by R2) ---")
        sorted_results = sorted(results_per_zip.items(), key=lambda item: item[1]['r2'], reverse=True)
        for i, (zip_code, metrics) in enumerate(sorted_results[:5]):
            print(f"  Zip {zip_code}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}")
    else:
        print("No models were successfully fitted for any zip code.")

    return results_per_zip

# --- Main Execution Flow ---
def main():
    """
    Main function to load, preprocess, and run ML models for AlphaCare.
    """
    print("Starting Machine Learning Modeling for AlphaCare Insurance Claims Data...")

    # Load data
    df = load_data(DATA_PATH, delimiter='|') # Adjust delimiter if your .txt file uses something else
    if df is None:
        return

    # Preprocess data for ML
    df_processed = preprocess_data_for_ml(df)

    # --- 1. Claim Severity Prediction ---
    # Filter for policies that actually had a claim
    df_claims_only = df_processed[df_processed['HadClaim'] == 1].copy()

    if df_claims_only.empty:
        print("No data with claims available for Claim Severity Prediction. Skipping.")
    else:
        # Define features (X) and target (y) for Claim Severity
        # Exclude 'TotalClaims', 'HadClaim', 'LossRatio', 'Margin' from features directly
        # 'PolicyID', 'UnderwrittenCoverID' would have been dropped in preprocessing
        # 'CalculatedPremiumPerTerm' could be a feature if it's not the target.
        # Ensure that target is not in features!
        features_severity = [col for col in df_claims_only.columns if col not in [
            'TotalClaims', 'HadClaim', 'LossRatio', 'Margin', 'TotalPremium'
        ]]
        
        # Filter for only existing columns in the df_claims_only
        features_severity = [f for f in features_severity if f in df_claims_only.columns]

        X_severity = df_claims_only[features_severity]
        y_severity = df_claims_only['TotalClaims']

        print(f"\nFeatures for Claim Severity Prediction: {X_severity.columns.tolist()}")

        # Models for Claim Severity
        models_severity = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10),
            "XGBoost Regressor": xgb.XGBRegressor(random_state=42, n_estimators=50, max_depth=5, objective='reg:squarederror')
        }

        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_severity_model_name = ""
        best_severity_model_pipeline = None

        results_severity = {}
        for name, model in models_severity.items():
            pipeline, rmse, r2 = train_and_evaluate_regression(X_severity, y_severity, name, model, "Claim Severity")
            results_severity[name] = {'rmse': rmse, 'r2': r2}
            if rmse < best_rmse:
                best_rmse = rmse
                best_r2 = r2
                best_severity_model_name = name
                best_severity_model_pipeline = pipeline

        print(f"\nBest Claim Severity Model: {best_severity_model_name} (RMSE: {best_rmse:.2f}, R-squared: {best_r2:.2f})")

        # Plot SHAP for the best severity model
        if best_severity_model_pipeline:
            # Need to create a sample for SHAP explanation
            # For simplicity, we'll use X_severity's head, but ideally a representative X_test sample is used
            # Ensure X_severity is properly formed with feature names for SHAP
            plot_shap_feature_importance(best_severity_model_pipeline, X_severity.columns.tolist(),
                                         f"SHAP Feature Importance for {best_severity_model_name} (Claim Severity)")
        
    # --- 2. Claim Probability Prediction (Advanced Task) ---
    if 'HadClaim' not in df_processed.columns:
        print("Skipping Claim Probability Prediction: 'HadClaim' column not found.")
    else:
        # Define features (X) and target (y) for Claim Probability
        features_probability = [col for col in df_processed.columns if col not in [
            'HadClaim', 'TotalClaims', 'LossRatio', 'Margin', 'TotalPremium'
        ]]
        # Filter for only existing columns in the df_processed
        features_probability = [f for f in features_probability if f in df_processed.columns]

        X_probability = df_processed[features_probability]
        y_probability = df_processed['HadClaim']

        print(f"\nFeatures for Claim Probability Prediction: {X_probability.columns.tolist()}")

        # Models for Claim Probability
        models_probability = {
            "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'), # liblinear for small datasets
            "Random Forest Classifier": RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10), # Using regressor, should be Classifier!
            "XGBoost Classifier": xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=5, use_label_encoder=False, eval_metric='logloss')
        }

        # Correcting the RandomForestRegressor to RandomForestClassifier
    

        best_roc_auc = -float('inf')
        best_probability_model_name = ""
        best_probability_model_pipeline = None

        results_probability = {}
        for name, model in models_probability.items():
            pipeline, accuracy, precision, recall, f1, roc_auc = train_and_evaluate_classification(
                X_probability, y_probability, name, model, "Claim Probability"
            )
            results_probability[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_probability_model_name = name
                best_probability_model_pipeline = pipeline

        print(f"\nBest Claim Probability Model: {best_probability_model_name} (ROC-AUC: {best_roc_auc:.2f})")

        # Plot SHAP for the best probability model
        if best_probability_model_pipeline:
            plot_shap_feature_importance(best_probability_model_pipeline, X_probability.columns.tolist(),
                                         f"SHAP Feature Importance for {best_probability_model_name} (Claim Probability)")
    
    # --- 3. Linear Regression per Zipcode ---
    # This task is separate and does not use the same preprocessor/pipeline setup
    fit_linear_regression_per_zipcode(df_processed)

    print("\nMachine Learning Modeling Completed.")

if __name__ == "__main__":
    main()