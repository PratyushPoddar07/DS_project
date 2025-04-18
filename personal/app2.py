# Data Science Assistant App
# app.py - Main application file

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import base64
from io import BytesIO
import re
from fuzzywuzzy import fuzz, process

# Set page configuration
st.set_page_config(page_title="Data Science Assistant", layout="wide")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'encoding_mapping' not in st.session_state:
    st.session_state.encoding_mapping = {}
if 'text_standardization_mappings' not in st.session_state:
    st.session_state.text_standardization_mappings = {}

# Header
st.title("🔍 Data Science Assistant")
st.markdown("Automate your machine learning workflow: Clean data, train models, and analyze results.")

# Navigation
tabs = st.tabs(["Clean", "Predict", "Compare & Visualize"])

# Helper Functions
def download_model(model, model_name):
    """Generate a download link for the model"""
    output = BytesIO()
    pickle.dump(model, output)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pkl">Download {model_name} Model</a>'
    return href

def download_dataframe(df, file_name):
    """Generate a download link for the dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download {file_name}</a>'
    return href

def get_numeric_and_categorical_columns(df):
    """Identify numeric and categorical columns"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

def standardize_text_values(df, categorical_cols, similarity_threshold=80):
    """Standardize text values in categorical columns with similar meaning"""
    df_processed = df.copy()
    standardization_mappings = {}
    
    for col in categorical_cols:
        if df_processed[col].dtype == 'object':
            # Get unique values
            unique_values = df_processed[col].dropna().unique()
            
            if len(unique_values) <= 1:
                continue
                
            # Create a mapping to standardize similar values
            value_mapping = {}
            processed_values = set()
            
            # For each unique value
            for value in unique_values:
                if value in processed_values:
                    continue
                    
                # Find similar values
                similar_values = []
                for other_value in unique_values:
                    if other_value != value and other_value not in processed_values:
                        # Calculate string similarity
                        similarity = fuzz.ratio(str(value).lower(), str(other_value).lower())
                        if similarity > similarity_threshold:
                            similar_values.append(other_value)
                            processed_values.add(other_value)
                
                # If similar values found, create mapping
                if similar_values:
                    for similar_value in similar_values:
                        value_mapping[similar_value] = value
                        
                    processed_values.add(value)
                    
            # Apply mapping if we found any
            if value_mapping:
                standardization_mappings[col] = value_mapping
                df_processed[col] = df_processed[col].replace(value_mapping)
                
    return df_processed, standardization_mappings

def detect_and_standardize_housing_specs(df, text_columns):
    """Specifically detect and standardize housing specifications like '2 bhk', '2 bedroom'"""
    df_processed = df.copy()
    standardization_mappings = {}
    
    for col in text_columns:
        if df_processed[col].dtype == 'object':
            value_mapping = {}
            
            # Function to standardize bedroom specs
            def standardize_bedroom(value):
                if pd.isna(value):
                    return value
                
                value_str = str(value).lower()
                
                # Check for bedroom patterns like "2 bhk", "2 bedroom", "2 br", etc.
                bedroom_patterns = [
                    (r'(\d+)\s*bhk', r'\1 bhk'),
                    (r'(\d+)\s*bedroom', r'\1 bhk'),
                    (r'(\d+)\s*bed', r'\1 bhk'),
                    (r'(\d+)\s*br', r'\1 bhk'),
                    (r'(\d+)\s*b\s*h\s*k', r'\1 bhk'),
                    (r'(\d+)\s*b\s*r', r'\1 bhk')
                ]
                
                for pattern, replacement in bedroom_patterns:
                    match = re.search(pattern, value_str)
                    if match:
                        standardized = re.sub(pattern, replacement, value_str)
                        if value != standardized:
                            value_mapping[value] = standardized
                        return standardized
                
                return value
            
            # Apply the standardization function
            df_processed[col] = df_processed[col].apply(standardize_bedroom)
            
            # Only store non-empty mappings
            if value_mapping:
                standardization_mappings[col] = value_mapping
    
    return df_processed, standardization_mappings

def encode_categorical_columns(df, categorical_cols, encoding_method='onehot'):
    """Encode categorical columns using one-hot encoding or label encoding"""
    df_processed = df.copy()
    encoding_mappings = {}
    
    for col in categorical_cols:
        if encoding_method == 'onehot':
            # One-hot encoding
            one_hot = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
            df_processed = pd.concat([df_processed, one_hot], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            encoding_mappings[col] = {'method': 'onehot', 'columns': one_hot.columns.tolist()}
        elif encoding_method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoding_mappings[col] = {'method': 'label', 'mapping': dict(zip(le.classes_, le.transform(le.classes_)))}
    
    return df_processed, encoding_mappings

def process_data(df, target_col, cleaning_options, test_size=0.2, random_state=42):
    """Process the data based on user selections"""
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Get column types
    numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df_processed)
    
    # Handle missing values
    if cleaning_options["handle_missing"] == "Drop rows":
        df_processed = df_processed.dropna()
    elif cleaning_options["handle_missing"] == "Fill with mean/median/mode":
        # Impute numeric values with mean
        for col in numeric_cols:
            if df_processed[col].isna().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        # Impute categorical values with mode
        for col in categorical_cols:
            if df_processed[col].isna().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Standardize text values for similar meanings (like '2 bhk' vs '2 bedroom')
    if cleaning_options["standardize_text"]:
        # First, general fuzzy matching for similar text
        df_processed, general_mappings = standardize_text_values(
            df_processed, categorical_cols, similarity_threshold=cleaning_options["similarity_threshold"]
        )
        
        # Then, specific rules for housing specs
        df_processed, housing_mappings = detect_and_standardize_housing_specs(df_processed, categorical_cols)
        
        # Store mappings
        st.session_state.text_standardization_mappings = {
            'general': general_mappings,
            'housing': housing_mappings
        }
    
    # Save categorical columns for encoding (excluding target if it's categorical)
    cat_cols_to_encode = [col for col in categorical_cols if col != target_col]
    
    # Encode categorical columns if requested
    if cleaning_options["encoding_method"] != "none" and cat_cols_to_encode:
        if cleaning_options["encoding_method"] == "auto":
            # Use one-hot for low cardinality, label for high cardinality
            onehot_cols = []
            label_cols = []
            
            for col in cat_cols_to_encode:
                unique_values = df_processed[col].nunique()
                if unique_values <= 10:  # Arbitrary threshold
                    onehot_cols.append(col)
                else:
                    label_cols.append(col)
            
            # Apply one-hot encoding
            if onehot_cols:
                df_processed, onehot_mappings = encode_categorical_columns(
                    df_processed, onehot_cols, encoding_method='onehot'
                )
                st.session_state.encoding_mapping.update(onehot_mappings)
            
            # Apply label encoding
            if label_cols:
                df_processed, label_mappings = encode_categorical_columns(
                    df_processed, label_cols, encoding_method='label'
                )
                st.session_state.encoding_mapping.update(label_mappings)
        else:
            # Apply the selected encoding method to all categorical columns
            df_processed, encodings = encode_categorical_columns(
                df_processed, cat_cols_to_encode, encoding_method=cleaning_options["encoding_method"]
            )
            st.session_state.encoding_mapping.update(encodings)
    
    # Prepare features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Create preprocessing pipelines for any remaining transformations
    numeric_cols = [col for col in X.columns if col in df_processed.select_dtypes(include=['int64', 'float64']).columns]
    categorical_cols = [col for col in X.columns if col in df_processed.select_dtypes(include=['object', 'category']).columns]
    
    preprocessor_steps = []
    
    if numeric_cols:
        # Fix: Create proper numeric transformer pipeline
        if cleaning_options["scale_features"]:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        else:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])
        preprocessor_steps.append(('num', numeric_transformer, numeric_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor_steps.append(('cat', categorical_transformer, categorical_cols))
    
    # Create column transformer if there are steps to perform
    if preprocessor_steps:
        preprocessor = ColumnTransformer(transformers=preprocessor_steps, remainder='passthrough')
    else:
        # Fix: Use None instead of 'passthrough' string as a transformer
        preprocessor = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save the preprocessor for future use
    st.session_state.preprocessor = preprocessor
    st.session_state.feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, df_processed

def train_model(X_train, y_train, model_type, params=None):
    """Train a model based on user selection"""
    if model_type == "Random Forest":
        model = RandomForestClassifier(**params) if params else RandomForestClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression(**params) if params else LogisticRegression(max_iter=1000)
    elif model_type == "SVM":
        model = SVC(**params, probability=True) if params else SVC(probability=True)
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate ROC curve and AUC (only for binary classification)
    roc_auc = None
    fpr = None
    tpr = None
    if len(np.unique(y_test)) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

def extract_feature_importance(model, feature_names):
    """Extract feature importance from the model if available"""
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        if len(model.coef_.shape) == 1:
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(model.coef_)
            }).sort_values('Importance', ascending=False)
        else:
            # For multiclass
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.mean(np.abs(model.coef_), axis=0)
            }).sort_values('Importance', ascending=False)
    return None

def generate_data_summary(df):
    """Generate summary statistics for the dataframe"""
    summary = {}
    summary['shape'] = df.shape
    summary['nulls'] = df.isnull().sum().to_dict()
    summary['dtypes'] = df.dtypes.to_dict()
    summary['numeric_summary'] = df.describe().to_dict()
    
    # Get unique counts for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        summary['categorical_counts'] = {col: df[col].value_counts().to_dict() for col in cat_cols}
    
    return summary

# 1. CLEAN TAB
with tabs[0]:
    st.header("1. Clean and Prepare Dataset")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        st.write("### Data Preview")
        st.dataframe(data.head())
        
        st.write("### Data Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
        with col2:
            missing_values = data.isnull().sum().sum()
            st.write(f"Missing values: {missing_values}")
            st.write(f"Data types: {', '.join(data.dtypes.astype(str).unique())}")
        
        st.write("### Configure Data Cleaning")
        
        target_col = st.selectbox("Select target column", data.columns.tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            handle_missing = st.selectbox(
                "Handle missing values",
                ["Drop rows", "Fill with mean/median/mode"]
            )
        with col2:
            scale_features = st.checkbox("Standardize/Scale numeric features", value=True)
        
        # Enhanced text standardization options
        standardize_text = st.checkbox("Standardize text values (group similar texts)", value=True)
        
        if standardize_text:
            similarity_threshold = st.slider(
                "Text similarity threshold (%)", 
                50, 
                100, 
                80, 
                help="Higher values require more similarity to group text values"
            )
        else:
            similarity_threshold = 80  # Default value if not used
        
        # Enhanced encoding options
        encoding_method = st.selectbox(
            "Categorical encoding method",
            ["auto", "onehot", "label", "none"],
            help="Auto will use one-hot for low cardinality and label for high cardinality"
        )
        
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        
        cleaning_options = {
            "handle_missing": handle_missing,
            "scale_features": scale_features,
            "standardize_text": standardize_text,
            "similarity_threshold": similarity_threshold,
            "encoding_method": encoding_method
        }
        
        if st.button("Process Dataset"):
            with st.spinner("Processing data..."):
                try:
                    X_train, X_test, y_train, y_test, cleaned_data = process_data(
                        data, target_col, cleaning_options, test_size
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.cleaned_data = cleaned_data
                    st.session_state.target_col = target_col
                    
                    st.success("Data processed successfully! You can now move to the Predict tab.")
                    
                    # Generate and display data summary
                    summary = generate_data_summary(cleaned_data)
                    
                    st.write("### Cleaned Data Summary")
                    st.write(f"Shape: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
                    
                    # Display counts of missing values after cleaning
                    st.write("#### Missing Values After Cleaning")
                    null_df = pd.DataFrame({
                        'Column': summary['nulls'].keys(),
                        'Missing Values': summary['nulls'].values()
                    })
                    st.dataframe(null_df)
                    
                    # Display text standardization results if applied
                    if cleaning_options["standardize_text"] and st.session_state.text_standardization_mappings:
                        st.write("#### Text Standardization Results")
                        
                        # Show general text standardization mappings
                        general_mappings = st.session_state.text_standardization_mappings['general']
                        if any(general_mappings.values()):
                            st.write("##### Similar Text Values Standardized:")
                            for col, mapping in general_mappings.items():
                                if mapping:
                                    st.write(f"**{col}** column:")
                                    for original, standardized in mapping.items():
                                        st.write(f"- '{original}' → '{standardized}'")
                        
                        # Show housing-specific standardization mappings
                        housing_mappings = st.session_state.text_standardization_mappings['housing']
                        if any(housing_mappings.values()):
                            st.write("##### Housing Specifications Standardized:")
                            for col, mapping in housing_mappings.items():
                                if mapping:
                                    st.write(f"**{col}** column:")
                                    for original, standardized in mapping.items():
                                        st.write(f"- '{original}' → '{standardized}'")
                    
                    # Display encoding results if applied
                    if cleaning_options["encoding_method"] != "none" and st.session_state.encoding_mapping:
                        st.write("#### Encoding Results")
                        for col, encoding_info in st.session_state.encoding_mapping.items():
                            if encoding_info['method'] == 'onehot':
                                st.write(f"**{col}** was one-hot encoded into {len(encoding_info['columns'])} columns")
                            elif encoding_info['method'] == 'label':
                                st.write(f"**{col}** was label encoded with {len(encoding_info['mapping'])} unique values")
                    
                    # Display cleaned data
                    st.write("### Cleaned Data Preview")
                    st.dataframe(cleaned_data.head())
                    
                    # Download link for cleaned data
                    st.markdown(download_dataframe(cleaned_data, "cleaned_data"), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.exception(e)

# 2. PREDICT TAB
with tabs[1]:
    st.header("2. Train and Evaluate Models")
    
    if st.session_state.X_train is not None:
        st.write(f"Data ready: {st.session_state.X_train.shape[0]} training samples and {st.session_state.X_test.shape[0]} test samples")
        
        model_options = ["Random Forest", "Logistic Regression", "SVM"]
        selected_model = st.selectbox("Select model", model_options)
        
        # Advanced options
        with st.expander("Advanced model settings"):
            if selected_model == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 500, 100)
                max_depth = st.slider("Maximum depth", 1, 50, 10)
                model_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            elif selected_model == "Logistic Regression":
                C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                model_params = {"C": C, "max_iter": 1000}
            elif selected_model == "SVM":
                C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                model_params = {"C": C, "kernel": kernel}
        
        # Cross-validation option
        do_cv = st.checkbox("Perform cross-validation", value=True)
        
        if st.button("Train Model"):
            with st.spinner(f"Training {selected_model}..."):
                try:
                    # Train the model
                    model = train_model(
                        st.session_state.X_train, 
                        st.session_state.y_train, 
                        selected_model, 
                        model_params
                    )
                    
                    # Store the model
                    st.session_state.models[selected_model] = model
                    
                    # Evaluate the model
                    results = evaluate_model(model, st.session_state.X_test, st.session_state.y_test)
                    st.session_state.results[selected_model] = results
                    
                    # Extract feature importance
                    feature_importance = extract_feature_importance(model, st.session_state.feature_names)
                    if feature_importance is not None:
                        st.session_state.feature_importance[selected_model] = feature_importance
                    
                    # Perform cross-validation if selected
                    if do_cv:
                        cv_scores = cross_val_score(model, 
                                                pd.concat([st.session_state.X_train, st.session_state.X_test]), 
                                                pd.concat([st.session_state.y_train, st.session_state.y_test]),
                                                cv=5)
                    
                    st.success(f"{selected_model} trained successfully!")
                    
                    # Display evaluation metrics
                    st.write("### Model Evaluation")
                    st.write(f"Test Accuracy: {results['accuracy']:.4f}")
                    
                    if do_cv:
                        st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                    # Display confusion matrix
                    st.write("### Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Display classification report
                    st.write("### Classification Report")
                    report_df = pd.DataFrame(results['classification_report']).drop('accuracy', axis=1).T
                    report_df = report_df.round(3)
                    st.dataframe(report_df)
                    
                    # Display ROC curve for binary classification
                    if results['roc_auc'] is not None:
                        st.write("### ROC Curve")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(results['fpr'], results['tpr'], label=f'AUC = {results["roc_auc"]:.3f}')
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax.legend(loc='lower right')
                        st.pyplot(fig)
                    
                    # Display feature importance if available
                    if selected_model in st.session_state.feature_importance:
                        st.write("### Feature Importance")
                        fi_df = st.session_state.feature_importance[selected_model].head(10)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax)
                        ax.set_title(f'Top 10 Features - {selected_model}')
                        st.pyplot(fig)
                    
                    # Model download link
                    st.markdown(download_model(model, selected_model), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.exception(e)
    else:
        st.info("Please process your data in the Clean tab first.")

# 3. COMPARE & VISUALIZE TAB
with tabs[2]:
    st.header("3. Compare Models & Visualize Results")
    
    if len(st.session_state.results) > 0:
        # Compare models accuracy
        st.write("### Model Comparison")
        model_accuracies = {model: results['accuracy'] for model, results in st.session_state.results.items()}
        
        # Create dataframe for comparison
        comparison_df = pd.DataFrame({
            'Model': list(model_accuracies.keys()),
            'Accuracy': list(model_accuracies.values())
        })
        
        # Display comparison table
        st.dataframe(comparison_df.sort_values('Accuracy', ascending=False))
        
        # Plot comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', data=comparison_df, ax=ax)
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        
        # Q&A Section
        st.write("### Ask Questions About Your Models")
        
        # Predefined questions
        question_templates = [
            "Which model performed best?",
            "Show me the confusion matrix for Random Forest",
            "What features are most important for Random Forest?",
            "Compare all models"
        ]
        
        # Let user select a question or type their own
        question_type = st.radio("Choose question type:", ["Select from templates", "Ask your own"])
        
        if question_type == "Select from templates":
            question = st.selectbox("Select a question:", question_templates)
        else:
            question = st.text_input("Ask a question about your models:")
        
        if question and st.button("Get Answer"):
            st.write(f"**Q: {question}**")
            
            # Process questions
            if "performed best" in question.lower():
                best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
                best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Accuracy']
                st.write(f"A: The best performing model is **{best_model}** with an accuracy of **{best_accuracy:.4f}**.")
                
                
                
                # Show a visual of the best model's confusion matrix
                st.write(f"Here's the confusion matrix for {best_model}:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(st.session_state.results[best_model]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
            elif "confusion matrix" in question.lower():
                # Extract the model name from the question
                for model in st.session_state.results.keys():
                    if model.lower() in question.lower():
                        st.write(f"A: Here's the confusion matrix for {model}:")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(st.session_state.results[model]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        break
                else:
                    st.write("A: Please specify which model's confusion matrix you want to see.")
                    
            elif "features" in question.lower() and "important" in question.lower():
                # Extract the model name from the question
                for model in st.session_state.feature_importance.keys():
                    if model.lower() in question.lower():
                        st.write(f"A: Here are the most important features for {model}:")
                        fi_df = st.session_state.feature_importance[model].head(10)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax)
                        ax.set_title(f'Top 10 Features - {model}')
                        st.pyplot(fig)
                        
                        # Display feature importance table
                        st.dataframe(fi_df)
                        break
                else:
                    st.write("A: Please specify which model's feature importance you want to see.")
                    
            elif "compare" in question.lower() and "models" in question.lower():
                st.write("A: Here's a comparison of all trained models:")
                
                # Show accuracy comparison
                st.write("#### Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Model', y='Accuracy', data=comparison_df, ax=ax)
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
                # If we have binary classification, compare ROC curves
                if all(res['roc_auc'] is not None for res in st.session_state.results.values()):
                    st.write("#### ROC Curve Comparison")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    for model, results in st.session_state.results.items():
                        ax.plot(results['fpr'], results['tpr'], label=f"{model} (AUC = {results['roc_auc']:.3f})")
                    
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve Comparison')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)
            else:
                st.write("A: I don't understand your question. Please try rephrasing or select from the template questions.")
        
        # Feature exploration section
        if st.session_state.cleaned_data is not None:
            st.write("### Explore Your Data")
            
            # Get columns for visualization
            numeric_cols, _ = get_numeric_and_categorical_columns(st.session_state.cleaned_data)
            
            if numeric_cols:
                st.write("#### Correlation Heatmap")
                corr_data = st.session_state.cleaned_data[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                st.pyplot(fig)
                
                # Distribution plots
                st.write("#### Feature Distributions")
                selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Histogram
                sns.histplot(st.session_state.cleaned_data[selected_feature], kde=True, ax=axes[0])
                axes[0].set_title(f'Distribution of {selected_feature}')
                
                # Boxplot
                sns.boxplot(y=st.session_state.cleaned_data[selected_feature], ax=axes[1])
                axes[1].set_title(f'Boxplot of {selected_feature}')
                
                st.pyplot(fig)
                
                # Scatter plot for selected features vs target (if target is numeric)
                if st.session_state.target_col in numeric_cols:
                    st.write("#### Feature vs Target")
                    selected_x = st.selectbox("Select feature for X-axis:", 
                                           [col for col in numeric_cols if col != st.session_state.target_col])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=selected_x, y=st.session_state.target_col, 
                                    data=st.session_state.cleaned_data, ax=ax)
                    ax.set_title(f'{selected_x} vs {st.session_state.target_col}')
                    st.pyplot(fig)
    else:
        st.info("Please train at least one model in the Predict tab first.")

# Add a footer
st.markdown("---")
st.markdown("Made with ❤️ by Data Science Assistant")