# Data Science Assistant App
# app.py - Main application file

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    
    # Save categorical columns for encoding
    cat_cols_to_encode = [col for col in categorical_cols if col != target_col]
    
    # Prepare features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) if cleaning_options["scale_features"] else ('passthrough', 'passthrough')
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols_to_encode) if cat_cols_to_encode else ('passthrough', 'passthrough', [])
        ], remainder='passthrough')
    
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
        
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        
        cleaning_options = {
            "handle_missing": handle_missing,
            "scale_features": scale_features
        }
        
        if st.button("Process Dataset"):
            with st.spinner("Processing data..."):
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
                
                # Display cleaned data
                st.write("### Cleaned Data Preview")
                st.dataframe(cleaned_data.head())
                
                # Download link for cleaned data
                st.markdown(download_dataframe(cleaned_data, "cleaned_data"), unsafe_allow_html=True)

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
                # Extract model name from question
                for model in st.session_state.models.keys():
                    if model.lower() in question.lower():
                        st.write(f"A: Here's the confusion matrix for {model}:")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(st.session_state.results[model]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        break
                else:
                    st.write("A: Please specify which model's confusion matrix you'd like to see.")
                    
            elif "features" in question.lower() and "important" in question.lower():
                # Extract model name from question
                for model in st.session_state.models.keys():
                    if model.lower() in question.lower() and model in st.session_state.feature_importance:
                        st.write(f"A: Here are the most important features for {model}:")
                        fi_df = st.session_state.feature_importance[model].head(10)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax)
                        ax.set_title(f'Top 10 Features - {model}')
                        st.pyplot(fig)
                        break
                else:
                    st.write("A: Please specify which model's feature importance you'd like to see, or the model doesn't support feature importance.")
                    
            elif "compare" in question.lower() and "models" in question.lower():
                st.write("A: Here's a comparison of all models:")
                
                # Show comparison chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Model', y='Accuracy', data=comparison_df, ax=ax)
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
                # Show detailed metrics in a table
                st.write("Detailed metrics:")
                
                # Create a more detailed comparison dataframe
                detailed_metrics = []
                for model, results in st.session_state.results.items():
                    if 'classification_report' in results:
                        report = results['classification_report']
                        if 'macro avg' in report:
                            metrics = {
                                'Model': model,
                                'Accuracy': results['accuracy'],
                                'Precision': report['macro avg']['precision'],
                                'Recall': report['macro avg']['recall'],
                                'F1-Score': report['macro avg']['f1-score']
                            }
                            detailed_metrics.append(metrics)
                
                if detailed_metrics:
                    detailed_df = pd.DataFrame(detailed_metrics)
                    st.dataframe(detailed_df)
                    
            else:
                st.write("A: I don't understand that question. Try asking about model performance, confusion matrices, or feature importance.")
    else:
        st.info("Please train at least one model in the Predict tab first.")

# Footer 
st.markdown("---")
st.markdown("Data Science Assistant - Made with ❤️ using Streamlit")