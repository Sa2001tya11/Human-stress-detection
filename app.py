import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, classification_report, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from flask import Flask, render_template, request, jsonify
import json

# Initialize Flask app
app = Flask(__name__)

# Global variables for models and data
dt_model = None
rf_model = None
scaler = None
model_metrics = None
X_test_global = None
y_test_global = None
feature_columns_global = None
df_global = None


def load_and_explore_data():
    """Load and explore the dataset"""
    try:
        # Load the dataset
        df = pd.read_csv("SaYoPillow.csv")
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")

        # Rename columns to match HTML interface
        df.rename(columns={
            'sr': 'snoring_rate',
            'rr': 'respiration_rate',
            't': 'body_temperature',
            'lm': 'limb_movement',
            'bo': 'blood_oxygen',
            'rem': 'eye_movement',
            'sr.1': 'sleeping_hours',
            'hr': 'heart_rate',
            'sl': 'stress_level'
        }, inplace=True)

        print("\nColumn names after renaming:")
        print(df.columns.tolist())

        return df

    except FileNotFoundError:
        print("Error: SaYoPillow.csv file not found. Please make sure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_data(df):
    """Prepare data for machine learning"""
    try:
        # Separate features and target (matching HTML order)
        feature_columns = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement',
                           'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate']

        X = df[feature_columns]
        y = df['stress_level']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None, None, None, None


def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate machine learning models"""
    try:
        # Define models
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        }

        trained_models = {}
        metrics = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)

            metrics[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist()
            }

            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"{name} Precision: {precision:.4f}")
            print(f"{name} Recall: {recall:.4f}")
            print(f"{name} F1-Score: {f1:.4f}")

            trained_models[name] = model

        return trained_models, metrics

    except Exception as e:
        print(f"Error training models: {e}")
        return None, None


def calculate_correlation_matrix(df, feature_columns):
    """Calculate correlation matrix for features"""
    try:
        correlation_matrix = df[feature_columns].corr()
        return correlation_matrix.values.tolist()
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None


def calculate_stress_distribution(df):
    """Calculate stress level distribution"""
    try:
        stress_counts = df['stress_level'].value_counts().sort_index()
        return stress_counts.tolist()
    except Exception as e:
        print(f"Error calculating stress distribution: {e}")
        return None


def save_models(models, scaler, metrics):
    """Save trained models, scaler, and metrics"""
    try:
        os.makedirs('models', exist_ok=True)

        with open('models/decision_tree_model.pkl', 'wb') as file:
            pickle.dump(models["Decision Tree"], file)

        with open('models/random_forest_model.pkl', 'wb') as file:
            pickle.dump(models["Random Forest"], file)

        with open('models/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        with open('models/model_metrics.pkl', 'wb') as file:
            pickle.dump(metrics, file)

        print("Models and metrics saved successfully!")

    except Exception as e:
        print(f"Error saving models: {e}")


def load_models():
    """Load saved models, scaler, and metrics"""
    try:
        with open('models/decision_tree_model.pkl', 'rb') as file:
            dt_model = pickle.load(file)

        with open('models/random_forest_model.pkl', 'rb') as file:
            rf_model = pickle.load(file)

        with open('models/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        with open('models/model_metrics.pkl', 'rb') as file:
            metrics = pickle.load(file)

        print("Models and metrics loaded successfully!")
        return dt_model, rf_model, scaler, metrics

    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None


def print_model_accuracy():
    """Print model accuracy in a formatted way"""
    global model_metrics

    if model_metrics is None:
        print("Model metrics not available. Please train the models first.")
        return

    print("\n" + "=" * 60)
    print("MODEL ACCURACY REPORT")
    print("=" * 60)

    for model_name, metrics in model_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    print("\n" + "=" * 60)

    # Compare models
    dt_acc = model_metrics['Decision Tree']['accuracy']
    rf_acc = model_metrics['Random Forest']['accuracy']

    if dt_acc > rf_acc:
        best_model = "Decision Tree"
        best_acc = dt_acc
    else:
        best_model = "Random Forest"
        best_acc = rf_acc

    print(f"Best Model: {best_model} with {best_acc:.4f} ({best_acc * 100:.2f}%) accuracy")
    print("=" * 60)


def quick_accuracy_check():
    """Quick function to just print accuracies"""
    global model_metrics

    if model_metrics is None:
        print("Model metrics not available.")
        return

    print("\nModel Accuracies:")
    for model_name, metrics in model_metrics.items():
        print(f"{model_name}: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")


def initialize_models():
    """Initialize or load models"""
    global dt_model, rf_model, scaler, model_metrics, X_test_global, y_test_global, feature_columns_global, df_global

    # Try to load existing models
    dt_model, rf_model, scaler, model_metrics = load_models()

    # Always load the dataset for correlation and distribution calculations
    df_global = load_and_explore_data()
    if df_global is None:
        print("Failed to load data. Please ensure SaYoPillow.csv is in the directory.")
        return False

    if dt_model is None or rf_model is None or scaler is None:
        print("Models not found. Training new models...")

        X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(df_global)
        if X_train is None:
            return False

        # Store test data globally for later use
        X_test_global = X_test
        y_test_global = y_test
        feature_columns_global = feature_columns

        # Train models
        models, metrics = train_models(X_train, X_test, y_train, y_test)
        if models is None:
            return False

        # Save models and metrics
        save_models(models, scaler, metrics)

        # Set global variables
        dt_model = models["Decision Tree"]
        rf_model = models["Random Forest"]
        model_metrics = metrics
    else:
        # Set feature columns for loaded models
        feature_columns_global = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement',
                                  'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate']

    return True


# Flask routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from request
        data = request.json

        # Extract features in the correct order
        features = [
            float(data['snoring']),
            float(data['respiration']),
            float(data['temperature']),
            float(data['limb']),
            float(data['oxygen']),
            float(data['eye']),
            float(data['hours']),
            float(data['heart'])
        ]

        # Scale the features
        features_scaled = scaler.transform([features])

        # Make predictions
        dt_prediction = int(dt_model.predict(features_scaled)[0])
        rf_prediction = int(rf_model.predict(features_scaled)[0])

        # Get probabilities from Random Forest
        rf_probabilities = rf_model.predict_proba(features_scaled)[0]
        confidence = float(max(rf_probabilities))

        # Calculate average prediction
        avg_prediction = round((dt_prediction + rf_prediction) / 2)

        # Return results
        return jsonify({
            'success': True,
            'decisionTree': dt_prediction,
            'randomForest': rf_prediction,
            'average': avg_prediction,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    try:
        if model_metrics is None:
            return jsonify({
                'success': False,
                'error': 'Model metrics not available'
            })

        # Format metrics for frontend
        formatted_metrics = {
            'dt': {
                'accuracy': model_metrics['Decision Tree']['accuracy'],
                'precision': model_metrics['Decision Tree']['precision'],
                'recall': model_metrics['Decision Tree']['recall'],
                'f1Score': model_metrics['Decision Tree']['f1_score']
            },
            'rf': {
                'accuracy': model_metrics['Random Forest']['accuracy'],
                'precision': model_metrics['Random Forest']['precision'],
                'recall': model_metrics['Random Forest']['recall'],
                'f1Score': model_metrics['Random Forest']['f1_score']
            }
        }

        return jsonify({
            'success': True,
            'metrics': formatted_metrics
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data"""
    try:
        if model_metrics is None:
            return jsonify({
                'success': False,
                'error': 'Confusion matrix data not available'
            })

        confusion_data = {
            'dt': model_metrics['Decision Tree']['confusion_matrix'],
            'rf': model_metrics['Random Forest']['confusion_matrix']
        }

        return jsonify({
            'success': True,
            'confusion_matrices': confusion_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/correlation-matrix', methods=['GET'])
def get_correlation_matrix():
    """Get feature correlation matrix"""
    try:
        if df_global is None or feature_columns_global is None:
            return jsonify({
                'success': False,
                'error': 'Correlation data not available'
            })

        correlation_matrix = calculate_correlation_matrix(df_global, feature_columns_global)

        return jsonify({
            'success': True,
            'correlation_matrix': correlation_matrix,
            'features': feature_columns_global
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/stress-distribution', methods=['GET'])
def get_stress_distribution():
    """Get stress level distribution"""
    try:
        if df_global is None:
            return jsonify({
                'success': False,
                'error': 'Distribution data not available'
            })

        distribution = calculate_stress_distribution(df_global)

        return jsonify({
            'success': True,
            'distribution': distribution
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': dt_model is not None and rf_model is not None and scaler is not None
    })


@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    """Get model accuracy as JSON"""
    try:
        if model_metrics is None:
            return jsonify({
                'success': False,
                'error': 'Model metrics not available'
            })

        # Format accuracy data
        accuracy_data = {}
        for model_name, metrics in model_metrics.items():
            accuracy_data[model_name] = {
                'accuracy': metrics['accuracy'],
                'accuracy_percentage': metrics['accuracy'] * 100
            }

        # Find best model
        dt_acc = model_metrics['Decision Tree']['accuracy']
        rf_acc = model_metrics['Random Forest']['accuracy']

        if dt_acc > rf_acc:
            best_model = "Decision Tree"
            best_acc = dt_acc
        else:
            best_model = "Random Forest"
            best_acc = rf_acc

        return jsonify({
            'success': True,
            'accuracies': accuracy_data,
            'best_model': {
                'name': best_model,
                'accuracy': best_acc,
                'accuracy_percentage': best_acc * 100
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def main():
    """Main function to run the application"""
    print("=" * 50)
    print("Sleep Stress Level Prediction Web Application")
    print("=" * 50)

    # Initialize models
    if initialize_models():
        print("\nModels initialized successfully!")

        # Print detailed model accuracy report
        print_model_accuracy()

        print("\nStarting Flask web server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Additional endpoints:")
        print("  - Model Metrics: http://localhost:5000/model-metrics")
        print("  - Accuracy Only: http://localhost:5000/accuracy")
        print("  - Health Check: http://localhost:5000/health")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)

        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize models. Please check your data and try again.")


# Additional utility functions that can be called directly
def standalone_accuracy_check():
    """Run a standalone accuracy check without starting the web server"""
    print("Loading models for accuracy check...")
    if initialize_models():
        print_model_accuracy()
        return True
    else:
        print("Failed to load models.")
        return False


if __name__ == "__main__":
    main()