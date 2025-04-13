from flask import Flask, request, jsonify,render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# Utility: Get model by name
def get_model(algorithm):
    if algorithm == 'LinearRegression':
        return LinearRegression()
    elif algorithm == 'LogisticRegression':
        return LogisticRegression()
    elif algorithm == 'DecisionTree':
        return DecisionTreeClassifier()
    elif algorithm == 'KNN':
        return KNeighborsClassifier()
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    X = np.array(data['X'])
    y = np.array(data['y'])
    test_size = float(data.get('test_size', 0.2))
    algorithm = data['algorithm']

    model = get_model(algorithm)
    if model is None:
        return jsonify({'error': 'Unsupported algorithm'}), 400

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    if algorithm == 'LinearRegression':
        score = r2_score(y_test, predictions)
    else:
        # Convert predictions to class labels for classification
        if predictions.dtype != int:
            predictions = np.round(predictions)
        score = accuracy_score(y_test, predictions)

    return jsonify({
        'predictions': predictions.tolist(),
        'score': score
    })

if __name__ == '__main__':
    app.run(debug=True)
