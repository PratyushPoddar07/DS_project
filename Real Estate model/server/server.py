from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        # Handle both form data and JSON
        if request.content_type == 'application/json':
            data = request.get_json()
            total_sqft = float(data['total_sqft'])
            location = data['location']
            bhk = int(data['bhk'])
            bath = int(data['bath'])
        else:
            # Form data
            total_sqft = float(request.form['total_sqft'])
            location = request.form['location']
            bhk = int(request.form['bhk'])
            bath = int(request.form['bath'])

        response = jsonify({
            'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400  # Bad Request


@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint to verify server is running"""
    return jsonify({'status': 'Server is up and running!'})


if __name__ == "__main__":
    print("Starting Python Flask server for home price prediction...")
    util.load_saved_artifacts()
    app.run(debug=True)  # Enable debug mode for better error messages