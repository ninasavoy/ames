from flask import Flask, request, jsonify
import pickle
import numpy as pd

app = Flask(__name__)

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)