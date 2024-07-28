from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Normalize the data
    df_scaled = scaler.transform(df.drop(['timestamp'], axis=1))
    
    # Make predictions
    predictions = model.predict(df_scaled)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
