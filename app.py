from flask import Flask,request, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    global model, encoder
    file = request.files['file']
    df = pd.read_csv(file)

    X = df[['code']]
    y = df['sentiment']

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)

    with open('model.pkl','wb') as f:
        pickle.dump((model,encoder),f)

    return jsonify({"status": "success", "trained_samples": len(df)})


@app.route('/predict', methods=['POST'])
def predict():
    global model, encoder
    data = request.get_json()
    code = [[data['code']]]
    X_encoded = encoder.transform(code)
    prediction = model.predict(X_encoded)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)


