from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

model_path = "./finetuned_model"
sentiment_analyzer = None

def load_model():
    global sentiment_analyzer
    sentiment_analyzer = pipeline("text-classification", model=model_path, tokenizer=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_analyzer(text, truncation=True, max_length=256)[0]

    return jsonify({
        "label": result['label'],  # e.g., "LABEL_0", "LABEL_1", "LABEL_2"
        "score": float(result['score'])
    })

@app.route('/reload', methods=['POST'])
def reload_model():
    load_model()
    return jsonify({"status": "reloaded"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)


