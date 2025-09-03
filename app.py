from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

model_path = "./fine_tuned_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = {
    0: "highly negative",
    1: "negative",
    2: "slightly negative",
    3: "neutral",
    4: "slightly positive",
    5: "positive",
    6: "highly positive"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text',None)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation= True, padding= True, max_length=1024)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        print(probs)  # 여기서 각 클래스 확률 확인 가능

        predicted_class = torch.argmax(logits, dim = 1).item()
        predicted_label = id2label[predicted_class]

    return jsonify({
        "text": text,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug = True)


