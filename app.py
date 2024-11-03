from flask import Flask, request, jsonify, render_template
from transformers import pipeline
app = Flask(__name__)

# Load Hugging Face model and tokenizer
model_id = "Subhajit01/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
def predict(text):
    max_prob_id = 0
    max_prob = 0
    preds = classifier(text, return_all_scores=True)
    for i in range(len(preds[0])):
        if (preds[0][i]["score"] > max_prob):
            max_prob = preds[0][i]["score"]
            max_prob_id = i
    return labels[max_prob_id]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = predict(text)
    return jsonify({"predicted_class": prediction})

if __name__ == "__main__":
    app.run(debug=True)
