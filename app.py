from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from chatbot import generate_response

app = Flask(__name__)

emotion_model_name = "monologg/bert-base-cased-goemotions-original"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    emotion_result = emotion_classifier(user_input)[0]
    emotion = emotion_result["label"]
    response = generate_response(user_input, emotion)
    return jsonify({"response": response, "emotion": emotion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)


