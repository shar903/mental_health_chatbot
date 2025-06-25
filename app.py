from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from chatbot import generate_response  

app = Flask(__name__)


emotion_model_name = "monologg/bert-base-cased-goemotions-original"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=1)


def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        emotion = result[0][0]['label'] if isinstance(result[0], list) else result[0]['label']
        return emotion
    except Exception as e:
        print("Emotion detection failed:", e)
        return "unknown"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    emotion = detect_emotion(user_input)
    bot_response = generate_response(user_input, emotion)
    return render_template("index.html", user_input=user_input, bot_response=bot_response, emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
