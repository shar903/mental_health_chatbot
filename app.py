from flask import Flask, render_template, request
from openai import OpenAI
import os
from dotenv import load_dotenv  


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)


VAGUE_REPLIES = ["I don't know", "I'm not sure", "That's interesting"]
def fallback():
    return "I'm here for you. It's okay to feel this way."


def generate_response(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a supportive mental health chatbot."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response.choices[0].message.content.strip()

        
        if any(x in reply.lower() for x in VAGUE_REPLIES):
            return fallback()

        return reply
    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    bot_response = generate_response(user_input)
    return render_template("index.html", user_input=user_input, bot_response=bot_response)


if __name__ == "__main__":
    app.run(debug=True)
