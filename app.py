from flask import Flask, render_template, request, session
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session storage

# Load BlenderBot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Fallback reply list
VAGUE_REPLIES = ["I don't know", "I'm not sure", "That's interesting", "Can you tell me more?"]

# Fallback response
def fallback():
    return "I'm here for you. It's okay to feel this way."

# Generate response from BlenderBot
def generate_response(user_input):
    # Optionally add a prompt to guide the bot’s tone
    guided_input = f"You are a friendly and supportive assistant helping users with their feelings.\nUser: {user_input}"

    # Tokenize input
    inputs = tokenizer([guided_input], return_tensors="pt")

    # Generate model response
    reply_ids = model.generate(
        **inputs,
        max_length=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode response
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    # Check for vague responses
    if any(phrase.lower() in response.lower() for phrase in VAGUE_REPLIES):
        return fallback()
    
    return response

# Homepage
@app.route("/")
def index():
    return render_template("index.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    bot_response = generate_response(user_input)
    return render_template("index.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run()

