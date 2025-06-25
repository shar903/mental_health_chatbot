from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


chat_model_name = "microsoft/DialoGPT-small"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name)


supportive_responses = {
    "sadness": "I'm really sorry you're feeling sad . You're not alone. Want to talk about it?",
    "anger": "It's okay to feel angry . I’m here to listen if you want to share.",
    "fear": "That sounds scary . I'm right here with you.",
    "grief": "Losing someone is hard . I'm here for you anytime you want to talk.",
    "nervousness": "It’s normal to feel nervous . Deep breaths—you’ve got this!",
    "remorse": "It’s okay to feel regret . Growth comes from reflection.",
    "disappointment": "That must have been tough . Want to share what happened?",
    "confusion": "Let’s sort through this together .",
    "joy": "That's amazing!  What made you feel this happy?",
    "love": "Love is beautiful . Tell me more!",
    "curiosity": "Curiosity is a sign of growth . Let’s explore!",
    "amusement": "Haha, that sounds funny !",
    "admiration": "It’s lovely to feel admiration . Who or what inspired you?",
    "gratitude": "Gratitude can brighten any day . What are you thankful for?",
    "optimism": "Keep that positive energy flowing !",
    "relief": "So glad you’re feeling better .",
    "pride": "You should be proud of yourself !",
    "realization": "Aha! I love lightbulb moments .",
    "desire": "That sounds important to you . Want to tell me more?",
    "annoyance": "That can be frustrating . I’m listening.",
    "embarrassment": "It happens to all of us . Don’t be too hard on yourself.",
    "surprise": "Whoa! That must have been unexpected .",
    "guilt": "It’s okay to feel guilty . Let’s work through it.",
    "loneliness": "You're not alone . I'm here for you.",
    "boredom": "Let’s do something fun ! What are you in the mood for?"
}

def fallback_response():
    return "I'm here for you 💬. Just tell me what's on your mind."

def generate_response(user_input, emotion):
    emotion = emotion.lower().strip()

    if emotion in supportive_responses:
        return supportive_responses[emotion]

   
    input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors='pt')
    output_ids = chat_model.generate(
        input_ids,
        max_length=100,
        pad_token_id=chat_tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = chat_tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    if not response or response.lower() in ["i don't know", "i'm not sure", "that's interesting"]:
        return fallback_response()

    return response
