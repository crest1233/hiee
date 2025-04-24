from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Lighter model for faster responses
print("🔄 Loading BlenderBot Small (90M)...")
model_name = "facebook/blenderbot_small-90M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✅ Model loaded on {device}")

# Health check route
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "up"})

# Chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        print(f"📥 User: {user_input}")
        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=60)

        reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"🤖 Bot: {reply}")
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Flask server running at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001)
