# api/index.py
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------------------
# Symptom Data
# ----------------------------
SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "voice_changes", "text": "تغيرات في جودة الصوت"},
    {"key": "hoarseness", "text": "بحة الصوت"},
    {"key": "taste_changes", "text": "تغير الطعم"},
    {"key": "low_appetite", "text": "انخفاض الشهية"},
    {"key": "vomiting", "text": "تقيؤ"},
    {"key": "heartburn", "text": "حرقة صدر"},
    {"key": "gas", "text": "الغازات"},
    {"key": "bloating", "text": "الانتفاخ"},
    {"key": "hiccups", "text": "زغطة"},
    {"key": "constipation", "text": "امساك"},
    {"key": "diarrhea", "text": "اسهال"},
    {"key": "fecal_incontinence", "text": "سلس برازي"},
    {"key": "breath_shortness", "text": "ضيق تنفس"},
]

SYMPTOM_QUESTIONS = {
    "dry_mouth": [{"question": "في الأيام السبعة الماضية، ما شدة جفاف الفم؟",
                   "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
    "headache": [{"question": "في الأيام السبعة الماضية، ما شدة الصداع؟",
                  "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
    "nausea": [{"question": "في الأيام السبعة الماضية، ما شدة الغثيان؟",
                "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
    "abdominal_pain": [{"question": "في الأيام السبعة الماضية، ما شدة ألم البطن؟",
                        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
    "hiccups": [{"question": "في الأيام السبعة الماضية، ما شدة الزغطة؟",
                 "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
    "heartburn": [{"question": "في الأيام السبعة الماضية، ما شدة حرقة الصدر؟",
                   "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]}],
}

# ----------------------------
# Load model
# ----------------------------
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

# ----------------------------
# Symptom detection
# ----------------------------
def detect_symptoms_embedding(user_text, top_k=5, threshold=0.15):
    import re
    parts = re.split(r"[,.!؟؛]", user_text)
    detected = set()
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        user_emb = model.encode([part])[0]
        similarities = [cosine_sim(user_emb, emb) for emb in symptom_embeddings]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        for idx in top_indices:
            if similarities[idx] > threshold:
                detected.add(SYMPTOMS[idx]["key"])
    return list(detected)

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
session_data = {"pending": [], "answers": {}, "completed": False, "chats": []}

# ----------------------------
# HTML_PAGE
# ----------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🤖 Symptom Checker</title>
<style>
body {font-family: Arial, sans-serif; background:#e5ddd5; display:flex; flex-direction:column; align-items:center; margin:0; padding:0;}
header {background:#075e54; color:white; width:100%; text-align:center; padding:20px;}
h1 { margin:0; }
main {background:white; width:90%; max-width:700px; margin-top:20px; padding:20px; border-radius:15px; box-shadow:0 0 15px rgba(0,0,0,0.1);}
.chat-box {max-height:60vh; overflow-y:auto; display:flex; flex-direction:column; gap:10px;}
.bubble {padding:12px 15px; margin:5px 0; border-radius:12px; max-width:85%; word-wrap:break-word;}
.user {background:#dcf8c6; align-self:flex-end;}
.ai {background:#f0f0f0; align-self:flex-start; border-left:4px solid #075e54;}
form {display:flex; flex-direction:column; gap:10px; margin-top:15px;}
button.option {padding:12px; border-radius:25px; border:none; background:#25d366; color:white; font-size:16px; cursor:pointer; transition:0.3s; width:100%; text-align:center;}
button.option:hover {background:#128c7e; transform: scale(1.05);}
textarea {flex:1; padding:12px; border-radius:10px; font-size:16px; border:1px solid #ccc; outline:none; transition:0.2s;}
textarea:focus {border-color:#075e54; box-shadow:0 0 5px rgba(7,94,84,0.3);}
.send-btn {background:#25d366; color:white; padding:12px; font-size:16px; border:none; border-radius:25px; cursor:pointer; transition:0.3s;}
.send-btn:hover {background:#128c7e; transform: scale(1.05);}
footer {text-align:center; padding:10px; color:#555; font-size:14px; margin-top:20px;}
</style>
</head>
<body>
<header><h1>🤖 Symptom Checker</h1></header>
<main>
<div class="chat-box" id="chat-box">
{% for q,a in chats %}
<div class="bubble user"><strong>You:</strong> {{ q }}</div>
<div class="bubble ai"><strong>Bot:</strong> {{ a }}</div>
{% endfor %}
{% if completed %}
<div class="bubble ai"><strong>Bot:</strong><br>شكراً على وقت حضرتكم.<br>برجاء احضار التحاليل المطلوبة غداً في الجلسة.<br>وشكراً.</div>
{% endif %}
</div>
<form method="POST">
{% if pending %}
  {% set symptom = pending[0] %}
  <p><strong>{{ symptom_question[symptom] }}</strong></p>
  {% for opt in severity_options %}
    <button class="option" type="submit" name="answer" value="{{ opt }}">{{ opt }}</button>
  {% endfor %}
{% elif not completed %}
  <textarea name="question" placeholder="اكتب الأعراض هنا..." required></textarea>
  <button type="submit" class="send-btn">إرسال</button>
{% endif %}
</form>
</main>
<footer>© 2025 Halmoushy</footer>
</body>
</html>
"""

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        session_data["pending"] = []
        session_data["answers"] = {}
        session_data["completed"] = False
        session_data["chats"] = []
        return render_template_string(HTML_PAGE, chats=session_data["chats"],
                                      pending=session_data["pending"],
                                      completed=session_data["completed"],
                                      symptom_question={k: v[0]["question"] for k, v in SYMPTOM_QUESTIONS.items()},
                                      severity_options=["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"])
    if request.method == "POST":
        if "answer" in request.form:
            if session_data["pending"]:
                sym = session_data["pending"].pop(0)
                answer = request.form["answer"]
                session_data["answers"][sym] = answer
                session_data["chats"].append((SYMPTOM_QUESTIONS[sym][0]["question"], answer))
                if not session_data["pending"]:
                    session_data["completed"] = True
        elif "question" in request.form:
            user_text = request.form["question"].strip()
            if user_text:
                session_data["chats"].append((user_text, ""))
                detected = detect_symptoms_embedding(user_text)
                session_data["pending"] = [d for d in detected if d in SYMPTOM_QUESTIONS]
                if not session_data["pending"]:
                    session_data["chats"].append(("Bot", "لم أتعرف على أي عرض."))
        return render_template_string(HTML_PAGE, chats=session_data["chats"],
                                      pending=session_data["pending"],
                                      completed=session_data["completed"],
                                      symptom_question={k: v[0]["question"] for k, v in SYMPTOM_QUESTIONS.items()},
                                      severity_options=["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"])

# ----------------------------
# Vercel handler
# ----------------------------
def handler(environ, start_response):
    from mangum import Mangum
    return Mangum(app)(environ, start_response)