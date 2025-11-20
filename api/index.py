
import requests
import json

# Configuration
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = "4_JENf9g9NVi7_332loZt65qIydiAJCPNHhbx0irqaHtJPkfqcUCpp8tp85SlqOU8QX1lYp4AsvLtKqgx0OXRQ"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def deepseek_chat(prompt, system_prompt=None, max_tokens=512, temperature=0.3):
    """
    Call DeepSeek-v3.1 model hosted on Huawei Cloud.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-v3.1",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    print("Sending request to DeepSeek-v3.1 ")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        print(f" Request failed with status {response.status_code}")
        print(response.text)
        return None

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

#  Test
answer = deepseek_chat("Explain supervised learning simply.")
print("✅ DeepSeek Response:\n", answer)



from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sentence_transformers import SentenceTransformer
import numpy as np

# قائمة الأعراض كنصوص عربية/إنجليزية
SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "Voice quality changes", "text": "تغيرات في جودة الصوت"},
    {"key": "Hoarseness", "text": "بحة الصوت"},
    {"key": "Taste changes ", "text": "تغير الطعم"},
    {"key": " Decreased appetite ", "text": "انخفاض الشهية"},
    {"key": "Vomiting", "text": "تقيؤ"},
    {"key": "Heartburn", "text": "حرقة صدر"},
    {"key": "Gas", "text": "الغازات"},
    {"key": "Bloating", "text": "الانتفاخ"},
    {"key": "Hiccups", "text": "زغطة"},
    {"key": "Constipation", "text": "امساك"},
    {"key": "Diarrhea", "text": "اسهال"},
    {"key": "Fecal incontinence", "text": "سلس برازي"},
    {"key": "Shortness of breath", "text": "ضيق تنفس"},
]

# نموذج يدعم العربية والإنجليزية
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# حساب تمثيلات الأعراض مسبقاً
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms_embedding(user_text, top_k=3):
    """
    ترجع قائمة أقرب الأعراض بناءً على Embeddings
    """
    user_embedding = model.encode([user_text])[0]

    # حساب التشابه الكوني (cosine similarity)
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_sim(user_embedding, emb) for emb in symptom_embeddings]

    # ترتيب من الأعلى تشابه
    top_indices = np.argsort(similarities)[::-1][:top_k]

    detected = []
    for idx in top_indices:
        detected.append({
            "key": SYMPTOMS[idx]["key"],
            "text": SYMPTOMS[idx]["text"],
            "similarity": similarities[idx]
        })
    return detected

user_input = "صدري واجعني "
detected = detect_symptoms_embedding(user_input)

print("تم اكتشاف الأعراض الأقرب:")
for d in detected:
    print(f"{d['text']} (Key: {d['key']}, Similarity: {d['similarity']:.2f})")

SYMPTOM_QUESTIONS = {
    "dry_mouth": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة جفاف الفم؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "difficulty_swallowing": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة صعوبة البلع؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "mouth_throat_sores": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة تقرحات الفم والحلق (إذا موجودة)؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر هذا على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "cheilosis": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة تشقق زوايا الفم؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "headache": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الصداع؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "nausea": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الغثيان؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر الغثيان على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "fatigue": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الإرهاق؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Voice quality changes": [
        {
            "question": "في الأيام السبعة الماضية، هل لاحظت أي تغييرات في جودة الصوت؟",
            "options": ["Ο نعم", "Ο لا"]
        }
    ],
    "Hoarseness": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة خشونة الصوت؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Taste changes": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة تغييرات الطعم؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Decreased appetite": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة فقدان الشهية؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر فقدان الشهية على كمية الطعام المتناولة؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Vomiting": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة القيء (إذا حصل)؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر القيء على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Heartburn": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة حرقة المعدة؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر حرقة المعدة على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Gas": [
        {
            "question": "في الأيام السبعة الماضية، هل عانيت من الغازات؟",
            "options": ["Ο نعم", "Ο لا"]
        }
    ],
    "Bloating": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة انتفاخ البطن؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر الانتفاخ على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Hiccups": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الحازوقة (قفزات المعدة)؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر الحازوقة على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Constipation": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الإمساك؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Diarrhea": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الإسهال؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "abdominal_pain": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة ألم البطن؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر ألم البطن على الأكل أو الشرب؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر ألم البطن على أنشطتك اليومية؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Fecal incontinence": [
        {
            "question": "في الأيام السبعة الماضية، هل عانيت من فقدان التحكم في البراز؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر فقدان التحكم في البراز على الأكل أو الشرب أو النشاط اليومي؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Shortness of breath": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة ضيق النفس؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر ضيق النفس على الأكل أو الشرب أو النشاط اليومي؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "cough": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة السعال (الجاف/الرطب)؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        },
        {
            "question": "في الأيام السبعة الماضية، هل أثر السعال على الأكل أو الشرب أو النشاط اليومي؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ],
    "Wheezing": [
        {
            "question": "في الأيام السبعة الماضية، ما شدة الصفير أثناء التنفس؟",
            "options": ["Ο لا أبدا", "Ο قليل", "Ο متوسط", "Ο شديد", "Ο شديد جدًا"]
        }
    ]

}

def ask_questions_for_detected_symptoms(detected_symptoms):
    responses = {}
    for symptom in detected_symptoms:
        key = symptom['key']
        responses[key] = {}

        # الحصول على قائمة الأسئلة من SYMPTOM_QUESTIONS
        if key in SYMPTOM_QUESTIONS:
            for question_dict in SYMPTOM_QUESTIONS[key]:
                question_text = question_dict['question']  # استخدم نص السؤال فقط
                options = question_dict.get('options', [])

                # عرض السؤال مع الخيارات
                print(f"\n{question_text}")
                for i, opt in enumerate(options, 1):
                    print(f"{i}. {opt}")

                # استقبال إجابة المستخدم
                answer = input("> ")

                # تخزين الإجابة
                responses[key][question_text] = answer
        else:
            print(f" لا توجد أسئلة معرفة لهذا العرض: {key}")
    return responses

# مثال كامل
user_input = input("اكتب الأعراض التي تشعر بها: ")
detected = detect_symptoms_embedding(user_input)

print("\nتم اكتشاف الأعراض الأقرب:")
for d in detected:
    print(f"{d['text']}")

# الآن نسأل الأسئلة لكل عَرَض
user_responses = ask_questions_for_detected_symptoms(detected)

print("\n تم تسجيل إجاباتك:")
for symptom_key, answers in user_responses.items():
    print(f"\n{symptom_key}:")
    for q, a in answers.items():
        print(f"{q} => {a}")



# --- Imports ---
from flask import Flask, request, render_template_string

from sentence_transformers import SentenceTransformer
import numpy as np



# ----------------------------------------------------
# 🔵 1) Symptom Data
# ----------------------------------------------------
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
    "dry_mouth": [{
        "question": "في الأيام السبعة الماضية، ما شدة جفاف الفم؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
    "headache": [{
        "question": "في الأيام السبعة الماضية، ما شدة الصداع؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
    "nausea": [{
        "question": "في الأيام السبعة الماضية، ما شدة الغثيان؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
    "abdominal_pain": [{
        "question": "في الأيام السبعة الماضية، ما شدة ألم البطن؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
    "hiccups": [{
        "question": "في الأيام السبعة الماضية، ما شدة الزغطة؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
    "heartburn": [{
        "question": "في الأيام السبعة الماضية، ما شدة حرقة الصدر؟",
        "options": ["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    }],
}


# ----------------------------------------------------
# 🔵 2) Load embedding model
# ----------------------------------------------------
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)


# ----------------------------------------------------
# 🔵 3) Symptom detection
# ----------------------------------------------------
def detect_symptoms_embedding(user_text, top_k=5, threshold=0.15):
    """
    تكتشف الأعراض من نص طويل حتى لو المستخدم كتب جملة مركبة.
    """
    # قسم الجملة على علامات الترقيم أو مسافات طويلة
    import re
    parts = re.split(r"[,.!؟؛]", user_text)
    detected = set()  # استخدام set لتجنب التكرار

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

# ----------------------------------------------------
# 🔵 4) Flask Memory
# ----------------------------------------------------
session_data = {
    "pending": [],
    "answers": {},
    "completed": False,
    "chats": []
}


# ----------------------------------------------------
# 🔵 5) HTML (WhatsApp style)
# ----------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🤖 Symptom Checker</title>
<style>
body {
    font-family: Arial, sans-serif;
    background:#e5ddd5;
    display:flex;
    flex-direction:column;
    align-items:center;
    margin:0;
    padding:0;
}
header {
    background:#075e54;
    color:white;
    width:100%;
    text-align:center;
    padding:20px;
}
h1 { margin:0; }
main {
    background:white;
    width:90%;
    max-width:700px;
    margin-top:20px;
    padding:20px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(0,0,0,0.1);
}
.chat-box {
    max-height:60vh;
    overflow-y:auto;
    display:flex;
    flex-direction:column;
    gap:10px;
}
.bubble {
    padding:12px 15px;
    margin:5px 0;
    border-radius:12px;
    max-width:85%;
    word-wrap:break-word;
}
.user {
    background:#dcf8c6;
    align-self:flex-end;
}
.ai {
    background:#f0f0f0;
    align-self:flex-start;
    border-left:4px solid #075e54;
}
form {
    display:flex;
    flex-direction:column;
    gap:10px;
    margin-top:15px;
}
button.option {
    padding:12px;
    border-radius:25px;
    border:none;
    background:#25d366;
    color:white;
    font-size:16px;
    cursor:pointer;
    transition:0.3s;
    width:100%;
    text-align:center;
}
button.option:hover {
    background:#128c7e;
    transform: scale(1.05);
}
textarea {
    flex:1;
    padding:12px;
    border-radius:10px;
    font-size:16px;
    border:1px solid #ccc;
    outline:none;
    transition:0.2s;
}
textarea:focus {
    border-color:#075e54;
    box-shadow:0 0 5px rgba(7,94,84,0.3);
}
.send-btn {
    background:#25d366;
    color:white;
    padding:12px;
    font-size:16px;
    border:none;
    border-radius:25px;
    cursor:pointer;
    transition:0.3s;
}
.send-btn:hover {
    background:#128c7e;
    transform: scale(1.05);
}
footer {
    text-align:center;
    padding:10px;
    color:#555;
    font-size:14px;
    margin-top:20px;
}
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


# ----------------------------------------------------
# 🔵 6) Flask Logic
# ----------------------------------------------------
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        # إعادة تهيئة الجلسة
        session_data["pending"] = []
        session_data["answers"] = {}
        session_data["completed"] = False
        session_data["chats"] = []
        return render_template_string(
            HTML_PAGE,
            chats=session_data["chats"],
            pending=session_data["pending"],
            completed=session_data["completed"],
            symptom_question={k: v[0]["question"] for k, v in SYMPTOM_QUESTIONS.items()},
            severity_options=["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
        )

    # --- POST request ---
    if request.method == "POST":
        # 1️⃣ إذا كان المستخدم يجيب على سؤال
        if "answer" in request.form:
            if session_data["pending"]:
                sym = session_data["pending"].pop(0)
                answer = request.form["answer"]
                session_data["answers"][sym] = answer
                session_data["chats"].append((SYMPTOM_QUESTIONS[sym][0]["question"], answer))
                if not session_data["pending"]:
                    session_data["completed"] = True
            else:
                session_data["chats"].append(("Bot", "لا يوجد سؤال للإجابة عليه. اكتب الأعراض من جديد."))

        # 2️⃣ إذا كان المستخدم يرسل أعراض جديدة
        elif "question" in request.form:
            user_text = request.form["question"].strip()
            if user_text:
                session_data["chats"].append((user_text, ""))
                detected = detect_symptoms_embedding(user_text)
                session_data["pending"] = [d for d in detected if d in SYMPTOM_QUESTIONS]
                if not session_data["pending"]:
                    session_data["chats"].append(("Bot", "لم أتعرف على أي عرض."))
            else:
                session_data["chats"].append(("Bot", "يرجى إدخال الأعراض قبل الإرسال."))

    # --- إعادة عرض الصفحة ---
    return render_template_string(
        HTML_PAGE,
        chats=session_data["chats"],
        pending=session_data["pending"],
        completed=session_data["completed"],
        symptom_question={k: v[0]["question"] for k, v in SYMPTOM_QUESTIONS.items()},
        severity_options=["لا أبدا", "قليل", "متوسط", "شديد", "شديد جدًا"]
    )


