from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from fastapi.responses import FileResponse
import uuid
from fastapi import Body
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

history = []

class InputData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([[ 
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi,
        data.diabetes_pedigree, data.age
    ]])

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    risk = round(prob*100,2)

    if risk > 70:
        level = "High"
    elif risk > 30:
        level = "Medium"
    else:
        level = "Low"

    result = {
        "prediction": "Diabetic" if pred else "Non-Diabetic",
        "risk_score": risk,
        "risk_level": level,
        "age": data.age
    }

    history.append(result)

    return result

@app.get("/history")
def get_history():
    return history

@app.post("/generate-report")
def generate_report(data: InputData):
    filename = f"{uuid.uuid4().hex}.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Diabetes Report", styles["Title"]))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Age: {data.age}", styles["Normal"]))
    content.append(Paragraph(f"Glucose: {data.glucose}", styles["Normal"]))
    content.append(Paragraph(f"BMI: {data.bmi}", styles["Normal"]))

    doc.build(content)

    return FileResponse(filename, media_type="application/pdf", filename="report.pdf")

# @app.post("/chat")
# async def chat(data: dict):
#     message = data.get("message")
#     context = data.get("context")

#     if not context:
#         prompt = f"""
# You are a helpful diabetes health assistant.

# User Question:
# {message}

# Instructions:
# - Explain simply
# - Give general health advice
# - No diagnosis
# - Keep it short
# """
#     else:
#         result = context.get("result", {})
#         health = context.get("data", {})

#         prompt = f"""
# User Health Data:
# Age: {health.get("age")}
# Glucose: {health.get("glucose")}
# BMI: {health.get("bmi")}

# Risk Level: {result.get("riskLevel")}

# User Question:
# {message}

# Give personalized answer.
# """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#     )

#     return {"reply": response.choices[0].message.content}

@app.post("/chat")
async def chat(data: dict):
    message = data.get("message")
    context = data.get("context", {})

    latest = context.get("latestResult")
    history = context.get("history", [])

    # 🧠 SYSTEM PROMPT (NO HARDCODE LOGIC)
#     system_prompt = """
# You are an intelligent diabetes assistant.

# You have access to:
# 1. User latest prediction
# 2. User past history

# Your job:
# - Answer user questions naturally max 3 sentences
# - Use available data when relevant
# - If user asks about history → analyze trends
# - If user asks about dashboard → summarize data
# - If no data → give general advice

# Rules:
# - Do NOT hallucinate
# - Do NOT assume missing data
# - Keep answers short and helpful
# """
    system_prompt = """
You are a friendly and intelligent diabetes assistant.

Style:
- Conversational, natural, and supportive
- Avoid repeating the same information
- Do NOT sound robotic or like a form
- Keep answers short (3–5 lines max)

Behavior:
- If user says "ok", "hmm", or short replies → guide them with helpful options
- Do NOT say "you didn’t ask a question"
- Do NOT repeat risk level unless necessary
- Give practical, simple advice

Capabilities:
- Explain diabetes simply
- Give lifestyle tips
- Analyze user data if available
- Suggest next actions

Never:
- Be rude or overly formal
- Repeat the same sentence again
- Claim access you don't have
"""

    # 📊 FORMAT CONTEXT (VERY IMPORTANT)
    context_text = f"""
LATEST RESULT:
{latest}

HISTORY (last records):
{history[-5:]}   # only last 5 for performance
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{context_text}\n\nUser Question:\n{message}"
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5,
    )

    return {"reply": response.choices[0].message.content}