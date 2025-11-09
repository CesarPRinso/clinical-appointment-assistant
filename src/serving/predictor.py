from flask import Flask, request, jsonify
from model_loader import model, tok
import torch, json, re

app = Flask(__name__)

SYSTEM = ("Eres el asistente de una clínica dental. "
          "Responde SOLO JSON {intent, service, date, time, name, dni}. "
          "Si falta info, haz UNA pregunta breve. "
          "Servicios: {limpieza, revisión, blanqueamiento, carillas, ortodoncia_consulta}. "
          "ISO dates y HH:mm.")


def extract_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None


@app.post("/predict")
def predict():
    body = request.get_json()
    instances = body.get("instances", [])
    replies = []
    for inst in instances:
        text = inst.get("text", "")
        prompt = f"### System:\n{SYSTEM}\n\n### User:\n{text}\n\n### Assistant:\n"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.7)
        raw = tok.decode(out[0], skip_special_tokens=True)
        js = extract_json(raw)
        replies.append(js if js else {"text": raw.strip()})
    return jsonify(predictions=replies)


@app.get("/health")
def health():
    return {"status": "ok"}
