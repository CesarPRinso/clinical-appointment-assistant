from flask import current_app
from google.cloud import aiplatform
from pathlib import Path
import json


def predict_vertex(user_text: str) -> dict:
    endpoint_id = current_app.config.get("VERTEX_ENDPOINT_ID")
    project = current_app.config.get("PROJECT_ID")
    region = current_app.config.get("REGION")
    timeout = current_app.config.get("PREDICT_TIMEOUT", 15)

    # Modo local sin endpoint: útil para dev
    if not (endpoint_id and project):
        return {"text": f"[LOCAL] {user_text}"}

    aiplatform.init(project=project, location=region)
    endpoint = aiplatform.Endpoint(endpoint_id)
    instances = [{"text": user_text, "system": SYSTEM_PROMPT}]
    try:
        resp = endpoint.predict(instances=instances, timeout=timeout)
        return resp.predictions[0] if resp.predictions else {"text": ""}
    except Exception as e:
        # Falla de red o del contenedor: devuelve texto para no romper UX
        return {"text": f"(temporalmente no disponible) {str(e)}"}

def _load_domain():
    root = Path(__file__).parent.parent
    intents = json.loads((root / "domain" / "intents.json").read_text())["intents"]
    services = json.loads((root / "domain" / "services.json").read_text())["services"]
    return intents, services


_INTENTS, _SERVICES = _load_domain()


def _build_system_prompt():
    intents_list = ", ".join(i["code"] for i in _INTENTS)
    services_list = ", ".join(s["code"] for s in _SERVICES)
    return (
        "Eres el asistente de una clínica dental.\n"
        f"Intents permitidos: {intents_list}.\n"
        f"Servicios permitidos: {services_list}.\n"
        "Responde SOLO JSON {intent, service, date, time, name, dni}.\n"
        "Fechas ISO (YYYY-MM-DD) y hora HH:mm (Europe/Berlin). "
        "Si falta info, haz UNA pregunta breve."
    )


SYSTEM_PROMPT = _build_system_prompt()
