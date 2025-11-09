# clinical-appointment-assistant

Flask API que envía texto a un LLM (Vertex AI Endpoint) y devuelve JSON estructurado para citas dentales.

## Ejecutar local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
# POST http://localhost:8000/parse  {"text":"Quiero limpieza el 25 de noviembre a las 5pm. Soy Carla, DNI 12345678X"}
