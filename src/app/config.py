# app/config.py
import os
from dotenv import load_dotenv

def load_settings(app):
    load_dotenv()
    app.config.update(
        PROJECT_ID=os.getenv("PROJECT_ID", ""),
        REGION=os.getenv("REGION", "europe-west4"),
        VERTEX_ENDPOINT_ID=os.getenv("VERTEX_ENDPOINT_ID", ""),
        # timeouts (segundos)
        PREDICT_TIMEOUT=float(os.getenv("PREDICT_TIMEOUT", "15")),
    )
