from flask import Flask
from .config import load_settings
from .routes import bp as api_bp


def create_app() -> Flask:
    app = Flask(__name__)
    load_settings(app)
    app.register_blueprint(api_bp)
    return app


'''
    Objetivo de este paquete: recibir requests, llamar al modelo (Vertex) y 
    validar/normalizar respuestas. Nada de entrenar ni cargar modelos aquí.
'''
