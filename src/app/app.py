from flask import Flask
from .config import load_settings
from .routes import bp


def create_app() -> Flask:
    app = Flask(__name__)
    load_settings(app)  # lee .env y setea app.config
    app.json.sort_keys = False
    app.register_blueprint(bp)
    return app
