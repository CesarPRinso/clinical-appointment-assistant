from flask import Blueprint, request, jsonify, current_app
from jsonschema import validate, ValidationError
from .schema_loader import load_output_schema
from .vertex_client import predict_vertex

bp = Blueprint("api", __name__)


@bp.get("/health")
def health():
    return {"status": "ok"}


@bp.post("/parse")
def parse():
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify(error="text is required"), 400

    result = predict_vertex(text)  # puede devolver dict(JSON) o {"text": "..."}
    if isinstance(result, dict) and "intent" in result:
        try:
            validate(instance=result, schema=load_output_schema())
        except ValidationError as e:
            return jsonify(error="invalid LLM JSON", details=str(e), raw=result), 422

    return jsonify(result=result)
