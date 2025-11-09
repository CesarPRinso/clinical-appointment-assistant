import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
ADAPTERS_GCS = os.getenv("ADAPTERS_GCS")  # ej: gs://tu-bucket/artifacts/adapters
ADAPTERS_LOCAL = "/model/adapters"

# Copia de GCS a disco en init (Vertex monta /model writable)
if ADAPTERS_GCS:
    os.system(f"gcloud storage cp -r {ADAPTERS_GCS} {ADAPTERS_LOCAL}")

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTERS_LOCAL)
model.eval()
