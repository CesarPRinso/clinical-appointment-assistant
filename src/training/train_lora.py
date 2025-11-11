import os, re, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers.trainer_utils import IntervalStrategy
from pathlib import Path



THIS_DIR  = Path(__file__).resolve().parent                  # .../src/training

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DATA_PATH = os.getenv("DATA_PATH", str(THIS_DIR / "data" / "data.sample.json"))
OUT_DIR    = os.getenv("OUT_DIR",   "/gcs/artifacts/adapters")

SYSTEM = (
    "Eres el asistente de una clínica dental.\n"
    "- Responde SOLO en JSON válido {intent, service, date, time, name, dni}.\n"
    "- Si falta info, haz UNA pregunta breve.\n"
    "- Servicios: {limpieza, revisión, blanqueamiento, carillas, ortodoncia_consulta}.\n"
    "- Fechas ISO y hora HH:mm (Europe/Berlin)."
)

def build_prompt(q: str, ctx: str, a: str) -> str:
    return (
        f"### System\n{SYSTEM}\n\n"
        "### Task\nDevuelve una respuesta en el formato indicado.\n"
        f"### Context\n{ctx or '(sin contexto)'}\n"
        f"### Input\n{q}\n"
        "### Output\n```json\n"
        f"{a}\n"
        "```"
    )

def main():
    # --- Modelo base 4-bit + tokenizer ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "32GiB"},
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False


    # --- LoRA ---
    peft_cfg = LoraConfig(
        r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","down_proj","up_proj","lm_head"]
    )
    model = get_peft_model(model, peft_cfg)

    # --- Dataset: split + construir campo `text` ---
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # Espera columnas: question, context, answer
    def to_text(ex):
        return {"text": build_prompt(ex.get("question",""), ex.get("context",""), ex.get("answer",""))}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.10, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]

    MAX_TRAIN = int(os.getenv("MAX_TRAIN", "2500"))
    MAX_EVAL  = int(os.getenv("MAX_EVAL",  "250"))
    train_ds = train_ds.select(range(min(len(train_ds), MAX_TRAIN)))
    eval_ds  = eval_ds.select(range(min(len(eval_ds),  MAX_EVAL)))


    # Collator que enmascara todo antes de ```json
    RESPONSE_TEMPLATE = "```json\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tok,
        mlm=False
    )

    # --- Métricas ---
    TARGET_FIELDS = ["intent","service","date","time","name","dni"]

    def safe_parse(s: str):
        m = re.search(r'\{.*\}', s, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_text = tok.batch_decode(preds, skip_special_tokens=True)
        label_text = tok.batch_decode(labels, skip_special_tokens=True)

        n = len(pred_text)
        ex_match = 0
        field_hits  = {k:0 for k in TARGET_FIELDS}
        field_count = {k:0 for k in TARGET_FIELDS}

        for p_txt, g_txt in zip(pred_text, label_text):
            p = safe_parse(p_txt); g = safe_parse(g_txt)
            if p and g and {k:p.get(k) for k in TARGET_FIELDS} == {k:g.get(k) for k in TARGET_FIELDS}:
                ex_match += 1
            for k in TARGET_FIELDS:
                if k in g:
                    field_count[k] += 1
                    if p.get(k) == g.get(k):
                        field_hits[k] += 1

        metrics = {
            "exact_match": ex_match / n if n else 0.0,
            "intent_acc":  (field_hits["intent"]/field_count["intent"]) if field_count["intent"] else 0.0,
            "service_acc": (field_hits["service"]/field_count["service"]) if field_count["service"] else 0.0,
            "date_acc":    (field_hits["date"]/field_count["date"]) if field_count["date"] else 0.0,
            "time_acc":    (field_hits["time"]/field_count["time"]) if field_count["time"] else 0.0,
        }
        # métrica objetivo para early stopping (puedes cambiar a exact_match o un promedio)
        metrics["eval_accuracy"] = metrics["intent_acc"]
        return metrics

    # --- Args + Early Stopping ---
    # 🔹 Configuración optimizada para GPU Tesla P4 (~7.8GB)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=int(os.getenv("EPOCHS", "1")),   # empieza con 1 para probar
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        bf16=False,
        learning_rate=float(os.getenv("LR", "1e-4")),
        logging_steps=50,
        eval_strategy=IntervalStrategy.NO,         # 🔹 sin evaluación intermedia
        save_strategy=IntervalStrategy.EPOCH,      # 🔹 guarda solo al final
        save_total_limit=2,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",                  # 🔹 optimizador 8-bit
        report_to="none",
    )

    sft_cfg = SFTConfig(
        max_seq_length=256,  # puedes bajar a 192 si hay OOM
        packing=False,  # incompatible con DataCollatorForCompletionOnlyLM
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,  # <-- añade tokenizer
        args=args,
        peft_config=peft_cfg,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",  # <-- CLAVE para evitar el KeyError
        formatting_func=None,  # (nos basamos en 'text', no en formateador)
        sft_config=sft_cfg,  # <-- quita warnings por deprecación
        compute_metrics=None,
    )
    
    trainer.train()

    metrics = trainer.evaluate()
    print("📊 Eval:", metrics)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Adapters guardados en", OUT_DIR)

if __name__ == "__main__":
    main()
