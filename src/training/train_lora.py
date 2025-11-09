import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json, re

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DATA_PATH = os.getenv("DATA_PATH", "/gcs/data/data.jsonl")  # montado en el container
OUT_DIR = os.getenv("OUT_DIR", "/gcs/artifacts/adapters")

SYSTEM = ("Eres el asistente de una clínica dental.\n"
          "- Responde SOLO en JSON válido {intent, service, date, time, name, dni}.\n"
          "- Si falta info, haz UNA pregunta breve.\n"
          "- Servicios: {limpieza, revisión, blanqueamiento, carillas, ortodoncia_consulta}.\n"
          "- Fechas ISO y hora HH:mm (Europe/Berlin).")



def build_prompt(q, ctx, a):
    return f"""
    ### Task Devuelve una respuesta en el formato indicado.
        ### Context
        {ctx or "(sin contexto)"}
        ### Input
        {q}
        ### Output
        ```json
        {a}
        ```
    """


def main():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft = LoraConfig(
        r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
    )
    model = get_peft_model(model, peft)

    # --- 1) Split ---
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.train_test_split(test_size=0.1, seed=42)  # 90% train, 10% eval
    train_ds, eval_ds = ds["train"], ds["test"]

    response_template_ids = tok.encode("```json", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tok)

    def safe_parse(s: str):
        # extrae primer bloque {...} (a veces el modelo agrega texto)
        m = re.search(r'\{.*\}', s, re.S)
        if not m: return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    TARGET_FIELDS = ["intent", "service", "date", "time", "name", "dni"]

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # decodifica a texto
        pred_text = tok.batch_decode(preds, skip_special_tokens=True)
        label_text = tok.batch_decode(labels, skip_special_tokens=True)

        n = len(pred_text)
        ex_match = 0
        field_hits = {k: 0 for k in TARGET_FIELDS}
        field_counts = {k: 0 for k in TARGET_FIELDS}

        for p_txt, g_txt in zip(pred_text, label_text):
            p = safe_parse(p_txt)
            g = safe_parse(g_txt)
            if p and g and {k: p.get(k) for k in TARGET_FIELDS} == {k: g.get(k) for k in TARGET_FIELDS}:
                ex_match += 1
            for k in TARGET_FIELDS:
                if k in g:
                    field_counts[k] += 1
                    if p.get(k) == g.get(k):
                        field_hits[k] += 1

        metrics = {
            "exact_match": ex_match / n if n else 0.0,
            "intent_acc": (field_hits["intent"] / field_counts["intent"]) if field_counts["intent"] else 0.0,
            "service_acc": (field_hits["service"] / field_counts["service"]) if field_counts["service"] else 0.0,
            "date_acc": (field_hits["date"] / field_counts["date"]) if field_counts["date"] else 0.0,
            "time_acc": (field_hits["time"] / field_counts["time"]) if field_counts["time"] else 0.0,
        }
        metrics["eval_accuracy"] = metrics["intent_acc"]  # o promedio de varios campos
        return metrics


    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=int(os.getenv("EPOCHS", "3")),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=float(os.getenv("LR", "2e-4")),
        bf16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,  # eval cada 200 steps
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        predict_with_generate=True,  # genera texto para poder evaluar JSON
        generation_max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        dataset_text_field="text",
        peft_config=peft,
        max_seq_length=1024,
        packing=False,
        args=args,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)]
    )

    trainer.train()

    response_template_ids = tok.encode("```json", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tok)

    args = TrainingArguments(
        output_dir="/tmp/out",
        num_train_epochs=int(os.getenv("EPOCHS", "3")),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=float(os.getenv("LR", "1e-4")),
        optim="paged_adamw_32bit",
        bf16=True, max_grad_norm=0.3,
        logging_steps=50, save_steps=500,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        report_to="none", evaluation_strategy="no",
    )

    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
                         dataset_text_field="text", data_collator=collator,
                         peft_config=peft, max_seq_length=1024, packing=False, args=args)
    trainer.train()

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Adapters guardados en", OUT_DIR)


if __name__ == "__main__":
    main()
