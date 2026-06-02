"""
Script — Fine-tune DistilBERT for Emotion Classification (Module 2).
Run: python scripts/train_emotion_model.py
Saves: models/emotion_distilbert/
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

SAVE_PATH   = ROOT / "models" / "emotion_distilbert"
MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 6
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

import torch
def _use_mps() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")


def main():
    print("Loading dair-ai/emotion …")
    ds = load_dataset("dair-ai/emotion")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    ds = ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    args = TrainingArguments(
        output_dir=str(SAVE_PATH),
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=False,               # fp16 not stable on MPS
        use_mps_device=_use_mps(), # Apple Silicon acceleration
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training …")
    trainer.train()

    results = trainer.evaluate(ds["test"])
    print(f"\nTest accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1:       {results['eval_f1']:.4f}")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SAVE_PATH))
    tokenizer.save_pretrained(str(SAVE_PATH))
    print(f"\nModel saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
