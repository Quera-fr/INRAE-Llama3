import sys
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM


# Import du modèle depuis hugging face


model_name = "meta-llama/Llama-3.2-1B-Instruct"
file_path = './data.jsonl'
n_epoches = 20


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# ✅ Charger le dataset JSONL
dataset = load_dataset('json', data_files=file_path)


# Prétraitement des données d'entrainement
def preprocess_function(examples):
    # 🔹 Construire le texte d'entrée à partir des messages
    text_inputs = []
    for message_set in examples["messages"]:
        text = ""
        for message in message_set:
            text += f"{message['content']}"
        text_inputs.append(text.strip())

    # 🔹 Tokenisation
    inputs = tokenizer(text_inputs, truncation=True, padding="max_length", max_length=35)

    # 🔹 Copier input_ids pour labels
    inputs["labels"] = inputs["input_ids"].copy()

    # 🔹 Remplacer les tokens de padding par -100 pour ignorer la perte
    padding_token_id = tokenizer.pad_token_id
    inputs["labels"] = [
        [(label if label != padding_token_id else -100) for label in labels] for labels in inputs["labels"]
    ]

    return inputs

# ✅ Appliquer la transformation
dataset = dataset.map(preprocess_function, batched=True)


# Entrainement du modèle
print("Entrainement du modèle :---------------------------------------------------------------")

# ✅ Config LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)


print("---------------------------------------------------------------")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("---------------------------------------------------------------")

training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    per_device_train_batch_size=5,  # ⚠️ Réduire la batch size car CPU est limité
    num_train_epochs=n_epoches,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-3,
    gradient_accumulation_steps=90,  # 🔹 Augmenter pour réduire la charge mémoire
    fp16=False,  # 🚫 Désactiver fp16 (inutile sur CPU)
    bf16=False,
    gradient_checkpointing=False,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    save_total_limit=2,
    weight_decay=0.01,
    report_to="tensorboard",
    torch_compile=False,  # ✅ Optimisation CPU
    no_cuda=True
)


# ✅ Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")


# ✅ Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['train'],
    data_collator=data_collator
)

trainer.train()


print("✅ Entrainement du modèle terminé !")

model.save_pretrained("llama3-finetuned")
tokenizer.save_pretrained("llama3-finetuned")


base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="cpu")
peft_model = PeftModel.from_pretrained(base_model, "llama3-finetuned")


# Fusionner les poids LoRA avec le modèle principal
peft_model = peft_model.merge_and_unload()

# Sauvegarde du modèle fusionné (sans LoRA)
peft_model.save_pretrained("llama3-finetuned-merged")
tokenizer.save_pretrained("llama3-finetuned-merged")

print("✅ Fusion et sauvegarde du modèle terminé !")

import os

os.system("python ./llama.cpp/convert_hf_to_gguf.py ./llama3-finetuned-merged")
os.system("ollama create llama-INRAE -f Modelfile")