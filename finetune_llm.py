import torch
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


parser = ArgumentParser()
parser.add_argument("--input_file", type=str, default="viet_muong_train.csv")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
parser.add_argument("--output_dir", type=str, default="./qwen_viet_muong")
parser.add_argument("--save_path", type=str, default="./qwen_lora")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--seed",type=int, default=42)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

df = pd.read_csv(args.input_file).dropna()
print("Total samples:", len(df))

train_df, val_df = train_test_split(df, test_size=0.05, random_state=args.seed)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

SYSTEM_PROMPT = """
    Convert Vietnamese text into Mường IPA phoneme sequences.
    Output only the phoneme sequence.
    Do not add explanations, labels, or extra text.
"""

def format_sft(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["viet"]},
            {"role": "assistant", "content": example["muong"]}
        ]
    }

train_dataset = train_dataset.map(format_sft)
val_dataset = val_dataset.map(format_sft)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=torch.cuda.is_bf16_supported(including_emulation=False),
    report_to="none",
    max_length=512,
    optim='paged_adamw_8bit',
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

print("Start training...")
trainer.train()

trainer.model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)

print("Saved:", args.save_path)