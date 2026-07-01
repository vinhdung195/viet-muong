import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from argparse import ArgumentParser
from peft import PeftModel
from TTS.api import TTS


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--lora_path", type=str, default="./qwen_lora")
parser.add_argument("--input_file", type=str, default="viet_muong_test.csv")
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="./eval/infer")

args = parser.parse_args()

def phoneme_to_speech(phoneme, model_path, output_file):
    tts = TTS(
        model_path=f"{model_path}/best_model.pth",
        config_path=f"{model_path}/config.json",
    )
    tts.tts_to_file(text=phoneme, file_path=output_file)

def text_to_phoneme(text, model, tokenizer):
    SYSTEM_PROMPT = """
        You convert Vietnamese text into Mường IPA phoneme sequences.
        Output only the phoneme sequence.
        Do not add explanations, labels, or extra text.
    """
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("assistant")[-1].strip()

def text_to_speech(text, model, tokenizer, model_path, output_file):
    phoneme = text_to_phoneme(text, model, tokenizer)
    phoneme_to_speech(phoneme, model_path, output_file)


tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
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

model = PeftModel.from_pretrained(model, args.lora_path)
model.eval()

os.makedirs(args.output_path, exist_ok=True)

df = pd.read_csv(args.input_file)
for idx, text in enumerate(df["viet"]):
    wav_name = f"{idx:06d}.wav"
    output_file = os.path.join(args.output_path, wav_name)
    text_to_speech(text, model, tokenizer, args.model_path, output_file)