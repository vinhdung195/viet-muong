import torch
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from sea_g2p import G2P


parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="strongpear/viet_muong_merged_0_200_denoise_silence_speaker101")
parser.add_argument("--model_name", type=str, default="vinai/PhoWhisper-medium")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

dataset = load_dataset(args.dataset)
splits = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
print(splits)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_name = args.model_name
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
g2p = G2P(lang="vi")

model.eval()

def extract_phonemes(audio):
    wav = audio["array"]

    input_features = processor(
        wav, 
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    phonemes = g2p.convert(transcription)

    phonemes = phonemes.strip()
    if phonemes.endswith("."):
        phonemes = phonemes[:-1].rstrip()

    return phonemes

def create_csv(split_data, split_name):
    pairs = []
    for sample in tqdm(split_data):
        pairs.append({"viet": sample["text"], "muong": extract_phonemes(sample["audio"])})

    df = pd.DataFrame(pairs)
    output_file = f"viet_muong_{split_name}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"Saved {len(df)} samples -> {output_file}")

create_csv(splits['train'], 'train')
create_csv(splits['test'], 'test')