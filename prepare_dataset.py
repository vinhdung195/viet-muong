import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset


parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="strongpear/viet_muong_merged_0_200_denoise_silence_speaker101")
parser.add_argument("--input_file", type=str, default="viet_muong_train.csv")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

dataset = load_dataset(args.dataset)
splits = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)

csv = pd.read_csv(args.input_file).dropna().reset_index(drop=True)
train_dataset = splits["train"]
test_dataset = splits["test"]

wav_folder = "dataset/wavs"
os.makedirs(wav_folder, exist_ok=True)

rows = []

for idx in tqdm(range(len(train_dataset))):
    audio = train_dataset[idx]["audio"]["array"]
    sample_rate = 16000
    muong_text = csv.loc[idx, "muong"]

    wav_name = f"{idx:06d}.wav"
    wav_path = os.path.join(wav_folder, wav_name)
    sf.write(wav_path, audio, sample_rate)
    
    rows.append({"audio_file": f"wavs/{wav_name}", "text": muong_text})

df = pd.DataFrame(rows)
path = "dataset/metadata_train.csv"
df.to_csv(path, sep="|", header=True, index=False, encoding="utf-8")
print(f"Saved metadata: {path}")


gt_folder = "eval/gt"
os.makedirs(gt_folder, exist_ok=True)

for idx in tqdm(range(len(test_dataset))):
    audio = test_dataset[idx]["audio"]["array"]
    sample_rate = 16000

    wav_name = f"{idx:06d}.wav"
    wav_path = os.path.join(gt_folder, wav_name)
    sf.write(wav_path, audio, sample_rate)