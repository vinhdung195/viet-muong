from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig, BaseAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--save_step", type=int, default=500)
parser.add_argument("--print_step", type=int, default=85)
parser.add_argument("--resume", type=str, default='')

args = parser.parse_args()

dataset_config = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="viet_muong",
    path="./dataset",
    meta_file_train="metadata_train.csv",
)

characters = CharactersConfig(
    characters="-124567abcdefhijklmnopstuvwxyzæðăđŋɐɑɔɗəɚɛɜɡɣɪɲɹɾʃʊʌʐʒˈˌː̪ᵻấ",
    punctuations=".,!? ",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)


config = VitsConfig(
    output_path="./vits_output",
    num_loader_workers=2,
    mixed_precision=True, 
    batch_size=args.batch_size,
    epochs=args.epochs,
    save_step=args.save_step,
    print_step=args.print_step,
    eval_batch_size=32,
    run_eval=True,
    datasets=[dataset_config],
    use_phonemes=False,
    characters=characters,
    test_sentences=[],
    audio=BaseAudioConfig(sample_rate=16000, mel_fmax=8000)
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.02,
)

model = Vits(config, ap, tokenizer)

trainer = Trainer(
    TrainerArgs(continue_path=args.resume),
    config,
    output_path=config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    gpu=0,
)

trainer.config.epochs = args.epochs

trainer.fit()