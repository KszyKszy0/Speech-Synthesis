import torch
import numpy as np

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path
import special_tokens
import speech_model
import HP
import os
from tokenizers import Tokenizer
import soundfile as sf
from speechbrain.inference.vocoders import HIFIGAN

def get_model_number(file):
    return file[-7:-4]

def load_model():
    global model
    files_list = os.listdir(MODEL_SAVE_PATH)
    files_list.sort(key=get_model_number, reverse=True)

    checkpoint_path = f'{MODEL_SAVE_PATH}/{files_list[0]}'
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model {checkpoint_path}")


# DIR SETUP
# Absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Now build all data paths from that, regardless of where you run from
DATA_DIR = SCRIPT_DIR / "data/LJSpeech-1.1"
MODEL_SAVE_PATH = f'{SCRIPT_DIR}/{HP.SAVEPATH}'

# DEVICE 
device = 'cuda' if HP.DEVICE == 'cuda' else 'cpu'

# MODEL INITIALIZATION
model = speech_model.SpeechSynthesisModel(
    embedding_size=HP.EMBEDDING_SIZE,
    mlp_hidden_size=HP.MLP_HIDDEN_SIZE,
    dropout_rate=HP.DROPOUT_RATE,
    attention_heads=HP.ATTENTION_HEADS,
    transformer_hidden=HP.TRANSFORMER_HIDDEN,
    transformer_dropout=HP.TRANSFORMER_DROPOUT,
    transformer_layers=HP.TRANSFORMER_LAYERS,
    mlp_lsm_hidden_size=HP.MLP_LSM_HIDDEN_SIZE,
    conv_size=HP.CONV_SIZE,
    conv_dropout_rate=HP.CONV_DROPOUT_RATE,
    mel_size=HP.MEL_SIZE
).to(device).eval()

load_model()

tokenizer = Tokenizer.from_file(f'{SCRIPT_DIR}/bpe_tokenizer.json')

def text_transform(text):
    ids = tokenizer.encode(text).ids
    ids = ids + [special_tokens.EOS_ID]
    return torch.tensor(ids, dtype=torch.long)

txt = input("")
text_tensor = text_transform(txt).unsqueeze(0).to(device)
mel_channels = 80
empty_prompt = torch.zeros(1, 0, mel_channels).to(device)

y_second = model.infer(text_tensor, empty_prompt)

# HiFi-GAN input is (n_mels, T)
mel = y_second.squeeze(0).transpose(0, 1)  

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="pretrained_models/tts-hifigan-libritts-22050Hz")
hifi_gan.eval()

with torch.no_grad():
    waveforms = hifi_gan.decode_batch(mel)

audio = waveforms.squeeze(1).detach().cpu().numpy()
audio = audio.squeeze()
sf.write(
    "output.wav",
    audio,
    22050,
    format="WAV",
    subtype="PCM_16",
)