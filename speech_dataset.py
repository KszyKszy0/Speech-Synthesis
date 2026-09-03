from torch.utils.data import Dataset
from tokenizers import Tokenizer
import pandas as pd
import soundfile as sf
import torchaudio
import torch
import csv
import special_tokens

class SpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, samplerate = 22050, n_fft = 1024, win_length=1024, hop_length=256, mel_number = 80):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id(special_tokens.EOS_TOKEN)
        assert self.eos_id is not None

        self.data = pd.read_csv(f'{self.data_path}/metadata.csv', sep="|", header=None, names=["filename", "text", "text_normalized"], quoting=csv.QUOTE_NONE)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(samplerate, n_fft, win_length, hop_length, n_mels=mel_number)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, text, text_normalized = self.data.iloc[index]
        filename = f'{self.data_path}/wavs/{filename}.wav'

        audio_spectogram = self.prepare_mel_spectogram(filename)

        text_tensor = self.text_transform(text_normalized)

        # (channels, n_mels, time), (seq_length)
        return (audio_spectogram, text_tensor)

    def prepare_mel_spectogram(self, filename):
        audio, samplerate = sf.read(filename, always_2d=True, dtype='float32')
        # Audio shape is (frames, channels) so we need to transpose to (channels, frames) for mel transform
        audio = torch.from_numpy(audio).transpose(0, 1)

        # This requires (**channels**, frames)
        audio_mel_spectogram = self.mel_transform(audio)

        # This returns (channels, n_mels, time)
        return audio_mel_spectogram

    def text_transform(self, text):
        ids = self.tokenizer.encode(text).ids
        ids = ids + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)