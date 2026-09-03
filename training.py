import torch
import torchaudio
from pathlib import Path
import speech_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import special_tokens
import speech_model
import HP

def collate_fn(batch):
    mel_list, text_list = list(zip(*batch))
    # (channels, n_mels, time), (seq_length)
    mel_lengths = [mel.shape[2] for mel in mel_list]
    text_lengths = [text.shape[0] for text in text_list]
    # (batch_size) where each value represents len

    max_text_length = max(text_lengths)
    max_mel_length = max(mel_lengths)

    padded_text = pad_sequence(text_list, batch_first=True, padding_value=special_tokens.PAD_ID)
    # Tensor is created by the function to result in (batch_size, max_text_length)

    text_lengths_tensor = torch.tensor(text_lengths)
    text_lengths_tensor = text_lengths_tensor.unsqueeze(1)
    # Match shape so that text_lengths_tensor is (batch_size, 1)
    text_positions_base = torch.arange(max_text_length)
    pad_text_mask = text_positions_base >= text_lengths_tensor
    # (max_text_length), (batch_size, 1) => (batch_size, max_text_length)

    mel_lengths_tensor = torch.tensor(mel_lengths)
    mel_lengths_tensor = mel_lengths_tensor.unsqueeze(1)
    # Match shape so that mel_lengths_tensor is (batch_size, 1)
    mel_positions_base = torch.arange(max_mel_length)
    pad_mel_mask = mel_positions_base >= mel_lengths_tensor
    # (max_mel_length), (batch_size, 1) => (batch_size, max_mel_length)

    mel_list_reshaped = [mel.squeeze(0).transpose(0, 1) for mel in mel_list]
    # Padding enforces (batch, time) [with batch first], we omit first dim as it is just channel which is mono (1, n_mels, time)
    # After this it becomes (time, n_mels)
    # List here is the size of a batch
    padded_mels = pad_sequence(mel_list_reshaped, batch_first=True, padding_value=special_tokens.MEL_PAD_VALUE)
    # This creates tensor by itself so it becomes (batch_size, time, n_mels)

    # (batch_size, max_text_length), (batch_size, max_mel_length)
    return padded_text, pad_text_mask, padded_mels, pad_mel_mask


# DIR SETUP
# Absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Now build all data paths from that, regardless of where you run from
DATA_DIR = SCRIPT_DIR / "data/LJSpeech-1.1"

# DEVICE 
device = 'cuda' if HP.DEVICE == 'cuda' else 'cpu'

# DATASET / DATALOADER SETUP
dataset = speech_dataset.SpeechDataset(f'{DATA_DIR}', f'{SCRIPT_DIR}/bpe_tokenizer.json')
dataloader = DataLoader(dataset, HP.BATCH_SIZE, True, num_workers=HP.NUM_WORKERS, collate_fn=collate_fn)

# MODEL INITIALIZATION
model = speech_model.SpeechSynthesisModel()

if HP.LOAD:
    pass

model = model.to(HP.DEVICE)

optimizer = torch.optim.Adam(model.parameters(), HP.LR)

for epoch in range(HP.EPOCHS):
    for i, (padded_text, pad_text_mask, padded_mels, pad_mel_mask) in enumerate(dataloader):
        print(i, len(dataloader))

        # Moving inputs to devices 
        padded_text = padded_text.to(device)
        pad_text_mask = pad_text_mask.to(device)
        padded_mels = padded_mels.to(device)
        pad_mel_mask = pad_mel_mask.to(device)