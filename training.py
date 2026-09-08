import torch
import numpy as np
import warnings

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

import torchaudio
from pathlib import Path
import speech_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import special_tokens
import speech_model
import HP
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math

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

    # (batch_size, max_text_length), (batch_size, max_text_length), (batch_size, max_mel_length, n_mels), (batch_size, max_mel_length), (batch_size)
    return padded_text, pad_text_mask, padded_mels, pad_mel_mask, mel_lengths_tensor

def get_model_number(file):
    return file[-7:-4]

def load_model():
    global epoch_start
    global model
    global global_step
    global scheduler

    files_list = os.listdir(MODEL_SAVE_PATH)
    if len(files_list) == 0:
        warnings.warn("No saved models found, starting training proccess from scratch.", UserWarning)
        return
        
    files_list.sort(key=get_model_number, reverse=True)

    checkpoint_path = f'{MODEL_SAVE_PATH}/{files_list[0]}'
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Loaded model {checkpoint_path}")


# DIR SETUP
# Absolute path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Now build all data paths from that, regardless of where you run from
DATA_DIR = SCRIPT_DIR / "data/LJSpeech-1.1"
MODEL_SAVE_PATH = f'{SCRIPT_DIR}/{HP.SAVEPATH}'

# DEVICE 
device = 'cuda' if HP.DEVICE == 'cuda' else 'cpu'

# DATASET / DATALOADER SETUP
dataset = speech_dataset.SpeechDataset(f'{DATA_DIR}', f'{SCRIPT_DIR}/bpe_tokenizer.json')
dataloader = DataLoader(dataset, HP.BATCH_SIZE, True, num_workers=HP.NUM_WORKERS, collate_fn=collate_fn)

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
).to(device)

optimizer = torch.optim.Adam(model.parameters(), HP.LR)

def lr_lambda(current_step):
    if current_step < HP.WARMUP_STEPS:
        return current_step / max(1, HP.WARMUP_STEPS)
    # linear decay after warmup
    progress = (current_step - HP.WARMUP_STEPS) / max(1, HP.TOTAL_STEPS - HP.WARMUP_STEPS)
    return max(0.0, 1.0 - progress)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

epoch_start = 1
global_step = 0

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if HP.LOAD:
    load_model()

# LOGGING
writer = SummaryWriter(f"runs/{HP.MODEL_NAME}")

# LOSSES
def masked_mean(loss_tensor, valid_mask):
    """
    loss_tensor: (batch, T, mel_size) - per-element loss, unreduced
    valid_mask:  (batch, T) - True where valid (not padding)
    """
    mask = valid_mask.unsqueeze(-1).float()  # (batch, T, 1), broadcasts over mel_size
    masked_loss = loss_tensor * mask
    return masked_loss.sum() / (mask.sum() * loss_tensor.shape[-1])

def regression_loss(y_first, y_second, target, valid_mask):
    l1_first  = torch.abs(target - y_first)
    l2_first  = (target - y_first) ** 2
    l1_second = torch.abs(target - y_second)
    l2_second = (target - y_second) ** 2

    loss = (masked_mean(l1_first, valid_mask)
          + masked_mean(l2_first, valid_mask)
          + masked_mean(l1_second, valid_mask)
          + masked_mean(l2_second, valid_mask))
    return loss

def kl_divergence_loss(mu, sigma, log_var, target, valid_mask):
    d = mu.shape[-1]  # mel_size

    sigma_sq_sum   = (sigma ** 2).sum(dim=-1)                # (batch, T)
    mu_diff_sq_sum = ((mu - target) ** 2).sum(dim=-1)         # (batch, T)
    log_var_sum    = log_var.sum(dim=-1)                      # (batch, T)

    per_timestep_kl = 0.5 * (sigma_sq_sum + mu_diff_sq_sum - d - log_var_sum)  # (batch, T)

    mask = valid_mask.float()
    return (per_timestep_kl * mask).sum() / mask.sum()

def flux_loss(mu, target, valid_mask):
    mu_shifted     = mu[:, 1:, :]        # positions t = 1..T-1
    target_shifted = target[:, :-1, :]   # positions t-1 = 0..T-2
    mask_shifted   = valid_mask[:, 1:]   # align mask to mu's positions

    diff = torch.abs(mu_shifted - target_shifted)
    return -masked_mean(diff, mask_shifted)

def stop_loss(stop_logits, stop_target, valid_mask, pos_weight=100.0):
    weight = torch.tensor(pos_weight, device=stop_logits.device)
    loss = F.binary_cross_entropy_with_logits(
        stop_logits, stop_target, pos_weight=weight, reduction='none'
    )  # (batch, T+1, 1)
    mask = valid_mask.unsqueeze(-1).float()
    return (loss * mask).sum() / mask.sum()

# Training
for epoch in range(epoch_start, HP.EPOCHS):
    epoch_losses = {"Total": 0.0, "Regression": 0.0, "KL": 0.0, "Flux": 0.0, "Stop": 0.0}

    for i, (padded_text, pad_text_mask, padded_mels, pad_mel_mask, mel_lengths_tensor) in enumerate(dataloader):

        # Moving inputs to devices 
        padded_text = padded_text.to(device)
        pad_text_mask = pad_text_mask.to(device)
        padded_mels = padded_mels.to(device)
        pad_mel_mask = pad_mel_mask.to(device)
        true_lengths = mel_lengths_tensor.to(device).squeeze(1)

        input_seq = padded_mels
        target_seq = padded_mels

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            (y_first, y_second, stop_prediction, mu, log_var, sigma) = model(padded_text, pad_text_mask, input_seq, pad_mel_mask)

            # There is one additional frame at the end - the frame that was generated alongside stop, we need to omit it.
            text_len = padded_text.shape[1]

            y_first  = y_first[:, text_len:-1, :]
            y_second = y_second[:, text_len:-1, :]
            mu       = mu[:, text_len:-1, :]
            log_var  = log_var[:, text_len:-1, :]
            sigma    = sigma[:, text_len:-1, :]

            stop_prediction = stop_prediction[:, text_len:, :]
            batch_size, stop_len, _ = stop_prediction.shape
            stop_target = torch.zeros(batch_size, stop_len, device=device)
            stop_target[torch.arange(batch_size, device=device), true_lengths] = 1.0
            stop_target = stop_target.unsqueeze(-1)  # (batch, T+1, 1)

            valid_mask = ~pad_mel_mask

            stop_len = stop_prediction.shape[1]
            positions = torch.arange(stop_len, device=device).unsqueeze(0)
            stop_valid_mask = positions <= true_lengths.unsqueeze(1)

            reg_loss  = regression_loss(y_first, y_second, target_seq, valid_mask)
            kl_loss   = kl_divergence_loss(mu, sigma, log_var, target_seq, valid_mask)
            flx_loss  = flux_loss(mu, target_seq, valid_mask)
            stp_loss = stop_loss(stop_prediction, stop_target, stop_valid_mask)

            lambda_kl   = 0.1   # per paper A.3, after warmup
            beta_flux   = 0.5   # per paper A.3
            gamma_stop  = 1.0   # per paper A.3

            total_loss = reg_loss + lambda_kl * kl_loss + beta_flux * flx_loss + gamma_stop * stp_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        batch_losses = {"Total": total_loss, "Regression": reg_loss, "KL": kl_loss, "Flux": flx_loss, "Stop": stp_loss}
        for name, value in batch_losses.items():
            epoch_losses[name] += value.item()

        print(f'Epoch: {epoch}, Batch: {i}/{len(dataloader)}, Step: {global_step}, Loss: {total_loss}')


    for name, value in epoch_losses.items():
        writer.add_scalar(f"Loss/{name}", value / len(dataloader), epoch)

    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "epoch_loss": epoch_losses["Total"] / len(dataloader),
        "optimizer": optimizer.state_dict(),
        'global_step': global_step,
        'scheduler': scheduler.state_dict(),
    }, f'{MODEL_SAVE_PATH}/model_{epoch:03d}.pth')