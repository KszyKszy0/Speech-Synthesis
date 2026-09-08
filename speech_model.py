import torch
import torch.nn as nn
import numpy as np

class SpeechSynthesisModel(torch.nn.Module):
    def __init__(self, vocab_size=1000, max_seq_len=5000, embedding_size=256, mlp_hidden_size=80, dropout_rate = 0.5, 
                 attention_heads = 4, transformer_hidden = 1024, transformer_dropout = 0.1, transformer_layers = 4, 
                 mlp_lsm_hidden_size = 160, conv_size = 256, conv_dropout_rate = 0.5, mel_size = 80):
        
        super().__init__()

        # First mel audio frame
        self.bos_frame = nn.Parameter(torch.randn(1, 1, mel_size) * 0.02)

        self.embed_layer = nn.Embedding(vocab_size, embedding_size)

        self.mlp_1 = nn.Linear(mel_size, mlp_hidden_size)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.mlp_2 = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.mlp_3 = nn.Linear(mlp_hidden_size, embedding_size)
        # Transpose (1,2) because linear and torchaudio swap the order 

        # Encoder
        self.encoder = nn.TransformerEncoderLayer(embedding_size, attention_heads, transformer_hidden, transformer_dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=transformer_layers)

        self.linear_post_transformer = nn.Linear(embedding_size, 2 * mel_size)
        self.linear_stop_prediction = nn.Linear(embedding_size, 1)

        self.mlp_lsm_1 = nn.Linear(mel_size, mlp_lsm_hidden_size)
        self.relu_lsm1 = nn.ReLU()
        self.mlp_lsm_2 = nn.Linear(mlp_lsm_hidden_size, mlp_lsm_hidden_size)
        self.relu_lsm2 = nn.ReLU()
        self.mlp_lsm_3 = nn.Linear(mlp_lsm_hidden_size, mel_size)

        self.conv1d_1 = nn.Conv1d(mel_size, conv_size, 5, padding=2)
        self.bn_conv_1 = nn.BatchNorm1d(conv_size)
        self.relu_conv_1 = nn.ReLU()
        self.dropout_conv_1 = nn.Dropout(conv_dropout_rate)
        self.conv1d_2 = nn.Conv1d(conv_size, conv_size, 5, padding=2)
        self.bn_conv_2 = nn.BatchNorm1d(conv_size)
        self.relu_conv_2 = nn.ReLU()
        self.dropout_conv_2 = nn.Dropout(conv_dropout_rate)
        self.conv1d_3 = nn.Conv1d(conv_size, conv_size, 5, padding=2)
        self.bn_conv_3 = nn.BatchNorm1d(conv_size)
        self.relu_conv_3 = nn.ReLU()
        self.dropout_conv_3 = nn.Dropout(conv_dropout_rate)
        self.conv1d_4 = nn.Conv1d(conv_size, conv_size, 5, padding=2)
        self.bn_conv_4 = nn.BatchNorm1d(conv_size)
        self.relu_conv_4 = nn.ReLU()
        self.dropout_conv_4 = nn.Dropout(conv_dropout_rate)
        self.conv1d_5 = nn.Conv1d(conv_size, mel_size, 5, padding=2)

        # Positional Encoding
        positional_encoding_lookup = torch.zeros(max_seq_len, embedding_size)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-np.log(10_000) / embedding_size))

        positional_encoding_lookup[:, 0::2] = torch.sin(position * div_term)
        positional_encoding_lookup[:, 1::2] = torch.cos(position * div_term)
        positional_encoding_lookup = positional_encoding_lookup.unsqueeze(0)
        self.register_buffer("positional_encoding_lookup", positional_encoding_lookup)

    def forward(self, padded_text, pad_text_mask, padded_mels, pad_mel_mask):
        batch, text_len = padded_text.shape
        batch, mel_len, mel_channels = padded_mels.shape
        model_device = pad_mel_mask.device
        
        # First mel frame adding and masking
        mel_len += 1
        bos_mask = torch.zeros(batch, 1, dtype=torch.bool, device=model_device)
        pad_mel_mask = torch.cat([bos_mask, pad_mel_mask], dim=1)

        batch_bos = self.bos_frame.expand(batch, -1, -1)
        first_frame_mels = torch.cat([batch_bos, padded_mels], dim=1)


        text_preprocess = self.embed_layer(padded_text)

        audio_preprocess = self.mlp_1(first_frame_mels)
        audio_preprocess = self.relu_1(audio_preprocess)
        audio_preprocess = self.dropout_1(audio_preprocess)
        audio_preprocess = self.mlp_2(audio_preprocess)
        audio_preprocess = self.relu_2(audio_preprocess)
        audio_preprocess = self.dropout_2(audio_preprocess)
        audio_preprocess = self.mlp_3(audio_preprocess)

        # Positional Encoding
        text_preprocess = text_preprocess + self.positional_encoding_lookup[:, :text_preprocess.size(1), :]
        audio_preprocess = audio_preprocess + self.positional_encoding_lookup[:, :audio_preprocess.size(1), :]

        transformer_input = torch.cat((text_preprocess, audio_preprocess), dim=1)
        padding_mask_combined = torch.cat((pad_text_mask, pad_mel_mask), dim=1)

        total_len = text_len + mel_len
        casual_mask = torch.triu(torch.ones(total_len, total_len, device=padded_text.device, dtype=torch.bool), diagonal=1)
        casual_mask[:text_len, :text_len] = False

        transformer_output = self.transformer(transformer_input, src_key_padding_mask=padding_mask_combined, mask=casual_mask)

        mu, log_var = torch.chunk(self.linear_post_transformer(transformer_output), chunks=2, dim=-1)
        stop_prediction = self.linear_stop_prediction(transformer_output)

        sigma = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        z_t = dist.rsample()

        lsm_data = self.mlp_lsm_1(z_t)
        lsm_data = self.relu_lsm1(lsm_data)
        lsm_data = self.mlp_lsm_2(lsm_data)
        lsm_data = self.relu_lsm2(lsm_data)
        y_first = self.mlp_lsm_3(lsm_data) + z_t

        transposed_conv_input = torch.transpose(y_first, 1, 2)
        conv_output = self.conv1d_1(transposed_conv_input)
        conv_output = self.bn_conv_1(conv_output)
        conv_output = self.relu_conv_1(conv_output)
        conv_output = self.conv1d_2(conv_output)
        conv_output = self.bn_conv_2(conv_output)
        conv_output = self.relu_conv_2(conv_output)
        conv_output = self.conv1d_3(conv_output)
        conv_output = self.bn_conv_3(conv_output)
        conv_output = self.relu_conv_3(conv_output)
        conv_output = self.conv1d_4(conv_output)
        conv_output = self.bn_conv_4(conv_output)
        conv_output = self.relu_conv_4(conv_output)
        conv_output = self.conv1d_5(conv_output)
        post_net_residual = torch.transpose(conv_output, 1, 2)
        y_second = y_first + post_net_residual

        '''
            OUTPUT SHAPE:
            y_first:  (batch, text_len + T + 1, mel_size)
            y_second: (batch, text_len + T + 1, mel_size)
            
            stop_prediction: (batch, text_len + T + 1, 1)

            mu:      (batch, text_len + T + 1, mel_size)
            log_var: (batch, text_len + T + 1, mel_size)
            sigma:   (batch, text_len + T + 1, mel_size)
        '''
        return y_first, y_second, stop_prediction, mu, log_var, sigma

    @torch.no_grad()
    def infer(self, padded_text, prompt_mels, max_len=2000, stop_threshold=0.5):
        """
        padded_text: (1, text_len)         -- text tokens (target text, optionally prompt text prepended)
        prompt_mels: (1, prompt_len, mel_channels) -- speech prompt mel frames (can be 0-length)
        """
        device = padded_text.device
        batch, text_len = padded_text.shape

        # --- Text embedding (fixed for the whole generation) ---
        text_preprocess = self.embed_layer(padded_text)
        text_preprocess = text_preprocess + self.positional_encoding_lookup[:, :text_len, :]

        # --- Init mel sequence with BOS frame + prompt frames ---
        batch_bos = self.bos_frame.expand(batch, -1, -1)  # (1, 1, mel_channels)
        generated_mels = torch.cat([batch_bos, prompt_mels], dim=1)  # (1, 1+prompt_len, mel_channels)

        prompt_len = prompt_mels.size(1)
        output_y_first = []  # store only the newly generated (non-prompt) frames

        for step in range(max_len):
            mel_len = generated_mels.size(1)

            # --- Audio pre-net (same as training) ---
            audio_preprocess = self.mlp_1(generated_mels)
            audio_preprocess = self.relu_1(audio_preprocess)
            audio_preprocess = self.dropout_1(audio_preprocess)  # eval() should disable this; fine either way
            audio_preprocess = self.mlp_2(audio_preprocess)
            audio_preprocess = self.relu_2(audio_preprocess)
            audio_preprocess = self.dropout_2(audio_preprocess)
            audio_preprocess = self.mlp_3(audio_preprocess)

            audio_preprocess = audio_preprocess + self.positional_encoding_lookup[:, :mel_len, :]

            transformer_input = torch.cat((text_preprocess, audio_preprocess), dim=1)
            total_len = text_len + mel_len

            causal_mask = torch.triu(
                torch.ones(total_len, total_len, device=device, dtype=torch.bool), diagonal=1
            )
            causal_mask[:text_len, :text_len] = False

            # no padding mask needed (batch=1, no padding)
            transformer_output = self.transformer(transformer_input, mask=causal_mask)

            # --- Only need the LAST time step's output to predict the next frame ---
            last_hidden = transformer_output[:, -1:, :]  # (1, 1, hidden)

            mu, log_var = torch.chunk(self.linear_post_transformer(last_hidden), chunks=2, dim=-1)
            stop_logit = self.linear_stop_prediction(last_hidden)

            sigma = torch.exp(0.5 * log_var)
            dist = torch.distributions.Normal(loc=mu, scale=sigma)
            z_t = dist.rsample()  # latent sampling module (Eq. 3) — keep sampling at inference too (Table 4 shows this helps)

            lsm_data = self.mlp_lsm_1(z_t)
            lsm_data = self.relu_lsm1(lsm_data)
            lsm_data = self.mlp_lsm_2(lsm_data)
            lsm_data = self.relu_lsm2(lsm_data)
            next_frame = self.mlp_lsm_3(lsm_data) + z_t  # this is y'_t, the coarse frame

            # --- Check stop condition ---
            stop_prob = torch.sigmoid(stop_logit)
            if stop_prob.item() > stop_threshold and step > 0:
                break

            # --- Feed frame back in, save it ---
            generated_mels = torch.cat([generated_mels, next_frame], dim=1)
            output_y_first.append(next_frame)

        y_first = torch.cat(output_y_first, dim=1)  # (1, T_generated, mel_channels)

        # --- Post-net applied ONCE at the end (paper: post-net processes y' after AR concludes) ---
        transposed = torch.transpose(y_first, 1, 2)
        conv_output = self.conv1d_1(transposed)
        conv_output = self.bn_conv_1(conv_output)
        conv_output = self.relu_conv_1(conv_output)
        conv_output = self.conv1d_2(conv_output)
        conv_output = self.bn_conv_2(conv_output)
        conv_output = self.relu_conv_2(conv_output)
        conv_output = self.conv1d_3(conv_output)
        conv_output = self.bn_conv_3(conv_output)
        conv_output = self.relu_conv_3(conv_output)
        conv_output = self.conv1d_4(conv_output)
        conv_output = self.bn_conv_4(conv_output)
        conv_output = self.relu_conv_4(conv_output)
        conv_output = self.conv1d_5(conv_output)
        post_net_residual = torch.transpose(conv_output, 1, 2)

        y_second = y_first + post_net_residual  # final refined mel-spectrogram

        return y_second