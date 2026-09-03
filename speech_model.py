import torch
import torch.nn as nn

class SpeechSynthesisModel(torch.nn.Module):
    def __init__(self, vocab_size=40, embedding_size=256, mlp_hidden_size=80, dropout_rate = 0.5, 
                 attention_heads = 4, transformer_hidden = 1024, transformer_dropout = 0.1, transformer_layers = 4, 
                 mlp_lsm_hidden_size = 160, conv_size = 256, conv_dropout_rate = 0.5, mel_size = 80):
        
        super().__init__()

        self.embed_layer = nn.Embedding(vocab_size, embedding_size)

        self.mlp_1 = nn.Linear(mel_size, mlp_hidden_size)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.mlp_2 = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.mlp_3 = nn.Linear(mlp_hidden_size, embedding_size)
        ### Transpose (1,2) bo linear i torchaudio wyrzucają inną kolejność 

        ### Encoder
        self.encoder = nn.TransformerEncoderLayer(embedding_size, attention_heads, transformer_hidden, transformer_dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=transformer_layers)

        self.linear_post_transformer = nn.Linear(embedding_size, 2 * mel_size)
        self.linear_stop_prediction = nn.Linear(embedding_size, 1)

        self.mlp_lsm_1 = nn.Linear(mel_size, mlp_lsm_hidden_size)
        self.relu_lsm1 = nn.ReLU()
        self.mlp_lsm_2 = nn.Linear(mlp_lsm_hidden_size, mel_size)

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