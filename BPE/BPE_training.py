from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path
import special_tokens

SCRIPT_DIR = Path(__file__).resolve().parent.parent

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=[special_tokens.PAD_TOKEN, special_tokens.UNK_TOKEN, special_tokens.EOS_TOKEN]
)

tokenizer.train(files=[f'{SCRIPT_DIR}/tokenizer_data.txt'], trainer=trainer)
tokenizer.save(f'{SCRIPT_DIR}/bpe_tokenizer.json')