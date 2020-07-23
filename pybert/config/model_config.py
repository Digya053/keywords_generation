from pathlib import Path

"""Add path location of pretrained and saved models"""

BASE_DIR = Path('models')

config = {
    'checkpoint_dir': BASE_DIR/ 'saved',

    'bert_vocab_path': BASE_DIR / 'pretrained/bert-base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrained/bert-base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrained/bert-base-uncased',

    'scibert_vocab_path': BASE_DIR / 'pretrained/scibert-base-uncased/bert_vocab.txt',
    'scibert_config_file': BASE_DIR / 'pretrained/scibert-base-uncased/config.json',
    'scibert_model_dir': BASE_DIR / 'pretrained/scibert-base-uncased',
}
