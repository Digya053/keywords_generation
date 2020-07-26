# keywords_generation

This repo generates the top ten keywords suggestion for earth science abstract data. BertMultiLabelClassification have been used with two embeddings: 'bert-base-uncased' and 'scibert-base-uncased'.

## Structure of the code
```
├── pybert
│   ├── config
│   │   ├── __init__.py
│   │   └── model_config.py
│   ├── data
│   │   ├── data_tabular.csv
│   │   ├── data.txt
│   │   └── __init__.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── pretrained
│   │   │   ├── bert-base-uncased
│   │   │   │   ├── bert_vocab.txt
│   │   │   │   ├── config.json
│   │   │   │   └── pytorch_model.bin
│   │   │   └── scibert-base-uncased
│   │   │       ├── bert_vocab.txt
│   │   │       ├── config.json
│   │   │       └── pytorch_model.bin
│   │   └── saved
│   ├── notebooks
│   │   ├── data_analysis_and_preprocessing
│   │   │   ├── data_analysis.ipynb
│   │   │   ├── data_creation_for_mixed_keywords.ipynb
│   │   │   ├── data_creation_for_multilabel_classification.ipynb
│   │   │   ├── data_preprocessing.ipynb
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── predict.py
│   ├── src
│   │   ├── data_preprocessing
│   │   │   ├── data_preparation
│   │   │   │   ├── __init__.py
│   │   │   │   └── prepare_csv.py
│   │   │   ├── download_data
│   │   │   │   ├── download_daac_metadata.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── pretrain
│   │   │   ├── __init__.py
│   │   │   ├── io
│   │   │   │   ├── bert_processor.py
│   │   │   │   └── __init__.py
│   │   │   └── model
│   │   │       ├── bert_for_multi_label.py
│   │   │       └── __init__.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── progressbar.py
│   │       └── tools.py
│   └── templates
│       └── keywords_generation.html
├── README.md
└── requirements.txt
```
#### Steps to use the code:
1. Download earth science dataset from Common Metadata Repository (CMR).
```
python pybert/src/data_preprocessing/download_data/download_daac_metadata.py --text_file=data.txt
```
2. Convert data to tabular format.
```
python pybert/src/data_preprocessing/data_preparation/prepare_csv.py --text_file=data.txt
```
The csv file will be saved to pybert/data/data_tabular.csv.

3. Run the notebooks present at 'pybert/notebooks/data_analysis_and_preprocessing' to obtain all the datasets required for multilabel classification.

4. The fine-tuned models for BertMultiLabelClassification using 'bert', 'fast-bert' and 'scibert' pretrained models have been prepared taking reference from [Bert-Multi-Label-Text-Classification](https://github.com/lonePatient/Bert-Multi-Label-Text-Classification) and [fast-bert](https://github.com/kaushaltrivedi/fast-bert). All the saved models and notebooks for creating them are available [here](https://drive.google.com/drive/folders/1lWEodLbNufo8u5k-DMq27ctdO0NoGFY4?usp=sharing).

5. Copy the saved models from 'used-models/saved' to 'pybert/models/saved'.

6. For predicting only one text, run:
```
cd pybert
python predict.py --text="Type text you want to predict" --model="Model you want to use from saved_models" --vocab_path='vocab path you want to use (can be 'bert_vocab_path' or 'scibert_vocab_path') ----max_seq_length = "maximum length of tokens" --do_lower_case=True(lower case) or False(as the words are present) --n="no of keywords to be extracted" --option="whether you want to extract keywords in term position, most depth position or mixed position"
```
Example:
```
cd pybert
python predict.py --text="Atmospheric winds" --model="bert_term_30" --vocab_path='bert_vocab_path' --option="term"
```

7. For running the web:
```
cd pybert
export FLASK_APP=main.py
python -m flask run
```
The web will be available at http://127.0.0.1:5000/.




