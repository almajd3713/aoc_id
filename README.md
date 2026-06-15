# Deep Models for Arabic Dialect Identification on Benchmarked Data. [[Paper](http://www.aclweb.org/anthology/W18-3930)].

*This is a simple text classification library focused on dialect identification, based on keras. Some Arabic text normalization utilities are included.*

# Current Implemented Models:

1- Word Level CNN based on [[Convultion Neural Network for Text Classificartion](http://www.aclweb.org/anthology/D14-1181)].

2- Word Level C-LSTM based on [[A C-LSTM Neural Network for Text Classification](https://arxiv.org/pdf/1511.08630.pdf)].

3- Recurrent Network and its variants (BiLSTM, LSTM, GRU, BiGRU, Attention-BiLSTM)

4- Models implemented but currently not supported in options (Attention-LSTM,Attention-BiGRU).

5- Not yet tested  (char level CNN). 

# Requirements
```
- keras (2.0 or above)
- gensim
- numpy
- pandas
```
# General Usage:
```
- * Tested with python 3.4 *
- python test_baselines.py --train training_file --Ar='True' --dev Dev_File --test test_file --model_type=model_selection --static=Trainable_embeddings --rand=Random_Embeddings --embedding=External_Embedding_model --model_file=Output_model_file_inJson
- put your training labels in [[link](https://github.com/UBC-NLP/aoc_id/edit/master/conf/label_list)].
```
# Options details #

- train: training file assuming in csv format, text, label
- Ar: if True then Arabic normalization is applied (should be true in case of external embeddings)
- dev: Development file in csv format 
- test: test file in csv format
- model_type: currently support those type of models: (cnn: word level cnn, clstm: word level clstm, lstm: vanilla lstm architecture, blstm: Vanilla bidirectional LSTM, bigru: Vanilla BiGated Recurrent unit, attbilstm: BiLSTM with self attention mechanism)
- static: used in case of external embedding, if True: External Embeddings are not fine tuned during training, if False: External EMbeddings are fine tuned during training). 
- rand: if True, No external embedding is applied, randomly initialized embedding 
- embedding: External embedding model in gensim format
- model_file: Output model file in Json. 
 -EMB_type: Choose whether fastText or CBOW or skipgram

*Note: final model score is dumped into a file with name_of_model_score with both dev and test scores*
# Example Project (Arabic Dialect Identification with Deep Models) #

- This project utilize 6 deep learning models applied on Arabic Online Commentary Dataset. [[Paper](https://www.cs.jhu.edu/~ccb/publications/arabic-dialect-corpus.pdf)]; [[Dataset](https://www.cis.upenn.edu/~ccb/data/AOC-dialectal-annotations.zip)]. 
- Make sure to cite AOC oringial paper if you are going to use it in your work. 
- This work currently accepted to VarDial Worshop 2018 co-located with COLING 2018 under the name (paper soon)
"Deep Models for Arabic Dialect Identification on Benchmarked Data"
- Training data: [[link](https://github.com/UBC-NLP/aoc_id/edit/master/data/train)]
- Dev data: [[link](https://github.com/UBC-NLP/aoc_id/edit/master/data/dev)]
- Test data: [[link](https://github.com/UBC-NLP/aoc_id/edit/master/data/test)]
- An example on how to use it is in:  [[link](https://github.com/UBC-NLP/aoc_id/edit/master/run.sh)]

- *If you are going to follow up on this project please cite this work using the following bibtext:*
```
@inproceedings{Elaraby2018,
  title={Deep Models for Arabic Dialect Identification on Benchmarked Data},
  author={Elaraby, Mohamed and Abdul-Mageed, Muhammad},
  booktitle={Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial5)},
  year={2018}
}
```
# External Embedding Models #

- For Arabic Dialects we release 2 embedding models 
- AOC embedding: [[Download URL](https://drive.google.com/open?id=1QEg9HotnTCI45-PT52g445bp5qYQ4RSm)]
- Twitter Embedding Model: [[Download URL](https://drive.google.com/open?id=1hEuNHn2PA7kIf1IK0FUGUskA77YZJ3vO)]
 - cite the following paper if you are planning to use city level dialect embedding model: 

```
@inproceedings{mageedYouTweet2018,
  title={You Tweet What You Speak: A City-Level Dataset of Arabic Dialects},
  author={Abdul-Mageed, Muhammad and Alhuzali, Hassan and Elaraby, Mohamed},
  booktitle={LREC},
  pages={3653--3659},
  year={2018}
}
```

# Masked Dataset Builder

Use `scripts/build_masked_dataset.py` to prepare an annotation-ready pool from `data/train/MultiTrain.Shuffled.csv`.

```bash
python3 scripts/build_masked_dataset.py
```

Default behavior:
- uses `MultiTrain.Shuffled.csv` as the source pool
- removes unusable rows, exact duplicates, and conservative near-duplicates
- samples a balanced pool across `MSA`, `DIAL_EGY`, `DIAL_GLF`, and `DIAL_LEV`
- writes annotation chunk files of `50` rows each to `artifacts/masked_dataset/chunks/`
- copies the first `2` chunks into `artifacts/masked_dataset/pilot/` for a pilot annotation pass
- creates `clean_candidates.csv`, `rejections.csv`, `chunk_manifest.csv`, and `summary.json`

The chunk files include blank `masked_text` and `mask_spans_json` fields so they can be annotated later and merged safely with stable `example_id` values.

Placeholder rules:
- use typed placeholders such as `[PERSON_1]` and `[LOC_1]`
- numbering restarts in each row
- repeated mentions of the same entity in one row should reuse the same placeholder
- numbering should be contiguous within a mask type in that row, e.g. `[PERSON_1]`, `[PERSON_2]`

After annotation, validate and merge approved chunks with:

```bash
python3 scripts/merge_masked_chunks.py
```

The masking rubric used by the merge validator is documented in `docs/masking_guidelines.md`.

# Annotation Coordinator

A local React + FastAPI coordinator now lives under `annotation_app/`. It is designed for the chunk workflow in `artifacts/masked_dataset/` and keeps accepted annotations in a managed workspace under `app_state/projects/` instead of overwriting source chunks.

Backend:

```bash
pip install -r requirements-annotation-app.txt
python3 -m annotation_app.backend
```

Frontend:

```bash
cd annotation_app/frontend
npm install
npm run dev
```

Key behavior:
- defaults to the `pilot` project from `app_state/config.json`
- generates a prompt for the current chunk
- can optionally include a reference `row_id` column in the prompt CSV to help external models keep row order and coverage straight
- accepts a pasted full-CSV response from an external AI
- previews validation and changed rows before acceptance
- writes accepted chunk copies to `app_state/projects/<project>/accepted/`
- auto-runs merge into `app_state/projects/<project>/merged/`

Important files:
- backend app: `annotation_app/backend/api.py`
- shared workflow services: `annotation_app/backend/workflow.py`
- frontend app: `annotation_app/frontend/src/App.tsx`
- app configuration: `app_state/config.json`
