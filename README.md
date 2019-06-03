
# CommonsenseQA

A question answering dataset for commonsense reasoning.

Check out the [website][commonsense-qa-website]!


## Downloading the Data

You can download the data from the [website][commonsense-qa-website],
which also has an evaluation script. The
[leaderboard][commonsense-qa-leaderboard] is for the `random` split of
the data.

## Running ESIM
Our implementation is based on [this code](https://github.com/rowanz/swagaf/tree/master/swag_baselines/esim). To run it, follow these steps:

1. Install ESIM dependencies:
    ```
    cd esim
    pip install -r requirements.txt
    cd ..
    ```
2. Place the dataset in ```data/``` folder.
3. Set PYTHONPATH to the `commonsenseqa` directory:
```export PYTHONPATH=$(pwd)```
4. Run the model either with pre-trained GloVe embeddings:
   ```
   python -m allennlp.run train esim/train-glove-csqa.json -s tmp --include-package esim
    ```
5. Alternatively, run the model with ELMo pretrained contextual embeddings:
    ```
    python -m allennlp.run train esim/train-elmo-csqa.json -s tmp --include-package esim
    ```

## Running BERT

To run BERT on CommonsenseQA, first install the BERT dependencies:

    cd bert/
    pip install -r requirements.txt

Then, [obtain the CommonsenseQA data](#downloading-the-data), and
download the [pretrained BERT weights][downloading-bert-weights]. For
the paper, we used [`BERT Large, Uncased`][bert-large-weights]. To train
BERT Large, you'll most likely need to [use a TPU][tpu-info]; BERT base
can be trained on a standard GPU.

To run training:

**GPU**

    python run_commonsense_qa.py
      --split=$SPLIT \
      --do_train=true \
      --do_eval=true \
      --data_dir=$DATA_DIR \
      --vocab_file=$BERT_DIR/vocab.txt \
      --bert_config_file=$BERT_DIR/bert_config.json \
      --init_checkpoint=$BERT_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$OUTPUT_DIR

**TPU**

    python run_commonsense_qa.py
      --split=$SPLIT \
      --use_tpu=true \
      --tpu_name=$TPU_NAME \
      --do_train=true \
      --do_eval=true \
      --data_dir=$DATA_DIR \
      --vocab_file=$BERT_DIR/vocab.txt \
      --bert_config_file=$BERT_DIR/bert_config.json \
      --init_checkpoint=$BERT_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$OUTPUT_DIR

For TPUs, all directories must be in Google Storage. The environment
variables have the following meanings:

  - `$SPLIT` should either be `rand` or `qtoken`, depending on the split
    you'd like to run.
  - `$DATA_DIR` is a location for the CommonsenseQA data.
  - `$BERT_DIR` is a location for the pre-trained BERT files.
  - `$TPU_NAME` is the name of the TPU.
  - `$OUTPUT_DIR` is the directory to write output to.

To predict on the test set, run:

**GPU (only)**

    python run_commonsense_qa.py \
      --split=$SPLIT \
      --do_predict=true \
      --data_dir=$DATA_DIR \
      --vocab_file=$BERT_DIR/vocab.txt \
      --bert_config_file=$BERT_DIR/bert_config.json \
      --init_checkpoint=$TRAINED_CHECKPOINT \
      --max_seq_length=128 \
      --output_dir=$OUTPUT_DIR

Prediction must be run on a GPU (including for BERT Large). All
environment variables have the same meanings, and the new variable
`$TRAINED_CHECKPOINT` is simply the prefix for your trained checkpoint
files from fine-tuning BERT. It should look something like
`$OUTPUT_DIR/model.ckpt-1830`.


[bert-large-weights]: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
[commonsense-qa-leaderboard]: https://www.tau-nlp.org/csqa-leaderboard
[commonsense-qa-website]: https://www.tau-nlp.org/commonsenseqa
[downloading-bert-weights]: https://github.com/google-research/bert#pre-trained-models
[tpu-info]: https://cloud.google.com/tpu/
