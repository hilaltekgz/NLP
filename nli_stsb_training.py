"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import nltk
import os

from scipy.spatial import distance
from laserembeddings import Laser
from sklearn.model_selection import train_test_split
import json
 

 
train_nli = pd.read_json('C:/Users/Msı/Desktop/IBSS/vestel_ariza_ongoru/data/snli_1.0/snli_1.0/snli_1.0_train.jsonl', lines=True)
dev_nli = pd.read_json('C:/Users/Msı/Desktop/IBSS/vestel_ariza_ongoru/data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl', lines=True)
test_nli = pd.read_json('C:/Users/Msı/Desktop/IBSS/vestel_ariza_ongoru/data/snli_1.0/snli_1.0/snli_1.0_test.jsonl', lines=True)

print(train_nli.columns)
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
#sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

DATA_IN_PATH = 'C:\\Users\\Msı\\Desktop\\IBSS\\100-ml-projects\\007\\STSb-TR'

TRAIN_STS_DF = os.path.join(DATA_IN_PATH, 'stsb_tr.tsv')


data  = pd.read_csv(TRAIN_STS_DF, header=0, delimiter = '\t', quoting = 3, keep_default_na=False)
print(data)

data_sts = data.filter(items=['split','score','sentence1','sentence2'])
print('sts',data_sts)

train_set  , dev_df = train_test_split(data_sts, test_size=0.33 )
test  , dev = train_test_split(dev_df, test_size=0.33 )

train_nli = data.filter(items=['gold_label','sentence1','sentence2'])
dev_nli = data.filter(items=['gold_label','sentence1','sentence2'])
test_nli = data.filter(items=['gold_label','sentence1','sentence2'])
nli = pd.concat([train_nli,dev_nli,test_nli])
nli['split'] = 'train'
train_df = pd.concat([train_set,nli])

print('traindf:',train_df)
# Read the dataset
#model_name = 'dbmdz/bert-base-turkish-cased'
#model_name = 'bert-base-multilingual-cased'
#model_name = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
#model_name = 'sentence-transformers/LaBSE'
#model_name = 'universal-sentence-encoder-multilingual'
#model_name = "DeepPavlov/bert-base-multilingual-cased-sentence"
#model_name = 'dbmdz/electra-small-turkish-mc4-cased-discriminator'
#model_name = "DeepPavlov/bert-base-multilingual-cased-sentence"
#model_name = "xlm-roberta-base"
model_name = 'sentence-transformers/stsb-xlm-r-multilingual'
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_continue_training_nli-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#laser = Laser()
#model = SentenceTransformer('dbmdz/bert-base-turkish-cased')
# model = SentenceTransformer('sentence-transformers/LaBSE')
# Load a pre-trained sentence transformer model
#model = SentenceTransformer("bert-base-multilingual-cased")
model = SentenceTransformer(model_name)
# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
for row in dev_df.itertuples(index=False):
        score = float(row[1]) / 5.0
        inp_example = InputExample(texts=[row[2], row[3]], label=score)
        if row[0] == 'dev':
            dev_samples.append(inp_example)
        elif row[0] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
