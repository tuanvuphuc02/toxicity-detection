from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings
warnings.filterwarnings('ignore')


import re
import os, sys
import json

import numpy as np # linear algebra
import spacy
from scipy.special import softmax


import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

from config import SEED, PARTIAL_TRAIN, TEST_SIZE, NUM_LABELS 
from config import MAX_SEQUENCE_LENGTH, NUM_EPOCH, LEARNING_RATE, BATCH_SIZE
from config import ACCUMULATION_STEPS, INPUT_DIR, WORK_DIR, TOXICITY_COLUMN, DATA_DIR
from config import BERT_MODEL_NAME, FINE_TUNED_MODEL_PATH

from utils import set_seed, convert_lines_onfly, preprocess

from flask import Flask
from flask import request


device = torch.device('cuda')

set_seed(SEED)

## instantiate bert pretrained model and tokenizer
toxic_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=NUM_LABELS)
toxic_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

## load saved model
toxic_model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
for p in toxic_model.parameters():
    p.requires_grad = False

## use gpu
toxic_model.cuda()


def predict_toxicity(sentence: str) -> float:
    """
    predict the toxicity level from a sentence
    """

    toxic_model.eval()

    X = np.array([str(sentence)])
    test_preds = torch.zeros((len(X)))

    Xp = convert_lines_onfly(X, MAX_SEQUENCE_LENGTH, toxic_tokenizer)
    y_pred = toxic_model(torch.from_numpy(Xp).to(device)).logits
    test_preds[0] = test_preds[0] + torch.sigmoid(y_pred[0, 0].cpu())

    return round(float(test_preds[0]), 4)

# Preprocess text (username and link placeholders)


def preprocess_sentiment(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary


task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# label mapping
NEG = 'negative'
NET = 'neutral'
POS = 'positive'
labels = [NEG, NET, POS]

label2id = {k: v for k, v in zip(labels, range(3))}
id2label = {k: v for k, v in zip(range(3), labels)}


def predict_sentiment(sentence: str):
    text = preprocess_sentiment(sentence)
    encoded_input = sentiment_tokenizer(text, return_tensors='pt')
    output = sentiment_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment = {id2label[idx]: round(s, 4) for idx, s in enumerate(scores)}

    return sentiment


nlp = spacy.load("en_core_web_lg")
print("Ready!")


def split_text(sentence):
    doc = nlp(sentence)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

def handle_edge_case(sentence: str, toxic_threshold=0.3) -> float:
    sentences = split_text(sentence)
    cases = []
    for i in range(len(sentences)):
        truncated = " ".join(sentences[:i] + sentences[i+1:])
        toxicity = predict_toxicity(truncated)
        sentiment = predict_sentiment(truncated)
        cases.append({
            "truncated_sentence": truncated,
            "toxicity": str(toxicity),
            "sentiment": str(sentiment)
        })
        if toxicity < toxic_threshold and sentiment[POS] > sentiment[NEG]:
            return toxicity, cases
    return predict_toxicity(sentence), cases


def predict_with_combined_toxicity_sentiment(sentence: str) -> float:
    toxicity = predict_toxicity(sentence)
    sentiment = predict_sentiment(sentence)
    data = {}
    data["analysis"] = {
        "toxicity": str(toxicity),
        "sentiment": str(sentiment)
    }

    score = toxicity

    if toxicity > 0.9:
        if sentiment[NEG] > sentiment[POS]:
            score = toxicity
        else:
            score, edge_case = handle_edge_case(sentence, toxic_threshold=0.1)
            data["analysis"]["edge_case"] = edge_case
    elif toxicity > 0.3:
        if sentiment[NEG] > sentiment[POS]:
            score = toxicity
        else:
            score, edge_case = handle_edge_case(sentence, toxic_threshold=0.5)
            data["analysis"]["edge_case"] = edge_case
    else:  # toxicity < 0.3
        score = toxicity

    data["combined_score"] = str(score)

    return data

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"
    
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == "GET" :
        sentence = request.args.get('sentence', 'hello world!')
        data = predict_with_combined_toxicity_sentiment(sentence)
        print(json.dumps(data))
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        return response

    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
