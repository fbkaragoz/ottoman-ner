import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
from datasets import load_metric

nltk.download('punkt')


# Load labeled data
def load_ner_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, label = line.strip().split()
                sentence.append((word, label))
        if sentence:
            sentences.append(sentence)
    return sentences


train_data = load_ner_data('../dsai/labeled_training_data/HisTR-main/train.txt')


# Tokenize raw text
def tokenize_text(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences


raw_text = "..."  # Load the raw text data
tokenized_raw_text = tokenize_text(raw_text)

tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=10)


def encode_data(sentences, tokenizer):
    texts = [[word for word, label in sentence] for sentence in sentences]
    labels = [[label for word, label in sentence] for sentence in sentences]
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=False, padding=True, truncation=True)

    labels = [[label_map[label] for label in doc] for doc in labels]
    max_len = max(len(enc['input_ids']) for enc in encodings)
    for i in range(len(labels)):
        labels[i] = labels[i] + [label_map['O']] * (max_len - len(labels[i]))

    encodings.update({'labels': labels})
    return encodings


label_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
train_encodings = encode_data(train_data, tokenizer)


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


train_dataset = NERDataset(train_encodings)
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_map[label] for label in doc] for doc in labels]
    true_predictions = [
        [label_map[pred] for (pred, label) in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, true_labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


results = trainer.evaluate()
print(results)