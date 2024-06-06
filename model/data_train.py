from transformers import BertTokenizerFast, BertForTokenClassification

tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=10)


def encode_data(sentences, tokenizer):
    texts = [[word for word, label in sentence] for sentence in sentences]
    labels = [[label for word, label in sentence] for sentence in sentences]
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

    labels = [[label_map[label] for label in doc] for doc in labels]
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

trainer.train()