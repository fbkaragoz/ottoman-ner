from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from .read import read_data, encode_data

# model and tokenizer
model_checkpoint = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=5)

# labels
label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

# locations for encoding and decoding
train_tokens, train_labels = read_data("../dsai/labeled_training_data/HisTR-main/train.txt")
dev_tokens, dev_labels = read_data("../dsai/labeled_training_data/HisTR-main/dev.txt")
test_tokens, test_labels = read_data("../dsai/labeled_training_data/HisTR-main/test.txt")

train_dataset = encode_data(train_tokens, train_labels, tokenizer, label_list)
dev_dataset = encode_data(dev_tokens, dev_labels, tokenizer, label_list)
test_dataset = encode_data(test_tokens, test_labels, tokenizer, label_list)

# arguments for training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# trainer and modules
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)

# training the model
trainer.train()

# evaluation
eval_results = trainer.evaluate()
print(eval_results)