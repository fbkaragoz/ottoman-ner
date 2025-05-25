from transformers import AutoTokenizer
import torch

def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tokens = []
    labels = []
    sentence_tokens = []
    sentence_labels = []

    for line in lines:
        if line.strip():
            token, label = line.strip().split()
            sentence_tokens.append(token)
            sentence_labels.append(label)
        else:
            if sentence_tokens:
                tokens.append(sentence_tokens)
                labels.append(sentence_labels)
                sentence_tokens = []
                sentence_labels = []

    if sentence_tokens:  #Add last sentence if not added
        tokens.append(sentence_tokens)
        labels.append(sentence_labels)

    return tokens, labels

def encode_data(tokens, labels, tokenizer, label_list):
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids = []
    attention_masks = []
    label_ids = []

    for i, sentence in enumerate(tokens):
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # padding sentences
            padding='max_length',
            return_attention_mask=True,  # attention masks.
            truncation=True,
            return_tensors='pt'  # Return pytorch tensors
        )

        # non of the [CLS] and [SEP] tokens.
        labels_adjusted = [-100] + [label_map.get(label, -100) for label in labels[i][:min(len(sentence), 510)]] + [-100]
        labels_adjusted = labels_adjusted + [-100] * (512 - len(labels_adjusted))  # Pad labels to max length

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        label_ids.append(torch.tensor(labels_adjusted))

    # Converting lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.stack(label_ids, dim=0)

    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': label_ids}


model_checkpoint = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#  label list
label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

#data files location
train_tokens, train_labels = read_data("../dsai/labeled_training_data/HisTR-main/train.txt")
dev_tokens, dev_labels = read_data("../dsai/labeled_training_data/HisTR-main/dev.txt")
test_tokens, test_labels = read_data("../dsai/labeled_training_data/HisTR-main/test.txt")

train_dataset = encode_data(train_tokens, train_labels, tokenizer, label_list)
dev_dataset = encode_data(dev_tokens, dev_labels, tokenizer, label_list)
test_dataset = encode_data(test_tokens, test_labels, tokenizer, label_list)
print(train_dataset['input_ids'][0])