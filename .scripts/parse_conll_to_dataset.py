from datasets import Dataset, DatasetDict
from pathlib import Path

def read_conll(filepath):
    tokens, labels = [], []
    sentence_tokens, sentence_labels = [], []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence_tokens:
                    tokens.append(sentence_tokens)
                    labels.append(sentence_labels)
                    sentence_tokens, sentence_labels = [], []
                continue
            splits = line.split()
            if len(splits) != 2:
                continue  # skip malformed lines
            word, label = splits
            sentence_tokens.append(word)
            sentence_labels.append(label)
        
        # Add last sentence
        if sentence_tokens:
            tokens.append(sentence_tokens)
            labels.append(sentence_labels)
    
    return tokens, labels


def load_dataset_from_conll(data_dir: str):
    data = {}
    for split in ["train", "dev", "test"]:
        filepath = Path(data_dir) / f"{split}.txt"
        tokens, labels = read_conll(filepath)
        data[split] = Dataset.from_dict({"tokens": tokens, "ner_tags": labels})
    
    return DatasetDict(data)


def create_label_mappings(dataset):
    """Create label2id and id2label mappings from the dataset."""
    unique_labels = set(label for example in dataset["train"]["ner_tags"] for label in example)
    label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def encode_labels(example, label2id):
    """Convert string labels to integer IDs."""
    example["labels"] = [label2id[label] for label in example["ner_tags"]]
    return example


if __name__ == '__main__':
    # Test the functions
    dataset = load_dataset_from_conll("data/raw")
    print("Dataset loaded:")
    print(dataset)
    print("\nFirst example from train set:")
    print(dataset["train"][0])
    
    # Create label mappings
    label2id, id2label = create_label_mappings(dataset)
    print(f"\nLabel mappings:")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    
    # Encode labels
    dataset = dataset.map(lambda example: encode_labels(example, label2id))
    print(f"\nDataset after encoding labels:")
    print(dataset["train"][0])

