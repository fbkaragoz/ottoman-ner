# nerpackage/training.py
def load_ner_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n\n')
    data = []
    for line in lines:
        tokens_labels = [token.split() for token in line.split('\n') if token]
        data.append([(token_label[0], token_label[1]) for token_label in tokens_labels if len(token_label) == 2])
    return data


train_data = load_ner_data('../dsai/labeled_training_data/HisTR-main/train.txt')
dev_data = load_ner_data('../dsai/labeled_training_data/HisTR-main/dev.txt')
test_data = load_ner_data('../dsai/labeled_training_data/HisTR-main/test.txt')

print(train_data[:2])