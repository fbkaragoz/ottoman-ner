import spacy
from sklearn.metrics import classification_report
import pandas as pd

"""
This function parses a text file and extracts sentences and named entity annotations.
The expected format of the text file is that each word is followed by a space and its label.
Each sentence is separated by a blank line.
"""
def parse_test_data(file_path):
    sentences = []
    entities = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        entity = []
        offset = 0
        for line in file:
            if line.strip():
                word, label = line.split()
                sentence.append(word)
                if label != 'O':
                    start = offset
                    end = start + len(word)
                    entity.append((start, end, label))
                offset += len(word) + 1  # +1 for the space
            else:
                if sentence:
                    sentences.append(' '.join(sentence))
                    entities.append({'entities': entity})
                    sentence = []
                    entity = []
                    offset = 0
    return list(zip(sentences, entities))


"""
This function aligns the true and predicted entities with tokens.
It assigns labels to the tokens based on their start and end indices.
"""

def get_token_labels(doc, entities):
    labels = ['O'] * len(doc)
    for start, end, label in entities:
        for token in doc:
            if token.idx >= start and token.idx < end:
                labels[token.i] = label
    return labels


# Path to the text file containing the test data
test_data_path = '../dsai/labeled_training_data/HisTR-main/test.txt'
test_data = parse_test_data(test_data_path)

# Load the trained NER model
model_path = 'ottoman_turkish_ner_model'
nlp = spacy.load(model_path)

# Evaluate the model
true_labels = []
pred_labels = []

for text, annotation in test_data:
    doc = nlp(text)
    true_labels_doc = get_token_labels(doc, annotation['entities'])
    pred_labels_doc = ['O'] * len(doc)

    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            pred_labels_doc[i] = ent.label_

    # Debugging output to check for mismatches in document lengths, as it occured
    if len(true_labels_doc) != len(pred_labels_doc):
        print(f"Mismatch in doc lengths: {len(true_labels_doc)} != {len(pred_labels_doc)} for text: {text}")

    true_labels.extend(true_labels_doc)
    pred_labels.extend(pred_labels_doc)

# Ensure lengths are consistent
if len(true_labels) != len(pred_labels):
    print(f"true_labels and pred_labels mismatch in length: {len(true_labels)} != {len(pred_labels)}")

# Calculate precision, recall, and F1 score
report = classification_report(true_labels, pred_labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Save the evaluation report to a CSV file
df_report.to_csv('ner_evaluation_report.csv', index=True)

# Print the report
print(df_report)