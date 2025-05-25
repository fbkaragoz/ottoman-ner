import spacy
from spacy.training import Example
from tqdm import tqdm
import docx

"""
This function parses a DOCX file and extracts sentences and named entity annotations.
The expected format of the DOCX file is that each word is followed by a '/' and its label.
If a word does not have a label, it is assumed to be 'O' (non-defined-entity).
"""
def parse_ner_data_from_docx(file_path):
    doc = docx.Document(file_path)
    sentences = []
    entities = []
    offset = 0

    for para in doc.paragraphs:
        if para.text.strip():  # Check if the paragraph is not empty
            words = para.text.split()
            sentence = []
            entity = []
            for word in words:
                if '/' in word:
                    token, label = word.rsplit('/', 1)
                else:
                    token, label = word, 'O'  # Default label if no delimiter
                sentence.append(token)
                if label != 'O':
                    start = offset
                    end = start + len(token)
                    entity.append((start, end, label))
                offset += len(token) + 1  # +1 for the space
            sentences.append(' '.join(sentence))
            entities.append({'entities': entity})
            offset = 0  # Reset offset for each sentence

    return list(zip(sentences, entities))


#the DOCX file containing the raw training data ruznamce

raw_data_path = '../dsai/raw_data/ruznamce_raw_text.docx'
train_data = parse_ner_data_from_docx(raw_data_path)

# Initialize a blank spaCy model
nlp = spacy.blank('xx')

# Create a new NER pipeline
ner = nlp.add_pipe('ner')

# Add labels to the NER pipeline based on the training data
for _, annotations in train_data:
    for ent in annotations['entities']:
        ner.add_label(ent[2])

# Disable other pipeline components and train the NER model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(100):
        losses = {}
        for text, annotations in tqdm(train_data):
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Iteration {itn} - Losses: {losses}")

        # Save the model at regular intervals as I stop it manually
        if itn % 10 == 0:  # Save every 10 iterations
            model_path = 'ottoman_turkish_ner_model'
            nlp.to_disk(model_path)

# Final save after training
model_path = 'ottoman_turkish_ner_model'
nlp.to_disk(model_path)