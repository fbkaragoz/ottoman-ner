from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("models/ottoman_ner_latin")
tokenizer = AutoTokenizer.from_pretrained("models/ottoman_ner_latin")

model.push_to_hub("fatihburakkaragoz/ottoman-ner-latin")
tokenizer.push_to_hub("fatihburakkaragoz/ottoman-ner-latin")
