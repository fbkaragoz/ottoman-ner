import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ner_evaluation_report.csv', index_col=0)

# Extract relevant data
entity_types = df.index.tolist()
precision = df['precision'].tolist()
recall = df['recall'].tolist()
f1_score = df['f1-score'].tolist()
support = df['support'].tolist()

# Convert support to integer
support = [int(s) for s in support]

# Precision Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=entity_types, y=precision, palette='Blues_d')
plt.title('Precision for Each Entity Type')
plt.ylabel('Precision')
plt.xlabel('Entity Type')
plt.show()

# Recall Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=entity_types, y=recall, palette='Greens_d')
plt.title('Recall for Each Entity Type')
plt.ylabel('Recall')
plt.xlabel('Entity Type')
plt.show()

# F1-Score Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=entity_types, y=f1_score, palette='Reds_d')
plt.title('F1-Score for Each Entity Type')
plt.ylabel('F1-Score')
plt.xlabel('Entity Type')
plt.show()

# Support Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=entity_types, y=support, palette='Purples_d')
plt.title('Support for Each Entity Type')
plt.ylabel('Support')
plt.xlabel('Entity Type')
plt.show()