from datasets import load_metric

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