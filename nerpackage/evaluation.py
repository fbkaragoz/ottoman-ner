# nerpackage/evaluation.py
from seqeval.metrics import precision_score, recall_score, f1_score

class Evaluation:
    @staticmethod
    def compute_metrics(p, label_list):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate(self, trainer, test_dataset, label_list):
        results = trainer.predict(test_dataset)
        metrics = self.compute_metrics(results, label_list)
        return metrics