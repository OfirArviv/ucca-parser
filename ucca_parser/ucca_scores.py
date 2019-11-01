from ucca import evaluation
from ucca.evaluation import Scores, LABELED, UNLABELED


class UccaScores():
    # TODO: Add Remote / Primary scores
    # TODO: Add a flag for compact score vs full scores. Compact score might be only F1 labeled and unlabeled
    #  while full will add the primary / remote distinction.
    def __init__(self):
        self.scores = {}

    def __call__(self, dataset_label, predicted_tree, gold_tree):
        score = evaluation.evaluate(predicted_tree, gold_tree)
        if dataset_label not in self.scores:
            self.scores[dataset_label] = []
        self.scores[dataset_label].append(score)

    def get_metric(self, reset: bool = False):
        metrics = {}
        score_list = [item for sublist in self.scores.values() for item in sublist]
        agg_score = Scores.aggregate(score_list)
        labeled_average_f1 = agg_score.average_f1(LABELED)
        unlabeled_average_f1 = agg_score.average_f1(UNLABELED)
        metrics[f'labeled_average_F1'] = labeled_average_f1
        metrics[f'unlabeled_average_F1'] = unlabeled_average_f1
        for dataset_label in self.scores:
            dataset_prefix = f'{dataset_label}_'  # if len(self.scores.keys()) > 1 else ""
            agg_score = Scores.aggregate(self.scores[dataset_label])
            labeled_average_f1 = agg_score.average_f1(LABELED)
            unlabeled_average_f1 = agg_score.average_f1(UNLABELED)
            metrics[f'{dataset_prefix}labeled_average_F1'] = labeled_average_f1
            metrics[f'{dataset_prefix}unlabeled_average_F1'] = unlabeled_average_f1
            titles = agg_score.titles()
            values = agg_score.fields()
            for title, value in zip(titles, values):
                metrics[f'{dataset_prefix}{title}'] = float(value)
        if reset:
            self.reset()
        return metrics

    def reset(self):
        self.scores = {}
