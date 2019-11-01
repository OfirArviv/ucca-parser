import os
import shutil
import torch
from ucca.convert import passage2file, xml2passage
import tempfile

from ucca_parser.ucca_scores import UccaScores


def read_passages(path):
    passages = []
    for file in sorted(os.listdir(path)):
        if "xml" not in file:
            print(f'Skipping {file}. Not an xml file.')
            continue
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            print(file_path)
        passages.append(xml2passage(file_path))
    return passages


def write_passages(dev_predicted, path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in dev_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


class UCCA_Evaluator(object):
    def __init__(
        self, parser, gold_dic=None, pred_dic=None, save_path=None
    ):
        self.parser = parser
        self.gold_dic = gold_dic
        self.pred_dic = pred_dic
        self.save_path = save_path

        self.temp_pred_dic = tempfile.TemporaryDirectory(prefix="ucca-eval-")
        self.best_F = 0

        self.dic_to_gold_dir_map = {}
        for idx, dic in enumerate(gold_dic):
            self.dic_to_gold_dir_map[dic] = tempfile.TemporaryDirectory(prefix=f'ucca-eval-gold-{idx}-')

        for dic in self.gold_dic:
            for file in sorted(os.listdir(dic)):
                src_path = os.path.join(dic, file)
                shutil.copy(src_path, self.dic_to_gold_dir_map[dic].name)

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        predicted = []
        for batch in loader:
            subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote = batch
            subword_idxs = subword_idxs.cuda() if torch.cuda.is_available() else subword_idxs
            subword_masks = subword_masks.cuda() if torch.cuda.is_available() else subword_masks
            token_starts_masks = token_starts_masks.cuda() if torch.cuda.is_available() else token_starts_masks
            lang_idxs = lang_idxs.cuda() if torch.cuda.is_available() else lang_idxs
            word_idxs = word_idxs.cuda() if torch.cuda.is_available() else word_idxs
            pos_idxs = pos_idxs.cuda() if torch.cuda.is_available() else pos_idxs
            dep_idxs = dep_idxs.cuda() if torch.cuda.is_available() else dep_idxs
            ent_idxs = ent_idxs.cuda() if torch.cuda.is_available() else ent_idxs
            ent_iob_idxs = ent_iob_idxs.cuda() if torch.cuda.is_available() else ent_iob_idxs

            pred_passages = self.parser.parse(subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs,
                                              pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages)
            predicted.extend(pred_passages)
        return predicted

    def remove_temp(self):
        for temp_dir in self.dic_to_gold_dir_map.values():
            temp_dir.cleanup()
        self.temp_pred_dic.cleanup()

    def compute_accuracy(self, loader):
        passage_predicted = self.predict(loader)
        write_passages(passage_predicted, self.temp_pred_dic.name)

        ucca_score = UccaScores()
        pred_trees = read_passages(self.temp_pred_dic.name)

        for key, item in self.dic_to_gold_dir_map.items():
            gold_trees = read_passages(item.name)
            gold_trees_ids = [p.ID for p in gold_trees]
            dic_pred_trees = filter(lambda x: x.ID in gold_trees_ids, pred_trees)
            for pred, gold in zip(dic_pred_trees, gold_trees):
                ucca_score(key, pred, gold)

        metrics, score = ucca_score.get_metric()

        metrics_path = os.path.join(self.save_path, "metrics.csv")
        is_new_file = not os.path.isfile(metrics_path)
        with open(metrics_path, 'a') as file:
            if is_new_file:
                file.write("".join([f'{title},' for title in metrics.keys()])+"\n")
            file.write("".join([f'{title},' for title in metrics.values()])+"\n")

        Fscore = metrics['labeled_f1']
        print("Fscore={}".format(Fscore))

        if Fscore > self.best_F:
            score.print()
            self.best_F = Fscore
            if self.pred_dic:
                write_passages(passage_predicted, self.pred_dic)

        return Fscore
