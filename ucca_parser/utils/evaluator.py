import os
import shutil
import subprocess
import torch
from ucca.convert import passage2file, xml2passage
import tempfile

from ucca_parser.ucca_scores import UccaScores


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
        self, parser, gold_dic=None, pred_dic=None,
    ):
        self.parser = parser
        self.gold_dic = gold_dic
        self.pred_dic = pred_dic

        self.temp_gold_dic = tempfile.TemporaryDirectory(prefix="ucca-eval-gold-")
        self.temp_pred_dic = tempfile.TemporaryDirectory(prefix="ucca-eval-")
        self.best_F = 0

        for dic in self.gold_dic:
            for file in sorted(os.listdir(dic)):
                if "xml" not in file:
                    print(f'Skipping {file}. Not an xml file.')
                    continue
                skip_files = ["90000", "82002", "422002", "423003", "772001", "776002", "776005"]
                if any(f'{passage_id}.xml' == file for passage_id in skip_files):
                    continue
                src_path = os.path.join(dic, file)
                shutil.copy(src_path, self.temp_gold_dic.name)

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        predicted = []
        for batch in loader:
            subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote, projections = batch
            subword_idxs = subword_idxs.cuda() if torch.cuda.is_available() else subword_idxs
            subword_masks = subword_masks.cuda() if torch.cuda.is_available() else subword_masks
            token_starts_masks = token_starts_masks.cuda() if torch.cuda.is_available() else token_starts_masks
            lang_idxs = lang_idxs.cuda() if torch.cuda.is_available() else lang_idxs
            word_idxs = word_idxs.cuda() if torch.cuda.is_available() else word_idxs
            pos_idxs = pos_idxs.cuda() if torch.cuda.is_available() else pos_idxs
            dep_idxs = dep_idxs.cuda() if torch.cuda.is_available() else dep_idxs
            ent_idxs = ent_idxs.cuda() if torch.cuda.is_available() else ent_idxs
            ent_iob_idxs = ent_iob_idxs.cuda() if torch.cuda.is_available() else ent_iob_idxs

            pred_passages = self.parser.parse(subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, projections=projections)
            predicted.extend(pred_passages)
        return predicted

    def remove_temp(self):
        self.temp_gold_dic.cleanup()
        self.temp_pred_dic.cleanup()

    @staticmethod
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

    def compute_accuracy(self, loader, res_path):
        passage_predicted = self.predict(loader)
        write_passages(passage_predicted, self.temp_pred_dic.name)

        ucca_score = UccaScores()
        pred_trees = self.read_passages(self.temp_pred_dic.name)
        gold_tress = self.read_passages(self.temp_gold_dic.name)
        for pred, gold in zip(pred_trees, gold_tress):
            ucca_score("dev", pred, gold)

        metrics = ucca_score.get_metric()
        is_new_file = not os.path.isfile(res_path)
        with open(res_path, 'a') as file:
            if is_new_file:
                file.write("labeled_average_F1,unlabeled_average_F1,dev_primary_labeled_f1,dev_remote_labeled_f1\n")
            file.write(f'{metrics["labeled_average_F1"]},{metrics["unlabeled_average_F1"]},'
                       f'{metrics["dev_primary_labeled_f1"]},{metrics["dev_remote_labeled_f1"]}\n')


        child = subprocess.Popen(
            "python -m scripts.evaluate_standard {} {} -f".format(
                self.temp_gold_dic.name, self.temp_pred_dic.name
            ),
            shell=True,
            stdout=subprocess.PIPE,
        )
        eval_info = str(child.communicate()[0], encoding="utf-8")
        try:
            Fscore = eval_info.strip().split("\n")[-2]
            Fscore = Fscore.strip().split()[-1]
            Fscore = float(Fscore)
            print("Fscore={}".format(Fscore))
        except IndexError:
            print("Unable to get FScore. Skipping.")
            Fscore = 0

        if Fscore > self.best_F:
            print('\n'.join(eval_info.split('\n')[1:]))
            self.best_F = Fscore
            if self.pred_dic:
                write_passages(passage_predicted, self.pred_dic)
        return Fscore