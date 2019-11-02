from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from torch.nn import init

from ucca_parser.module import Feedforward

import torch
import torch.nn as nn
from ucca.layer0 import Terminal

from ucca_parser.convert import to_UCCA
from ucca_parser.convert import InternalParseNode, LeafParseNode
from ucca_parser.convert import get_position


class Chart_Span_Parser(nn.Module):
    def __init__(self, vocab, lstm_dim, label_hidden_dim, drop=0, norm=False):
        super(Chart_Span_Parser, self).__init__()
        self.vocab = vocab
        self.label_ffn = Feedforward(lstm_dim, label_hidden_dim, vocab.num_parse_label, drop, norm)

    def forward(self, span):
        label_score = self.label_ffn(span)
        return label_score

    def trace_back(self, left, right, best_split_matrix, label_index_matrix):
        length = right - left
        label_index = int(label_index_matrix[length - 1][left])
        if length == 1:
            tree = LeafParseNode(int(left), "pos", "word")
            if label_index != self.vocab.NULL_index:
                tree = InternalParseNode(self.vocab.id2parse_label(label_index), [tree])
            return [tree]
        else:
            best_split = best_split_matrix[length - 1][left]
            children = self.trace_back(
                left, best_split, best_split_matrix, label_index_matrix
            ) + self.trace_back(
                best_split, right, best_split_matrix, label_index_matrix
            )
            if label_index != self.vocab.NULL_index:
                tree = [
                    InternalParseNode(self.vocab.id2parse_label(label_index), children)
                ]
            else:
                tree = children
            return tree

    def CKY(self, label_scores, sen_len):
        label_index_matrix = label_scores.new_zeros(
            (sen_len, sen_len), dtype=torch.long
        )
        best_split_matrix = label_scores.new_full(
            (sen_len, sen_len), -1, dtype=torch.long
        )
        accum_score_matrix = label_scores.new_zeros((sen_len, sen_len))

        for length in range(1, sen_len + 1):
            for left in range(0, sen_len + 1 - length):
                right = left + length

                label_score = label_scores[get_position(sen_len, left, right)]

                if length == sen_len:
                    label_score[0] = float("-inf")

                argmax_label_score, argmax_label_index = torch.max(label_score, dim=0)
                label_index_matrix[length - 1, left] = argmax_label_index

                if length == 1:
                    accum_score_matrix[length - 1, left] = argmax_label_score
                    continue
                span_scores = (
                    accum_score_matrix[range(0, length - 1), left]
                    + accum_score_matrix[
                        range(length - 2, -1, -1), range(left + 1, right)
                    ]
                )
                accum_score, best_split = torch.max(span_scores, dim=0)
                best_split_matrix[length - 1][left] = best_split + left + 1
                if int(argmax_label_index) == self.vocab.NULL_index:
                    accum_score_matrix[length - 1][left] = accum_score
                else:
                    accum_score_matrix[length - 1][left] = (
                        accum_score + argmax_label_score
                    )
        tree = self.trace_back(0, sen_len, best_split_matrix, label_index_matrix)
        assert len(tree) == 1
        return tree[0].convert()

    def get_loss(self, spans, sen_lens, trees):
        loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
        batch_loss = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score = self.forward(spans[i][:span_num])
            gold_label = self.get_gold_labels(trees[i], length)
            gold_label = torch.tensor(gold_label, device=spans[i].device)
            batch_loss.append(loss_func(label_score, gold_label))
        return batch_loss

    def predict(self, spans, sen_lens):
        pred_trees = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score = self.forward(spans[i][:span_num])
            pred_tree = self.CKY(label_score, length)
            pred_trees.append(pred_tree)
        return pred_trees

    def get_gold_labels(self, tree, sen_len):
        span_num = (1 + sen_len) * sen_len // 2
        gold_label = [0] * span_num
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, InternalParseNode):
                label = node.label
                labelindex = self.vocab.parse_label2id(label)
                left = node.left
                right = node.right
                gold_label[get_position(sen_len, left, right)] = labelindex
                nodes.extend(reversed(node.children))
        return gold_label


class Topdown_Span_Parser(nn.Module):
    def __init__(
        self, vocab, lstm_dim, label_hidden_dim, split_hidden_dim, drop=0, norm=False, projection_dim=0
    ):
        super(Topdown_Span_Parser, self).__init__()
        self.vocab = vocab
        self.label_ffn = Feedforward(lstm_dim+projection_dim, label_hidden_dim, vocab.num_parse_label, drop, norm)
        self.split_ffn = Feedforward(lstm_dim+projection_dim, split_hidden_dim, 1, drop, norm)

        if projection_dim > 0:
            # the + 1 is for labels in the projections dataset that were not obvserved in the training and validation
            # datasets
            self.projection_embedding = nn.Embedding(len(self.vocab._parse_label) + 1, projection_dim)
            init.normal_(self.projection_embedding.weight)
        else:
            self.projection_embedding = None

    def forward(self, span, sen_len, projection=None):
        if projection:
            span_list = enumerate_spans(list(range(sen_len)))
            assert len(span_list) == span.shape[0]
            i = 0
            for left, right in span_list:
                position = get_position(sen_len, left, right + 1)
                assert i == position
                i = i + 1

            # 0 index will represent unseen labels. Even thou you don't have to do it. You learn them all from 0 anyway.
            # Need to change it. In the vocab there is need to be another property. TODO:
            label2embedd = lambda x: self.vocab.parse_label2id(x) + 1 if x in self.vocab._parse_label else 0
            # +1 to the right value because the enumerate_spans value is inclusive and the oracle requires
            # exclusive label.
            projection_labels = [projection.tree.oracle_label(left, right + 1) for left, right in span_list]
            projection_labels = [label2embedd(label) for label in projection_labels]
            projection_labels = torch.FloatTensor(projection_labels).long()
            projection_labels = self.projection_embedding(projection_labels)
            span = torch.cat([projection_labels, span], dim=1)

        label_score = self.label_ffn(span)
        split_score = self.split_ffn(span).squeeze(-1)
        return label_score, split_score

    def helper(self, label_scores, split_scores, sen_len, left, right, gold=None):
        position = get_position(sen_len, left, right)
        label_score = label_scores[position]
        if self.training:
            oracle_label = gold.oracle_label(left, right)
            oracle_label_index = self.vocab.parse_label2id(oracle_label)
            label_score = self.augment(label_score, oracle_label_index)
            oracle_label_score = label_score[oracle_label_index]

        if right - left == sen_len:
            label_score[0] = float("-inf")

        argmax_label_score, argmax_label_index = torch.max(label_score, dim=0)
        argmax_label = self.vocab.id2parse_label(int(argmax_label_index))

        if self.training:
            label = oracle_label
            label_loss = argmax_label_score - oracle_label_score
        else:
            label = argmax_label
            label_loss = label_score[argmax_label_index]

        if right - left == 1:
            tree = LeafParseNode(left, "pos", "word")
            if label:
                tree = InternalParseNode(label, [tree])
            return [tree], label_loss

        left_positions = get_position(sen_len, left, range(left + 1, right))
        right_positions = get_position(sen_len, range(left + 1, right), right)
        splits = split_scores[left_positions] + split_scores[right_positions]

        if self.training:
            oracle_splits = gold.oracle_splits(left, right)
            oracle_split = min(oracle_splits)
            oracle_split_index = oracle_split - (left + 1)
            splits = self.augment(splits, oracle_split_index)
            oracle_split_score = splits[oracle_split_index]

        argmax_split_score, argmax_split_index = torch.max(splits, dim=0)
        argmax_split = argmax_split_index + (left + 1)

        if self.training:
            split = oracle_split
            split_loss = argmax_split_score - oracle_split_score
        else:
            split = argmax_split
            split_loss = splits[argmax_split_index]

        left_trees, left_loss = self.helper(
            label_scores, split_scores, sen_len, left, int(split), gold
        )
        right_trees, right_loss = self.helper(
            label_scores, split_scores, sen_len, int(split), right, gold
        )

        children = left_trees + right_trees
        loss = label_loss + split_loss + left_loss + right_loss
        if label:
            children = [InternalParseNode(label, children)]
        return children, loss

    def get_loss(self, spans, sen_lens, trees, projections=None):
        batch_loss = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score, split_score = self.forward(spans[i][:span_num], sen_lens[i],
                                                    projections[i] if projections else None)
            _, loss = self.helper(label_score, split_score, length, 0, length, trees[i])
            batch_loss.append(loss)
        return batch_loss

    def predict(self, spans, sen_lens, projections):
        pred_trees = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score, split_score = self.forward(spans[i][:span_num], sen_lens[i],
                                                    projections[i] if projections else None)
            pred_tree, _ = self.helper(label_score, split_score, length, 0, length)
            pred_trees.append(pred_tree[0].convert())
        return pred_trees

    def augment(self, scores, oracle_index):
        increment = torch.ones_like(scores)
        increment[oracle_index] = 0
        return scores + increment


class Global_Chart_Span_Parser(nn.Module):
    def __init__(self, vocab, lstm_dim, label_hidden_dim, drop=0, norm=False):
        super(Global_Chart_Span_Parser, self).__init__()
        self.vocab = vocab
        self.label_ffn = Feedforward(lstm_dim, label_hidden_dim, vocab.num_parse_label, drop, norm)

    def forward(self, span):
        label_score = self.label_ffn(span)
        return label_score

    def trace_back(self, left, right, best_split_matrix, label_index_matrix):
        length = right - left
        label_index = int(label_index_matrix[length - 1][left])
        if length == 1:
            tree = LeafParseNode(int(left), "pos", "word")
            if label_index != self.vocab.NULL_index:
                tree = InternalParseNode(self.vocab.id2parse_label(label_index), [tree])
            return [tree]
        else:
            best_split = best_split_matrix[length - 1][left]
            children = self.trace_back(
                left, best_split, best_split_matrix, label_index_matrix
            ) + self.trace_back(
                best_split, right, best_split_matrix, label_index_matrix
            )
            if label_index != self.vocab.NULL_index:
                tree = [
                    InternalParseNode(self.vocab.id2parse_label(label_index), children)
                ]
            else:
                tree = children
            return tree

    def CKY(self, label_scores, sen_len, gold=None):
        label_index_matrix = label_scores.new_zeros(
            (sen_len, sen_len), dtype=torch.long
        )
        best_split_matrix = label_scores.new_full(
            (sen_len, sen_len), -1, dtype=torch.long
        )
        accum_score_matrix = label_scores.new_zeros((sen_len, sen_len))

        for length in range(1, sen_len + 1):
            for left in range(0, sen_len + 1 - length):
                right = left + length
                label_score = label_scores[get_position(sen_len, left, right)]

                if self.training:
                    oracle_label = gold.oracle_label(left, right)
                    oracle_label_index = self.vocab.parse_label2id(oracle_label)
                    label_score = self.augment(label_score, oracle_label_index)

                if length == sen_len:
                    label_score[0] = float("-inf")

                argmax_label_score, argmax_label_index = torch.max(label_score, dim=0)
                label_index_matrix[length - 1, left] = argmax_label_index

                if length == 1:
                    accum_score_matrix[length - 1, left] = argmax_label_score
                    continue

                span_scores = (
                    accum_score_matrix[range(0, length - 1), left]
                    + accum_score_matrix[
                        range(length - 2, -1, -1), range(left + 1, right)
                    ]
                )
                accum_score, best_split = torch.max(span_scores, dim=0)
                best_split_matrix[length - 1][left] = best_split + left + 1

                accum_score_matrix[length - 1][left] = accum_score + argmax_label_score
        if self.training:
            return None, accum_score_matrix[-1][0]

        tree = self.trace_back(0, sen_len, best_split_matrix, label_index_matrix)
        assert len(tree) == 1
        return tree[0].convert(), accum_score_matrix[-1][0]

    def get_gold_score(self, label_scores, sen_len, gold):
        accum_score_matrix = label_scores.new_zeros((sen_len, sen_len))

        for length in range(1, sen_len + 1):
            for left in range(0, sen_len + 1 - length):
                right = left + length
                label_score = label_scores[get_position(sen_len, left, right)]

                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.vocab.parse_label2id(oracle_label)
                oracle_label_score = label_score[oracle_label_index]

                if length == 1:
                    accum_score_matrix[length - 1, left] = oracle_label_score
                    continue

                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)

                left_score = accum_score_matrix[oracle_split - left - 1, left]
                right_score = accum_score_matrix[right - oracle_split - 1, oracle_split]

                accum_score_matrix[length - 1][left] = (
                    left_score + right_score + oracle_label_score
                )
        return accum_score_matrix[-1][0]

    def get_loss(self, spans, sen_lens, trees):
        batch_loss = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score = self.forward(spans[i][:span_num])

            _, pred_score = self.CKY(label_score, length, trees[i])
            gold_score = self.get_gold_score(label_score, length, trees[i])

            batch_loss.append(pred_score - gold_score)
        return batch_loss

    def predict(self, spans, sen_lens):
        pred_trees = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score = self.forward(spans[i][:span_num])
            pred_tree, _ = self.CKY(label_score, length)
            pred_trees.append(pred_tree)
        return pred_trees

    def augment(self, scores, oracle_index):
        increment = torch.ones_like(scores)
        increment[oracle_index] = 0
        return scores + increment
