import argparse
import json
import os
from ucca_parser import UCCA_Parser
import torch.optim as optim
import torch
import torch.utils.data as Data

from ucca_parser.utils import (
    Corpus,
    Trainer,
    Vocab,
    collate_fn,
    get_config,
    Embedding,
    UCCA_Evaluator,
    MyScheduledOptim,
)


class Train(object):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, help="Train a model.")
        subparser.add_argument("--fr_train_path", required=True, help="fr train data dir")

        subparser.add_argument("--fr_dev_path", required=True, help="fr dev data dir")

        subparser.add_argument("--fr_train_projection_path", required=False, help="fr train projection data dir")
        subparser.add_argument("--fr_dev_projection_path", required=False, help="fr dev projection data dir")

        subparser.add_argument("--save_path", required=True, help="dic to save all file")
        subparser.add_argument("--config_path", required=True, help="init config file")

        subparser.add_argument("--fr_test_path", help="fr data dir", default="")
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        config = get_config(args.config_path)
        assert config.ucca.type in ["chart", "top-down", "global-chart"]

        with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=4)

        assert isinstance(args.fr_train_projection_path, str) == isinstance(args.fr_dev_projection_path, str)
        use_projections = isinstance(args.fr_train_projection_path, str) and isinstance(args.fr_dev_projection_path, str)

        print("save all files to %s" % (args.save_path))
        # read training , dev file
        print("loading datasets and transforming to trees...")
        fr_train = Corpus(args.fr_train_path, "fr", args.fr_train_projection_path)
        print(fr_train)

        fr_dev = Corpus(args.fr_dev_path, "fr", args.fr_dev_projection_path)
        print(fr_dev)

        # init vocab
        print("collecting words and labels in training dataset...")
        vocab = Vocab(config.ucca.bert_vocab, [fr_train])
        print(vocab)
        fr_train.filter(512, vocab)

        vocab_path = os.path.join(args.save_path, "vocab.pt")
        torch.save(vocab, vocab_path)

        # init ucca_parser
        print("initializing model...")
        ucca_parser = UCCA_Parser(vocab, config.ucca, use_projections=use_projections)
        if torch.cuda.is_available():
            ucca_parser = ucca_parser.cuda()

        # prepare data
        train_dataset = Data.ConcatDataset([fr_train.generate_inputs(vocab, True)])
        dev_dataset = Data.ConcatDataset([fr_dev.generate_inputs(vocab, False)])
        print("preparing input data...")
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=config.ucca.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_size=10,
            shuffle=False,
            collate_fn=collate_fn,
        )

        optimizer = optim.Adam(ucca_parser.parameters(), lr=config.ucca.lr)
        ucca_evaluator = UCCA_Evaluator(
            parser=ucca_parser,
            gold_dic=[args.fr_dev_path]
        )

        trainer = Trainer(
            parser=ucca_parser,
            optimizer=optimizer,
            evaluator=ucca_evaluator,
            batch_size=config.ucca.batch_size,
            epoch=config.ucca.epoch,
            patience=config.ucca.patience,
            path=args.save_path,
        )
        trainer.train(train_loader, dev_loader)

        # reload ucca_parser
        del ucca_parser
        torch.cuda.empty_cache()
        print("reloading the best ucca_parser for testing...")
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        state_path = os.path.join(args.save_path, "ucca_parser.pt")
        config_path = os.path.join(args.save_path, "config.json")
        ucca_parser = UCCA_Parser.load(vocab_path, config_path, state_path)

        if args.fr_test_20k_path:
            print("evaluating fr 20K test data : %s" % (args.fr_test_20k_path))
            test = Corpus(args.fr_test_20k_path, "fr")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.fr_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()