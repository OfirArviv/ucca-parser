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

        subparser.add_argument("--save_path", required=True, help="dic to save all file")
        subparser.add_argument("--config_path", required=True, help="init config file")

        subparser.add_argument("--en_test_wiki_path", help="en wiki test data dir", default="")
        subparser.add_argument("--en_test_20k_path", help="en 20k data dir", default="")
        subparser.add_argument("--fr_test_20k_path", help="fr 20k data dir", default="")
        subparser.add_argument("--de_test_20k_path", help="de 20k data dir", default="")
        subparser.set_defaults(func=self)

        return subparser

    @staticmethod
    def load_parser(args):
        print("reloading the best ucca_parser...")
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        state_path = os.path.join(args.save_path, "ucca_parser.pt")
        config_path = os.path.join(args.save_path, "config.json")
        ucca_parser, vocab = UCCA_Parser.load(vocab_path, config_path, state_path)

        print(vocab)

        return ucca_parser, vocab

    @staticmethod
    def existing_parser_exists(args):
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        state_path = os.path.join(args.save_path, "ucca_parser.pt")
        config_path = os.path.join(args.save_path, "config.json")
        return os.path.isfile(vocab_path) and os.path.isfile(state_path) and os.path.isfile(config_path)

    def __call__(self, args):
        config = get_config(args.config_path)

        assert all(path is None for path in [config.en_train_path, config.en_dev_path]) or \
               all(path is not None for path in [config.en_train_path, config.en_dev_path])
        assert all(path is None for path in [config.fr_train_path, config.fr_dev_path]) or \
               all(path is not None for path in [config.fr_train_path, config.fr_dev_path])
        assert all(path is None for path in [config.de_train_path, config.de_dev_path]) or \
               all(path is not None for path in [config.de_train_path, config.de_dev_path])
        assert any(path is not None for path in [config.en_train_path, config.fr_train_path, config.de_train_path])

        assert config.ucca.type in ["chart", "top-down", "global-chart"]

        with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=4)

        print("save all files to %s" % (args.save_path))

        # read training , dev file
        train_corpora = []
        dev_corpora = []
        print("loading datasets and transforming to trees...")
        if config.projections_path:
            print("using projections...")
        if config.en_train_path and config.en_dev_path:
            en_train = Corpus(config.en_train_path, "en", config.projections_path)
            train_corpora.append(en_train)
            en_dev = Corpus(config.en_dev_path, "en", config.projections_path)
            dev_corpora.append(en_dev)

        if config.fr_train_path and config.fr_dev_path:
            fr_train = Corpus(config.fr_train_path, "fr", config.projections_path)
            train_corpora.append(fr_train)
            fr_dev = Corpus(config.fr_dev_path, "fr", config.projections_path)
            dev_corpora.append(fr_dev)

        if config.de_train_path and config.de_dev_path:
            de_train = Corpus(config.de_train_path, "de", config.projections_path)
            train_corpora.append(de_train)
            de_dev = Corpus(config.de_dev_path, "de", config.projections_path)
            dev_corpora.append(de_dev)

        print("Train Corpora:")
        for corpus in train_corpora:
            print(f'{corpus}')
        print("Dev Corpora:")
        for corpus in dev_corpora:
            print(f'{corpus}')

        if self.existing_parser_exists(args):
            ucca_parser, vocab = self.load_parser(args)
        else:
            # init vocab
            print("collecting words and labels in training dataset...")
            vocab = Vocab(config.ucca.bert_vocab, train_corpora)
            print(vocab)

            vocab_path = os.path.join(args.save_path, "vocab.pt")
            torch.save(vocab, vocab_path)

            # init ucca_parser
            print("initializing model...")
            ucca_parser = UCCA_Parser(vocab, config.ucca, use_projections=args.projections_path is not None)

        for corpus in train_corpora:
            corpus.filter(512, vocab)

        if torch.cuda.is_available():
            ucca_parser = ucca_parser.cuda()

        # prepare data
        train_dataset = Data.ConcatDataset([corpus.generate_inputs(vocab, True) for corpus in train_corpora])
        dev_dataset = Data.ConcatDataset([corpus.generate_inputs(vocab, False) for corpus in dev_corpora])
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
            gold_dic=list(filter(lambda x: x is not None, [args.en_dev_path, args.fr_dev_path, args.de_dev_path])),
            save_path=args.save_path
        )
        if self.existing_parser_exists(args):
            print("computing accuracy of the saved model for calibration...")
            ucca_evaluator.compute_accuracy(dev_loader)

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
        self.load_parser(args)

        if args.en_test_wiki_path:
            print("evaluating en wiki test data : %s" % (args.en_test_wiki_path))
            test = Corpus(args.en_test_wiki_path, "en")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.en_test_wiki_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()

        if args.en_test_20k_path:
            print("evaluating en 20K test data : %s" % (args.en_test_20k_path))
            test = Corpus(args.en_test_20k_path, "en")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.en_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()

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

        if args.de_test_20k_path:
            print("evaluating de 20K test data : %s" % (args.de_test_20k_path))
            test = Corpus(args.de_test_20k_path, "de")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.de_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()
