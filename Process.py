import pandas as pd
import torch
import torchtext
from torchtext import data
from Tokenize import moltokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import numpy as np


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

    if opt.src_data_te is not None:
        try:
            opt.src_data_te = open(opt.src_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data_te + "' file not found")
            quit()

    if opt.trg_data_te is not None:
        try:
            opt.trg_data_te = open(opt.trg_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data_te + "' file not found")
            quit()


def create_fields(opt):
    lang_formats = ['SMILES', 'SELFIES']
    if opt.lang_format not in lang_formats:
        print('invalid src language: ' + opt.lang_forma + 'supported languages : ' + lang_formats)

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer)
    TRG = data.Field(tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))

        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG, PROP, tr_te):
    # masking data longer than max_strlen
    if tr_te == "tr":
        print("\n* creating [train] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    if tr_te == "te":
        print("\n* creating [test] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data_te], 'trg': [line for line in opt.trg_data_te]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    df = pd.concat([df, PROP], axis=1)

    # if tr_te == "tr":  #for code test
    #     df = df[:30000]
    # if tr_te == "te":
    #     df = df[:3000]

    if opt.lang_format == 'SMILES':
        mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # if opt.lang_format == 'SELFIES':
    #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    df = df.loc[mask]
    if tr_te == "tr":
        print("     - # of training samples:", len(df.index))
        df.to_csv("DB_transformer_temp.csv", index=False)
    if tr_te == "te":
        print("     - # of test samples:", len(df.index))
        df.to_csv("DB_transformer_temp_te.csv", index=False)

    logP = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)

    data_fields = [('src', SRC), ('trg', TRG), ('logP', logP), ('tPSA', tPSA), ('QED', QED)]

    if tr_te == "tr":
        toklenList = []
        train = data.TabularDataset('./DB_transformer_temp.csv', format='csv', fields=data_fields, skip_header=True)
        for i in range(len(train)):
            toklenList.append(len(vars(train[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv("toklen_list.csv", index=False)
        if opt.verbose == True:
            print("     - tokenized training sample 0:", vars(train[0]))
    if tr_te == "te":
        train = data.TabularDataset('./DB_transformer_temp_te.csv', format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print("     - tokenized testing sample 0:", vars(train[0]))

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.logP), len(x.tPSA), len(x.QED)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)
    try:
        os.remove('DB_transformer_temp.csv')
    except:
        pass
    try:
        os.remove('DB_transformer_temp_te.csv')
    except:
        pass

    if tr_te == "tr":
        if opt.load_weights is None:
            print("     - building vocab from train data...")
            SRC.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of SRC: {}\n        -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of TRG: {}\n        -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = get_len(train_iter)

    if tr_te == "te":
        opt.test_len = get_len(train_iter)

    return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

