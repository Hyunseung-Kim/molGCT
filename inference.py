import sys
from io import StringIO
import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import re
import numpy as np
import math
import moses
from rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from dataDistibutionCheck import checkdata

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def gen_mol(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()

    robustScaler = joblib.load(opt.load_weights + '/scaler.pkl')
    if opt.conds == 'm':
        cond = cond.reshape(1, -1)
    elif opt.conds == 's':
        cond = cond.reshape(1, -1)
    elif opt.conds == 'l':
        cond = cond.reshape(1, -1)
    else:
        cond = np.array(cond.split(',')[:-1]).reshape(1, -1)

    cond = robustScaler.transform(cond)
    cond = Variable(torch.Tensor(cond))
    
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence


def inference(opt, model, SRC, TRG):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    if opt.conds == 'm':
        print("\nGenerating molecules for MOSES benchmarking...")
        n_samples = 30000
        nBins = [1000, 1000, 1000]

        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)

        start = time.time()
        for idx in range(n_samples):
            toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
            z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
            molecule_tmp = gen_mol(conds[idx], model, opt, SRC, TRG, toklen, z)
            toklen_gen.append(molecule_tmp.count(' ')+1)
            molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

            molecules.append(molecule_tmp)
            conds_trg.append(conds[idx])
            # toklen-3: due to cond dim
            toklen_check.append(toklen-3)
            m = Chem.MolFromSmiles(molecule_tmp)
            if m is None:
                val_check.append(0)
                conds_rdkit.append([None, None, None])
            else:
                val_check.append(1)
                conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))

            if (idx+1) % 100 == 0:
                print("*   {}m: {} / {}".format((time.time() - start)//60, idx+1, n_samples))

            if (idx+1) % 2000 == 0:
                np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
                gen_list = pd.DataFrame(
                    {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
                gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

        print("Please check the file: 'moses_bench2_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))


    elif opt.conds == 's':
        print("\nGenerating molecules for 10 condition sets...")
        n_samples = 10
        n_per_samples = 200
        nBins = [1000, 1000, 1000]

        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples*n_per_samples)

        print("conds:\n", conds)
        start = time.time()
        for idx in range(n_samples):
            for i in range(n_per_samples):
                toklen = int(toklen_data[idx*n_per_samples + i]) + 3  # +3 due to cond2enc
                z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
                molecule_tmp = gen_mol(conds[idx], model, opt, SRC, TRG, toklen, z)
                toklen_gen.append(molecule_tmp.count(" ") + 1)
                molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

                molecules.append(molecule_tmp)
                conds_trg.append(conds[idx])

                toklen_check.append(toklen-3) # toklen -3: due to cond size
                m = Chem.MolFromSmiles(molecule_tmp)
                if m is None:
                    val_check.append(0)
                    conds_rdkit.append([None, None, None])
                else:
                    val_check.append(1)
                    conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))

                if (idx*n_per_samples+i+1) % 100 == 0:
                    print("*   {}m: {} / {}".format((time.time() - start)//60, idx*n_per_samples+i+1, n_samples*n_per_samples))

                if (idx*n_per_samples+i+1) % 200 == 0:
                    np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
                    gen_list = pd.DataFrame(
                        {"set_idx": idx, "mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
                    gen_list.to_csv('moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

        print("Please check the file: 'moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))

    else:
        conds = opt.conds.split(';')
        toklen_data = pd.read_csv(opt.load_toklendata)
        toklen= int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

        for cond in conds:
            molecules.append(gen_mol(cond + ',', model, opt, SRC, TRG, toklen, z))
        toklen_gen = molecules[0].count(" ") + 1
        molecules = ''.join(molecules).replace(" ", "")
        m = Chem.MolFromSmiles(molecules)
        target_cond = conds[0].split(',')
        if m is None:
            #toklen-3: due to cond dim
            print("   --[Invalid]: {}".format(molecules))
            print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-3))
        else:
            logP_v, tPSA_v, QED_v = Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)
            print("   --[Valid]: {}".format(molecules))
            print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-3))
            print("   --From RDKit: logP={:,.4f}, tPSA={:,.4f}, QED={:,.4f}, GenToklen={}".format(logP_v, tPSA_v, QED_v, toklen_gen))

    return molecules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default="saved_model")
    parser.add_argument('-load_traindata', type=str, default="data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=80) #max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)

    parser.add_argument('-epochs', type=int, default=1111111111111)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.max_logP, opt.min_logP, opt.max_tPSA, opt.min_tPSA, opt.max_QED, opt.min_QED = checkdata(opt.load_traindata)

    while True:
        opt.conds =input("\nEnter logP, tPSA, QED to generate molecules (refer the pop-up data distribution)\
        \n* logP: {:.2f} ~ {:.2f}; tPSA: {:.2f} ~ {:.2f}; QED: {:.2f} ~ {:.2f} is recommended.\
        \n* Typing sample: 2.2, 85.3, 0.8\n* Enter the properties (Or type m: MOSES benchmark, s: 10-Condition set test, q: quit):".format(opt.min_logP, opt.max_logP, opt.min_tPSA, opt.max_tPSA, opt.min_QED, opt.max_QED))

        if opt.conds=="q":
            break
        if opt.conds == "m":
            molecule = inference(opt, model, SRC, TRG)
            break
        if opt.conds == "s":
            molecule = inference(opt, model, SRC, TRG)
            break
        else:
            molecule = inference(opt, model, SRC, TRG)


if __name__ == '__main__':
    main()
