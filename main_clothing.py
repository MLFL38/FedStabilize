# python version 3.10.15
# -*- coding: utf-8 -*-
import os
import cv2
cv2.setNumThreads(0)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest, globaltest_cos_similarity
from util.fedavg import FedAvg
from util.util import add_noise, get_output, generate
from util.dataset import get_dataset
from model.build_model import build_model
from sklearn.manifold import TSNE
import pickle

np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    args = args_parser()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "../output_folder/"




    outputs_path = rootpath + 'outputs/%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_%d_ep_%d_Frac_%.2f_LR_%.3f_ConT_%.2f_ClT_%.1f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds1,
        args.rounds2, args.local_ep, args.frac, args.lr,
        args.confidence_thres, args.clean_set_thres, args.seed)

    if args.iid:
        outputs_path += "_IID"
    else:
        outputs_path += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.mixup:
        outputs_path += "_Mix_%.1f" % (args.alpha)
    outputs_path = outputs_path + '/'

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)


    dataset_train, dataset_train_noaug, dataset_test, dict_users = get_dataset(args)

    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy.copy()
    dataset_train_noaug.targets = y_train_noisy.copy()

    print(args)

    with open(outputs_path + "dict_users.pkl", "wb") as f:
        pickle.dump(dict_users, f)

    if os.path.exists(outputs_path + 'y_train_clean.npy'):
        print(f"File y_train_clean.npy already exists. Skipping save.")
    else:
        # Save the file if it doesn't exist
        np.save(outputs_path + 'y_train_clean.npy', y_train)

    if os.path.exists(outputs_path + 'y_train_noisy.npy'):
        print(f"File y_train_clean.npy already exists. Skipping save.")
    else:
        # Save the file if it doesn't exist
        np.save(outputs_path + 'y_train_noisy.npy', y_train_noisy)

    if os.path.exists(outputs_path + 'gamma_s.npy'):
        print(f"File gamma_s.npy already exists. Skipping save.")
    else:
        # Save the file if it doesn't exist
        np.save(outputs_path + 'gamma_s.npy', gamma_s)


    if not os.path.exists(rootpath + 'txtsave/'):
        os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_%d_ep_%d_Frac_%.2f_LR_%.3f_ConT_%.2f_ClT_%.1f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds1,
        args.rounds2, args.local_ep, args.frac, args.lr,
        args.confidence_thres, args.clean_set_thres, args.seed)

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.mixup:
        txtpath += "_Mix_%.1f" % (args.alpha)

    f_acc = open(txtpath + '_acc.txt', 'a')

    #tensorboard
    
    if not os.path.exists(rootpath + 'runs/'):
        os.makedirs(rootpath + 'runs/')
    tensorboard_path = rootpath + 'runs/%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_%d_ep_%d_Frac_%.2f_LR_%.3f_ConT_%.2f_ClT_%.1f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds1,
        args.rounds2, args.local_ep, args.frac, args.lr,
        args.confidence_thres, args.clean_set_thres, args.seed)

    if args.iid:
        tensorboard_path += "_IID"
    else:
        tensorboard_path += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.mixup:
        tensorboard_path += "_Mix_%.1f" % (args.alpha)
    writer = SummaryWriter(tensorboard_path)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    LID_accumulative_client = np.zeros(args.num_users)
    estimated_noisy_level = np.zeros(args.num_users)

    best_acc = 0.0
    # ---------------------------- Stage 1 -------------------------------
    m = max(int(args.frac * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds1):
        w_locals = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
           
            net_local = copy.deepcopy(netglob).to(args.device)
            w_local, loss_local = local.update_weights(net=net_local, w_g=netglob, epoch=args.local_ep)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            del net_local, w_local

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc = globaltest(netglob, dataset_test, args)
        if acc > best_acc:
            best_acc = acc
        f_acc.write("Stage 1 round %d, test acc (logits) %.4f \n" % (rnd, acc))
        f_acc.flush()

    LID_whole = np.zeros(len(y_train))
    loss_whole = np.zeros(len(y_train))
    LID_client = np.zeros(args.num_users)

    loss_avg_client_localmodel = np.zeros(args.num_users)


    if args.model == 'resnet50':
        prototypes = generate(2048, args.num_classes, seed=args.seed)
    else:
        prototypes = generate(512, args.num_classes, seed=args.seed)
        
    prototypes = torch.tensor(prototypes).to(args.device)

    
    # Client selection round: all clients participate
    w_locals = []
    for idx in range(args.num_users): 
        sample_idx = np.array(list(dict_users[idx]))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
        
        net_local = copy.deepcopy(netglob).to(args.device)
        w_local, loss_local = local.update_weights(net=net_local, w_g=netglob, epoch=args.local_ep)


        dataset_client = Subset(dataset_train_noaug, sample_idx)
        loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False, num_workers=args.num_workers)


        local_output, loss, local_feature = get_output(loader, net_local, args, True, criterion)
        loss_avg_client_localmodel[idx] = np.mean(loss)

        del net_local, w_local


    loss_avg_client_localmodel = (loss_avg_client_localmodel-loss_avg_client_localmodel.min())/(loss_avg_client_localmodel.max()-loss_avg_client_localmodel.min())

    gmm_loss_avg_client = GaussianMixture(n_components=2, random_state=args.seed).fit(
        np.array(loss_avg_client_localmodel).reshape(-1, 1))
    labels_loss_avg_client = gmm_loss_avg_client.predict(np.array(loss_avg_client_localmodel).reshape(-1, 1))
    clean_label = np.argsort(gmm_loss_avg_client.means_[:, 0])[0]

    noisy_set = np.where(labels_loss_avg_client != clean_label)[0]
    clean_set = np.where(labels_loss_avg_client == clean_label)[0]
    
    #torch.save(netglob.state_dict(), outputs_path + "model_endstage1.pth")


    torch.cuda.empty_cache()
    # ---------------------------- Stage 2 -------------------------------
    
    m = max(int(args.frac * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for i in range(args.num_users)]

    for rnd in range(args.rounds2):
        w_locals, dict_locals = [], []
        dataset_train.targets = y_train_noisy.copy()
        dataset_train_noaug.targets = y_train_noisy.copy()
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in idxs_users:  # training over the subset
            if idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train_noaug, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                output_whole, loss_whole, feature_whole = get_output(loader, netglob, args, True, criterion)


                loss = (loss_whole-loss_whole.min())/(loss_whole.max()-loss_whole.min())
                gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
                labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
                gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]
                pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
                
                feature_whole = torch.tensor(feature_whole).to(args.device)
                feature_whole = F.normalize(feature_whole, p=2, dim=1)
                cos_sim = F.cosine_similarity(feature_whole.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
                y_predicted = torch.argmax(cos_sim, dim=1).cpu().numpy().astype(np.int64)
                confident_idx = np.where(torch.max(cos_sim.cpu(), axis=1).values > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets, copy=True)

                y_train_noisy_new[sample_idx[confident_idx]] = y_predicted[confident_idx]
                dataset_train.targets = y_train_noisy_new.copy()
                dataset_train_noaug.targets = y_train_noisy_new.copy()

                #for Clothing1M: all samples are used for local training, where only confident samples are relabelized for identified noisy samples
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                dict_locals.append(len(dict_users[idx]))
                net_local = copy.deepcopy(netglob).to(args.device)
                w_local, loss_local = local.update_weights(net=net_local, w_g=netglob, epoch=args.local_ep)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                if idx==0 or idx==1:
                        writer.add_scalar(f'Client_{idx} Loss', loss_local, rnd + args.rounds1)
                        writer.flush()
                del net_local, w_local
                
                    
            if idx in clean_set:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                dict_locals.append(len(dict_users[idx]))
                net_local = copy.deepcopy(netglob).to(args.device)
                w_local, loss_local = local.update_weights(net=net_local, w_g=netglob, epoch=args.local_ep)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                if idx==0 or idx==1:
                        writer.add_scalar(f'Client_{idx} Loss', loss_local, rnd + args.rounds1)
                        writer.flush()
                del net_local, w_local
                    
        if len(dict_locals)>0:
            w_glob_fl = FedAvg(w_locals, dict_locals)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            acc = globaltest(netglob, dataset_test, args)
            f_acc.write("Stage 2 round %d, test acc (logits) %.4f \n" % (rnd, acc))
            f_acc.flush()
            writer.add_scalar('Accuracy', acc, rnd + args.rounds1)
            writer.flush()
            if acc > best_acc:
                best_acc = acc


    f_acc.write("Best test acc: %.4f \n" % best_acc)
    f_acc.flush()

    #torch.save(netglob.state_dict(), outputs_path + "model_endstage2.pth")
    torch.cuda.empty_cache()

