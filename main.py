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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest, globaltest_cos_similarity
from util.fedavg import FedAvg
from util.util import add_noise, get_output, generate, extract_features
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

        acc = globaltest_cos_similarity(netglob, dataset_test, args)
        if acc > best_acc:
            best_acc = acc
        f_acc.write("Stage 1 round %d, test acc %.4f \n" % (rnd, acc))
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

    clients_noise_less_thres = np.zeros(len(real_noise_level))
    clients_noise_less_thres[real_noise_level > args.clean_set_thres] = 1
    #for the ground truth: we consider clients to be clean if their noise ratio is <clean_set_thres (default 10) 
    ground_truth = clients_noise_less_thres

    # Predicted labels from GMM (noisy = 1, clean = 0)
    predicted_labels = np.ones_like(ground_truth)
    predicted_labels[clean_set] = 0
    # Reverse the roles: clean =1, noisy = 0
    ground_truth_positive = np.where(ground_truth == 0, 1, 0)  # Clean clients as the positive class
    predicted_labels_positive = np.where(predicted_labels == 0, 1, 0)  # Predicted clean clients as positive

    # Confusion matrix (for clean clients as positive)
    tn, fp, fn, tp = confusion_matrix(ground_truth_positive, predicted_labels_positive).ravel()

    accuracy_client_selection = accuracy_score(ground_truth_positive, predicted_labels_positive)
    precision_clean_client = precision_score(ground_truth_positive, predicted_labels_positive)
    recall_clean_client = recall_score(ground_truth_positive, predicted_labels_positive)
    f_acc.write("Clean client selection local model loss avg, Accuracy: %.4f, Precision: %.4f, Recall: %.4f \n" % (accuracy_client_selection, precision_clean_client, recall_clean_client))
    f_acc.flush()
    print("Clean client selection local model loss avg, Accuracy: %.4f, Precision: %.4f, Recall: %.4f \n" % (accuracy_client_selection, precision_clean_client, recall_clean_client))

    
    torch.save(netglob.state_dict(), outputs_path + "model_endstage1.pth")

    #t-SNE plot:
    '''
    features, labels = extract_features(args, netglob, dataset_test)

    features_np = features.astype(np.float64) 

    if args.model == 'resnet50':
        prototypes_numpy = generate(2048, args.num_classes, seed=args.seed)
    else:
        prototypes_numpy = generate(512, args.num_classes, seed=args.seed)
    
    combined_features = np.vstack([features_np, prototypes_numpy])

    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_features)  

    features_2d = combined_2d[:features_np.shape[0]]
    prototypes_2d = combined_2d[features_np.shape[0]:]

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 8))

    cmap = plt.get_cmap('tab20', args.num_classes) 

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, alpha=0.7, label='Data Points')
    
    plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], c='black', marker='x', s=100, label='Prototypes')
    plt.legend()
    
    if args.dataset=="cifar10":
        cbar = plt.colorbar(scatter, ticks=range(10))  
        cbar.ax.set_yticklabels(cifar10_classes)  #
    else:
        plt.colorbar(scatter)  


    plt.title('t-SNE Visualization of Feature Representations')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.savefig(outputs_path + 'tsne_visualization_stage1.png', dpi=300)

    print(f"t-SNE plot saved")
    '''

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
                
                feature_whole = torch.tensor(feature_whole).to(args.device)
                feature_whole = F.normalize(feature_whole, p=2, dim=1)
                cos_sim = F.cosine_similarity(feature_whole.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
                y_predicted = torch.argmax(cos_sim, dim=1).cpu().numpy().astype(np.int64)
                confident_idx = np.where(torch.max(cos_sim.cpu(), axis=1).values > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets, copy=True)

                y_train_noisy_new[sample_idx[confident_idx]] = y_predicted[confident_idx]
                dataset_train.targets = y_train_noisy_new.copy()
                dataset_train_noaug.targets = y_train_noisy_new.copy()
                f_acc.write("Noise ratio before relab: %.4f\n" % (np.sum(y_train[sample_idx] != y_train_noisy[sample_idx])/len(y_train[sample_idx])))
                f_acc.write("Ratio confident_samples/all_samples: %.4f\n" % (len(confident_idx)/len(sample_idx)))

                confident_samples = sample_idx[confident_idx]
                if len(confident_samples)>0:
                    
                    noise_ratio_union = np.sum(y_train[confident_samples] != y_train_noisy_new[confident_samples]) / len(confident_samples)

                    f_acc.write("Noise ratio in confident samples: %.4f\n" % noise_ratio_union)
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=confident_samples)
                    dict_locals.append(len(confident_samples))
                    net_local = copy.deepcopy(netglob).to(args.device)
                    w_local, loss_local = local.update_weights(net=net_local, w_g=netglob, epoch=args.local_ep)
                    w_locals.append(copy.deepcopy(w_local))  # store every updated model
                    if idx==0 or idx==1:
                            writer.add_scalar(f'Client_{idx} Loss', loss_local, rnd + args.rounds1)
                            writer.flush()
                    del net_local, w_local
                else:
                    f_acc.write("No confident predictions \n")
            f_acc.flush()
                    
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

            acc = globaltest_cos_similarity(netglob, dataset_test, args)
            f_acc.write("Stage 2 round %d, test acc %.4f \n" % (rnd, acc))
            f_acc.flush()
            writer.add_scalar('Accuracy', acc, rnd + args.rounds1)
            writer.flush()
            if acc > best_acc:
                best_acc = acc
        
    f_acc.write("Best test acc: %.4f \n" % best_acc)
    f_acc.flush()

    torch.save(netglob.state_dict(), outputs_path + "model_endstage2.pth")

    #t-SNE plot:
    '''
    features, labels = extract_features(args, netglob, dataset_test)

    features_np = features.astype(np.float64) 

    if args.model == 'resnet50':
        prototypes_numpy = generate(2048, args.num_classes, seed=args.seed)
    else:
        prototypes_numpy = generate(512, args.num_classes, seed=args.seed)
    
    combined_features = np.vstack([features_np, prototypes_numpy])

    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_features)  

    features_2d = combined_2d[:features_np.shape[0]]
    prototypes_2d = combined_2d[features_np.shape[0]:]

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 8))

    cmap = plt.get_cmap('tab20', args.num_classes) 

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, alpha=0.7, label='Data Points')
    
    plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], c='black', marker='x', s=100, label='Prototypes')
    plt.legend()
    
    if args.dataset=="cifar10":
        cbar = plt.colorbar(scatter, ticks=range(10))  
        cbar.ax.set_yticklabels(cifar10_classes)  #
    else:
        plt.colorbar(scatter)  


    plt.title('t-SNE Visualization of Feature Representations')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.savefig(outputs_path + 'tsne_visualization_stage2.png', dpi=300)

    print(f"t-SNE plot saved")
    '''

    torch.cuda.empty_cache()

