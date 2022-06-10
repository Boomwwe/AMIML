import torch.nn as nn
import os
import copy
import pickle
import random
import glob
from AMIML_model import AMIML
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Earlystopping import EarlyStopping
from dataloader_function import dataloader_function

def fit_model(
        cluster,
        t_gene,
        cv=2,
        # cv_splits='/data/gbw/WSI_embeddings/cvsplits/cv_splits.pkl',
        weights=[0.4, 0.6],
        use_weights=True,
        n_epochs=50,
        lr=1e-4,
        gpu_id=3,
):
    print("GPU_id: %d" % (gpu_id))
    print("CV_spilt: %d" % (cv))
    print("cluster: %d" % (cluster))
    print(t_gene)
    torch.cuda.set_device(gpu_id)

    cv_splits = pickle.load(
        open('./Gene_Mut/'+file+'/TCGA/cluster/CLUSTER' + str(cluster) + "/" + t_gene + '/5fold_splits.pkl', 'rb'))
    test_id = [feature_paths[i] for i in cv_splits[cv]["test_set"]]
    train_id = [feature_paths[j] for j in cv_splits[cv]["train_set"]]
    val_id = [feature_paths[m] for m in cv_splits[cv]["val_set"]]

    data = My_dataloader(train_id, train=True)
    train_loader = data.get_loader()

    val_data = My_dataloader(val_id, train=False)
    test_data = My_dataloader(test_id, train=False)
    val_loader = val_data.get_loader()
    test_loader = test_data.get_loader()
    weights = weights

    model = HE2RNA(input_dim=256)
    os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu_id'
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(30, verbose=True)
    loss_func = nn.CrossEntropyLoss(weight=torch.tensor(weights).float() if use_weights else None)
    loss_function = loss_func.cuda()

    # initialize val saving
    save_mod = True
    past_performance = [10]
    n_total_batches = 0
    save_folder_path = './Gene_Mut/'+file+'/TCGA/cluster/CLUSTER' + str(cluster) + "/" + t_gene +"/"+model_name+str(seed)+"model"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    for epoch in range(n_epochs):
        train_ACC, train_AUC, train_Loss = [], [], []
        for i_batch, sample_batch in enumerate(train_loader):
            model.train(True)
            inputs = sample_batch["feat"].cuda()
            labels = sample_batch["label"].squeeze(-1).cuda()
            y_p = model(inputs)  # 输出的label
            loss = loss_function(y_p, labels.long())  # 损失函数
            loss.backward()  # 反向传播
            optimizer.step()
            optimizer.zero_grad()
            train_acc = accuracy_score(torch.max(y_p, dim=1)[1].detach().cpu(), labels.detach().cpu())
            train_ACC.append(train_acc)
            train_Loss.append(loss.data.item())
            del inputs, labels, y_p, train_acc, loss
        vali_ACC, vali_AUC, vali_Loss = [], [], []
        model.eval()
        for i_batch, sample_batch in enumerate(val_loader):
            model.train(False)
            inputs = sample_batch["feat"].cuda()
            labels = sample_batch["label"].squeeze(-1).cuda()
            y_p = model(inputs)  # 输出的label
            loss = loss_function(y_p, labels.long())  # 损失函数
            vali_acc = accuracy_score(torch.max(y_p, dim=1)[1].detach().cpu(), labels.detach().cpu())
            vali_auc = roc_auc_score(labels.detach().cpu(), y_p[:, 1].detach().cpu())
            vali_ACC.append(vali_acc)
            vali_AUC.append(vali_auc)
            vali_Loss.append(loss.data.item())
            del inputs, labels, y_p, loss
        scheduler.step(np.mean(vali_Loss))
        early_stopping(np.mean(vali_Loss), model)

        if epoch >= 5:
            if save_mod and np.mean(vali_Loss) <= min(past_performance):
                best_model_dict = copy.deepcopy(model.state_dict())
                past_performance.append(np.mean(vali_Loss))
        print("epoch: %d" % (epoch), "train_loss： %f" % (np.mean(train_Loss)),
              "val_loss: %f" % (np.mean(vali_Loss)))
        print("epoch: %d" % (epoch), "train_acc： %f" % (np.mean(train_ACC)), "val_acc: %f" % (np.mean(vali_ACC)),
              "val_auc: %f" % (np.mean(vali_AUC)))
        if epoch % 20 == 0 and epoch > 1:
            torch.save(model.state_dict(), save_folder_path + '/weight_MLP_model' + str(epoch) + str(cv) + '.pkl')
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break

    model.load_state_dict(best_model_dict)
    torch.save(model.state_dict(),
               save_folder_path + '/weight_Best_MLP_model' + str(n_epochs) + str(cv) + '.pkl')

