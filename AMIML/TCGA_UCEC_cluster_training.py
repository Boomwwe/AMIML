import torch.nn as nn
import os
import copy
import pickle
import random
import glob
from model import AMIML
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
from fit_model import fit_model
# file = "BRCA"
# gene_rank=["TP53","CDH1","ERBB2","BRCA1","BRCA2","PIK3CA"]
# file = "GBM"
# gene_rank=["IDH1","TRRAP","KMT2C","RB1","ATRX","ZFHX3"]
file = "UCEC"
gene_rank=["TP53","PTEN","POLE","JAK1","MTOR","ATM"]
# file ="KIRC"
# gene_rank=["TP53","BAP1","SETD2","PBRM1","KMT2C","ATM"]
gpu_id = 1
model_name = "AMIML"
file_name = "cluster"
seed = 12345
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

list_before=gene_rank
list_after = [val for val in list_before for i in range(4)]
cluster=["cluster1","cluster2","cluster3","cluster4"]
all_cluster=[]
for i in range(len(list_before)):
    for j in cluster:
        all_cluster.append(j)
output_execl = pd.DataFrame(np.zeros(len(list_before)*4*4).reshape(len(list_before)*4,4), index = list_after, columns = ['TCGA_mean_ACC','TCGA_mean_AUC','TCGA_BEST_ACC',"TCGA_BEST_AUC"])
output_execl["cluster"]=all_cluster

for t_gene in list_before:
    for c in [1,2,3,4]:
        My_dataloader = dataloader_function(c,t_gene,file)
        # 设置随机数种子
        setup_seed(seed)

        c_weight=pd.read_csv("./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER"+str(c) +"/" + str(t_gene)+"/path_label_ALL_" + str(t_gene) + "_"+file+".csv")
        c_label=c_weight["label"]
        weights=[np.sum(c_label)/len(c_label),1-np.sum(c_label)/len(c_label)]

        feature_folder_path = "./Gene_Mut/"+file+"/TCGA/cluster/2048_cluster_"+ str(c)
        # feature_paths = glob.glob(feature_folder_path + "/*.npy")
        # feature_index = range(len(feature_paths))
        path_label_data = pd.read_csv("./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER"+str(c) +"/" + str(t_gene)+"/path_label_ALL_" + str(t_gene) + "_"+file+".csv")
        feature_paths = path_label_data["path"].tolist()

        for m in range(5):
            fit_model(
                cluster=c,
                t_gene=t_gene,
                cv=m,
                weights=weights,
                use_weights=True,
                n_epochs=300,
                lr=1e-3,
                gpu_id=gpu_id,
            )


count_triger=-1
for t_gene in gene_rank:
    for c in [1,2,3,4]:
        count_triger=count_triger+1
        AUC = []
        ACC = []
        torch.cuda.set_device(gpu_id )
        for m in range(5):
            cv_splits = pickle.load(open("./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER" + str(c) + "/" + t_gene + '/5fold_splits.pkl', 'rb'))
            feature_folder_path = "./Gene_Mut/"+file+"/TCGA/cluster/2048_cluster_"+ str(c)
            path_label_data = pd.read_csv("./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER" + str(c) + "/" + str(t_gene) + "/path_label_ALL_" + str(t_gene) + "_"+file+".csv")
            feature_paths = path_label_data["path"].tolist()
            test_id = [feature_paths[i] for i in cv_splits[m]["test_set"]]
            My_dataloader = dataloader_function(c, t_gene,file)
            test_data = My_dataloader(test_id, train=False)
            test_loader = test_data.get_loader()
            model = HE2RNA(input_dim=256)
            model = model.cuda()
            model.load_state_dict(torch.load(
                "./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER" + str(c) + "/" + str(t_gene) + "/"+model_name+"model/model" + str(m) + ".pkl"))
            model.train(False)
            for i_batch, sample_batch in enumerate(test_loader):
                inputs = sample_batch["feat"].cuda()
                labels = sample_batch["label"].squeeze(-1).cuda()
                model.eval()
                y_p = model(inputs)
                test_acc = accuracy_score(torch.max(y_p, dim=1)[1].detach().cpu(), labels.detach().cpu())
                test_auc = roc_auc_score(labels.detach().cpu(), y_p[:, 1].detach().cpu())
                AUC.append(test_auc)
                ACC.append(test_acc)
        output_execl.iloc[count_triger,0] = np.mean(ACC)
        output_execl.iloc[count_triger,1] = np.mean(AUC)
        output_execl.iloc[count_triger,2] = np.max(ACC)
        output_execl.iloc[count_triger,3] = np.max(AUC)
        output_execl.to_csv("./Gene_Mut/"+file+"/TCGA/cluster/"+model_name+str(seed)+"_CLUSTER_result.csv")