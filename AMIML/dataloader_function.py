import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
def dataloader_function(c,t_gene,file):
    class My_dataloader():
        def __init__(self, data_path, train=True):

            if train:
                X_train = data_path
                traindataset = My_dataset(list_path=X_train, train=True,
                                          transform=transforms.Compose([ToTensor()]))  #
                traindataloader = DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=4)

                valdataset = My_dataset(list_path=X_val, train=False, transform=transforms.Compose([ToTensor()]))
                valdataloader = DataLoader(valdataset, batch_size=len(valdataset), shuffle=False)

                self.dataloader = traindataloader

            else:

                testdataset = My_dataset(list_path=data_path, train=False,
                                         transform=transforms.Compose([ToTensor()]))
                testdataloader = DataLoader(testdataset, batch_size=len(testdataset), shuffle=False)

                self.dataloader = testdataloader

        def get_loader(self):
            return self.dataloader


    class My_dataset(Dataset):

        def __init__(self, list_path, train=False, transform=None):
            self.list_path = list_path
            self.random = train
            self.transform = transform
            self.label_path = "./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER" + str(c) + "/" + str(
                t_gene) + "/path_label_ALL_" + str(t_gene) + "_"+file+".csv"

        def __len__(self):
            return len(self.list_path)

        def __getitem__(self, idx):
            img_path = self.list_path[idx]
            Batch_set = []

            train_file = np.load(img_path)
            img_path_suffix = img_path[img_path.rindex("/") + 1:]
            img_path_new = "./Gene_Mut/"+file+"/TCGA/cluster/2048_cluster_" + str(c) + "/" + img_path_suffix
            path_label_data = pd.read_csv("./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER" + str(c) + "/" + str(
                t_gene) + "/path_label_ALL_" + str(t_gene) + "_"+file+".csv")
            train_label = path_label_data[(path_label_data['path'] == img_path_new)]["label"]
            train_label = train_label.squeeze()
            Batch_set.append((train_file, train_label))

            sample = {'feat': train_file, 'label': np.asarray([train_label])}

            if self.transform:
                sample = self.transform(sample)

            return sample


    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            image, label = sample['feat'], sample['label']

            return {'feat': torch.from_numpy(image), 'label': torch.FloatTensor(label)}
    return(My_dataloader)