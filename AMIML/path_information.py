#%%

import pandas as pd
import os
import shutil
rare_gene = gene_rank
"""
This code is used to generate path information
"""
for c in [1,2,3,4]:
    for t_gene in rare_gene:
        mutation=pd.read_csv("./Gene_Mut/"+file+"/TCGA/TCGA_mutation.csv")
        gene=mutation["sample"]
        feature_path="./Gene_Mut/"+file+"/TCGA/cluster/2048_cluster_"+str(c)
        out_big_path = "./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER"+str(c)
        if not os.path.exists(out_big_path):
            os.mkdir(out_big_path)
        out_path = "./Gene_Mut/"+file+"/TCGA/cluster/CLUSTER"+str(c)+"/"+t_gene
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        ge_len = len(gene)
        m_gene = t_gene
        index=gene[gene==m_gene].index
        mut_list=list(mutation)
        mut_list=mut_list[1:]
        mut_data=pd.DataFrame(mut_list,columns=["ID"])
        value=mutation.iloc[index].values.squeeze(0)
        value.tolist()
        value=value[1:]
        value=pd.DataFrame(value,columns=[m_gene])
        final = pd.concat([mut_data,value],axis=1)

        datalist=os.listdir(feature_path)
        len1=len(datalist)
        len2=len(mut_list)
        common_name=[]
        common_label=[]
        for i in range(len1):
            dataname1=datalist[i][0:12]
            tigger=0
            for j in range(len2):
                dataname2 = mut_list[j][0:12]
                if dataname1 == dataname2:
                    tigger=1
                    #shutil.copy("/data/TCGA_CPTAC/BRCA/CLUSTER/CLUSTER3/feature2000_1/"+datalist[i],
                    #           "/data/TCGA_CPTAC/BRCA/CLUSTER/CLUSTER3/feature2000_1_common/"+dataname1+".npy")
                    common_name.append(datalist[i][0:-4])
                    common_label.append(final[final["ID"]==mut_list[j]][m_gene].item())
        common_name=pd.DataFrame(common_name,columns=["ID"])
        common_label=pd.DataFrame(common_label,columns=[m_gene])
        final_execl = pd.concat([common_name,common_label],axis=1)
        final_execl.to_csv(out_path+"/tcga_"+m_gene+"_"+file+"_mutation_with_label.csv")

        datalist = os.listdir(feature_path)
        mutation_label = pd.read_csv(out_path+"/tcga_"+m_gene+"_"+file+"_mutation_with_label.csv")
        path_name=[]
        all_label=[]
        length=len(mutation_label)
        for i in range(length):
            ID = mutation_label.iloc[i]["ID"]
            longname=feature_path+"/"+ID+".npy"
            path_name.append(longname)
            all_label.append(mutation_label.iloc[i][m_gene])
        path_name= pd.DataFrame(path_name,columns=["path"])
        all_label= pd.DataFrame(all_label,columns=["label"])
        path_label=pd.concat([path_name,all_label],axis=1)
        path_label.to_csv(out_path+"/path_label_ALL_"+m_gene+"_"+file+".csv")
