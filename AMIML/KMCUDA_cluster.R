library(class)
sample = read.csv("./TCGA_CPTAC/BRCA/sample_feature.csv")
dyn.load("/home/ustc/kmcuda/src/libKMCUDA.so")
result = .External("kmeans_cuda", as.matrix(sample[,-1]),4, tolerance=0.01,seed=12345, verbosity=1, average_distance=TRUE)
paths='./TCGA_CPTAC/cluster_csv/'
path1 = "./Gene_Mut/BRCA/TCGA/csv/"
dataset<-list.files(paths)
dataset1<-list.files(path1)
dataset2<-setdiff(dataset,dataset1)
write.table(result$centroids,file = "./Gene_Mut/BRCA/TCGA/center.txt")
for (i in 1:length(dataset2)){
  feat=read.csv(file = paste0(paths,dataset2[i]))[,-1]
  clusterlab<-knn(result$centroids, as.matrix(feat), c(1,2,3,4), k = 1, l = 0, prob = FALSE, use.all = TRUE)
  feat$cluster=clusterlab
  write.csv(feat,paste0('./Gene_Mut/BRCA/TCGA/csv/',dataset2[i]),row.names = F)
}


center=read.table("./Gene_Mut/BRCA/TCGA/center.txt")
paths='./Gene_Mut/BRCA/TCGA/csv/'
path1 = "./Gene_Mut/BRCA/TCGA/cluster/all"
dataset<-list.files(paths)
dataset1<-list.files(path1)
dataset2<-setdiff(dataset,dataset1)
for (i in 1:length(dataset2)){
  feat=read.csv(file = paste0(paths,dataset2[i]))[,-1]
  clusterlab<-knn(center, as.matrix(feat), c(1,2,3,4), k = 1, l = 0, prob = FALSE, use.all = TRUE)
  feat$cluster=clusterlab
  write.csv(feat,paste0('./Gene_Mut/BRCA/TCGA/cluster/all/',dataset2[i]),row.names = F)
}




