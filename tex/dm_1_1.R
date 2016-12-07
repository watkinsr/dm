library(ggbiplot)
library(ggfortify)
require(ggplot2)
library(cluster)
library(clValid)

# Plot 1: PCA applied to wine dataset
wined <- read.csv("data/wine.csv",head=TRUE,sep=",")
df <- data.frame(wined)
df.pca <- prcomp(df[,2:14])
df.pca$x[,1:2]
classes <- as.character(c(df$class))
plot1 <- ggbiplot(df.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#c0392b"))+
    geom_point(aes(X=df.pca$x[,1], Y=df.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot1
dev.copy(pdf, 'report/images/p1_un.pdf')
dev.off()

# K-means clustering w/ PCA
# Plot 2 
cd_k_unnorm <- kmeans(wined[,2:14], 3)
autoplot(cd_k_unnorm, data=scale(wined[,2:14]))
dev.copy(pdf, 'report/images/p2_un.pdf')
dev.off()

# Attempt to perform cluster validation
intern_unnorm <- clValid(wined[,2:14], 3, clMethods=c("clara", "kmeans", "hierarchical"), validation="internal")
summary(intern_unnorm)


# Plot3a-3c
df["clusid"] <- cd_k_unnorm$cluster

# Plot3a
df_clus1_unnorm <- df[,][which(cd_k_unnorm$cluster==1),]
cl1_unnorm.pca <- prcomp(df_clus1_unnorm[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus1_unnorm$class))

plot3a <- ggbiplot(cl1_unnorm.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#FFFFFF")) + geom_point(aes(X=cl1_unnorm.pca$x[,1], Y=cl1_unnorm.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3a
dev.copy(pdf, 'report/images/p3a_un.pdf')
dev.off()

# Plot3b
df_clus2_unnorm <- df[,][which(cd_k_unnorm$cluster==2),]
cl2_unnorm.pca <- prcomp(df_clus2_unnorm[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus2_unnorm$class))
classes
plot3b <- ggbiplot(cl2_unnorm.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#c0392b")) + geom_point(aes(X=cl2_unnorm.pca$x[,1], Y=cl2_unnorm.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3b
dev.copy(pdf, 'report/images/p3b_un.pdf')
dev.off()

# Plot3c
df_clus3_unnorm <- df[,][which(cd_k_unnorm$cluster==3),]
cl3_unnorm.pca <- prcomp(df_clus3_unnorm[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus3_unnorm$class))
plot3c <- ggbiplot(cl3_unnorm.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#c0392b")) + geom_point(aes(X=cl3_unnorm.pca$x[,1], Y=cl3_unnorm.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3c
dev.copy(pdf, 'report/images/p3c_un.pdf')
dev.off()
