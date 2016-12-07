library(ggbiplot)
library(ggfortify)
require(ggplot2)
library(cluster)
library(clValid)

# Plot 1: PCA applied to wine dataset
wined <- read.csv("data/wine.csv",head=TRUE,sep=",")
df <- data.frame(wined)
df.pca <- prcomp(df[,2:14], scale. = TRUE)
df.pca$x[,1:2]
classes <- as.character(c(df$class))
plot1 <- ggbiplot(df.pca, var.axes=FALSE, var.scale=1, obs.scale=1) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#c0392b"))+
    geom_point(aes(X=df.pca$x[,1], Y=df.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot1
dev.copy(pdf,'report/images/p1_n.pdf')
dev.off()

# K-means clustering w/ PCA
# Plot 2 
cd_kmeans <- kmeans(scale(wined[,2:14]), 3)
autoplot(cd_kmeans, data=scale(wined[,2:14]))
dev.copy(pdf,'report/images/p2_n.pdf')
dev.off()

# Attempt to perform cluster validation
intern_norm <- clValid(scale(wined[,2:14]), 3, clMethods=c("clara", "kmeans", "hierarchical"), validation="internal")
summary(intern_norm)

# Plot3a-3c
df["clusid"] <- cd_kmeans$cluster
df_clus1 <- df[,][which(cd_kmeans$cluster==1),]
cl1.pca <- prcomp(df_clus1[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus1$class))
# Plot3a
plot3a <- ggbiplot(cl1.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c", "#FFFFFF")) + geom_point(aes(X=cl1.pca$x[,1], Y=cl1.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3a
dev.copy(pdf,'report/images/p3a_n.pdf')
dev.off()

# Plot3b
df["clusid"] <- cd_kmeans$cluster
df_clus2 <- df[,][which(cd_kmeans$cluster==2),]
cl2.pca <- prcomp(df_clus2[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus2$class))
plot3b <- ggbiplot(cl2.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c")) + geom_point(aes(X=cl2.pca$x[,1], Y=cl2.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3b
dev.copy(pdf,'report/images/p3b_n.pdf')
dev.off()

# Plot3c
df["clusid"] <- cd_kmeans$cluster
df_clus3 <- df[,][which(cd_kmeans$cluster==3),]
cl3.pca <- prcomp(df_clus3[,2:14], scale.=TRUE)
classes <- as.character(c(df_clus3$class))
classes
plot3c <- ggbiplot(cl3.pca, var.axes=FALSE) + scale_colour_manual(values=c("#9b59b6", "#1abc9c")) + geom_point(aes(X=cl3.pca$x[,1], Y=cl3.pca$x[,2], colour=classes, size=1)) + scale_size_identity()
plot3c
dev.copy(pdf,'report/images/p3c_n.pdf')
dev.off()

