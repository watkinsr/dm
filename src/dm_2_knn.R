library(class)
library(gmodels)
library(caret) 
library(pROC)

# Wine dataset
wined <- read.csv("data/wine.csv",head=TRUE,sep=",")
wined <- data.frame(wined)
wined <- cbind(wined[,1],scale(wined[,2:14]))
colnames(wined)[1] = "class"

# Seed allows us to always get same results in random sampling
set.seed(2)

# Randomly shuffle the data
wined <- wined[sample(nrow(wined)),]

# Create 10 equal size folds
folds <- cut(seq(1, nrow(wined)),breaks=10,labels=FALSE)

# Perform 10 cross fold validation
wined.err <- c()
wined.pred <- c()
test.class <- c()
wined.all.pred <- c()

for (i in 1:10){
    # Segment your data by using the which() function
    test.ind = which(folds==i,arr.ind=TRUE)
    test.data <- wined[test.ind, 2:14]
    train.data  <- wined[-test.ind, 2:14]
    test.class <- wined[test.ind,1]
    cl=as.factor(wined[-test.ind,1])
    wined.pred = knn(train.data, test.data, cl, k=13)
    wined.all.pred = c(wined.all.pred,wined.pred)
    print(confusionMatrix(wined.pred, test.class))
    wined.err <- c(wined.err,mean(wined.pred != test.class))
}
multiclass.roc(wined.all.pred, wined[,1])$auc
cat("Generalisation error rate: ", sum(wined.err)/10)
