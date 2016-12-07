# To fit decision trees
library(tree) 
library(rpart)
library(rattle)

# Wine dataset scaled for decision trees
wined <- read.csv("data/wine.csv",head=TRUE,sep=",")

# Seed allows us to always get same results in random sampling
set.seed(2)

# Randomly shuffle the data
wined <- wined[sample(nrow(wined)),]

# Get tree model
wined.rpart <- rpart(class~., data=wined,method="class")
plotcp(wined.rpart)
dev.copy(pdf,'report/images/t2_tree_prune.pdf')
dev.off()

tree.predict(wined.rpart,wined)
wined.rpart <- prune(wined.rpart, cp=0.017)
tree.predict(wined.rpart, wined)

fancyRpartPlot(wined.rpart)
dev.copy(pdf,'report/images/t2_tree.pdf')
dev.off()


tree.predict <- function(model, ds) {
  cat("Error rate on tree prediction:" , mean(predict(model,ds,type="class") != ds$class))
}

# Perform 10 cross fold validation on decision trees
ten.fold.tree <- function() {
    folds <- cut(seq(1, nrow(wined)),breaks=10,labels=FALSE)
    means <- c()
    wined.err <- c()
    for (i in 1:10){
      # Segment your data by using the which() function
      test.ind = which(folds==i,arr.ind=TRUE)
      test.data <- wined[test.ind, ]
      train.data  <- wined[-test.ind, ]
      test.class  <- test.data$class
      
      # Test the tree model using test data
      wined.pred = predict(wined.rpart, test.data, type="class")
      wined.err = c(wined.err,mean(wined.pred!=test.class))
    }
    cat("Generalisation error rate: ", sum(wined.err)/10)
}

ten.fold.tree()

