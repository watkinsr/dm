# Task 3 - Binary Classification Challenge
# Author: Ryan Watkins

library(class)
library(gmodels)
library(cvTools)
library(caret)
library(hydroGOF)
library(MASS)
library(ROCR)
library(nnet)

# Get and clean dataset ready to be worked with
clean.datasets <- function() {
  train100k <<- read.csv("data/training100Ku.csv",head=TRUE,sep=",")
  colnames(train100k) <<- paste("C", 1:21, sep = "")
  colnames(train100k)[22:28] <<- paste("H", 1:7, sep="")
  colnames(train100k)[29] <<- "class"
  train100k[,29] <<- as.numeric(train100k[,29]) - 1
  test1K <<- read.csv("data/test1K.csv",head=TRUE,sep=",")
  colnames(test1K)[2:22] <<- paste("C", 1:21, sep="")
  test1K <<- test1K[2:22]
}

reset.holdout.data <- function(folds,i,dataset) {
  test.ind <<- which(folds==i,arr.ind=TRUE)
  test <<- dataset[test.ind,1:28]
  test.class <<- dataset[test.ind,29]

  train <<- dataset[-test.ind,1:28]
  train.copy <<- train
  train.class <<- dataset[-test.ind,29]

  test.Hl <<- c()
  train.Hl <<- c()
  rmse.errors <<- c()
  hl.rmse.errors <<- c()
  pred.Low.res <<- c()
  pred.Low.fit <<- c()
  pred.HL.res <<- c()
  pred.Hl.fit <<- c()
}

get.high.levels <- function(hl.ind,fold.ind) {
  train.ind <<- hl.ind + 21
  hl.model <<- lm(train[,train.ind]~., data=train[,1:21])
  train.Hl <<- cbind(train.Hl, predict(hl.model, train))
  test.Hl <<- cbind(test.Hl, predict(hl.model, test))
  train.copy[,train.ind] <<- predict(hl.model,train)
  hl.error <- rmse(train.Hl[,hl.ind], train[,train.ind])
  hl.rmse.errors <<- c(hl.rmse.errors, hl.error)
  hl.misclasses[,hl.ind][fold.ind] <<- hl.error
}

rename.hl.cols <- function() {
  colnames(train.Hl)[1:7] <<- paste("H", 1:7, sep="")
  colnames(test.Hl)[1:7]  <<- paste("H", 1:7, sep="")
}

get.roc.and.acc <- function(pred,fn1,fn2) {
  perf <- performance(pred,"tpr","fpr")
  plot(perf)
  dev.copy(pdf,fn1)
  dev.off()

  perf <- performance(pred,"acc")
  plot(perf)
  dev.copy(pdf,fn2)
  dev.off()
}

log.class.results <- function(type, roc.filename, acc.filename, fit, res,i,type.n) {
  cat(type, "results: \n")
  print(confusionMatrix(res,test.class)$table)
  pred <- prediction(abs(fit), test.class)
  get.roc.and.acc(pred,roc.filename,acc.filename)  
  cat("Absolute errors: ", sum(abs(res-test.class)),"\n")
  accuracy[i,type.n] <<- 1 - mean(res!=test.class)
  cat("Accuracy: ", accuracy[i,type.n],"\n")
  retrieved <- sum(as.numeric(res))
  precision[i,type.n] <<- sum(as.numeric(res) & test.class) / retrieved
  recall[i,type.n] <<- sum(as.numeric(res) & test.class) / sum(test.class)
  Fmeasure[i,type.n] <<- 2 * precision[i,type.n] * recall[i,type.n]/ precision[i,type.n] + recall[i,type.n]
  cat ("Precision:" , precision[i,type.n],"\n")
  cat ("Recall: " , recall[i,type.n],"\n")
  cat ("F-measure: " , Fmeasure[i,type.n],"\n")
}

even.class.distribution <- function() {
  train100k <- train100k[sample(nrow(train100k)),] # Shuffles data

  classd <- train100k[which(train100k[,29]==0),]
  based <- train100k[which(train100k[,29]==1),]

  ind <- sample(1:(100000-nrow(based)), 10000)
  train100k.evenSplit <<- rbind(classd[ind,], based)

  train100k.evenSplit <<- train100k.evenSplit[sample(nrow(train100k.evenSplit)),] # Shuffle dataset
}

get.logit.models <- function() {
  train.Low.model <<- glm(as.factor(train.class)~.,family=binomial(link=logit), data=as.data.frame(train[,1:21]))
  
  train.Hl.model <<- glm(as.factor(train.class)~., family=binomial(link=logit), data=as.data.frame(train.Hl))
 
  train.scaled.model <<- glm(as.factor(train.class)~.,family=binomial(link=logit),data=as.data.frame(cbind(scale(train[,1:21]), train[,22:28])))

  train.default.hl.model <<- glm(as.factor(train.class)~.,family=binomial(link=logit),data=as.data.frame(cbind(train[,1:21],train.Hl)))
}

do.logit.predictions <- function() {
  test.class <- as.data.frame(test.class)  
  pred.Low.fit <<- predict(train.Low.model,test,type="response")
  pred.Low.res <<- ifelse(pred.Low.fit > 0.5, 1,0)

  pred.Hl.fit <<- predict(train.Hl.model,test,type="response")
  pred.Hl.res <<- ifelse(pred.Hl.fit > 0.5, 1,0)

  pred.scaled.fit <<- predict(train.scaled.model,test,type="response")
  pred.scaled.res <<- ifelse(pred.scaled.fit >0.5,1,0)

  pred.default.hl.fit <<- predict(train.default.hl.model,test,type="response")
  pred.default.hl.res <<- ifelse(pred.default.hl.fit >0.5,1,0)
}

# Gather HL components and classify with logistic regression
logit.classification <- function(dataset) {
  folds <- cut(seq(1,nrow(dataset)),breaks=10,labels=FALSE)
  for (i in 1:10) {
    reset.holdout.data(folds,i,dataset)
    for (hl.ind in 1:7) {
      get.high.levels(hl.ind,i)
    }

    rename.hl.cols()
    get.logit.models()
    do.logit.predictions()

    log.class.results("low", 'report/images/t3/roc_low.pdf', 'report/images/t3/acc_low.pdf', pred.Low.fit, pred.Low.res,i,1)
    log.class.results("reg.hl", 'report/images/t3/roc_reg_hl.pdf', 'report/images/t3/acc_reg_hl.pdf', pred.Hl.fit, pred.Hl.res,i,2)
    log.class.results("scaled", 'report/images/t3/roc_scaled.pdf', 'report/images/t3/acc_scaled.pdf', pred.scaled.fit, pred.scaled.res,i,3)
    log.class.results("low.hl", 'report/images/t3/roc_low_reg_hl.pdf', 'report/images/t3/acc_low_reg_hl.pdf', pred.default.hl.fit, pred.default.hl.res,i,4)
    
    cat("HL Misclass rates:",hl.rmse.errors,"\n")
    cat("Avg HL Misclass rate:",sum(hl.rmse.errors)/7,"\n")

    cat("\n----END FOLD----\n\n")
  }
  finish.log()
}

scale.data <- function() {
  train <<- scale(train)
  test <<- scale(test)
}
 
nn.with.log <- function(ds.train, ds.test, type,i,type.n) {
  mod <<- nnet(as.factor(train.class)~ ., data=ds.train, size=10, maxit=100)
  out <- predict(mod, ds.test, type="class")
  confmatrix <- table(as.factor(as.numeric(out)),as.factor(test.class))
  print(confmatrix)
  cat("Type: ", type,"\n")
  cat("RMSE: ", rmse(as.numeric(out),test.class),"\n")
  res <- abs(as.numeric(out)-test.class)

  accuracy[i,type.n] <<- length(which(res == 0)) / length(test.class)
  cat("Accuracy: ", accuracy[i,type.n],"\n")
  
  # Saving model
  if (type=="low.reg.hl") {
    prev.best <<- best.accuracy
    best.accuracy <<- ifelse(accuracy[i,type.n] > best.accuracy,accuracy[i,type.n],best.accuracy)
    if(prev.best != best.accuracy) {
      save(mod, file="models/best_nnet.rda")
      saved.test <<- ds.test
      saved.test.class <<- test.class
    }
  }
  cat("Best accuracy so far: " , best.accuracy,"\n")

  retrieved <- sum(as.numeric(out))
  precision[i,type.n] <<- sum(as.numeric(out) & test.class) / retrieved
  recall[i,type.n] <<- sum(as.numeric(out) & test.class) / sum(test.class)
  Fmeasure[i,type.n] <<- 2 * precision[i,type.n] * recall[i,type.n]/ precision[i,type.n] + recall[i,type.n]

  cat ("Precision:" , precision[i,type.n],"\n")
  cat ("Recall: " , recall[i,type.n],"\n")
  cat ("F-measure: " , Fmeasure[i,type.n],"\n")
}

form.matrix <- function(attr){
  matrix(c(mean(attr[,1]),mean(attr[,2]),mean(attr[,3]),mean(attr[,4])),ncol=4)
}

form.all.measures <- function() {
  accur_mat <<- form.matrix(accuracy)
  precision_mat <<- form.matrix(precision)
  recall_mat <<- form.matrix(recall)
  Fmeasure_mat <<- form.matrix(Fmeasure)
}

print.measures <- function(mat,measure) {
  colnames(mat) <- c("low","reg.hl","scaled","low.reg.hl")
  rownames(mat) <- c(measure)
  print(mat)
}

print.all.measures <- function() {
  print.measures(accur_mat,"accuracy")
  print.measures(precision_mat,"precision")
  print.measures(recall_mat,"recall")
  print.measures(Fmeasure_mat,"F-measure")
}

finish.log <- function() {
  form.all.measures()
  print.all.measures()
  cat("Average Misclass Per HL: ", colMeans(hl.misclasses),"\n")
  cat("HL Avg Misclass rate:", mean(colMeans(hl.misclasses)))
}

nn.classification <- function(dataset) {  
  folds <- cut(seq(1,nrow(dataset)),breaks=10,labels=FALSE)
  for (i in 1:10) {
    reset.holdout.data(folds,i,dataset)
    for (hl.ind in 1:7) {
      get.high.levels(hl.ind,i)
    }  
    rename.hl.cols()
    nn.with.log(train[,1:21],test[,1:21],"low",i,1)
    nn.with.log(train.Hl,test.Hl,"reg.hl",i,2)
    nn.with.log(cbind(scale(train[,1:21]),train[,22:28]), cbind(scale(test[,1:21]),test[,22:28]),"scaled",i,3)   
    nn.with.log(cbind((train[,1:21]),train.Hl), cbind(test[,1:21],test.Hl),"low.reg.hl",i,4)  
    cat("\n----END FOLD----\n\n")
  }
  finish.log()
}

# Classify test1k
newdata.classify <- function() {
  reset.holdout.data(1,1,train100k)
  for (hl.ind in 1:7) {
    train.ind <- 21 + hl.ind
    hl.model <<- lm(train100k[,train.ind]~., data=train100k[,1:21])
    train.Hl <<- cbind(train.Hl, predict(hl.model, train))
    test.Hl <<- cbind(test.Hl, predict(hl.model, test1K))
  }
  colnames(test.Hl)[1:7]  <<- paste("H", 1:7, sep="")
  newdata <<- cbind(test1K[,1:21],test.Hl) 
  load("models/best_nnet.rda")
  out <<- predict(mod, newdata, type="class")
  # Output to csv
  write.table(out, file = "Task3-predictions.csv", sep = ",", qmethod = "double", col.names=FALSE, row.names=FALSE)
}

set.seed(3)
clean.datasets()
even.class.distribution()

hl.misclasses <<- matrix(NA,nrow=10,ncol=7)
train <- c()

accuracy <<- matrix(NA,nrow=10,ncol=4)
precision <<- matrix(NA,nrow=10,ncol=4)
recall <<- matrix(NA,nrow=10,ncol=4)
Fmeasure <<- matrix(NA,nrow=10,ncol=4)
best.accuracy <<- 0
# --- Classifications ---
# nn.classification(train100k) Takes too long - 90% on one class
logit.classification(train100k.evenSplit)
nn.classification(train100k.evenSplit)
newdata.classify()
