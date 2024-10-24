rm(list=ls())

library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(msm)
library(caret)
library(vctrs)
library(ipw)
library(tidyverse)
library(survival)

workdir <- "XX"
datafile <- file.path(workdir,"XX.csv")
lipidsdata <- read.csv(datafile, stringsAsFactors=FALSE, encoding = "UTF-8")
lipidsdata <- transform(lipidsdata, outcome = factor(outcome))
lipidsdata$Covariate1=factor(lipidsdata$Covariate1)
lipidsdata$Covariate2<-as.numeric(lipidsdata$Covariate2)

Covariatesname <- c("xx","xx")
covar <- paste(covname,collapse = "+")
formula <- as.formula(paste("outcome~", covar))
lipidsdata$predict<-NA
namevector<-paste0('prediction',1:1000)
lipidsdata[,namevector]<-NA


for (i in 1:1000){
  set.seed(i) 
  folds <- createFolds(y=lipidsdata$outcome,k=10)
  for (k in 1:10){
    lipidstest <- lipidsdata[folds[[k]],]
    lipidstrain <- lipidsdata[-folds[[k]],]
    fit<-glm(formula,family=binomial(link="logit"),data=lipidstrain)
    predict<-predict.glm(fit,type="response",newdata=lipidstest, Weight1s = lipidsdata$Weight1)
    lipidsdata[folds[[k]],]$predict=predict
    print(predict)
    print(k)}
  lipidsdata[,namevector[i]]<-lipidsdata$predict
  print(i)}
write.csv(lipidsdata,"xx.csv")