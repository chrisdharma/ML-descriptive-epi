#custom_rf is a program to customize rf where you can tune both mtry and 
#ntree directly from the caret package
source("custom_rf.R")

set.seed(4578)
load("impute_Table2.Rdata")
colnames(df1)

#Separate x and y
alcoholout<-c("alcohol_abuse","alcohol_abusec")
predictors <- names(df1)[!(names(df1) %in% alcoholout)]
x<-df1[predictors]
yc<-df1[,"alcohol_abusec"]
y<-df1[,"alcohol_abuse"]
df1$alcohol_abuse<-as.factor(df1$alcohol_abuse)
df2<- subset(df1,select=-c(alcohol_abusec))

library(caret)
library(randomForest)
library(dplyr)

library(epiR)
library(pROC)

library(ggplot2)
library(tidyr)
library(corrplot)

library(elasticnet)
library(klaR)
library(glmnet)

library(ggplot2)

library(vip)
library(pdp)

outerfold <- 10
innerfold <- 5

#The control is for inner fold
control <- trainControl(method='cv',number=innerfold,search="random",summaryFunction=twoClassSummary,classProbs=TRUE)

#Folds for outer and inner
folds <- rep_len(1:outerfold, nrow(x))
folds <- sample(folds, nrow(x))

#1. Random forest
mtry <- round(sqrt(ncol(x))) 
rf.grid <- expand.grid(.mtry=c((mtry/2):mtry),.ntree=c(500,700,900,1200))

#To record the parameter assessments
nested.cv.rf <- list(auc=NULL, mtry=NULL,ntree=NULL)

start.time <- Sys.time()
for(i in 1:outerfold){
  # The outer fold is 10
  #actual split of the data
  fold <- which(folds == i)
  rf_default <- train(y = yc[-fold], x = x[-fold,],method=customRF,metric="ROC",tuneGrid=rf.grid,trControl=control)  
  rf_pred <- predict(rf_default, x[fold,])
  rf_predn<-as.numeric(ifelse(rf_pred=="class0",0,1))
  aucroc <- roc(rf_predn,as.numeric(y[fold]))
  nested.cv.rf$auc <- c(nested.cv.rf$auc,aucroc$auc[1])
  nested.cv.rf$mtry <- c(nested.cv.rf$mtry, rf_default$bestTune[,1])
  nested.cv.rf$ntree <- c(nested.cv.rf$ntree, rf_default$finalModel$ntree)
}
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 1.03 hours

mtry.ncv<-nested.cv.rf$mtry[which.max(nested.cv.rf$auc)]
ntree.ncv<-nested.cv.rf$ntree[which.max(nested.cv.rf$auc)]
rf.grid.ncv <- expand.grid(.mtry=mtry.ncv)
rf.ncv<- train(alcohol_abuse ~ . ,data=df2,method="rf",tuneGrid=rf.grid.ncv,ntree=ntree.ncv)
rf_pred.ncv <- predict(rf.ncv, x)
varImp.rf.ncv<-vip(rf.ncv,scale=TRUE)
impframe.rf.ncv <- data.frame(varImp.rf.ncv$data)
impframe.rf.ncv 

###2. Neural network
nn.grid <- expand.grid(size = seq(from = 1, to = 10, by = 1),decay = seq(from = 0.01, to = 0.1, by = 0.03))
maxSize <- max(nn.grid$size)
numWts <- 1000
nested.cv.nn<-list(auc=NULL,decay = NULL, size=NULL)

start.time <- Sys.time()
for(i in 1:outerfold){
  fold <- which(folds == i)
  nn.m <- train(y = yc[-fold], x = x[-fold,],method='nnet',metric="ROC",trace = FALSE,MaxNWts = numWts,tuneGrid=nn.grid,trControl=control,maxit=1000,preProc = c("center","scale"))  
  nn_pred <- predict(nn.m, x[fold,])
  nn_predn<-as.numeric(ifelse(nn_pred=="class0",0,1))
  aucroc <- roc(nn_predn,as.numeric(y[fold]))
  nested.cv.nn$auc <- c(nested.cv.nn$auc,aucroc$auc[1])
  nested.cv.nn$decay <- c(nested.cv.nn$decay, nn.m$bestTune$decay)
  nested.cv.nn$size <- c(nested.cv.nn$size, nn.m$bestTune$size)
}
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 8.63 hours

decay.ncv<-nested.cv.nn$decay[which.max(nested.cv.nn$auc)]
size.ncv<-nested.cv.nn$size[which.max(nested.cv.nn$auc)]
nn.grid.ncv <- expand.grid(size = size.ncv,decay = decay.ncv)
nn.ncv <- train(alcohol_abuse ~.,data=df2,method='nnet',tuneGrid=nn.grid.ncv,maxit=1000,MaxNWts = numWts,trace = FALSE,preProc = c("center","scale"))  
varImp.nn.ncv<-vip(nn.ncv,scale=TRUE)
impframe.nn.ncv <- data.frame(varImp.nn.ncv$data)

###3. LASSO Regression

x.matrix.all <- model.matrix(alcohol_abuse ~ .,df2)[,-1]
y.all<-df2$alcohol_abuse

folds <- rep_len(1:outerfold, nrow(x.matrix.all))
folds <- sample(folds, nrow(x.matrix.all))
nested.cv.lasso <- list(auc=NULL, lambda=NULL)

for(i in 1:outerfold){
  fold <- which(folds == i)
  cv.lasso <- cv.glmnet(x.matrix.all,y.all,alpha=1,family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
  pred <- as.numeric(predict(cv.lasso,newx=x.matrix.all[fold,],s=cv.lasso$lambda.min,type="class"))
  aucroc<-roc(pred,as.numeric(y.all[fold]))
  nested.cv.lasso$auc <- c(nested.cv.lasso$auc,aucroc$auc[1])
  nested.cv.lasso$lambda <- c(nested.cv.lasso$lambda, cv.lasso$lambda.min)
}

lambda.min.ncv<-nested.cv.lasso$lambda[which.max(nested.cv.lasso$auc)]
lasso.grid.ncv <- expand.grid(alpha = 1,lambda = lambda.min.ncv)
lasso.ncv <- train(y = yc, x = x.matrix.all,method='glmnet',tuneGrid=lasso.grid.ncv)  

varimp.lasso.ncv<-vip(lasso.ncv,scale=TRUE)
impframe.lasso.ncv <- data.frame(varimp.lasso.ncv$data)

###4. NCV for EN (elastic net)

alpha <- seq(0.01, 0.99, 0.02)
bestout <- list(a=NULL, lambda=NULL, auc=NULL)
bestinner <- list(a=NULL, lambda=NULL, auc=NULL)

###This is tuning for alpha at every possible value
start.time <- Sys.time()
for(i in 1:outerfold){
  fold <- which(folds == i)
  for (j in 1:length(alpha)) 
  {
    cvg <- cv.glmnet(x.matrix.all,y.all,alpha=alpha[j],family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
    bestinner$a <- c(bestinner$a, alpha[j])
    bestinner$lambda <- c(bestinner$lambda, cvg$lambda.min)
    bestinner$auc <- c(bestinner$auc, max(cvg$cvm))
  }
  index <- which.min(bestinner$auc)
  best_alpha <- bestinner$a[index]
  best_lambda <- bestinner$lambda[index]
  bestout$a[i] <- best_alpha
  bestout$lambda[i] <- best_lambda
  cvg.EN <- glmnet(x.matrix.all,y.all,alpha=best_alpha,lambda=best_lambda,family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
  pred <- predict(cvg.EN,newx=x.matrix.all[fold,],s=cvg.EN$lambda.min,type="class")  
  aucroc<-roc(pred,as.numeric(y.all[fold]))
  bestout$auc[i] <- aucroc$auc[1]
  bestinner <- list(a=NULL, lambda=NULL, cvm=NULL)
}  
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 14.84 mins
# > time.taken
# Time difference of 1.71 hours

bestout
index <- which.max(bestout$auc)
best_alpha <- bestout$a[index]
best_lambda <- bestout$lambda[index]
EN.grid.ncv <- expand.grid(alpha = best_alpha,lambda = best_lambda)
model.EN <- train(y = yc, x = x.matrix.all,method='glmnet',tuneGrid=EN.grid.ncv) 
varimp.EN.ncv<-vip(model.EN,scale=TRUE)
impframe.EN.ncv <- data.frame(varimp.EN.ncv$data)

#########Partial Dependence Plots#####
pdp_plots <- function(model,xvar,label,binary=F) {
  pdp.partial <- partial(model, xvar,which.class=2)
  pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  if (xvar == 'cannabis') {
     pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(0,1,2), labels= c('Never', 'Frequently', 'Sometimes'))
     pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  if (xvar == 'plan_quit') {
    pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(0,1,2), labels= c('Not planning to quit', 'Plan to quit', 'Already quit'))
    pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  if (xvar == 'covid') {
    pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(0,1,2,3,4), labels= c('Non-smoker',"smoke more","smoke less","smoke same","stopped smoking"))
    pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  if (xvar == 'house_income') {
    pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(1,2,3,4,5,6), labels= c('< 15k','15k - 30k','30k - 60k','60k - 80k','80k - 100k','100k+'))
    pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  if (xvar == 'rural_city') {
    pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(1,2,3,4), labels= c('Large urban',"medium city","small city","rural city"))
    pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  if (binary == T) {
    pdp.partial[,1] <- factor(pdp.partial[,1], levels = c(0,1), labels= c('No','Yes'))
    pdp.plot <- plotPartial(pdp.partial,  xlab= label, ylab= 'PD')
  }
  return(print(pdp.plot))
}

##1. Random Forest
pdp_rf1<-pdp_plots(model=rf.ncv,xvar='outness',label = 'overall outness')
pdp_rf2<-pdp_plots(model=rf.ncv,xvar='age',label = 'age')
pdp_rf3<-pdp_plots(model=rf.ncv,xvar='phobia',label = 'internalized homophobia')
pdp_rf4<-pdp_plots(model=rf.ncv,xvar='per_stigma',label = 'perceived stigma')
pdp_rf5<-pdp_plots(model=rf.ncv,xvar='connect_com',label = 'community connectedness')
pdp_rf6<-pdp_plots(model=rf.ncv,xvar='ace',label = 'adverse childhood experience')
pdp_rf7<-pdp_plots(model=rf.ncv,xvar='en_stigma',label = 'enacted stigma')
pdp_rf8<-pdp_plots(model=rf.ncv,xvar='cen_identity',label = 'centralized identity')
pdp_rf9<-pdp_plots(model=rf.ncv,xvar='cesd_score',label = 'depressive symptoms score')
pdp_rf10<-pdp_plots(model=rf.ncv,xvar='cigs_smoked',label = 'cigarettes smoked')

library(ggpubr)

ggarrange(pdp_rf1,pdp_rf2,pdp_rf3,pdp_rf4,pdp_rf5,pdp_rf6,pdp_rf7,pdp_rf8,pdp_rf9,pdp_rf10,
          ncol = 5, nrow = 2)

###2. Neural Network
pdp_nn1<-pdp_plots(model=nn.ncv,xvar='cannabis',label = 'cannabis use')
pdp_nn2<-pdp_plots(model=nn.ncv,xvar='suicidal',label = 'suicidal thinking score')
pdp_nn3<-pdp_plots(model=nn.ncv,xvar='cocaine',label = 'cocaine use',binary=T)
pdp_nn5<-pdp_plots(model=nn.ncv,xvar='ace',label = 'adverse childhood experience')

pdp_nn4<-pdp_plots(model=nn.ncv,xvar='house_income',label = 'household income')
pdp_nn6<-pdp_plots(model=nn.ncv,xvar='covid',label = 'covid impact on smoking')
pdp_nn7<-pdp_plots(model=nn.ncv,xvar='rural_city',label = 'living environment')

ggarrange(pdp_nn1,pdp_nn2,pdp_nn3,pdp_nn5,
          ncol = 2, nrow = 2)

ggarrange(pdp_nn4,pdp_nn6,pdp_nn7,
          ncol = 1, nrow = 3)

#4. EN
pdp_EN1<-pdp_plots(model=model.EN,xvar='GHB1',label = 'GHB use',binary=T)
pdp_EN2<-pdp_plots(model=model.EN,xvar='cannabis2',label = 'Cannabis use - sometimes',binary=T)
pdp_EN3<-pdp_plots(model=model.EN,xvar='cocaine1',label = 'Cocaine use',binary=T)
pdp_EN4<-pdp_plots(model=model.EN,xvar='cannabis1',label = 'Cannabis use - frequent',binary=T)
pdp_EN5<-pdp_plots(model=model.EN,xvar='residence2',label = 'Time in Canada - 3-5 years',binary=T)
pdp_EN6<-pdp_plots(model=model.EN,xvar='rural_city2',label = 'Medium city (30k - 99k people)',binary=T)
pdp_EN7<-pdp_plots(model=model.EN,xvar='curr_orient23',label = 'Sexual orientation - gay',binary=T)
pdp_EN8<-pdp_plots(model=model.EN,xvar='con_eating1',label = 'Eating disorders diagnosis',binary=T)
pdp_EN9<-pdp_plots(model=model.EN,xvar='plan_quit2',label = 'Quit smoking cigarettes',binary=T)
pdp_EN10<-pdp_plots(model=model.EN,xvar='poc1',label = 'Person of colour',binary=T)

ggarrange(pdp_EN1,pdp_EN2,pdp_EN3,pdp_EN4,pdp_EN5,
          pdp_EN6,pdp_EN7,pdp_EN8,pdp_EN9,pdp_EN10,
          ncol = 5, nrow = 2)

####Interaction

###1. Random forest interactions
##See interactions between all the demographic vars, as well those with the top 10 vars

demovars<-c('age','curr_orient2','gender','education','house_income')
top10.rf<-impframe.rf.ncv$Variable[!(impframe.rf.ncv$Variable %in% demovars)]
rf.int<-c(demovars,top10.rf)

start.time <- Sys.time()
interact.rf <- vint(rf.ncv, feature_names = rf.int)
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
print(interact.rf,n=91)
# > time.taken
# Time difference of 4.16 hours

#2 Neural network
demovars<-c('age','curr_orient2','gender','education','house_income')

top10.nn<-impframe.nn.ncv$Variable[!(impframe.nn.ncv$Variable %in% demovars)]
top10.nn2<-gsub("[[:digit:]]", "",top10.nn)
nn.int1<-c(demovars,top10.nn2)
#Remove duplicates
nn.int<-nn.int1[!duplicated(nn.int1)]

start.time <- Sys.time()
interact.nn <- vint(nn.ncv, feature_names = nn.int)
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 4.64 mins

print(interact.nn,n=67)

###3. LASSO

x.matrix.int.all <- model.matrix(alcohol_abuse ~ .^2,df2)[,-1]
y.all<-df2$alcohol_abuse

folds <- rep_len(1:outerfold, nrow(x.matrix.int.all))
folds <- sample(folds, nrow(x.matrix.int.all))
nested.cv.lasso.int <- list(auc=NULL, lambda=NULL)

for(i in 1:outerfold){
  fold <- which(folds == i)
  cv.lasso.int <- cv.glmnet(x.matrix.int.all,y.all,alpha=1,family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
  pred <- as.numeric(predict(cv.lasso.int,newx=x.matrix.int.all[fold,],s=cv.lasso.int$lambda.min,type="class"))
  aucroc<-roc(pred,as.numeric(y.all[fold]))
  nested.cv.lasso.int$auc <- c(nested.cv.lasso.int$auc,aucroc$auc[1])
  nested.cv.lasso.int$lambda <- c(nested.cv.lasso.int$lambda, cv.lasso.int$lambda.min)
}

lambda.min.int.ncv<-nested.cv.lasso.int$lambda[which.max(nested.cv.lasso.int$auc)]
lasso.ncv.int<- glmnet(x.matrix.int.all,y.all,family="binomial",alpha=1,lambda=lambda.min.int.ncv)
coef.min1 <- coef(lasso.ncv.int, s ="lambda.min.int.ncv")
varimp.lasso.ncv.int<-vip(lasso.ncv.int)
impframe.lasso.ncv.int <- data.frame(varimp.lasso.ncv.int$data)

#####4. Elastic Net Interaction

alpha <- seq(0.01, 0.99, 0.02)
bestout.int <- list(a=NULL, lambda=NULL, auc=NULL)
bestinner.int <- list(a=NULL, lambda=NULL, auc=NULL)

start.time <- Sys.time()
for(i in 1:outerfold){
  fold <- which(folds == i)
  for (j in 1:length(alpha)) 
  {
    cvg <- cv.glmnet(x.matrix.int.all,y.all,alpha=alpha[j],family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
    bestinner.int$a <- c(bestinner.int$a, alpha[j])
    bestinner.int$lambda <- c(bestinner.int$lambda, cvg$lambda.min)
    bestinner.int$auc <- c(bestinner.int$auc, max(cvg$cvm))
  }
  index <- which.min(bestinner.int$auc)
  best_alpha.int <- bestinner.int$a[index]
  best_lambda.int <- bestinner.int$lambda[index]
  bestout.int$a[i] <- best_alpha.int
  bestout.int$lambda[i] <- best_lambda.int
  cvg.EN <- glmnet(x.matrix.int.all,y.all,alpha=best_alpha.int,lambda=best_lambda.int,family = "binomial", nfolds=innerfold,type.measure = "auc",subset=-fold)
  pred <- predict(cvg.EN,newx=x.matrix.int.all[fold,],s=cvg.EN$lambda.min,type="class")  
  aucroc<-roc(pred,as.numeric(y.all[fold]))
  bestout.int$auc[i] <- aucroc$auc[1]
  bestinner.int <- list(a=NULL, lambda=NULL, cvm=NULL)
}  
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 1.58 hours

bestout.int
index <- which.max(bestout.int$auc)
best_alpha.int <- bestout.int$a[index]
best_lambda.int <- bestout.int$lambda[index]
model.EN.int <- glmnet(x.matrix.int.all,y.all, alpha = best_alpha.int, family = "binomial",lambda = best_lambda.int)
varimp.EN.ncv.int<-vip(model.EN.int)
impframe.EN.ncv.int <- data.frame(varimp.EN.ncv.int$data)

interact.rf.10<-interact.rf.save[1:10,]
interact.nn.10<-interact.nn.save[1:10,]
interact.EN.10 <- impframe.EN.ncv.int %>%
  rename(Interaction=Importance,Variables=Variable) %>%
  dplyr::select(Variables,Interaction)

interact.rf.10<-interact.rf.10[order(-interact.rf.10$Interaction),]
interact.nn.10<-interact.nn.10[order(-interact.nn.10$Interaction),]
interact.EN.10<-interact.EN.10[order(-interact.EN.10$Interaction),]

interaction_plot <- function(dat,upper) {
plot<- ggplot(data=dat, aes(x=reorder(Variables,Interaction),y=Interaction)) + 
    geom_bar(stat="identity") +
    scale_x_discrete(name="Variable Name") +
    scale_y_continuous(name="Interaction Score", limits = c(0, upper)) +
    coord_flip() +
    theme_minimal()
return(plot)
} 
rf.int.plot<-interaction_plot(dat=interact.rf.10,upper=0.02)
nn.int.plot<-interaction_plot(dat=interact.nn.10,upper=0.12)
EN.int.plot<-interaction_plot(dat=interact.EN.10,upper=0.25)

ggarrange(rf.int.plot, nn.int.plot, EN.int.plot,
          labels = c("Random Forest", "Neural Network",  "Elastic Net"),
          ncol = 3, nrow = 1,
          font.label=list(size=11,color="red"))

#Partial dependence plot for the important interaction from neural network
pd1 <- partial(nn.ncv, pred.var = c("cocaine", "rural_city"),which.class=2)
pd1$rural_city <- factor(pd1$rural_city, levels = c(1,2,3,4), 
                          labels= c('Large urban', 'Medium city', 'Small city', 'Rural area'))
pd1$cocaine <- factor(pd1$cocaine, levels = c(0,1), 
                             labels= c('No','Yes'))
pd1_plot <- plotPartial(pd1,xlab= 'cocaine use', ylab="PD")
print(pd1_plot)

pd2 <- partial(nn.ncv, pred.var = c("cannabis", "suicidal"),which.class=2)
pd2$cannabis <- factor(pd2$cannabis, levels = c(0,1), 
                      labels= c('No cannabis use','cannabis use'))
pd2_plot <- plotPartial(pd2,xlab= 'Suicidality', ylab="PD")
print(pd2_plot)

ggarrange(pd1_plot, pd2_plot, ncol = 2,font.label=list(size=10),
    labels = c("Figure 4a. Cocaine Use x Rurality", "Figure 4b. Cannabis Use x Suicidality"))


######Old logistic regression
library("stringr")
log.model <- glm(y.all ~ x.matrix.all, family = 'binomial')
summary<-summary(log.model)
coeffs<-abs(coef(log.model))
low<-coeffs-(1.96*summary$coefficients[,2])
upp<-coeffs+(1.96*summary$coefficients[,2])
coeffsord<-coeffs[order(-coeffs)][1:11]
coefdat<-as.data.frame(coeffsord)
coefdat$var<-rownames(coefdat)
LR_res<-as.data.frame(cbind(low=low[names(low) %in% rownames(coefdat)],
                            coef=coef(log.model)[rownames(summary$coefficients) %in% rownames(coefdat)],
                            upper=upp[names(upp) %in% rownames(coefdat)]))
LR_res$var<-str_replace(rownames(LR_res),"x.matrix.all","")

####Bootstrap
library(boot)
library(tidyverse)
library(rsample)

#Take 100 samples
B = 100
bt_samples <- bootstraps(df2, times = B)
nfeature = 10

#Given that we already optimized it, we can use the parameters from the NCV to be bootstrapped

##1. Random forest
implist.rf<-vector("list",nfeature)

start.time <- Sys.time()
for (i in 1:B) {
  x<-analysis(bt_samples$splits[[i]]) %>% as_tibble()
  rf.boot<- train(alcohol_abuse ~. ,data=x,method="rf",tuneGrid=rf.grid.ncv,ntree=ntree.ncv)
  vip.rf.boot<-vip(rf.boot,scale=TRUE,num_features=ncol(x))
  imp.rf.boot <- data.frame(vip.rf.boot$data)
  for (j in 1:nfeature) {
    implist.rf[[j]] <- c(implist.rf[[j]],imp.rf.boot$Importance[imp.rf.boot$Variable %in% impframe.rf.ncv$Variable[j]])
  }
}
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 13.93 hours

rf.ci<-setNames(data.frame(matrix(ncol = 3, nrow = nrow(impframe.rf.ncv))), c("lower", "estimate", "upper"))
rownames(rf.ci) <- impframe.rf.ncv$Variable
for (k in 1:nrow(impframe.rf.ncv)) {
  rf.ci$lower[k]<-quantile(implist.rf[[k]],.025)
  rf.ci$estimate[k]<-impframe.rf.ncv$Importance[k]
  rf.ci$upper[k]<-quantile(implist.rf[[k]],.975)
}
rf.ci
rf.ci$variable <- rownames(rf.ci)

rf.ci$variable<-reorder(rf.ci$variable, rf.ci$estimate)
rf.ci$mean<-unlist(lapply(implist.rf,mean))

rf.plot <- ggplot(rf.ci) + 
  geom_point(aes(x=variable,y=estimate,shape='estimate'),size = 4,position=position_dodge(.5)) +
  geom_point(aes(x=variable,y=mean,shape='mean'),size = 4,position=position_dodge(.5)) +
  scale_shape_discrete(guide="none") +
  geom_errorbar(aes(x=variable,ymin=lower, ymax=upper),position=position_dodge(.5)) +
  scale_x_discrete(name="Variable Name") +
  scale_y_continuous(name="Variable Importance Scores (%)", limits = c(0, 100)) +
  coord_flip() +
  theme_minimal() 
rf.plot

##2. Neural network
implist.nn<-vector("list",nfeature)
start.time <- Sys.time()
for (i in 1:B) {
  x<-analysis(bt_samples$splits[[i]]) %>% as_tibble()
  nn.boot <- train(alcohol_abuse ~.,data=x,method='nnet',tuneGrid=nn.grid.ncv,maxit=1000,MaxNWts = numWts,trace = FALSE,preProc = c("center","scale"))   
  vip.nn.boot<-vip(nn.boot,scale=TRUE,num_features=ncol(x))
  imp.nn.boot <- data.frame(vip.nn.boot$data)
  for (j in 1:nfeature) {
    implist.nn[[j]] <- c(implist.nn[[j]],imp.nn.boot$Importance[imp.nn.boot$Variable %in% impframe.nn.ncv$Variable[j]])
  }
}
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken
# > time.taken
# Time difference of 38.77 mins

nn.ci<-setNames(data.frame(matrix(ncol = 4, nrow = nrow(impframe.nn.ncv))), c("variable","lower", "estimate", "upper"))
nn.ci$variable <- impframe.nn.ncv$Variable
for (k in 1:nrow(impframe.nn.ncv)) {
  nn.ci$lower[k]<-quantile(implist.nn[[k]],.025)
  nn.ci$estimate[k]<-impframe.nn.ncv$Importance[k]
  nn.ci$upper[k]<-quantile(implist.nn[[k]],.975)
}
nn.ci

nn.ci$variable <- rownames(nn.ci)
nn.ci$variable<-reorder(nn.ci$variable, nn.ci$estimate)
nn.ci$mean<-unlist(lapply(implist.nn,mean))

nn.plot <- ggplot(nn.ci) + 
  geom_point(aes(x=variable,y=estimate,shape='estimate'),size = 4,position=position_dodge(.5)) +
  geom_point(aes(x=variable,y=mean,shape='mean'),size = 4,position=position_dodge(.5)) +
  scale_shape_discrete(guide="none") +
  geom_errorbar(aes(x=variable,ymin=lower, ymax=upper),position=position_dodge(.5)) +
  scale_x_discrete(name="Variable Name") +
  scale_y_continuous(name="Variable Importance Scores (%)", limits = c(0, 100)) +
  coord_flip() +
  theme_minimal() 
nn.plot

###3. LASSO
implist.lasso<-vector("list",nfeature)

for (i in 1:B) {
  x<-analysis(bt_samples$splits[[i]]) %>% as_tibble()
  x.matrix.all.boot <- model.matrix(alcohol_abuse ~ .,x)[,-1]
  y.boot<-x$alcohol_abuse
  lasso.boot<- glmnet(x.matrix.all.boot,y.boot,family="binomial",alpha=1,lambda=lambda.min.ncv)
  vip.lasso.boot<-vip(lasso.boot,scale=TRUE,num_features=ncol(x.matrix.all.boot))
  imp.lasso.boot <- data.frame(vip.lasso.boot$data)
  for (j in 1:nfeature) {
    implist.lasso[[j]] <- c(implist.lasso[[j]],imp.lasso.boot$Importance[imp.lasso.boot$Variable %in%  impframe.lasso.ncv$Variable[j]])
  }
}
implist.lasso

lasso.ci<-setNames(data.frame(matrix(ncol = 4, nrow = nrow(impframe.lasso.ncv))), c("variable","lower", "estimate", "upper"))
lasso.ci$variable <- impframe.lasso.ncv$Variable
for (k in 1:nrow(impframe.lasso.ncv)) {
  lasso.ci$lower[k]<-quantile(implist.lasso[[k]],.025)
  lasso.ci$estimate[k]<-impframe.lasso.ncv$Importance[k]
  lasso.ci$upper[k]<-quantile(implist.lasso[[k]],.975)
}
lasso.ci

lasso.ci$variable<-reorder(lasso.ci$variable, lasso.ci$estimate)
lasso.ci$mean<-unlist(lapply(implist.lasso,mean))

lasso.plot <- ggplot(lasso.ci) + 
  geom_point(aes(x=variable,y=estimate,shape='estimate'),size = 4,position=position_dodge(.5)) +
  geom_point(aes(x=variable,y=mean,shape='mean'),size = 4,position=position_dodge(.5)) +
  geom_errorbar(aes(x=variable,ymin=lower, ymax=upper),position=position_dodge(.5)) +
  scale_shape_discrete(guide="none") +
  scale_x_discrete(name="Variable Name") +
  scale_y_continuous(name="Variable Importance Scores (%)", limits = c(0, 100)) +
  coord_flip() +
  theme_minimal() 
lasso.plot

###4. Elastic Net (EN)
implist.EN<-vector("list",nfeature)

for (i in 1:B) {
  x<-analysis(bt_samples$splits[[i]]) %>% as_tibble()
  x.matrix.all.boot <- model.matrix(alcohol_abuse ~ .,x)[,-1]
  y.boot<-x$alcohol_abuse
  EN.boot <- glmnet(x.matrix.all.boot,y.boot, alpha = best_alpha, family = "binomial",lambda = best_lambda)
  vip.EN.boot<-vip(EN.boot,scale=TRUE,num_features=ncol(x.matrix.all.boot))
  imp.EN.boot <- data.frame(vip.EN.boot$data)
  for (j in 1:nfeature) {
    implist.EN[[j]] <- c(implist.EN[[j]],imp.EN.boot$Importance[imp.EN.boot$Variable %in%  impframe.EN.ncv$Variable[j]])
  }
}
implist.EN

EN.ci<-setNames(data.frame(matrix(ncol = 4, nrow = nrow(impframe.EN.ncv))), c("variable","lower", "estimate", "upper"))
EN.ci$variable <- impframe.EN.ncv$Variable
for (k in 1:nrow(impframe.EN.ncv)) {
  EN.ci$lower[k]<-quantile(implist.EN[[k]],.025)
  EN.ci$estimate[k]<-impframe.EN.ncv$Importance[k]
  EN.ci$upper[k]<-quantile(implist.EN[[k]],.975)
}
EN.ci

EN.ci$variable<-reorder(EN.ci$variable, EN.ci$estimate)
EN.ci$mean<-unlist(lapply(implist.EN,mean))

EN.plot <- ggplot(EN.ci) + 
  geom_point(aes(x=variable,y=estimate,shape='estimate'),size = 4,position=position_dodge(.5)) +
  geom_point(aes(x=variable,y=mean,shape='mean'),size = 4,position=position_dodge(.5)) +
  geom_errorbar(aes(x=variable,ymin=lower, ymax=upper),position=position_dodge(.5)) +
  scale_shape_discrete(labels = c("NCV estimate", "Mean bootstrap")) +
  scale_x_discrete(name="Variable Name") +
  scale_y_continuous(name="Variable Importance Scores (%)", limits = c(0, 100)) +
  coord_flip() +
  theme_minimal() 
EN.plot
  
ggarrange(rf.plot, nn.plot, lasso.plot, EN.plot,
          labels = c("Random Forest", "Neural Network", "LASSO", "Elastic Net"),
          ncol = 2, nrow = 2,
          font.label=list(size=12,color="red"))