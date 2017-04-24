library(dplyr)
library(caret)
library(reshape2)
library(stringr)
library(ggplot2)
library(data.table)
library(doParallel)
library(foreach)
library(impute)
library(class)

#setwd("~/R_repository/kaggle_titanic")



# load data
test_set = read.csv('test.csv')
train_set = read.csv('train.csv')


# preprocessing train set
train_set = train_set %>% select(-Name, -Ticket, -Cabin)
train_set = train_set[complete.cases(train_set),]

non_empty = rep(TRUE, nrow(train_set))
for(i in 1:ncol(train_set)){non_empty = non_empty & str_wrap(as.character(train_set[,i])) != ""}
train_set = train_set[non_empty,]

train_set$male = as.numeric(train_set$Sex == 'male')
embarked = data.frame(PassengerId = train_set$PassengerId, Embarked = train_set$Embarked, count = 1)
embarked_dummy = dcast(embarked, PassengerId~Embarked, fun.aggregate = sum, value.var = "count")

train_set = merge(train_set, embarked_dummy, by = "PassengerId")
train_set$log_fare = log(train_set$Fare + 1)

train_set = train_set %>% select(-PassengerId, -Sex, -Embarked, -Fare)

train_set_X = train_set %>% select(-Survived)
train_set_Y = train_set$Survived

# select variables with random forest 
total_cores <- detectCores()
cl <- makeCluster(total_cores)
registerDoParallel(cl)
control = rfeControl(functions=rfFuncs, method="cv", number = 50)
rfe_results = rfe(x = train_set_X, y = train_set_Y, rfeControl=control, sizes = c(2,4,6,8,10,20,40,60))
stopCluster(cl)

input_vars = rownames(varImp(rfe_results))

train_set_X = subset(train_set_X, select = input_vars)

cor(train_set_X)

################  replace fare and Pclass with PCA 
preProcRf = preProcess(subset(train_set_X, select = c(log_fare, Pclass)), method="pca", pcaComp = 1)
value_pca = predict(preProcRf, subset(train_set_X, select = c(log_fare, Pclass)))
colnames(value_pca) = "value"

train_set_X = cbind(train_set_X, value_pca) %>% select(-Pclass, -log_fare)

min_values = apply(train_set_X, 2, min)
max_values = apply(train_set_X, 2, max)

train_set_X = as.data.frame(apply(train_set_X,2, function(x) (x - min(x))/(max(x) - min(x)) ))


ggplot(train_set_X, aes(value, Age)) + geom_point(aes(colour =train_set_Y))


# logistic regression cross validation:
lr_xfit = function(iteration){
  library(caret)
  train_set_index <- sample(1:nrow(train_set_X), round(0.9 * nrow(train_set_X)))
  train_set_index <- (1:nrow(train_set_X)) %in% train_set_index
  #train_set <- input_data[train_set_index,]
  #test_set <- input_data[!train_set_index, ]
  
  y = train_set_Y
  
  #logregr.poly <- function(){
  alpha = 0.01
  
  m = length(y)
  X = as.matrix(cbind(x0_norm = rep(1,m),train_set_X))
  X_test = X[!train_set_index,]
  y_test = y[!train_set_index]
  
  X = X[train_set_index,]
  Y = y[train_set_index]
  m = length(Y)
  n = ncol(X)
  THETA = rep(0, n)
  
  H_theta = function(i){
    1/(1+exp(-(t(THETA) %*% X[i,]) ) )
  }
  
  J_theta <- function(){ -(1/m)*(  sum(  Y*log(mapply(H_theta, 1:m))  +   (1-Y)*log(1 - (mapply(H_theta, 1:m)) )   ) ) }
  
  gradient_j <- function(j){sum((mapply(H_theta, 1:m) - Y)*X[,j]) }
  
  #error_summary <- data.table()
  
  for(i in 1:150){
    NEW_THETA <- THETA - alpha*mapply(gradient_j, 1:n)
    THETA <- NEW_THETA
    #print(J_theta()) 
    cost = J_theta()
    #error_summary <- rbind(error_summary, data.table(i, cost))
  }
  y_poly = mapply(function(i) 1/(1+exp(-(t(THETA) %*% X_test[i,]) ) ), 1:nrow(X_test))>0.5
  
  return(as.numeric(confusionMatrix(as.logical(y[!train_set_index]), y_poly)[[3]][1]))
}


total_cores <- detectCores()
cl <- makeCluster(total_cores)
registerDoParallel(cl)
log_regr_x_val =  foreach(i=1:30, .combine = "c") %dopar% lr_xfit()
stopCluster(cl)

mean(log_regr_x_val)
#hist(log_regr_x_val)



# knn cross validation
knn_xfit = function(k = 3){
  library(caret)
  library(class)
  
  train_set_index <- sample(1:nrow(train_set_X), round(0.9 * nrow(train_set_X)))
  train_set_index <- (1:nrow(train_set_X)) %in% train_set_index

  y = train_set_Y
  X = train_set_X
  X_test = X[!train_set_index,]
  y_test = y[!train_set_index]
  X = X[train_set_index,]
  Y = y[train_set_index]
  
  y_poly = as.logical(as.numeric(as.character(knn(X, X_test, Y, k = 4, prob=FALSE))))
  
  return(as.numeric(confusionMatrix(as.logical(y[!train_set_index]), as.logical(y_poly))[[3]][1]))
}


total_cores <- detectCores()
cl <- makeCluster(total_cores)
registerDoParallel(cl)
knn_x_val =  foreach(i=1:30, .combine = "rbind") %dopar% mapply(knn_xfit, 1:50)
stopCluster(cl)
knn_x_val = apply(knn_x_val,1,mean)


optimal_k = which(knn_x_val == max(knn_x_val))
knn_x_val[7]



####  preprocessing test set

test_set = test_set %>% select(-Name, -Ticket, -Cabin)
test_set$male = as.numeric(test_set$Sex == 'male')

embarked = data.frame(PassengerId = test_set$PassengerId, Embarked = test_set$Embarked, count = 1)
embarked_dummy = dcast(embarked, PassengerId~Embarked, fun.aggregate = sum, value.var = "count")
test_set = merge(test_set, embarked_dummy, by = "PassengerId")

test_set = test_set %>% select(-PassengerId, -Sex, -Embarked)

# impute missing values:
imputed_matrix <- impute.knn(data = as.matrix(test_set) ,k = 10, rowmax = 0.5, colmax = 0.5, maxp = 1500, rng.seed=362436069)
imputed_set <- as.data.frame(imputed_matrix$data)

# transform Fare
imputed_set$log_fare = log(imputed_set$Fare + 1)
value_pca_test = predict(preProcRf, subset(imputed_set, select = c(log_fare, Pclass)))
colnames(value_pca_test) = "value"
imputed_set = cbind(imputed_set, value_pca_test)
imputed_set = imputed_set %>% select(male, Age, SibSp, Parch, value)
imputed_normalized_set = as.data.frame(t(apply(imputed_set, 1, function(x) (x - min_values)/(max_values - min_values))))

# normalize

#########  build logistic regr model to predict:

lr = function(){
  library(caret)
  alpha = 0.01
  Y = train_set_Y
  m = length(Y)
  X = as.matrix(cbind(x0_norm = rep(1,m),train_set_X))
  n = ncol(X)
  THETA = rep(0, n)
  
  H_theta = function(i){
    1/(1+exp(-(t(THETA) %*% X[i,]) ) )
  }
  
  #J_theta <- function(){ -(1/m)*(  sum(  Y*log(mapply(H_theta, 1:m))  +   (1-Y)*log(1 - (mapply(H_theta, 1:m)) )   ) ) }
  
  gradient_j <- function(j){sum((mapply(H_theta, 1:m) - Y)*X[,j]) }
  
  for(i in 1:500){
    NEW_THETA <- THETA - alpha*mapply(gradient_j, 1:n)
    THETA <- NEW_THETA
    #print(J_theta()) 
    #cost = J_theta()
    #error_summary <- rbind(error_summary, data.table(i, cost))
  }
  return(list(THETA, data.frame(name = c("intercept", colnames(X)[-1]), value =THETA)))
}

logregr_model = lr()
logregr_model[[2]]
THETA = logregr_model[[1]]

# predict with logistic regression
X_new = as.matrix(cbind(x0_norm = rep(1,nrow(imputed_normalized_set)),imputed_normalized_set))
prediction_test_set = as.numeric(mapply(function(i) 1/(1+exp(-(t(THETA) %*% X_new[i,]) ) ), 1:nrow(X_new)) >= 0.5)
actual_test_set_survived = read.csv('gender_submission.csv')$Survived
confusionMatrix(prediction_test_set, actual_test_set_survived)

# predict with knn
prediction_test_set = knn(as.matrix(train_set_X), imputed_normalized_set, cl = train_set_Y, k = optimal_k, prob=FALSE)
confusionMatrix(prediction_test_set, actual_test_set_survived)

