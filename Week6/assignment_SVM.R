# SVD aasigment
# R 3.
# To use SVM in R, we have a package e1071. The package is not preinstalled, 
# hence one needs to run the line “install.packages(“e1071”) to install the package
# and then import the package contents using the library command. 
# The syntax of svm package is quite similar to linear regression
# documentation is at https://www.rdocumentation.org/packages/e1071/versions/1.6-8/topics/svm

library("e1071")

#This part IRSI serves as an example how to use R code for SVM

attach(iris)

#Divide Iris data to x (containt the all features) and y only the classes
x <- subset(iris, select=-Species)
y <- Species

#Create SVM Model, by default kernel is radial exp(-gamma*|u-v|^2), and penalty C for missclassification is 1.
#we will use first linear

svm_model <- svm(x,y, kernel="linear")
summary(svm_model)

#Run Prediction 
pred <- predict(svm_model,x)
#Linear
tl<-table(pred,y)


svm_model <- svm(Species ~ ., data=iris)
summary(svm_model)

#Run Prediction 
pred <- predict(svm_model,x)
#Radila SVM
tr<-table(pred,y)


# Tuning SVM to find the best cost and gamma 
# The optimal values are obtained using the tune.svm function (also available in e1071),
# which essentially builds models for multiple combinations of parameter values and selects the best.
#------------------------------------------
svm_tune <- tune(svm, train.x=x, train.y=y, 
              kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)

#After you find the best cost and gamma, you can create svm model again and try to run again
svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
summary(svm_model_after_tune)

#Run Prediction again with new tuned model
pred <- predict(svm_model_after_tune,x)
ttune<-table(pred,y)


#See the confusion matrix result of prediction, using command table to compare the result of 
#SVM prediction linear, radila and tuned

tl
tr
ttune

##########################################################################################
###############Assigment starts here 

dataset <- read.csv('wdbc.csv',head = FALSE)
index <- 1:nrow(dataset)

testindex <- sample(index, trunc(length(index)*30/100))

testset <- dataset[testindex,]

trainset <- dataset[-testindex,]
x_train <- trainset[,3:31]
y_train <- trainset[,2]
x_test <- testset[,3:31]
y_test <- testset[,2]


#Assigment : find the optimal SVM for calssification (try linear and  radial)
#start code
svm_linear <-  ## edit here
summary(svm_linear)
pred_linear <- predict(svm_linear,x_test)
t_linear <- table(pred_linear,y_test)

#Radial
svm_radial <- ## edit here
summary(svm_radial)
pred_radial <-  ## edit here
t_radial<- ## edit here

#Tuning
svm_tune <- tune(svm, train.x=x_train, train.y=y_train, 
  
t_linear  
t_radial

#end code



