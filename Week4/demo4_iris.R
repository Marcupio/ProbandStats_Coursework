# neuralDemo42.R
# R 3.3.0

library(nnet)

cat("\nBegin neural network using nnet demo \n\n")

cat("Original Fisher's Iris data: \n\n");
origdf <- read.table("IrisData.txt", header=TRUE, sep=",")
write.table(origdf[1:2,], col.names=TRUE, sep="\t", quote=FALSE)
cat(". . . \n")
write.table(origdf[51:52,], col.names=FALSE, sep="\t", quote=FALSE)
cat(". . . \n")
write.table(origdf[101:102,], col.names=FALSE, sep="\t", quote=FALSE)
cat("\n")

col1 <- origdf$Sepal.length
col2 <- origdf$Sepal.width
col3 <- origdf$Petal.length
col4 <- origdf$Petal.width
col5 <- factor(c(rep("s",50), rep("c",50), rep("v",50)))

irisdf <- data.frame(col1, col2, col3, col4, col5)
names(irisdf) <- c("SepLen", "SepWid", "PetLen", "PetWid", "Species")

cat("Factor-ized data: \n\n");
write.table(irisdf[1:2, ], col.names=TRUE, sep="\t", quote=FALSE)
cat(". . . \n")
write.table(irisdf[51:52, ], col.names=FALSE, sep="\t", quote=FALSE)
cat(". . . \n")
write.table(irisdf[101:102, ], col.names=FALSE, sep="\t", quote=FALSE)
cat("\n")

# configure the training sample
set.seed(1)
sampidx <- c(sample(1:50,10), sample(51:100,10), sample(101:150,10))



# create and train nn
cat("\nCreating and training a neural network . . \n")
mynn <- nnet(Species ~ SepLen + SepWid + PetLen + PetWid, data=irisdf, subset = sampidx, size=2, decay=1.0e-5, maxit=500)

# evaluate accuracy of nn model with a confusion matrix
cm <- table(irisdf$Species[-sampidx], predict(mynn, irisdf[-sampidx, ], type="class"))
actual <- irisdf$Species[-sampidx]

preds <- predict(mynn, irisdf[-sampidx, ], type="class")
cm <- table(actual, preds)
cat("\nConfusion matrix for resulting nn model is: \n")
print(cm)


#######
### Classification with logistic regression and family binomial 



iris<-irisdf  #origdf
x <-  iris[sampidx,]

# Virginica
x$virginica <- x$Species == "v"
x$Species <-NULL
plot(x, col=x$virginica+1)

model <- glm(virginica ~ .,family = binomial(logit), data=x)
summary(model)
 
#Aboiut the warning: glm.fit: fitted probabilities numerically 0 or 1 occurred 
#means that the data is possibly linearely separable.
x<-irisdf[-sampidx, ]
x$virginica <- x$Species == "v"

prv <- predict(model, x, type="response")


#Setosa
x$setosa <- x$Species == "s"
model <- glm(setosa ~ .,family = binomial(logit), data=x)
summary(model)
prs <- predict(model, x, type="response")
# Versicolor
x$versicolor <- x$Species == "c"
model <- glm(versicolor ~ .,family = binomial(logit), data=x)
summary(model)
prc <- predict(model, x, type="response")


cat("\n Versicolor Test Result")
table(actual=x$versicolor, predicted=prc>.5)
cat("\n Setosa Test Result")
table(actual=x$setosa, predicted=prs>.5)
cat("\n Virginica Test Result")
table(actual=x$virginica, predicted=prv>.5)


cat("\n Neural Network Confusion Matrix Test Result")
cm




