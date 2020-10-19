set.seed(1234567890)
library("neuralnet")
dataset <- read.csv("creditset.csv")
head(dataset)
## extract a set to train the NN
trainset <- dataset[1:800, ]

## select the test set
testset <- dataset[801:2000, ]

## build the neural network (NN)
creditnet <- neuralnet(default10yr ~ LTI + age, trainset, hidden = 4, lifesign = "minimal", 
    linear.output = FALSE, threshold = 0.1)
## hidden: 4    thresh: 0.1    rep: 1/1    steps:    7266   error: 0.79202  time: 9.32 secs

## plot the NN
plot(creditnet, rep = "best")


## test the resulting output
temp_test <- subset(testset, select = c("LTI", "age"))

creditnet.results <- compute(creditnet, temp_test)
head(temp_test)

results <- data.frame(actual = testset$default10yr, prediction = creditnet.results$net.result)
results[100:115, ]
results$prediction <- round(results$prediction)
results[100:115, ]



