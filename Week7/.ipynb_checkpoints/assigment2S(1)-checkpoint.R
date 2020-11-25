library(nutshell)
data(spambase)
library(sampling)
table(spambase$is_spam)
spambase.strata <- strata(spambase,stratanames=c("is_spam"),size=c(1269,1951),method="srswor")
names(spambase.strata)
spambase.training <- spambase[rownames(spambase) %in% spambase.strata$ID_unit,]
spambase.validation <- spambase[!(rownames(spambase) %in% spambase.strata$ID_unit),]
nrow(spambase.training)
nrow(spambase.validation)

library(nnet)

#your code start
spam.nnet <- < ---- insert code here  ---- >

#your code end

table(actual=spambase.training$is_spam, predicted=predict(spam.nnet, type="class"))

table(actual=spambase.validation$is_spam, predicted=predict(spam.nnet, newdata=spambase.validation,type="class"))



