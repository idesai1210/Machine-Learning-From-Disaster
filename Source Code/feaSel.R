#install.packages(randomForest)

library(ggplot2)
library(randomForest)
set.seed(1)

#read training data file
train <- read.csv("C:\\Users\\silpy\\Desktop\\ML Project\\train.csv")

#handle null data
train$Cabin[as.character(train$Cabin) == ""] <- "C148"
train$Age[is.na(train$Age)] = 0
train$Fare[is.na(train$Fare)] <- median(train$Fare, na.rm=TRUE)
train$Embarked[train$Embarked==""] <- "S"

new_train <- train[,c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")]

#find feature importance
rfModel <- randomForest(new_train, as.factor(train$Survived), ntree=1000, importance=TRUE)
imp <- importance(rfModel,type =1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

#Plot the graph for importance
plot(featureImportance,uniform=TRUE, main="Feature Selection Importance")