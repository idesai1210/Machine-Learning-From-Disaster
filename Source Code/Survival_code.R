#install.packages("randomForest")
#install.packages("rpart")
#install.packages("adabag",dependencies = TRUE)
# install.packages("e1071")
# install.packages("nnet")

set.seed(1)

library(adabag)
library(class)
library(randomForest)
library(rpart)
library(nnet)
library(e1071)


#loading train and test data files
trainingData <- read.csv("C:\\Users\\silpy\\Desktop\\ML Project\\train.csv")
testData <- read.csv("C:\\Users\\silpy\\Desktop\\ML Project\\test.csv")

#Select only the important features
feature_labels <- c("Pclass","Age","Sex","Parch","SibSp","Fare","Embarked")

# create a subset of train data based on the important features
new_training_data <- trainingData[,feature_labels]

# create a subset of train data based on the important features
new_test_data <- testData[,feature_labels]


#handling nulls in the subset created(pre-processing)
new_training_data$Age[is.na(new_training_data$Age)] <- 0
new_training_data$Fare[is.na(new_training_data$Fare)] <- median(new_training_data$Fare,na.rm=TRUE)
new_training_data$Embarked[new_training_data$Embarked==""] <- "S"
new_training_data$Sex      <- as.factor(new_training_data$Sex)
new_training_data$Embarked <- as.factor(new_training_data$Embarked)

new_training_data$Embarked <- droplevels(new_training_data$Embarked)

new_test_data$Age[is.na(new_test_data$Age)] <- 0
new_test_data$Fare[is.na(new_test_data$Fare)] <- median(new_test_data$Fare,na.rm=TRUE)
new_test_data$Embarked[new_test_data$Embarked==""] <- "S"
new_test_data$Sex      <- as.factor(new_test_data$Sex)
new_test_data$Embarked <- as.factor(new_test_data$Embarked)

#create output csv file

Survival_output <- data.frame(PassengerId = testData$PassengerId,Name = testData$Name)

#Random Forest Model
rfModel <- randomForest(new_training_data,as.factor(trainingData$Survived),ntree=1000)
Survival_output$Survived_RF <- predict(rfModel,new_test_data)

#Decision Tree
dt_formula <- trainingData$Survived~ Sex + Age + Pclass + Fare + Parch 
dtModel <- rpart(dt_formula, data = new_training_data, minbucket = 5, method="class")
Survival_output$Survived_DT <- predict(dtModel,new_test_data, type="class") 

#SVM
svmModel <- svm(trainingData$Survived~ ., kernel="linear", data = new_training_data,type="C-classification")
Survival_output$Survived_SVM <- predict(svmModel,new_test_data, type="class",decision.values = TRUE)

#Neural Net
nnModel <- nnet(as.factor(trainingData$Survived)~., new_training_data,size=8,maxit=1000,decay=0.001,trace = FALSE)
Survival_output$Survived_NN <- predict(nnModel,new_test_data,type="class")

#kNN
fitctrl <- trainControl(method="repeatedcv",number = 10,repeats = 3)
knnModel <- train(as.factor(trainingData$Survived)~., data = new_training_data, method = "knn", trControl = fitctrl, preProcess = c("center","scale"), tuneLength = 20)
Survival_output$Survived_KNN <- predict(knnModel,new_test_data)


# Find accurate classification for each classifier
# Take row mean value
Survival_output[,3:7] <- lapply(Survival_output[,3:7], function(x) as.numeric(as.character(x)))
Survival_output$Survival_count <- round(rowMeans(Survival_output[,3:7],na.rm=TRUE))

#write output file
write.csv(Survival_output,file="C:\\Users\\silpy\\Desktop\\ML Project\\Survival_op.csv",row.names = FALSE)

#predict accuracy for the classifiers based on mean output
Accuracy_DT <- (mean(Survival_output$Survived_DT == Survival_output[ ,8])) * 100
Accuracy_RF <- (mean(Survival_output$Survived_RF == Survival_output[ ,8])) * 100
Accuracy_SVM <- (mean(Survival_output$Survived_SVM == Survival_output[ ,8])) * 100
Accuracy_NN <- (mean(Survival_output$Survived_NN == Survival_output[ ,8])) * 100
Accuracy_KNN <- (mean(Survival_output$Survived_KNN == Survival_output[ ,8])) * 100

cat("Accuracy for Decision Tree is:",Accuracy_DT,"% \n")
cat("Accuracy for Random Forest is:",Accuracy_RF,"% \n")
cat("Accuracy for SVM is:",Accuracy_SVM,"% \n")
cat("Accuracy for Neural Net is:",Accuracy_NN,"% \n")
cat("Accuracy for kNN is:",Accuracy_KNN,"% \n")

plot(dtModel, uniform=TRUE, main="Decision Tree for Survival on Titanic")
text(dtModel, use.n=TRUE, all=TRUE, cex=.8)