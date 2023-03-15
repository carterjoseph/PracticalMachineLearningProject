library(dplyr)
library(caret)
library(randomForest)
library(parallel)
library(doParallel)


fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
filename1 <- "pml-training.csv"

if(!file.exists(filename1)) {
          download.file(fileUrl1,destfile=filename1,method="curl")
}

                              #Cleaning the data

data <- read.csv("pml-training.csv", header = TRUE, na.strings = "NA")


data <- select(data, -(1:5))

# If a column has less than 50% NA values, then it will be subset

data <- data[,colMeans(is.na(data)) < .5]

# Removes near zero variance variables
nvz <- nearZeroVar(data)
data <- data[,-nvz]

data$classe <- as.factor(data$classe)


#Splitting the data into training and test sets 

Index <- createDataPartition(data$classe, p = .7,  list = FALSE, times = 1)
training <- data[Index,]
testing <- data[-Index,]




#Setting up parallel processing, and allowing a free CPU core for other tasks

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

# Creating the cross validation, including the parallel processing to speed up computation

RF_ctrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Training the random forest model including the cross validation, and 500 trees.

system.time(RF <- train(classe~., method="rf",data=training, trControl = RF_ctrl, ntrees = 500))

#  Ending parallel processing. clean up the parallel backend used for the parallel computations and reset the R environment to use the default sequential backend for future computations.

stopCluster(cluster)
registerDoSEQ()


#Checking accuracy of training model

RF$finalModel


#Determining accuracy on model against test set

predictiontest <- predict(RF, newdata = testing)
confusionMatrix(predictiontest, testing$classe)




#Loading Assignment test data

fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filename2 <- "pml-testing.csv"
if(!file.exists(filename2)) {
          download.file(fileUrl2,destfile=filename2,method="curl")
}

assignment <- read.csv("pml-testing.csv", header = TRUE, na.strings = "NA")

#Printing results

prediction <- predict(RF, newdata = assignment)

print(prediction)


#Appendix

#N
plot(RF)

#Number of trees vs error rate: The optimal number of trees is between 50 and 100.

plot(x = 1:length(RF$finalModel$err.rate[,1]), y = RF$finalModel$err.rate[,1], 
     xlab = "Number of trees", ylab = "OOB error rate", type = "l")


#Confusion matrix

cm <- confusionMatrix(predictiontest, testing$classe)
cm_df <- as.data.frame(cm$table)
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) + 
          geom_tile() + 
          scale_fill_gradient(low = "white", high = "blue") + 
          labs(x = "Actual", y = "Predicted", fill = "Frequency") +
          geom_text(aes(label = Freq), color = "black", size = 4)

varImp(RF)

            