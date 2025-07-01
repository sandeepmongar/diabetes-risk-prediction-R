setwd("Set your directory path")

# Install required packages 
# install.packages("tidyverse")
# install.packages("caTools")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("caret")
# install.packages("e1071")
# install.packages("pROC")

# Load libraries
library(tidyverse)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(randomForest)
library(pROC)

# Load dataset
Diabetes <- read.csv("diabetes_data_upload.csv")

# Explore data
summary(Diabetes)
str(Diabetes)
head(Diabetes)

# Check for missing values
cat("Total missing values: ", sum(is.na(Diabetes)), "\\n")

# Encode target variable
Diabetes <- Diabetes %>%
  mutate(class = ifelse(tolower(class) == "positive", "At Risk", "No Risk"))

# Select relevant features
Diabetes <- select(Diabetes,
                   Age, Gender, Polyuria, weakness, Irritability,
                   Polydipsia, Alopecia, class)

Diabetes <- Diabetes %>%
  mutate(Gender = as.factor(Gender),
         Polyuria = as.factor(Polyuria),
         weakness = as.factor(weakness),
         Irritability = as.factor(Irritability),
         Polydipsia = as.factor(Polydipsia),
         Alopecia = as.factor(Alopecia),
         class = as.factor(class),
         Age = as.numeric(Age))

# Split data
set.seed(123)
split <- sample.split(Diabetes$class, SplitRatio = 0.70)
training_data <- subset(Diabetes, split == TRUE)
testing_data <- subset(Diabetes, split == FALSE)


# Naive Bayes Classifier
NB_Model <- naiveBayes(class ~ ., data = training_data)
NB_Predictions <- predict(NB_Model, testing_data, type = "class")

# Confusion Matrix
nb_cm <- confusionMatrix(NB_Predictions, testing_data$class)
print(nb_cm)

# Extract TP, FP, FN, TN from the confusion matrix
cm_table <- nb_cm$table
TP_nb <- cm_table[1,1]
TN_nb <- cm_table[2,2]
FP_nb <- cm_table[2,1]
FN_nb <- cm_table[1,2]

# Metrics
precision_nb <- TP_nb / (TP_nb + FP_nb)
recall_nb <- TP_nb / (TP_nb + FN_nb)
f1_nb <- 2 * (precision_nb * recall_nb) / (precision_nb + recall_nb)

cat("\\nNaive Bayes Metrics:\\n")
cat("Precision:", precision_nb, "\\n")
cat("Recall:", recall_nb, "\\n")
cat("F1 Score:", f1_nb, "\\n")

# ROC Curve for NB
NB_probs <- predict(NB_Model, testing_data, type = "raw")[, "At Risk"]
nb_roc <- roc(testing_data$class, NB_probs, levels = c("No Risk", "At Risk"))
plot(nb_roc, main = "ROC Curve - Naive Bayes", col = "blue")
cat("AUC (NB):", auc(nb_roc), "\\n")


# Random Forest Classifier
RF_Model <- randomForest(class ~ ., data = training_data, proximity = TRUE)
RF_Predictions <- predict(RF_Model, testing_data, type = "class")

# Confusion Matrix
rf_cm <- confusionMatrix(RF_Predictions, testing_data$class)
print(rf_cm)

# Extract TP, FP, FN, TN
cm_rf <- rf_cm$table
TP_rf <- cm_rf[1,1]
TN_rf <- cm_rf[2,2]
FP_rf <- cm_rf[2,1]
FN_rf <- cm_rf[1,2]

# Metrics
precision_rf <- TP_rf / (TP_rf + FP_rf)
recall_rf <- TP_rf / (TP_rf + FN_rf)
f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

cat("\\nRandom Forest Metrics:\\n")
cat("Precision:", precision_rf, "\\n")
cat("Recall:", recall_rf, "\\n")
cat("F1 Score:", f1_rf, "\\n")

# ROC Curve for RF
RF_probs <- predict(RF_Model, testing_data, type = "prob")[, "At Risk"]
rf_roc <- roc(testing_data$class, RF_probs, levels = c("No Risk", "At Risk"))
plot(rf_roc, main = "ROC Curve - Random Forest", col = "green")
cat("AUC (RF):", auc(rf_roc), "\\n")


# Final Comparison Table
results <- data.frame(
  Model = c("Naive Bayes", "Random Forest"),
  Accuracy = c(
    mean(NB_Predictions == testing_data$class),
    mean(RF_Predictions == testing_data$class)
  ),
  Precision = c(precision_nb, precision_rf),
  Recall = c(recall_nb, recall_rf),
  F1_Score = c(f1_nb, f1_rf),
  AUC = c(auc(nb_roc), auc(rf_roc))
)
print(results)

