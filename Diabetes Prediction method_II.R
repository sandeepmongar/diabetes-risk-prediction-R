
# Set working directory
setwd("")

# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)
library(e1071)

# Load dataset
data <- read.csv("diabetes/data.csv")

# Check data
head(data)
str(data)
cat("Missing values per column:\\n")
print(colSums(is.na(data)))

# Encode categorical variables to binary
dataset <- data %>%
  mutate(
    Polyuria = ifelse(Polyuria == "Yes", 1, 0),
    Polydipsia = ifelse(Polydipsia == "Yes", 1, 0),
    sudden.weight.loss = ifelse(sudden.weight.loss == "Yes", 1, 0),
    weakness = ifelse(weakness == "Yes", 1, 0),
    Obesity = ifelse(Obesity == "Yes", 1, 0),
    Alopecia = ifelse(Alopecia == "Yes", 1, 0),
    partial.paresis = ifelse(partial.paresis == "Yes", 1, 0),
    Polyphagia = ifelse(Polyphagia == "Yes", 1, 0),
    delayed.healing = ifelse(delayed.healing == "Yes", 1, 0),
    Genital.thrush = ifelse(Genital.thrush == "Yes", 1, 0),
    muscle.stiffness = ifelse(muscle.stiffness == "Yes", 1, 0),
    Itching = ifelse(Itching == "Yes", 1, 0),
    Irritability = ifelse(Irritability == "Yes", 1, 0),
    visual.blurring = ifelse(visual.blurring == "Yes", 1, 0),
    Gender = ifelse(Gender == "Male", 1, 0),
    class = ifelse(class == "Positive", 1, 0)
  )

# Set proper factor levels
dataset$class <- factor(dataset$class, levels = c(0, 1))
dataset$Gender <- factor(dataset$Gender, levels = c(0, 1))

# Convert all binary columns to factor
binary_columns <- names(dataset)[names(dataset) != "Age"]
dataset[binary_columns] <- lapply(dataset[binary_columns], as.factor)

str(dataset)

# Visualizations
ggplot(dataset, aes(class, Age)) + geom_boxplot()
ggplot(dataset, aes(Gender, fill = class)) +
  geom_bar() +
  labs(title = "Distribution of Diabetes Classes by Gender", x = "Gender", y = "Count")

# Train-test split
set.seed(255)
ind <- sample(2, nrow(dataset), replace = TRUE, prob = c(0.7, 0.3))
trainset <- dataset[ind == 1, ]
testset <- dataset[ind == 2, ]
label <- testset$class

model_formula <- class ~ .


# Logistic Regression Model (1)
LGModel <- glm(model_formula, family = binomial(), data = trainset)
summary(LGModel)

# Predictions
prob_LG <- predict(LGModel, testset, type = "response")
pred_LG <- round(prob_LG)

# Evaluation
cm_LG <- confusionMatrix(factor(pred_LG, levels = c(0, 1)), factor(label, levels = c(0, 1)), positive = "1")
print(cm_LG)

TP_LG <- sum(pred_LG == 1 & label == 1)
FP_LG <- sum(pred_LG == 1 & label == 0)
FN_LG <- sum(pred_LG == 0 & label == 1)

Precision_LG <- TP_LG / (TP_LG + FP_LG)
Recall_LG <- TP_LG / (TP_LG + FN_LG)
F1_LG <- 2 * (Precision_LG * Recall_LG) / (Precision_LG + Recall_LG)

cat("\\nLogistic Regression Performance:\\n")
cat("Precision:", Precision_LG, "\\n")
cat("Recall:", Recall_LG, "\\n")
cat("F1 Score:", F1_LG, "\\n")

# ROC & AUC
LG.roc <- roc(label, prob_LG)
plot(LG.roc, main = "ROC Curve - Logistic Regression", col = "blue")
cat("AUC (Logistic Regression):", auc(LG.roc), "\\n")


# Support Vector Machine (2)
svm.model <- svm(model_formula, data = trainset, probability = TRUE)
svm.pred <- predict(svm.model, testset)
svm.prob <- attr(predict(svm.model, testset, probability = TRUE), "probabilities")[, "1"]

# Evaluation
cm_SVM <- confusionMatrix(factor(svm.pred, levels = c(0, 1)), factor(label, levels = c(0, 1)), positive = "1")
print(cm_SVM)

TP_SVM <- sum(svm.pred == 1 & label == 1)
FP_SVM <- sum(svm.pred == 1 & label == 0)
FN_SVM <- sum(svm.pred == 0 & label == 1)

Precision_SVM <- TP_SVM / (TP_SVM + FP_SVM)
Recall_SVM <- TP_SVM / (TP_SVM + FN_SVM)
F1_SVM <- 2 * (Precision_SVM * Recall_SVM) / (Precision_SVM + Recall_SVM)

cat("\\nSVM Performance:\\n")
cat("Precision:", Precision_SVM, "\\n")
cat("Recall:", Recall_SVM, "\\n")
cat("F1 Score:", F1_SVM, "\\n")

# ROC & AUC
svm.roc <- roc(label, svm.prob)
plot(svm.roc, main = "ROC Curve - SVM", col = "green")
cat("AUC (SVM):", auc(svm.roc), "\\n")


# Model Comparison Summary
results <- data.frame(
  Model = c("Logistic Regression", "SVM"),
  Accuracy = c(mean(pred_LG == label), mean(svm.pred == label)),
  Precision = c(Precision_LG, Precision_SVM),
  Recall = c(Recall_LG, Recall_SVM),
  F1_Score = c(F1_LG, F1_SVM),
  AUC = c(auc(LG.roc), auc(svm.roc))
)
print(results)

