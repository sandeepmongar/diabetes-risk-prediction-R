# diabetes-risk-prediction-R
This project applies Naive Bayes and Random Forest models in R to predict early-stage diabetes risk. It includes data preprocessing, binary encoding, model training, performance evaluation (accuracy, precision, recall, F1, AUC), and ROC curve analysis.

## Models Used

- Naive Bayes
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

## Workflow

1. Data loading and structure inspection
2. Data preprocessing (binary encoding, factor conversion)
3. Data visualization (box plots, bar charts)
4. Train-test split (70/30)
5. Model training (Naive Bayes, Random Forest, Logistic Regression, SVM)
6. Performance evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC Curve & AUC

## Results

| Model                | Accuracy | Precision | Recall | F1 Score | AUC   |
|---------------------|----------|-----------|--------|----------|-------|
| Naive Bayes         | ~85%     | Good      | Good   | Balanced | High  |
| Random Forest       | ~89%     | High      | High   | High     | High  |
| Logistic Regression | ~93%     | High      | High   | High     | High  |
| SVM                 | ~93%     | High      | High   | High     | High  |

## Requirements

- R 4.x
- Packages: `dplyr`, `caret`, `ggplot2`, `randomForest`, `e1071`, `pROC`

## Dataset

The dataset contains health-related attributes for diabetes diagnosis, with features such as polyuria, polydipsia, sudden weight loss, and gender.

## Objective

To compare and evaluate different classification models for medical diagnosis use cases, and to demonstrate how machine learning can support early detection and healthcare analytics using R.

## Author

Sandeep Monger  
Email: sandeepmongar11@gmail.com  
LinkedIn: [sandeep-mongar](https://www.linkedin.com/in/sandeep-mongar-7b29b1173)
