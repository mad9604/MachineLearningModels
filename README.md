a. **Problem Statement**:-
   
   The project aims to implement and compare multiple machine learning classification models to predict the price range of mobile phones based on their technical specifications. In addition to model development and evaluation, an interactive Streamlit web application is built and deployed to demonstrate the models in a real-world deployment setting.

b. **Dataset Description:-**
   
   The dataset used for this assignment is a publicly available Mobile Price Classification dataset obtained from Kaggle. The dataset contains technical specifications of mobile phones such as battery power, RAM, internal memory, camera features, and connectivity options.

  **Dataset Characteristics:**
   
   Number of instances: > 500
   
   Number of features: > 12
   
   Target variable: price_range
   
   Type: Multi-class classification (4 classes: 0, 1, 2, 3)
   
   No missing values
   
   No duplicate records
   
   All features are numerical

   The dataset was found to be clean and did not require explicit data cleansing. Basic preprocessing such as feature scaling was applied where required.

c. **Models Used:-**

| ML Model Name            | Accuracy   | AUC      | Precision | Recall | F1 Score | MCC      |
| ------------------------ | --------   | ----     | --------- | ------ | -------- | -------- |
| Logistic Regression      | 0.9650     | 0.998667 | 0.965045  | 0.9650 | 0.964986 | 0.953357 |
| Decision Tree            | 0.8300     | 0.886667 | 0.831883  | 0.8300 | 0.830168 | 0.773811 |
| KNN                      | 0.5000     | 0.769750 | 0.521130  | 0.5000 | 0.505355 | 0.334993 |
| Naive Bayes              | 0.8100     | 0.950567 | 0.811326  | 0.8100 | 0.810458 | 0.746804 |
| Random Forest (Ensemble) | 0.8775     | 0.979608 | 0.877649  | 0.8775 | 0.877400 | 0.836785 |
| XGBoost (Ensemble)       | 0.9225     | 0.993842 | 0.922631  | 0.9225 | 0.922482 | 0.896719 |

 **Observations:-**

| ML Model Name            | Observation                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| Logistic Regression      | Provided a strong baseline performance and worked well with scaled numerical features.         |
| Decision Tree            | Captured non-linear relationships but showed signs of overfitting compared to ensemble models. |
| KNN                      | Achieved good accuracy but was sensitive to feature scaling and choice of neighbors.           |
| Naive Bayes              | Fast and efficient but showed lower accuracy due to the feature independence assumption.       |
| Random Forest (Ensemble) | Demonstrated robust performance with high accuracy and reduced overfitting.                    |
| XGBoost (Ensemble)       | Achieved the best overall performance due to boosting, regularization, and efficient learning. |

   
