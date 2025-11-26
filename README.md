# Machine-Learning-Model-for-Diabetes-Prediction
Diabetes dataset is used to predict whether someone will develop diabetes over the next five years. Prediction is based on various Machine Learning models and shows the efficiency and reliability of AI models in early disease detection. 

## Overview: 
Diabetes is a chronic disease with long‑lasting effects, and early detection is crucial to help patients receive timely medical intervention and manage the condition effectively. I built this project to explore how machine learning can be applied to predict diabetes risk, with the goal of supporting preventative healthcare and encouraging healthier lifestyle changes before the disease develops.

### The real‑world problem it addresses:
In many cases, diabetes can be prevented if the risk is identified well in advance. By accurately outlining risk factors, patients and clinicians can take proactive steps to reduce the likelihood of developing the disease. This project uses the Diabetes dataset to predict whether someone will develop diabetes over the next five years. It frames the challenge as a binary classification problem, applying supervised machine learning to provide reliable predictions that could inform early interventions.

### What makes this project unique and impactful:
The project demonstrates the efficiency and reliability of AI models in early disease detection. A very thorough data preprocessing pipeline was implemented to ensure accuracy: missing values and duplication were addressed, outliers were handled using the Interquartile Range (IQR), and skewed features were transformed using log functions. As the dataset was imbalanced, appropriate measures were taken to balance the target variable ‘Outcome’ for more precise predictions. The code was developed in Jupyter Notebook using Python, with robust support from libraries such as Scikit‑learn, Pandas, NumPy, and Matplotlib. This combination of careful preprocessing, model experimentation, and reproducible coding makes the project impactful as a demonstration of how AI can contribute to preventative healthcare.

## Results & Key Insights:

### Performance Table:
The following chart summarises the performance metrics of six machine learning models applied to diabetes prediction.

![Model Comparison](visuals/Model_comparison.png) 

Voting Ensemble achieved the highest recall (0.889), which is especially important in medical prediction because it helps minimise false negatives ensuring fewer at‑risk patients are missed.

Logistic Regression recorded the highest ROC AUC (0.833), showing strong overall discrimination between diabetic and non‑diabetic cases.

Random Forest delivered the best specificity (0.867), meaning it performed well in correctly identifying non‑diabetic cases and reducing false positives.





### Model Selection for next five years:
For next five years diabetes prediction, the Voting Ensemble model showed the highest recall (0.889), making it the most suitable for clinical data where identifying diabetic cases is critical. Even though Random Forest achieved the highest specificity, and Logistic Regression offered balanced performance with high interpretability, the Voting Ensemble strikes the best balance between sensitivity and overall predictive power, supporting its selection as the final model.

These results highlight how different models excel in different areas: some are better at catching positive cases, while others are stronger at ruling out negatives. Together, they demonstrate the value of comparing multiple approaches when tackling healthcare prediction problems.

### Ensemble Model
To enhance the predictive ability of the disease detection model, I combined the learning from multiple existing models to create an Ensemble Model. By integrating two or more classifiers, there is a strong potential to improve overall performance, particularly recall and F1 score. Since Logistic Regression, Balanced Random Forest Classifier, Random Forest, and SVC demonstrated consistently high accuracy, these models were selected to form the pipeline for the ensemble approach.

![Confusion_Matrix](visuals/Confusion_Matrix - Ensemble_Model.png)

### Logistic Regression:
Logistic Regression is used as a baseline model because of its reliability in binary classification tasks (Diabetic vs Non-Diabetic). It works well with a moderate number of features. With class_weight = ‘balanced’, it can handle imbalanced data well by adjusting the penalties on misclassifications, and it seeks to minimise the cost function during training.
The Logistic Regression model achieved an accuracy of 74%, with a sensitivity (recall) of 70%, indicating it correctly identifies 70% of diabetic cases. The specificity of 76% shows that the model reliably identifies non-diabetic individuals. Having a high recall rate is important in medical data, where False Negatives can be more critical than False Positives. The precision and recall for the Non-Diabetic class(0) also suggest that the model is correctly identifying healthy individuals.

### Random Forest:
Random Forest is chosen as one of the classification models as it’s robust and can handle complex relationships in the data. It is an ensemble method that combines multiple decision trees for prediction. The feature importance map is used to visualise which features played an important role in the model’s prediction. Glucose, BMI and Age played an important role in the decision-making process for the Random Forest Classifier. 

![RF Feature Importance](visuals/Random_Forest.png)

The Random Forest model achieved an accuracy of 76%, demonstrating strong sensitivity (87%) in correctly identifying non-diabetic cases. However, its sensitivity for diabetic cases is lower (57%), suggesting that while the model is reliable in detecting non-diabetics, it may underperform in catching all positive cases, and this is an important consideration in disease prediction, where recall is often prioritised.

### Balanced Random Forest Classifier (BRF)
BRF from the imblearn library is used as it automatically balances the dataset by under sampling the majority class at each tree during training. The Balanced Random Forest model achieved an overall accuracy of 74%, providing balanced sensitivity and specificity across both classes. With a recall (sensitivity) of 64% for diabetic cases, the model is effective in identifying diabetic individuals, though precision suggests potential room for reducing false positives. There is a reliable balance between detecting diabetic patients and minimising false positives compared to the standard Random Forest model, which underperformed in detecting positive cases.


## Model Evaluation:
### Stratified K-fold Validation:

Stratified K‑Fold cross‑validation is applied to evaluate model performance on unseen data. Rather than depending on a single train‑test split, each model is trained and tested across multiple folds, ensuring that every subset of the data is used for both training and validation. This approach provides a more reliable and robust estimate of how the models generalise to new cases.

![K-fold Validation](visuals/K-fold.png) 



