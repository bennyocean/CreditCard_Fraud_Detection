tbd.
<!--
# Introduction

Full project on Predicting the hospitalization urgency of COVID-19 patients using Python. I developed different classification models (kNN & logistic regression) to predict a patient's need for hospitality (target) based on several symptoms and evaluated them by various score metrics, e.g., accuracy, recall, F1-score.

Check the detailled machine learning algorithms out here: [Covid_Hospitalization_Classification](/)

### Problem Setting:
At the peak of the COVID-19 pandemic, hospital authorities had to make a call about who to admit and who to send home given the limited available resources. Our problem is to have a classifier that suggests whether a patient should be immediately admitted to the hospital or sent home.

### Goal:
The goal of this project is to predict the urgency with which a COVID-19 patient will need to be admitted to the hospital from the time of onset of symptoms. The dataset contains some COVID-19 symptoms and demographic information. Notably, this dataset was collected in the peak of a COVID-19 wave and hence may have some errors and missing data.

While this case study tries to mimic a real-life problem, it is important to note that this analysis is for educational purposes only.

### Background:
This in-depth machine learning project resides on my **Capstone project** of the course *[Introduction to Data Science with Python](https://www.edx.org/learn/data-science/harvard-university-introduction-to-data-science-with-python)* taught by Professor Pavlos Protopapas, the Scientific Program director at the Harvard School of Engineering and Applied Sciences (SEAS).

### Tools I Used:
For my deep dive into the development of a classification machine learning model of the data, I harnessed the power of several key tools:

- **Python** 
- **Jupyter Notebook**
- **Sklearn**
- **Seaborn**

# Data Pre-Processing and EDA
To clean and manipulate the COVID-19 data, I used the [Covid_Hospitalization_Classification](/covid.csv.rtf) file, filling missing data using sklearn's ```KNNImputer```, and run an explorative data analysis (EDA).

![AgeDistribution](assets/distribution_age_groups.png)

*Bar chart visualizing the distribution of age groups needing hospital beds*


![CoughCounts](assets/counts_of_cough_by_urgency.png)

*Bar chart visualizing the counts of cough by urgency*

Here's the breakdown of the COVID-19 EDA:
- **Patients at risk:** The age group of 41-50 has the most urgent need for a hosphital bed.
- **Feature Selection:** Fever is the most common symptom for urgent hospitalization, followed by cough.
- **Feature Inspection:** Patients with no urgent need of hospitalization have cough as more common symptom than patients with no urgency.

# Train / Test Split 
To split the COVID-19 data, I used the ```train_test_split``` from from *sklearn.model_selection library*. The data is split itno training and a test set of 30% and a random state of 60 for reproducibility.

```python
df_train, df_test = train_test_split(df, test_size=0.3, random_state=60)
```

# Prediction

**Classification Model**

I use a K-neighbour classifier model with k = 10 to fit and predict the urgency of covid patients.

```python
model = KNeighborsClassifier(n_neighbors = 10)

model.fit(X_train,y_train)
```

```python
y_pred = model.predict(X_test)

```

```python
from sklearn.metrics import accuracy_score

model_accuracy = model.score(X_test, y_test)
print(f"Model Accuracy is {model_accuracy}")
```

The kNN-model accuracy is ~0.691.

# Evaluation

Computing metrics other than accuracy to judge the efficacy of the model's predictions.

**Fit Other Models:**
- Classification model using a kNN classfier, with k = 7
- Logistic model using regression, with c = 0.01, where c = inverse of regularization strength

**Evaluation of the New Models:**
Creating a dictionary with different metric scores:
- Accuracy
- Recall
- Specificity
- Precision
- F1-score


### Classification Model Performance Comparison

The following table summarizes the performance of the two kNN classifiers with different values of \(k\) (7 and 10) and a Logistic Regression model based on various classification metrics.

| Classification Metric | kNN Classification (k=7) | kNN Classification (k=10) | Logistic Regression |
|-----------------------|--------------------------|---------------------------|---------------------|
| Accuracy              | 0.6910                   | 0.6811                    | 0.6079              |
| Recall                | 0.6525                   | 0.7376                    | 0.7163              |
| Specificity           | 0.7250                   | 0.6313                    | 0.5125              |
| Precision             | 0.6765                   | 0.6380                    | 0.5642              |
| F1-score              | 0.6643                   | 0.6842                    | 0.6313              |

### Results & Interpretation

- The results show a notable improvement in the Accuracy and Specificity for the kNN classifier with \(k=7\), suggesting that this model now outperforms the other models in correctly identifying both positive and negative classes.
- The Recall has decreased for the kNN classifier with \(k=7\), but its significant improvement in Specificity and Precision indicates a more balanced performance between identifying true positives and true negatives.
- Logistic Regression remains the least accurate model, but its Recall and F1-score suggest it is still relatively good at identifying true positives and balancing precision and recall.
- Comparing the F1-scores, we see that the kNN classifier with \(k=7\) has a higher score than Logistic Regression, indicating a better overall balance of precision and recall, a critical factor in many classification tasks.

The improvement in the kNN classifier with \(k=7\)'s performance suggests that parameter tuning and model selection should be iterative processes, taking into account the full range of performance metrics.

In general, the choice between these models depend on the specific application and whether the emphasis is on reducing false positives (higher precision) or false negatives (higher recall).

![EvaluationMetrics](assets/evaluation_metrics_comparison.png)

*Bar charts comparing the model performance metrics*

# Selection of Best Metrics

It is not clear how to choose which metric(s) to use to pick the best model. For this reason, I calculate the AUC scores and plot the ROC curves for both the kNN (k=7) and the logistic regression models and use that information to decide which model is ideal for each of the scenarios.

### Prediction of Probabilities

I predict the probabiities for the positive class on the test data using the ```predict_proba``` method. 

```python
y_pred_knn = knn_model.predict_proba(X_test)[:, 1]

y_pred_logreg = log_model.predict_proba(X_test)[:,1]
```

To optimize model performance, the ```get_thresholds``` function evaluates a range of thresholds tailored to the model's predicted probabilities, discarding evenly spaced values that don't impact classification outcomes. This ensures only relevant threshold adjustments are considered, enhancing model accuracy.

```python
def get_thresholds(y_pred_proba):
    unique_probas = np.unique(y_pred_proba)
    unique_probas_sorted = np.sort(unique_probas)[::-1]
    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    thresholds = np.append(thresholds, 0)
    return thresholds
```

```python
knn_thresholds = get_thresholds(y_pred_knn)

logreg_thresholds = get_thresholds(y_pred_logreg)
```
### Plotting ROC

We then calculate the **False Posotive Rate (FPR)** and the **True Posotive Rate (TPR)** and plott the TPR agaisnt the FPR at various threshod levels.

![ROC_Plot](assets/roc_plot.png)

*ROC plot of the two models evaluated*

# COVID-19 Scenario Analysis

The choice of a threshold in ROC curve interpretation is heavily dependent on the specific application and its tolerance for false positives versus false negatives. This concept is illustrated through hypothetical scenarios involving different countries' responses to the Covid-19 pandemic, each with unique constraints to optimize their strategies:
- Brazil aims to minimize bad press with a combined TPR and FPR of less than 0.5,
- Germany focuses on reducing fatalities by maintaining a TPR between 0.8 and 0.9, and
- India prioritizes managing limited hospital beds with a combined TPR and FPR of up to 1.

These examples, while not reflective of actual policies, demonstrate how application-specific needs dictate the selection of the optimal classifier and threshold settings.

![ROC_Scenarios](assets/roc_scenarios.png)

*ROC plot on various scenarios*

### Choice of Classifier
- BRAZIL : Logistic regression with a high threshold
- GERMANY : Logistic regression with a low threshold
- INDIA : kNN classifier with a moderate threshold


# What I Learned

During my capstone ML project on predicting the hospitalization urgency of COVID-19 patients, I undertook several critical steps and learned valuable lessons that enhanced my understanding and application of machine learning techniques.

### Key Learnings and Achievements:
- **Model Development:** I developed both kNN and logistic regression models to predict hospitalization urgency based on COVID-19 symptoms.
- **Evaluation Metrics:** These models were rigorously evaluated using metrics such as accuracy, recall, and F1-score, allowing me to gauge their effectiveness comprehensively.
- **Data Handling:** I honed my skills in handling missing data and conducting exploratory data analysis (EDA), which are fundamental aspects of preparing data for machine learning models.
- **Model Selection:** The project emphasized the importance of careful model selection based on performance metrics, ensuring the chosen models were well-suited for the task at hand.
- **Threshold Tailoring:** A significant insight from this project was the necessity of tailoring threshold selection to specific scenarios. This was illustrated through hypothetical COVID-19 policy responses from countries like Brazil, Germany, and India.
- **Practical Application:** The use of ROC curves and the emphasis on precision in model performance highlighted the practical applications of my work, demonstrating how machine learning can inform real-world decisions and policies.

### Insights

This capstone project illuminated the complexity and criticality of **machine learning** in healthcare, particularly in crisis situations like the COVID-19 pandemic. Through the development and evaluation of kNN and logistic regression models, the project showcased the importance of data preprocessing, exploratory data analysis, and the nuanced selection of classification models. The analysis demonstrated that different scenarios require tailored approaches in model selection and threshold setting to balance the trade-offs between false positives and negatives, reflecting real-world decision-making challenges.

### Closing Thoughts

The project not only reinforced my technical skills in machine learning and data science but also emphasized ethical and practical considerations in applying these technologies to public health. It served as a powerful reminder of the potential impact of data science in informing policy and operational decisions during health emergencies, highlighting the importance of adaptability, precision, and contextual understanding in developing solutions that cater to diverse and dynamic challenges.

-->
