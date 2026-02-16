# 📩 Spam Mail Detection using TF-IDF & Logistic Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)

---

## 🔍 Project Overview

Spam email detection is a classic Natural Language Processing (NLP) problem.  
In this project, we build a machine learning model capable of classifying emails as:

- **Ham (Not Spam)**
- **Spam**

The model is built using:

- Text preprocessing techniques  
- TF-IDF feature extraction  
- Logistic Regression classifier  
- Performance evaluation & feature importance analysis  

The main focus of this project is achieving high **Recall** to minimize false negatives, which is critical in spam detection systems.

---

## 🎯 Problem Statement

Spam emails pose serious risks including fraud, phishing, and malware distribution.  
The objective is to develop a reliable classification model that:

- Accurately identifies spam messages  
- Minimizes false negatives  
- Maintains good generalization on unseen data  

Challenges addressed in this project:

- Text cleaning and preprocessing  
- High-dimensional sparse features  
- Class imbalance  
- Model interpretability  

---

## 📊 Dataset Description

The dataset consists of labeled email messages:

| Label | Meaning |
|-------|----------|
| 0     | Ham (Not Spam) |
| 1     | Spam |

Each sample contains:
- Raw email text  
- Corresponding target label  

We performed exploratory data analysis to understand:
- Class distribution  
- Message length patterns  
- Potential imbalance  

---

## 🔎 Exploratory Data Analysis (EDA)

EDA steps included:

- Visualizing class distribution  
- Analyzing message length distribution  
- Observing imbalance between spam and ham messages  

Understanding the dataset helped in selecting appropriate evaluation metrics beyond simple accuracy.

---

## 🧹 Text Preprocessing

Before training the model, the following preprocessing steps were applied:

- Converting text to lowercase  
- Removing punctuation  
- Removing stopwords  
- Tokenization  

These steps reduce noise and improve model performance.

---

## 🔤 Feature Engineering (TF-IDF)

Text data must be converted into numerical form before feeding it into a machine learning model.

We used:

### **TF-IDF (Term Frequency – Inverse Document Frequency)**

Why TF-IDF?

- Reduces the impact of very common words  
- Emphasizes important and discriminative words  
- Works efficiently with linear models  

---

## ⚖️ Handling Class Imbalance

Spam detection datasets are often imbalanced.

To address this issue, we used:

```python
class_weight='balanced'


This ensures that spam samples receive appropriate importance during training, preventing bias toward the majority class.

---

## 🤖 Model Training

A **Logistic Regression** classifier was trained on TF-IDF features extracted from the email text.

### Why Logistic Regression?

- Fast and computationally efficient  
- Performs exceptionally well on high-dimensional sparse data  
- Provides interpretable coefficients  
- Strong baseline model for NLP tasks  

The model achieved:

- **Training Accuracy: 99.05%**

This indicates that the model successfully captured meaningful patterns in the dataset.

---

## 📈 Model Evaluation

The model was evaluated on a held-out test set using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### 📊 Test Performance

- **Test Accuracy:** 97.39%  
- **Precision (Spam):** 89%  
- **Recall (Spam):** 92%  
- **F1-score (Spam):** 90%  

---

## 🚨 Why Recall Matters in Spam Detection

In spam detection systems, **Recall is more important than Accuracy**.

A false negative (classifying spam as ham) allows a harmful email to reach the user's inbox.

### Confusion Matrix

- [[949 17]
- [ 12 137]]

- 137 spam emails were correctly detected  
- Only 12 spam emails were missed (false negatives)  
- 17 ham emails were incorrectly classified as spam  

The strong **92% recall** demonstrates that the model effectively detects the majority of spam messages.

Additionally, the small gap between training accuracy (99.05%) and test accuracy (97.39%) indicates good generalization with no significant overfitting.

---

## 🔥 Feature Importance Analysis

One of the main advantages of Logistic Regression is interpretability.

Each word in the TF-IDF representation is assigned a coefficient:

- Words with high positive coefficients strongly indicate **Spam**  
- Words with negative coefficients are associated with **Ham**  

By analyzing these coefficients, we identified the most influential words driving the model’s decisions.

This makes the model transparent and explainable rather than a black-box system.

---

## 📊 Results Summary

- High overall test accuracy (97.39%)  
- Strong spam recall (92%)  
- Balanced precision-recall tradeoff  
- Low number of missed spam messages  
- Interpretable and explainable predictions  
- Good generalization capability  

TF-IDF combined with Logistic Regression provides a powerful and reliable baseline for spam detection tasks.

---

## 🚀 Future Improvements

Potential enhancements include:

- Hyperparameter tuning using GridSearchCV  
- Adding n-grams (bigrams & trigrams)  
- Testing alternative models (Naive Bayes, SVM, XGBoost)  
- Deploying the model using Flask or FastAPI  
- Building a real-time spam detection web interface  
- Experimenting with Deep Learning models (LSTM, BERT)  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn 
- pickle 
- Matplotlib  
- Seaborn  

