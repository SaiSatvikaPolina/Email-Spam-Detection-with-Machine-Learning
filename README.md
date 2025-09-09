📧 #Email Spam Detection with Machine Learning

This project is about building a spam email detection system using Python and machine learning. The model can classify emails as either:

Spam (junk/unwanted mail)

Ham (legitimate mail)


##📝 Problem Statement

Spam emails continue to flood inboxes with scams, phishing attempts, and unwanted promotions. Our goal is to build a reliable system that can automatically detect spam emails, improve email security, and give users a cleaner inbox.


##🎯 Project Objectives

Data Preprocessing → Clean and prepare email text data.

Feature Engineering → Extract useful features from subject lines, email bodies, and text content.

Model Selection → Train and test different ML models like Decision Trees, SVM, and Naïve Bayes.

Model Evaluation → Use metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Optimization → Tune hyperparameters for best performance.

Validation → Apply cross-validation to check model generalization.

Deployment → Explore how the model can be used in real-world email filtering systems.

Ethics → Handle email data carefully to respect privacy.



##📊 Dataset & Insights

Around 13.41% of emails were spam, and 86.59% were ham.

Common spam words included: free, call, text, txt, now.

These keywords helped improve model performance.



##⚙️ Methodology

Load and clean the dataset.

Preprocess text (remove stopwords, lowercase, stemming/lemmatization).

Convert text into numerical format using Bag of Words or TF-IDF.

Train machine learning models (Decision Tree, SVM, Naïve Bayes).

Evaluate with accuracy, precision, recall, F1-score.

Select the best model and fine-tune it.



##🏆 Results

Best Model: Multinomial Naïve Bayes

Accuracy: 98.49%

Strength: Excellent recall, meaning it catches almost all spam emails.

✅ Inboxes become safer and cleaner with this system.



##📌 Project Summary

Data preprocessing and feature engineering were crucial steps.

Multinomial Naïve Bayes gave the best results among tested models.

The project shows how ML can successfully fight spam emails.

Future improvements may include deep learning (e.g., LSTMs, BERT) for even better performance.



##📚 Tools & Technologies

Python

Pandas, NumPy → Data handling

Scikit-learn → Machine learning models

Matplotlib, Seaborn, WordCloud → Visualization

