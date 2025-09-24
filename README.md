# Email_spam_detector
Email Spam Detector 📨🚫  A machine learning-based email spam classifier that automatically identifies spam and legitimate (ham) emails using advanced NLP techniques and ensemble models. Built with Python, scikit-learn, and TF-IDF vectorization, this project demonstrates a robust, real-world approach to email filtering.
# 📬 Email Spam Detector

> A smart machine learning-based email spam classifier that helps separate spam from legitimate emails using Python, scikit-learn, and NLP techniques.  

---

## 🚀 Why This Project?

Spam emails are everywhere, and they can be annoying or even dangerous.  
This project lets you:

- Automatically detect spam emails.  
- Reduce false positives (never mark legit emails as spam unnecessarily).  
- Learn how classical ML and ensemble methods can be combined for real-world applications.  

---

## 🛠 Features

- **Text preprocessing & TF-IDF vectorization**  
- **Multiple ML classifiers**:
  - Multinomial Naive Bayes  
  - Bernoulli Naive Bayes  
  - Support Vector Classifier (SVC)  
  - Extra Trees Classifier  
- **Voting Classifier ensemble** → combines multiple models for better accuracy  
- **Metrics & Evaluation**:
  - Accuracy, Precision, Recall, F1-score  
  - Confusion matrix visualization  
- **Optional advanced tools**:
  - Hyperparameter tuning using GridSearchCV  
  - Cross-validation for more reliable results  
  - Feature selection to remove irrelevant words  

---

## 🍪 How It Works

1. **Load & clean data** – emails labeled spam/ham.  
2. **Convert text to numerical features** using TF-IDF.  
3. **Train multiple ML models** and combine them via a Voting Classifier.  
4. **Evaluate performance** using accuracy, precision, recall, F1, and confusion matrix.  
5. (Optional) **Tune hyperparameters** using GridSearchCV for optimal performance.  

---

## 📊 Example Performance

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|---------|-----------|--------|----------|
| MultinomialNB      | 95.4%   | 95.6%     | 100%   | 0.81     |
| BernoulliNB        | 96.6%   | 96.6%     | 97.2%  | 0.87     |
| Voting Classifier  | TBD     | TBD       | TBD    | TBD      |

> MultinomialNB catches **all spam** but may mark some good emails as spam.  
> BernoulliNB is **balanced**, fewer false positives.  
> Voting Classifier combines strengths of all models for the **best overall performance**.

---

## ⚡ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/email-spam-detector.git
cd email-spam-detector

# Install dependencies
pip install -r requirements.txt
from spam_detector import SpamClassifier

# Initialize classifier
classifier = SpamClassifier()

# Train the model on your dataset
classifier.train(df)  # df = your email dataframe with 'text' and 'label'

# Predict new emails
emails = ["You won a free prize!", "Meeting at 10 AM tomorrow"]
predictions = classifier.predict(emails)
print(predictions)  # Output: [1, 0] -> 1=Spam, 0=Ham
# For simple Data Visualizations
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(classifier.model, X_test, y_test)
