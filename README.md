# 📰 News Article Classification (NLP)

This project builds a **multi-class text classification model** to categorize news articles into topics such as **sports, politics, technology, business**, etc., using Natural Language Processing (NLP) and machine learning.

---

## 🎯 Objective

Develop a machine learning model that can automatically assign each news article to the most likely **category** based on its text content.

Example categories:

- `sports`
- `politics`
- `technology`
- `business`
- (and others, depending on the dataset)

---

## 🗂️ Dataset

- **Dataset name:** `data_news`
- **Records:** ~10,000+ news articles (edit this as per your actual data)
- **Target variable:** `category` → article topic/label

**Main columns:**

- `text` – the full content of the news article  
- `category` – the label for each article (e.g., sports, politics, tech)

---

## 🛠 Tech Stack

- **Language:** Python  
- **Environment:** Jupyter Notebook  

**Libraries:**

- Data handling: `pandas`, `numpy`
- NLP & preprocessing: `nltk`, `re`, `string`
- Machine learning: `scikit-learn`
- Visualization: `matplotlib`, `seaborn`

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning & Preprocessing

Applied the following preprocessing steps to the `text` column:

- Lowercased all text  
- Removed:
  - Punctuation  
  - Numbers  
  - Special characters  
  - Extra whitespace  
- Tokenization (splitting text into words)  
- Stopword removal using **NLTK**  
- Lemmatization / stemming to reduce words to their base form  

This prepared clean text suitable for feature extraction.

---

### 2️⃣ Exploratory Data Analysis (EDA)

- Checked dataset shape and basic info  
- Verified if there were **missing values**  
- Analyzed **class distribution** across categories (sports, politics, tech, etc.)  
- Visualized:
  - Category counts (bar plots)  
  - Common words or phrases in each category  

---

### 3️⃣ Feature Extraction

Converted preprocessed text into numerical features using:

- **TF-IDF (Term Frequency–Inverse Document Frequency)**  
  - Uni-grams and bi-grams

Optionally, additional ideas (if implemented):

- Bag-of-Words representation  
- Word embeddings like **Word2Vec** / **GloVe**

The main representation for modeling in this project is **TF-IDF vectors**.

---

### 4️⃣ Model Development

Trained and compared multiple machine learning models for **multi-class classification**:

- Logistic Regression  
- Multinomial Naive Bayes  
- Linear Support Vector Machine (SVM)  

Steps followed:

- Train–test split (e.g., 80% train, 20% test)  
- Cross-validation to ensure robustness  
- Hyperparameter tuning for the best model using grid search / manual tuning  

---

### 5️⃣ Model Evaluation

Evaluated model performance using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score** (macro and weighted)
- Confusion matrix for class-wise performance

Visualizations included:

- Confusion matrix heatmap  
- Category-wise precision/recall/F1 comparison  
- Support (number of samples per class)

---

## 📊 Example Results (Replace with Your Actual Metrics)

> These are placeholder numbers – update them from your notebook.

- **Best model:** Linear SVM / Logistic Regression with TF-IDF  
- **Test accuracy:** ~**85–88%**  
- **Macro F1-score:** ~**0.84–0.87**  

**Key insights:**

- **Sports** articles were classified with the highest accuracy due to distinct vocabulary (e.g., “match”, “goal”, “tournament”, “score”).  
- Some misclassification occurred between **politics** and **business** articles because of overlapping terms (e.g., “policy”, “budget”, “market”).  
- Using **bi-grams** improved performance by capturing meaningful phrases like “prime minister”, “stock market”, “artificial intelligence”, etc.

---

## 📁 Files in This Repository

- `NLP2Project.ipynb` – main Jupyter Notebook containing:
  - Data loading  
  - Text preprocessing  
  - Feature extraction (TF-IDF)  
  - Model training and evaluation  

- `data_news.csv` (optional) – a small sample of the news dataset for demo

- `reports/News_Classification_Report.pdf` (optional) – summary report of the project

---

## 🚀 How to Run the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MukulSSrank/news-article-classification-nlp.git
   cd news-article-classification-nlp
