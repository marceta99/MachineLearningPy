# **Multi-Model ML Playground** &middot; ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

An experimental repository exploring various Machine Learning models (Regression, Classification, Neural Networks, etc.) with Python. Built for quick prototyping, benchmarking, and showcasing foundational ML concepts.

---

## **Table of Contents**
1. [Overview](#overview)  
2. [Features](#features)  
3. [Quick Start](#quick-start)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Examples](#examples)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## **Overview**

This project contains multiple Machine Learning models implemented in Python. Each model demonstrates how to:
- Preprocess data  
- Train and evaluate performance  
- Visualize metrics (confusion matrix, ROC curves, etc.)  

Whether you’re **learning** ML or **prototyping** advanced algorithms, this playground provides an easy way to experiment with different approaches.

---

## **Features**

- **Variety of ML Models**  
  Logistic Regression, SVM, Random Forest, Neural Networks, and more.

- **Modular Pipeline**  
  Each model has its own script or notebook with clear separation of data loading, preprocessing, and evaluation.

- **Configurable Hyperparameters**  
  Tweak learning rates, number of layers, or other parameters for quick experimentation.

- **Easy Plots and Metrics**  
  Built-in functions to visualize accuracy, precision, recall, and more.

---

## **Quick Start**

1. **Clone the repo** (via SSH or HTTPS):
   ```bash
   git clone https://github.com/marceta99/MachineLearningPy.git
   cd MachineLearningPy
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a sample model**:
   ```bash
   cd "Association Rule Learning"
   python Apriori.py
   ```

---

## **Usage**

### **1. Data Preparation**
Place your datasets in the `data/` folder or adjust file paths within each model’s script. By default, each script assumes a CSV format.

### **2. Training & Evaluation**
Run any of the model files in `models/`:
```bash
python models/random_forest.py
```
This trains the model and outputs metrics like accuracy and F1-score in the console.

### **3. Hyperparameter Tuning**
Experiment by modifying hyperparameters (e.g., `n_estimators` for Random Forest, `learning_rate` for Neural Networks) directly in the script, or pass them as command-line arguments if supported:
```bash
python models/random_forest.py --n_estimators 200 --max_depth 10
```

---

## **Project Structure**

```
MachineLearningPy/
├── Association Rule Learning/
│   ├── Apriori.py
│   └── Market_Basket_Optimisation.csv
├── Classification/
│   ├── LogisticRegression.py
│   └── Social_Network_Ads.csv
├── Clustering/
│   ├── KMeans.py
│   └── Mall_Customers.csv
├── Datapreprocessing/
│   ├── Data.csv
│   └── DataPreprocessing.py
├── DeepLearning/
│   ├── Artificial Neural Network/
│   │   ├── Artificial Neural Networks.py
│   │   └── Churn_Modelling.csv
│   └── Convolutional Neural Network/
│       ├── __MACOSX/       (System folder, can be ignored or removed)
│       ├── dataset/
│       └── Convolutional Neural Networks.py
├── Fast API/
│   ├── __pycache__/
│   ├── APIClassification.py
│   ├── Classification.py
│   ├── diabetes_model.sav
│   └── diabetes.csv
├── Natural Language Processing/
│   ├── Natural Language Processing.py
│   └── Restaurant_Reviews.tsv
├── Regression/
│   ├── 50_Startups.csv
│   ├── Data.csv
│   ├── DesicionTreeRegression.py
│   ├── MultipleLinearRegression.py
│   ├── PolynomialRegression.py
│   ├── Position_Salaries.csv
│   ├── RandomForestRegression.py
│   ├── RegressionModelSelection.py
│   ├── Salary_Data.csv
│   ├── SimpleLinearRegression.py
│   └── SupportVectorRegression.py
├── Reinforcement Learning/
│   └── KMeans.py
└── requirements.txt
└── README.md
```

- **data/**: Contains CSV files or other datasets.  
- **models/**: Each Python script implements one ML model or approach.  
- **notebooks/**: Jupyter Notebooks for exploratory analysis or demonstrations.

---

## **Examples**

### **Confusion Matrix Plot**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# After model predictions
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

<p align="center">
  <img src="https://via.placeholder.com/300/09f/fff.png" alt="Placeholder Confusion Matrix" width="40%" />
</p>

### **ROC Curve**

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"Model (AUC = {roc_auc:.2f})")
plt.legend(loc="lower right")
plt.show()
```

---

## **Contributing**

Contributions, issues, and feature requests are welcome!  
- Fork the project  
- Create a new branch for your feature (`git checkout -b feature/something`)  
- Commit changes (`git commit -m 'Add awesome feature'`)  
- Push to your branch (`git push origin feature/something`)  
- Open a Pull Request  

---

<div align="center">
  <em>Happy Coding & Data Crunching!</em>  
</div>