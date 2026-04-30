# Handwritten Digit Recognition Using MNIST Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iogJp_1jGp3csVtF3cw61U8PE36KGcOU?usp=sharing)

> **A comprehensive machine learning project implementing multiple algorithms for handwritten digit recognition with 95%+ accuracy**

---

## 📋 Project Overview

This project implements and compares three different machine learning algorithms for handwritten digit recognition using the MNIST dataset. The implementation includes both scikit-learn models and custom implementations from scratch, along with optimization techniques like ensemble learning and dimensionality reduction.

### Key Features

| Feature | Status |
|---------|--------|
| Complete data preprocessing pipeline | ✅ |
| KNN implementation (sklearn + from scratch) | ✅ |
| SVM with hyperparameter tuning | ✅ |
| Decision Tree (sklearn + from scratch) | ✅ |
| Voting Ensemble (95.74% accuracy) | ✅ |
| PCA Dimensionality Reduction | ✅ |
| Comprehensive evaluation & visualization | ✅ |

---

## 📊 Dataset Information

| Property | Value |
|----------|-------|
| **Dataset** | MNIST (Modified National Institute of Standards and Technology) |
| **Total Images** | 70,000 handwritten digits |
| **Image Size** | 28 × 28 pixels |
| **Features** | 784 pixels per image |
| **Classes** | 10 digits (0 through 9) |
| **Training Set** | 56,000 images (80%) |
| **Testing Set** | 14,000 images (20%) |

---

## 🏆 Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM (RBF Kernel)** | **95.92%** | Best individual model |
| **KNN (sklearn, k=3)** | 97.14% | Best sklearn accuracy |
| **KNN (from scratch)** | 94.10% | Custom implementation |
| **Decision Tree** | 87.42% | max_depth=15 |
| **Voting Ensemble** | **95.74%** | Combined all 3 models |
| **KNN + PCA** | 97.76% | +2.96% improvement |
| **SVM + PCA** | 95.71% | +0.73% improvement |

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn |
| **Numerical Computing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, Graphviz |
| **Dimensionality Reduction** | PCA |
| **Environment** | Google Colab |

## 🚀 Getting Started

### Option 1: Run on Google Colab (Recommended)

Click the badge at the top of this README or use the link below:

🔗 **Open in Colab:** https://colab.research.google.com/drive/1iogJp_1jGp3csVtF3cw61U8PE36KGcOU?usp=sharing

### Option 2: Run Locally

```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn graphviz

# Launch Jupyter Notebook
jupyter notebook Handwritten_Digit_Recognition.ipynb

## 📁 Project Structure
mnist_classical_ml/
│
├── mnist_classical_ml.ipynb   ⭐ MAIN
├── README.md
├── data/
├── outputs/ (images, plots)

📈 Model Performance Details
Per-Digit Accuracy (Best Model - SVM)
Digit	Accuracy
0	98.99%
1	98.41%
2	94.92%
3	94.33%
4	95.68%
5	95.09%
6	97.53%
7	96.09%
8	94.58%
9	93.24%
Common Misclassifications
Actual → Predicted	Count	Reason
4 → 9	52	Similar curved shapes
8 → 3	49	Similar loop patterns
2 → 7	37	Stroke angle variations
8 → 5	35	Shape similarity
9 → 4	34	Symmetrical confusion
🎯 Bonus Implementations
1. Voting Ensemble
Method: Majority voting combining KNN, SVM, and Decision Tree

Result: 95.74% accuracy (outperformed all individual models)

2. PCA Dimensionality Reduction
Original features: 784 pixels

PCA components: 50

Variance preserved: 82.55%

Impact on KNN: +2.96% accuracy improvement

Impact on SVM: +0.73% accuracy improvement

📊 Key Findings
SVM with RBF kernel performed best among individual models (95.92%)

Ensemble methods consistently outperform single classifiers

PCA significantly benefits distance-based algorithms like KNN (+2.96%)

Most challenging digits: 9 and 8 (confused with 4, 3, and 5)

From scratch KNN achieved 94.10% accuracy (comparable to sklearn)

🔧 Implementation Highlights
KNN From Scratch
python
def predict(self, X):
    for test_point in X:
        distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        prediction = np.bincount(self.y_train[k_indices]).argmax()
        predictions.append(prediction)
SVM Hyperparameter Tuning
python
param_grid = {'C': [0.1, 1], 'gamma': ['scale'], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=2, scoring='accuracy')
Voting Ensemble
python
def ensemble_vote(predictions_list):
    for i in range(len(predictions_list[0])):
        votes = [pred[i] for pred in predictions_list]
        final_predictions.append(Counter(votes).most_common(1)[0][0])

📈 Visual Outputs
The notebook generates the following visualizations:

Output	Description
Sample digits	10 random images from dataset
Confusion matrices	Heatmaps for all 3 models
Model comparison	Bar chart comparing accuracies
Misclassified samples	10 errors with actual/predicted labels
PCA comparison	Before/after performance visualization
Flow diagram	Complete project workflow

💡 Future Improvements
Improvement	Expected Benefit
Convolutional Neural Networks (CNN)	>99% accuracy
Data augmentation	Better generalization
Expanded hyperparameter grid	+1-2% accuracy
Cross-validation for all models	More reliable evaluation
XGBoost / Random Forest	Additional benchmarks

📚 References
LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST Database of Handwritten Digits

scikit-learn Documentation: https://scikit-learn.org/

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification

👨‍💻 Author
Bhavana
Project: Handwritten Digit Recognition

📄 License
This project is for educational purposes.

🙏 Acknowledgments
MNIST dataset creators

scikit-learn contributors

Google Colab for free GPU/CPU resources

🔗 Quick Links
Link	Purpose
Open in Colab	Run the notebook
scikit-learn	ML library documentation
MNIST Dataset	Original dataset
