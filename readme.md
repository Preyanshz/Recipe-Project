# What’s Cooking? – Predicting Cuisine from Ingredients

This repository contains all the code, analysis, and submission files for the [Kaggle “What’s Cooking?” challenge](https://www.kaggle.com/competitions/whats-cooking/). The objective is to **predict a recipe’s cuisine** given a list of ingredients.

We explored multiple classifiers and feature engineering techniques to accurately categorize cuisines. Ultimately, a **tuned Support Vector Classifier (SVC)** emerged as our best performer with a **Kaggle public leaderboard score of 0.80410**.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Data Source & Preprocessing](#data-source--preprocessing)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Feature Engineering](#feature-engineering)  
6. [Modeling Approach](#modeling-approach)  
7. [Results & Evaluation](#results--evaluation)  
8. [Usage & Reproduction](#usage--reproduction)  
9. [References & Credits](#references--credits)

---

## Project Overview
- **Goal**: Predict the cuisine (e.g., Italian, Indian, Mexican, etc.) purely from the ingredients in a recipe.
- **Key Insight**: Different cuisines rely on characteristic ingredients and spice blends. By modeling ingredient usage patterns, we can classify which cuisine a recipe belongs to.
- **Approach**:
  1. Perform thorough data cleaning and preprocessing.
  2. Use vectorization (TF-IDF, CountVectorizer) to transform text ingredients into numerical features.
  3. Experiment with various classification models (Logistic Regression, Random Forest, SVC).
  4. Optimize the best model (SVC) using hyperparameter tuning (GridSearch).

---

## Repository Structure
Here is a brief overview of the important files in this repository:

```
.
├── DATA6100_FinalProject.ipynb     # Jupyter Notebook with code and analysis
├── DATA6100_Finalproject_Report.pdf  # PDF report summarizing the project
├── submission.csv                  # Baseline SVM (untuned) submission (score ~ 0.79646)
├── submission_svm_tuned.csv        # Tuned SVC submission (score ~ 0.80410)
├── README.md                       # This README file
└── ...
```

**Key CSV files**  
- **`submission.csv`**:  
  - Generated using a baseline Support Vector Machine without hyperparameter tuning.  
  - Kaggle Leaderboard Score: **0.79646**.

- **`submission_svm_tuned.csv`**:  
  - Generated after applying hyperparameter tuning (GridSearch) on the SVM model.  
  - Kaggle Leaderboard Score: **0.80410** (final, best).

---

## Data Source & Preprocessing
1. **Data Source**:  
   - The dataset comes from the “What’s Cooking?” Kaggle competition. It includes a list of recipes, each labeled with its cuisine and featuring a list of ingredients.

2. **Data Cleaning & Preparation**:  
   - Removal of irrelevant characters and normalization of ingredient text.  
   - Consolidation of similar ingredient names (e.g., “chopped garlic” vs. “garlic”).  
   - Ensured consistent formatting for modeling (lowercasing, stripping punctuation, etc.).

3. **Split**:
   - Training set used for model training and validation.
   - A holdout/test set was used to produce Kaggle submissions.

---

## Exploratory Data Analysis (EDA)
- **Distribution of Cuisines**: Visualized the frequency of each cuisine to identify the dominant categories (e.g., Italian, Mexican, Indian).  
- **Top Ingredients**: Identified ingredients most frequently used within each cuisine. This helped guide feature engineering (e.g., focusing on uniquely characteristic spices).
- **Insights**:  
  - Some cuisines rely heavily on specific flavor profiles (e.g., soy sauce for Asian cuisines, basil & tomato for Italian).  
  - The data had an unbalanced distribution, meaning some cuisines had more samples than others.

(See the [DATA6100_Finalproject_Report.pdf](./DATA6100_Finalproject_Report.pdf) for in-depth charts and analysis.)

---

## Feature Engineering
- **Tokenization & Cleaning**: Split ingredient text into tokens, removing special characters and lemmatizing/stemming (if necessary).  
- **Vectorization**:  
  - **CountVectorizer**: Basic frequency-based vectorization.  
  - **TF-IDF**: Used to give higher weight to more unique ingredients, reducing the impact of very common words.  
- **Dimensionality Reduction** (optional exploration):  
  - In some trials, we tested reducing the feature space to speed up model training. Ultimately, the raw or modestly reduced feature space performed best.

---

## Modeling Approach
1. **Models Tried**:  
   - **Logistic Regression** – Quick to train, baseline.  
   - **Random Forest Classifier** – Ensemble method to capture ingredient interactions.  
   - **Support Vector Classifier** – Good track record on text classification tasks, can handle sparse features well.  

2. **Hyperparameter Tuning (GridSearch)**:  
   - **SVC**: Explored `C`, `gamma`, and kernel types.  
   - Achieved improved accuracy after systematically searching the parameter space.

---

## Results & Evaluation
- **Baseline SVM**:  
  - **Local Validation Accuracy**: ~79%  
  - **Kaggle Score**: `0.79646`  
  - Submission file: [`submission.csv`](./submission.csv)

- **Tuned SVC**:  
  - **Local Validation Accuracy**: ~81%  
  - **Kaggle Score**: `0.80410`  
  - Submission file: [`submission_svm_tuned.csv`](./submission_svm_tuned.csv)

**Confusion Matrix** (top-level findings):
- Most classes (cuisines) predicted well; some confusion where cuisines share many common ingredients.

**Precision, Recall, F1-score**:
- Balanced across major cuisines, though slight dips in underrepresented cuisines.

For full metrics and analysis, see the [PDF Report](./DATA6100_Finalproject_Report.pdf).

---

## Usage & Reproduction
1. **Environment Setup**  
   - Recommended: Python 3.8+  
   - Required libraries (common ones):
     ```bash
     pip install numpy pandas scikit-learn matplotlib seaborn
     ```
2. **How to Run**  
   - Clone this repository.  
   - Download the “What’s Cooking?” dataset from [Kaggle](https://www.kaggle.com/competitions/whats-cooking/data) and place it in a `data/` folder.  
   - Open `DATA6100_FinalProject.ipynb` in Jupyter or VSCode and run all cells in order.  
   - Adjust hyperparameters if desired to replicate or improve the results.  

3. **Generating Predictions**  
   - Once the notebook finishes training, the final predictions are saved as CSV files.  
   - To submit to Kaggle, upload the relevant CSV file (e.g., `submission_svm_tuned.csv`).

---

## References & Credits
- [Kaggle: What’s Cooking?](https://www.kaggle.com/competitions/whats-cooking/)
- Scikit-learn Documentation – [Link](https://scikit-learn.org/stable/user_guide.html)
- [DATA6100_Finalproject_Report.pdf](./DATA6100_Finalproject_Report.pdf) – Detailed project report.
- Project by *[Preyansh Yadav & Pratham Sankhala]* as part of *[DATA*6100-Introduction to Data Science]*.
