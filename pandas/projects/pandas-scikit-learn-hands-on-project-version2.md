# Pandas + scikit-learn Hands-On Learning Guide 🐼🤖  
*A single source of truth for data preparation and classical machine learning*

This document is designed to be followed **top to bottom**.
It explains **what you do, why you do it, and what happens next**.

You start with **pandas (data preparation)**  
→ then move to **scikit-learn (model training & evaluation)**

------------------------------------------------------------
------------------------------------------------------------

## 0. Prerequisites

### What you need installed

- Python 3.8+
- Basic Python knowledge
- Libraries:
  - pandas
  - scikit-learn
  - (optional) jupyter

Install everything with:

    pip install pandas scikit-learn jupyter

------------------------------------------------------------

## How this file is structured

Each dataset follows the **same lifecycle**:

1. Load raw data (pandas)
2. Inspect & clean data
3. Engineer features
4. Save a clean dataset
5. Train a model (scikit-learn)
6. Evaluate and understand results

This mirrors **real ML workflows** used in industry.

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 1  
## Titanic Dataset — Data Cleaning → Machine Learning

### Business / ML Question

> Given passenger information, can we predict whether a passenger survived?

This is a **binary classification** problem.

------------------------------------------------------------

## PART A — Pandas: Preparing the Data

------------------------------------------------------------

## 1. Load the dataset

We first load the raw Titanic CSV into a pandas DataFrame.
A DataFrame is a table-like structure (rows & columns).

    import pandas as pd

    df = pd.read_csv("data/titanic/train.csv")

------------------------------------------------------------

## 2. Inspect the data

Before doing anything, we must *understand* the data.

### Look at first rows

    df.head()

This tells us:
- Column names
- Data types (numeric vs text)
- Example values

### Check shape

    df.shape

This tells us:
- How many rows (passengers)
- How many columns (features)

### Check column types & missing values

    df.info()

Pay attention to:
- Which columns have missing values
- Which columns are integers, floats, or objects (strings)

### Check numeric summaries

    df.describe()

This shows:
- Mean, min, max
- Helps identify outliers (e.g., very high fares)

------------------------------------------------------------

## 3. Identify missing data

Missing data is common in real datasets.
We must handle it **before ML**.

    df.isna().sum()

Look especially at:
- Age
- Cabin
- Embarked

------------------------------------------------------------

## 4. Drop irrelevant or unusable columns

Some columns are either:
- Mostly missing (`Cabin`)
- Not useful right now (`Ticket`)

Removing them simplifies the problem.

    df = df.drop(columns=["Cabin", "Ticket"])

------------------------------------------------------------

## 5. Handle missing values

### 5.1 Categorical column: Embarked

`Embarked` shows where passengers boarded.
Missing values can confuse models.

Strategy:
- Replace missing values with the **most common category (mode)**.

    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

------------------------------------------------------------

### 5.2 Numeric column: Age

`Age` is numeric.
A common, simple strategy is to use the **median**.

Why median?
- Less sensitive to outliers than mean.

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

------------------------------------------------------------

### Verify missing data again

    df.isna().sum()

At this point:
- Age ✅
- Embarked ✅

------------------------------------------------------------

## 6. Remove invalid data

Fare values should be positive.
Zero or negative fares do not make sense here.

    df = df[df["Fare"] > 0]

------------------------------------------------------------

## 7. Fix data types

Some columns represent categories, not numbers.

Why this matters:
- Clarity
- Memory efficiency
- Correct modeling assumptions

    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    df["Pclass"] = df["Pclass"].astype("category")
    df["Survived"] = df["Survived"].astype("int8")

------------------------------------------------------------

## 8. Exploratory analysis

Exploration helps us understand **patterns** before modeling.

### Overall survival rate

    df["Survived"].mean()

### Survival rate by gender

    df.groupby("Sex")["Survived"].mean()

### Survival rate by passenger class

    df.groupby("Pclass")["Survived"].mean()

### Average age by survival

    df.groupby("Survived")["Age"].mean()

------------------------------------------------------------

## 9. Feature engineering

### 9.1 FamilySize

Why:
- Family presence may affect survival.

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

------------------------------------------------------------

### 9.2 IsAlone

Why:
- Simpler signal derived from FamilySize.

    df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")

------------------------------------------------------------

## 10. Save cleaned dataset

We save the cleaned data so we don’t repeat work.

    df.to_csv("data/titanic/titanic_clean.csv", index=False)

------------------------------------------------------------

## 11. Prepare ML-ready dataset

We select only features useful for modeling.

    ml_columns = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone"
    ]

    df_ml = df[ml_columns]
    df_ml.to_csv("data/titanic/titanic_ml_ready.csv", index=False)

------------------------------------------------------------
------------------------------------------------------------

# PART B — scikit-learn: Training Your First Model

------------------------------------------------------------

## What happens now?

Until now:
- You prepared data using pandas

Now:
- scikit-learn will **learn patterns** from the data

------------------------------------------------------------

## 1. Load ML-ready data

    import pandas as pd

    df = pd.read_csv("data/titanic/titanic_ml_ready.csv")

------------------------------------------------------------

## 2. Define features (X) and target (y)

ML models learn:
- X → input variables
- y → output we want to predict

    X = df.drop(columns="Survived")
    y = df["Survived"]

------------------------------------------------------------

## 3. Encode categorical variables

Models only understand numbers.
We convert categories into numeric dummy variables.

    X = pd.get_dummies(X, drop_first=True)

------------------------------------------------------------

## 4. Train-test split

Why split?
- Train model on one part
- Test on unseen data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

------------------------------------------------------------

## 5. Train a Logistic Regression model

Why Logistic Regression?
- Simple
- Interpretable
- Excellent first baseline model

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

------------------------------------------------------------

## 6. Make predictions

    y_pred = model.predict(X_test)

------------------------------------------------------------

## 7. Evaluate the model

### Accuracy

    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)

### Detailed report

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

### Confusion matrix

    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, y_pred)

------------------------------------------------------------

## 8. Interpret model coefficients

Logistic Regression learns weights for each feature.

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    feature_importance

Positive coefficient → increases survival probability  
Negative coefficient → decreases survival probability

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 2  
## Netflix Dataset — Exploration & Simple Modeling

(This project is exploratory-first, modeling-second.)

------------------------------------------------------------

## PART A — Pandas Exploration

    df = pd.read_csv("data/netflix/netflix_titles.csv")

------------------------------------------------------------

## Clean missing values

    df["country"] = df["country"].fillna("Unknown")
    df = df.dropna(subset=["title", "type"])

------------------------------------------------------------

## Dates

    df["date_added"] = pd.to_datetime(df["date_added"])
    df["year_added"] = df["date_added"].dt.year

------------------------------------------------------------

## Simple analysis

    df["type"].value_counts()
    df["release_year"].value_counts().head()

------------------------------------------------------------

## Save curated dataset

    df[
        ["show_id", "type", "country", "release_year", "rating", "year_added"]
    ].to_csv("data/netflix/netflix_curated.csv", index=False)

------------------------------------------------------------
------------------------------------------------------------

# PART B — Simple ML Model (Netflix)

Goal:
- Predict Movie vs TV Show

------------------------------------------------------------

    df = pd.read_csv("data/netflix/netflix_curated.csv")

    X = df.drop(columns="type")
    y = df["type"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy_score(y_test, y_pred)

------------------------------------------------------------

# Final Summary

You have learned:

✅ How to clean real data with pandas  
✅ How to engineer features  
✅ How to prepare ML-ready datasets  
✅ How scikit-learn trains models  
✅ How to evaluate predictions  
✅ How to interpret model behavior  

This is **exactly the right foundation** for:
- NumPy
- More ML algorithms
- Real-world ML projects

------------------------------------------------------------
