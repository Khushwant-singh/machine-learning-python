# Pandas + scikit-learn Hands-On Projects 🐼🤖

This file is your **single source of truth** for:

- Practicing **pandas** with real datasets
- Training your **first ML models** using **scikit-learn**

It is designed to be followed **top to bottom**, at your own pace.

------------------------------------------------------------
------------------------------------------------------------

## 0. Prerequisites

### 0.1. Python & Libraries

You need:

- Python 3.8+
- Basic Python knowledge
- Libraries:
  - pandas
  - scikit-learn
  - (optional) jupyter

Install:

    pip install pandas scikit-learn jupyter

---

### 0.2. Suggested Folder Structure

You can adapt this, but it keeps things clean:

    machine-learning-python/
      data/
        titanic/
          train.csv
          titanic_clean.csv           # created later
          titanic_ml_ready.csv        # created later
        netflix/
          netflix_titles.csv
          netflix_curated.csv         # created later
      notebooks/
        titanic_pandas.ipynb
        titanic_sklearn.ipynb
        netflix_pandas.ipynb
        netflix_sklearn.ipynb
      pandas_sample_projects.md       # this file
      README.md

Use either Jupyter notebooks (`.ipynb`) or Python scripts (`.py`) as you like.

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 1  
## Titanic Dataset – Pandas Data Cleaning & Exploration

### Goal

Use pandas to:

- Load a real dataset (Titanic)
- Inspect and understand its structure
- Handle missing and invalid data
- Engineer simple features
- Explore survival patterns

Later, you will use this cleaned data in **Project 1B** with scikit-learn.

------------------------------------------------------------

## 1. Get the Dataset

1. Go to Kaggle.
2. Search: “Titanic - Machine Learning from Disaster”.
3. Download `train.csv`.
4. Place it at:

       data/titanic/train.csv

------------------------------------------------------------

## 2. Create a File / Notebook

Create **one** of:

- `notebooks/titanic_pandas.ipynb` (recommended), or
- `notebooks/titanic_pandas.py`

At the top of your file:

    import pandas as pd

------------------------------------------------------------

## 3. Load & Inspect the Data

    df = pd.read_csv("data/titanic/train.csv")

    print(df.head())      # first 5 rows
    print(df.shape)       # (rows, columns)
    print(df.info())      # column types + non-null counts
    print(df.describe())  # numeric summary

Check missing values per column:

    print(df.isna().sum())

Pay attention to: `Age`, `Cabin`, `Embarked`.

------------------------------------------------------------

## 4. Drop Unnecessary Columns

To keep it simple, we drop columns that are sparse or not used now:

    df = df.drop(columns=["Cabin", "Ticket"])
    print(df.columns)

------------------------------------------------------------

## 5. Handle Missing Values

### 5.1. Fill Embarked

Fill missing `Embarked` with the most frequent value (mode):

    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(df["Embarked"].isna().sum())

### 5.2. Fill Age

Fill missing `Age` with the median age:

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print(df["Age"].isna().sum())

Check missing values again:

    print(df.isna().sum())

You should have 0 missing for `Age` and `Embarked`.

------------------------------------------------------------

## 6. Remove Obviously Invalid Data (Fare)

Remove rows where `Fare` is non-positive:

    before_rows = len(df)
    df = df[df["Fare"] > 0]
    after_rows = len(df)

    print("Rows removed with non-positive Fare:", before_rows - after_rows)

------------------------------------------------------------

## 7. Fix Data Types

Convert some columns to `category` for clarity and efficiency:

    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    df["Pclass"] = df["Pclass"].astype("category")

Treat `Survived` as a small integer:

    df["Survived"] = df["Survived"].astype("int8")

Check:

    print(df.dtypes)

------------------------------------------------------------

## 8. Basic Exploration

### 8.1. Overall Survival Rate

    total_passengers = len(df)
    total_survived = df["Survived"].sum()
    survival_rate = total_survived / total_passengers

    print("Total passengers:", total_passengers)
    print("Survived:", total_survived)
    print("Survival rate:", survival_rate)

### 8.2. Survival by Sex

    survival_by_sex = df.groupby("Sex")["Survived"].mean()
    print(survival_by_sex)

### 8.3. Survival by Class

    survival_by_class = df.groupby("Pclass")["Survived"].mean()
    print(survival_by_class)

### 8.4. Average Age of Survivors vs Non-survivors

    avg_age_by_survival = df.groupby("Survived")["Age"].mean()
    print(avg_age_by_survival)

------------------------------------------------------------

## 9. Save Clean Dataset

    df.to_csv("data/titanic/titanic_clean.csv", index=False)

✅ You now have a **cleaned Titanic dataset**.

------------------------------------------------------------

## 10. Extra Pandas Tasks – Titanic (Recommended)

These improve your feature engineering skills and help your future ML model.

### 10.1. Create FamilySize

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(df[["SibSp", "Parch", "FamilySize"]].head())

    print(df.groupby("FamilySize")["Survived"].mean())

### 10.2. Create IsAlone

    df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")
    print(df[["FamilySize", "IsAlone"]].head())

    print(df.groupby("IsAlone")["Survived"].mean())

### 10.3. Create AgeGroup

    bins = [0, 12, 18, 50, 100]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    print(df[["Age", "AgeGroup"]].head())
    print(df.groupby("AgeGroup")["Survived"].mean())

### 10.4. Create FareBucket

    df["FareBucket"] = pd.qcut(df["Fare"], 4, labels=["Low", "Med-Low", "Med-High", "High"])
    print(df[["Fare", "FareBucket"]].head())

    print(df.groupby("FareBucket")["Survived"].mean())

### 10.5. Prepare ML-Ready Subset

We choose a small set of useful columns for modeling:

    ml_columns = [
        "Survived",      # target
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone"
    ]

    df_ml = df[ml_columns].copy()
    df_ml.to_csv("data/titanic/titanic_ml_ready.csv", index=False)

✅ Now you are ready to train an ML model on Titanic.

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 1B  
## Titanic – Your First ML Model with scikit-learn

### Goal

Use **scikit-learn** to:

- Treat `Survived` as the target (0/1)
- Use other columns as features
- Train a **Logistic Regression** classifier
- Evaluate and interpret the model

------------------------------------------------------------

## 1. Conceptual Overview

- Target (y): `Survived` → 0 or 1  
- Features (X): passenger data (class, sex, age, fare, etc.)  
- Task: **binary classification**

We will:

1. Load the ML-ready data from pandas stage.
2. Encode categorical variables.
3. Split into training set and test set.
4. Train a Logistic Regression model.
5. Evaluate accuracy, precision, recall, etc.
6. Inspect feature importance via coefficients.

------------------------------------------------------------

## 2. Create a New Notebook / Script

Create:

- `notebooks/titanic_sklearn.ipynb`, or
- `notebooks/titanic_sklearn.py`

Imports:

    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

------------------------------------------------------------

## 3. Load the ML-Ready Titanic Data

If you created `titanic_ml_ready.csv`:

    df = pd.read_csv("data/titanic/titanic_ml_ready.csv")

    print(df.head())
    print(df.info())

If not, you can start from `titanic_clean.csv` and recompute features:

    df = pd.read_csv("data/titanic/titanic_clean.csv")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")

    df = df[[
        "Survived",
        "Pclass", "Sex", "Age", "Fare",
        "Embarked", "FamilySize", "IsAlone"
    ]].copy()

------------------------------------------------------------

## 4. Split into Features (X) and Target (y)

    target_col = "Survived"

    feature_cols = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone"
    ]

    X = df[feature_cols].copy()
    y = df[target_col]

Check:

    print(X.head())
    print(y.head())

------------------------------------------------------------

## 5. Encode Categorical Features

`scikit-learn` works with numeric arrays, so we use one-hot encoding:

    X_encoded = pd.get_dummies(X, drop_first=True)

    print(X_encoded.head())
    print(X_encoded.columns)
    print(X_encoded.shape)

Now, `X_encoded` contains only numeric columns:
- Dummy columns for `Pclass`, `Sex`, `Embarked`
- Numeric columns: `Age`, `Fare`, `FamilySize`, `IsAlone`

------------------------------------------------------------

## 6. Train/Test Split

We reserve 20% of the data as a test set:

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keep class balance similar in train & test
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

------------------------------------------------------------

## 7. Train Logistic Regression Model

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

After `fit`, the model has learned a set of coefficients for each feature.

------------------------------------------------------------

## 8. Make Predictions

    y_pred = model.predict(X_test)

Optional: predicted probabilities for the positive class (Survived = 1):

    y_proba = model.predict_proba(X_test)[:, 1]
    print(y_proba[:10])

------------------------------------------------------------

## 9. Evaluate the Model

### 9.1. Accuracy

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

### 9.2. Detailed Metrics

    print(classification_report(y_test, y_pred))

You will see:

- precision
- recall
- f1-score
- support

for each class: 0 (did not survive), 1 (survived).

### 9.3. Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

Make it more readable:

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"]
    )
    print(cm_df)

------------------------------------------------------------

## 10. Inspect Feature Importance (Coefficients)

For Logistic Regression:

    feature_names = X_encoded.columns
    coef = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef
    }).sort_values(by="coefficient", ascending=False)

    print(coef_df)

Interpretation:

- Positive coefficient → increases probability of survival.
- Negative coefficient → decreases probability of survival.

------------------------------------------------------------

## 11. Extra ML Tasks – Titanic

1. Remove some features (e.g. `IsAlone`) and retrain:
   - How does accuracy change?
2. Add `AgeGroup` or `FareBucket` as additional features (encode them with `get_dummies`) and retrain.
3. Try a `RandomForestClassifier`:

       from sklearn.ensemble import RandomForestClassifier

       rf_model = RandomForestClassifier(
           n_estimators=200,
           random_state=42
       )
       rf_model.fit(X_train, y_train)

       y_pred_rf = rf_model.predict(X_test)
       print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
       print(classification_report(y_test, y_pred_rf))

4. Compare Logistic Regression vs Random Forest:
   - Which has higher test accuracy?
   - Which seems to overfit (if train performance >> test performance)?

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 2  
## Netflix Dataset – Pandas Filtering, Text & Grouping

### Goal

Practice with pandas on a media catalog dataset:

- Handle missing values
- Work with text columns (genres, casts)
- Group and aggregate
- Work with dates
- Prepare a curated dataset for possible ML

------------------------------------------------------------

## 1. Get the Dataset

1. Go to Kaggle.
2. Search: “Netflix Movies and TV Shows”.
3. Download `netflix_titles.csv`.
4. Place it at:

       data/netflix/netflix_titles.csv

------------------------------------------------------------

## 2. Create a File / Notebook

Create:

- `notebooks/netflix_pandas.ipynb`, or
- `notebooks/netflix_pandas.py`

------------------------------------------------------------

## 3. Load & Inspect

    import pandas as pd

    df = pd.read_csv("data/netflix/netflix_titles.csv")

    print(df.head())
    print(df.shape)
    print(df.info())

------------------------------------------------------------

## 4. Handle Missing Data

Check missing values:

    print(df.isna().sum())

Fill missing `country`:

    df["country"] = df["country"].fillna("Unknown")

Drop rows missing `title` or `type`:

    before_rows = len(df)
    df = df.dropna(subset=["title", "type"])
    after_rows = len(df)

    print("Rows dropped (missing title/type):", before_rows - after_rows)

------------------------------------------------------------

## 5. Convert date_added to datetime

    df["date_added"] = pd.to_datetime(df["date_added"])

    print(df["date_added"].head())

Create a `year_added` column:

    df["year_added"] = df["date_added"].dt.year

------------------------------------------------------------

## 6. Basic Exploration

### 6.1. Movies vs TV Shows

    type_counts = df["type"].value_counts()
    print(type_counts)

### 6.2. Titles Released in a Specific Year (e.g., 2020)

    titles_2020 = df[df["release_year"] == 2020]
    print("Titles released in 2020:", len(titles_2020))
    print(titles_2020[["title", "type"]].head(10))

### 6.3. Titles Added per Year

    titles_per_year_added = df["year_added"].value_counts().sort_index()
    print(titles_per_year_added)

------------------------------------------------------------

## 7. Country Analysis

Extract the main (first) country:

    df["main_country"] = (
        df["country"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

Top 10 countries by number of titles:

    top_countries = df["main_country"].value_counts().head(10)
    print(top_countries)

------------------------------------------------------------

## 8. TV Shows with Many Seasons

Filter to TV shows:

    tv = df[df["type"] == "TV Show"].copy()

Extract season count from `duration` (e.g. "3 Seasons", "1 Season"):

    tv["seasons"] = tv["duration"].str.extract(r"(\d+)").astype(float)

Filter shows with 5+ seasons:

    long_shows = tv[tv["seasons"] >= 5]
    print("TV shows with 5+ seasons:", len(long_shows))
    print(long_shows[["title", "seasons"]].head(10))

------------------------------------------------------------

## 9. Genre Filtering

Find all titles that contain "Comedies" in `listed_in`:

    comedies = df[df["listed_in"].str.contains("Comedies", na=False)]
    print("Total comedies:", len(comedies))
    print(comedies[["title", "type", "listed_in"]].head(10))

------------------------------------------------------------

## 10. Save a Curated Dataset

Create a smaller dataset with selected columns:

    export_columns = [
        "show_id", "type", "title",
        "main_country", "release_year",
        "rating", "duration",
        "listed_in", "date_added", "year_added"
    ]

    df_curated = df[export_columns].copy()
    df_curated.to_csv("data/netflix/netflix_curated.csv", index=False)

✅ You now have `netflix_curated.csv` for further analysis or ML.

------------------------------------------------------------

## 11. Extra Pandas Tasks – Netflix

### 11.1. Top Genres

    genres = (
        df["listed_in"]
        .str.split(",")
        .explode()
        .str.strip()
    )

    genre_counts = genres.value_counts().head(20)
    print(genre_counts)

### 11.2. Top 10 Directors

    directors = df.dropna(subset=["director"])
    director_counts = directors["director"].value_counts().head(10)
    print(director_counts)

### 11.3. Top Actors (More Advanced)

    cast_series = (
        df["cast"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )

    top_actors = cast_series.value_counts().head(20)
    print(top_actors)

### 11.4. Movies vs TV Shows per Rating

    rating_type_counts = df.groupby(["rating", "type"])["show_id"].count()
    print(rating_type_counts)

    print(rating_type_counts.unstack(fill_value=0))

### 11.5. Titles Added per Year (Non-missing)

    titles_per_year = df.dropna(subset=["year_added"])["year_added"].value_counts().sort_index()
    print(titles_per_year)

------------------------------------------------------------
------------------------------------------------------------

# PROJECT 2B  
## Netflix – Simple ML Model with scikit-learn (Movie vs TV Show)

### Goal

Use **scikit-learn** to:

- Predict whether a title is a **Movie** or a **TV Show**
- Use simple metadata like country, years, rating
- Practice the same ML steps as Titanic on a different dataset

This is more for **practice** than for serious modeling, but it uses real data.

------------------------------------------------------------

## 1. Create a New Notebook / Script

Create:

- `notebooks/netflix_sklearn.ipynb`, or
- `notebooks/netflix_sklearn.py`

Imports:

    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

------------------------------------------------------------

## 2. Load Curated Netflix Data

    df = pd.read_csv("data/netflix/netflix_curated.csv")

    print(df.head())
    print(df.info())

Expect columns like:

- type (Movie / TV Show) – target
- main_country
- release_year
- rating
- duration
- listed_in
- date_added
- year_added

------------------------------------------------------------

## 3. Choose Features and Target

We will be simple and use:

- Target (`y`): `type`  
- Features (`X`): `main_country`, `release_year`, `rating`, `year_added`

    target_col = "type"
    feature_cols = ["main_country", "release_year", "rating", "year_added"]

    X = df[feature_cols].copy()
    y = df[target_col]

Check:

    print(X.head())
    print(y.head())

------------------------------------------------------------

## 4. Handle Missing year_added (If Any)

    print(X.isna().sum())

If `year_added` has missing values, fill with 0 as a simple placeholder:

    X["year_added"] = X["year_added"].fillna(0)

------------------------------------------------------------

## 5. Encode Categorical Features

Categorical: `main_country`, `rating`  
Numeric: `release_year`, `year_added`

    X_encoded = pd.get_dummies(X, drop_first=True)

    print(X_encoded.head())
    print(X_encoded.shape)

------------------------------------------------------------

## 6. Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

------------------------------------------------------------

## 7. Train Logistic Regression

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

------------------------------------------------------------

## 8. Predict & Evaluate

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print(classification_report(y_test, y_pred))

Confusion matrix:

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Movie", "Actual_TV_Show"],
        columns=["Pred_Movie", "Pred_TV_Show"]
    )
    print(cm_df)

------------------------------------------------------------

## 9. Extra ML Tasks – Netflix

1. Add more features:
   - For example, extract whether `duration` contains "Season" or "min" (seasons vs minutes).
2. Add one-hot encoded **top genres** as features (from `listed_in`).
3. Train a `RandomForestClassifier` like in the Titanic project and compare results:

       from sklearn.ensemble import RandomForestClassifier

       rf_model = RandomForestClassifier(
           n_estimators=200,
           random_state=42
       )
       rf_model.fit(X_train, y_train)

       y_pred_rf = rf_model.predict(X_test)
       print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
       print(classification_report(y_test, y_pred_rf))

4. Compare:
   - Logistic Regression vs Random Forest
   - Which is more accurate?
   - Which might be overfitting?

------------------------------------------------------------
------------------------------------------------------------

# Summary – What You’ve Practiced

By working through this file, you will have gained:

## Pandas Skills

- Loading CSV files
- Inspecting data (`head`, `info`, `describe`)
- Handling missing values:
  - `fillna`, `dropna`
- Dropping and selecting columns
- Filtering rows with conditions
- Grouping and aggregation (`groupby`, `value_counts`)
- Working with:
  - Categorical data
  - Text columns (`str.split`, `str.contains`, `explode`)
  - Dates (`to_datetime`, `.dt.year`)
- Feature engineering:
  - FamilySize, IsAlone
  - AgeGroup, FareBucket
  - main_country from country
  - year_added from date_added

## scikit-learn Skills

- Defining features (X) and target (y)
- Encoding categorical variables using one-hot encoding (`get_dummies`)
- Splitting data into train and test (`train_test_split`)
- Training:
  - LogisticRegression
  - (optionally) RandomForestClassifier
- Evaluating models using:
  - Accuracy
  - Classification report (precision, recall, f1-score)
  - Confusion matrix
- Inspecting feature importance (coefficients in Logistic Regression)
- Comparing different models on the same dataset

You now have a strong, practical base to continue with:

- More **NumPy**
- More ML algorithms (SVM, gradient boosting, XGBoost)
- Cross-validation, hyperparameter tuning, pipelines, etc.

Happy coding and learning 🚀
