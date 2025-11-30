# Pandas Hands-On Practice Projects 🐼

This file contains **two beginner-friendly, real-world pandas projects**
with **clear step-by-step instructions** and **extra tasks** to deepen your practice.

You can follow this exactly as written.
No prior ML knowledge required.

----------------------------------------------------

## Prerequisites

- Python 3.8+
- Basic Python knowledge
- Pandas installed

Install pandas if needed:

    pip install pandas

You can use:
- Jupyter Notebook, OR
- VS Code + `.py` files

Directory suggestion:

    project/
      data/
        titanic/
        netflix/
      notebooks_or_scripts_here...

----------------------------------------------------
----------------------------------------------------

# PROJECT 1
## Titanic Dataset – Data Cleaning & Exploration

### Goal

Learn how pandas is used to:
- Load real data
- Identify missing / invalid data
- Clean data step by step
- Filter rows
- Answer simple questions
- (Extra tasks) engineer features and build more complex queries

----------------------------------------------------

## Step 1 – Get Dataset

1. Go to **Kaggle**
2. Search: **Titanic - Machine Learning from Disaster**
3. Download `train.csv`
4. Place it here:

    project/
      data/
        titanic/
          train.csv

----------------------------------------------------

## Step 2 – Create File

Create one of the following:

- `project1_titanic.ipynb` (recommended)
OR
- `project1_titanic.py`

----------------------------------------------------

## Step 3 – Load & Inspect Data

    import pandas as pd

    df = pd.read_csv("data/titanic/train.csv")

    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())

----------------------------------------------------

## Step 4 – Find Missing Values

    print(df.isna().sum())

Focus on:
- Age
- Cabin
- Embarked

Optionally, inspect some rows:

    print(df[df["Embarked"].isna()])
    print(df[df["Age"].isna()].head())

----------------------------------------------------

## Step 5 – Drop Unnecessary Columns

Cabin has too many missing values for now.

    df = df.drop(columns=["Cabin", "Ticket"])
    print(df.columns)

----------------------------------------------------

## Step 6 – Fix Missing Values

### Fill Embarked with most common value

    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

### Fill Age with median

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    print(df.isna().sum())

----------------------------------------------------

## Step 7 – Remove Invalid Data

Remove passengers with invalid fare values:

    before_rows = len(df)
    df = df[df["Fare"] > 0]
    after_rows = len(df)

    print("Rows removed with non-positive Fare:", before_rows - after_rows)

----------------------------------------------------

## Step 8 – Fix Data Types

    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    df["Pclass"] = df["Pclass"].astype("category")
    df["Survived"] = df["Survived"].astype("int8")

    print(df.dtypes)

----------------------------------------------------

## Step 9 – Data Exploration (Core Tasks)

### 9.1 Overall survival rate

    total = len(df)
    survived = df["Survived"].sum()
    print("Total passengers:", total)
    print("Survived:", survived)
    print("Survival Rate:", survived / total)

### 9.2 Survival by gender

    survival_by_sex = df.groupby("Sex")["Survived"].mean()
    print(survival_by_sex)

### 9.3 Survival by class

    survival_by_class = df.groupby("Pclass")["Survived"].mean()
    print(survival_by_class)

### 9.4 Average age of survivors vs non-survivors

    avg_age_by_survival = df.groupby("Survived")["Age"].mean()
    print(avg_age_by_survival)

### 9.5 Survival rate for women in 3rd class

    women_third = df[(df["Sex"] == "female") & (df["Pclass"] == 3)]
    print("Women in 3rd class:", len(women_third))
    print("Survival rate:", women_third["Survived"].mean())

----------------------------------------------------

## Step 10 – Save Clean Dataset

    df.to_csv("data/titanic/titanic_clean.csv", index=False)

✅ **Core Project 1 complete**

----------------------------------------------------

## EXTRA TASKS – Titanic 🧠

These are **optional but highly recommended** for more practice.

### Task 1 – Create a FamilySize feature

Add a column that shows how many people from the same family are aboard:
`FamilySize = SibSp + Parch + 1`

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(df[["SibSp", "Parch", "FamilySize"]].head())

Then:

- Compute survival rate by `FamilySize`:

      print(df.groupby("FamilySize")["Survived"].mean())

- Observe: do small families have better survival?

----------------------------------------------------

### Task 2 – Create an IsAlone feature

Use `FamilySize` to create a new boolean/int feature:

- 1 if passenger is alone (`FamilySize == 1`)
- 0 otherwise

    df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")
    print(df[["FamilySize", "IsAlone"]].head())

Compare survival:

    print(df.groupby("IsAlone")["Survived"].mean())

----------------------------------------------------

### Task 3 – Age groups / bins

Use `pd.cut` to split passengers into age groups:

- Child (0–12)
- Teen (12–18)
- Adult (18–50)
- Senior (50+)

Example:

    bins = [0, 12, 18, 50, 100]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    print(df[["Age", "AgeGroup"]].head())

Then:

- Survival by `AgeGroup`:

      print(df.groupby("AgeGroup")["Survived"].mean())

----------------------------------------------------

### Task 4 – Combined group analysis

Compute survival rate grouped by **Sex + Pclass**:

    combo = df.groupby(["Sex", "Pclass"])["Survived"].mean()
    print(combo)

You can also convert this to a small table (unstacked):

    print(combo.unstack())

----------------------------------------------------

### Task 5 – Explore Fare distribution

- Create a new column `FareBucket` using `pd.qcut` to split Fare into quartiles:

      df["FareBucket"] = pd.qcut(df["Fare"], 4, labels=["Low", "Med-Low", "Med-High", "High"])
      print(df[["Fare", "FareBucket"]].head())

- Compare survival rate per `FareBucket`:

      print(df.groupby("FareBucket")["Survived"].mean())

----------------------------------------------------

### Task 6 – Top 5 families by count

- Extract last name from the `Name` column.
  Hint: last name is the string before the comma.

      df["LastName"] = df["Name"].str.split(",").str[0]
      print(df[["Name", "LastName"]].head())

- Count how many passengers by `LastName`:

      family_counts = df["LastName"].value_counts().head(5)
      print(family_counts)

----------------------------------------------------

### Task 7 – Export a subset for ML

Export only the columns you think are useful as ML features:

- Survived
- Pclass
- Sex
- Age
- Fare
- Embarked
- FamilySize
- IsAlone
- AgeGroup
- FareBucket

Example:

    ml_columns = [
        "Survived", "Pclass", "Sex", "Age", "Fare",
        "Embarked", "FamilySize", "IsAlone",
        "AgeGroup", "FareBucket"
    ]

    df_ml = df[ml_columns].copy()
    df_ml.to_csv("data/titanic/titanic_ml_ready.csv", index=False)

----------------------------------------------------
----------------------------------------------------

# PROJECT 2
## Netflix Dataset – Filtering, Text & Grouping

### Goal

Practice:
- Filtering rows
- Text operations on string columns
- Grouping & counting
- Working with dates
- (Extra tasks) deeper analysis with genres, actors, and time

----------------------------------------------------

## Step 1 – Get Dataset

1. Go to **Kaggle**
2. Search: **Netflix Movies and TV Shows**
3. Download `netflix_titles.csv`
4. Place it here:

    project/
      data/
        netflix/
          netflix_titles.csv

----------------------------------------------------

## Step 2 – Create File

Create:

- `project2_netflix.ipynb`
OR
- `project2_netflix.py`

----------------------------------------------------

## Step 3 – Load Data

    import pandas as pd

    df = pd.read_csv("data/netflix/netflix_titles.csv")

    print(df.head())
    print(df.shape)
    print(df.info())

----------------------------------------------------

## Step 4 – Handle Missing Data

Check missing values:

    print(df.isna().sum())

Fill missing country values:

    df["country"] = df["country"].fillna("Unknown")

Drop rows missing `title` or `type`:

    before_rows = len(df)
    df = df.dropna(subset=["title", "type"])
    after_rows = len(df)

    print("Rows dropped (missing title/type):", before_rows - after_rows)

----------------------------------------------------

## Step 5 – Convert Dates

Convert `date_added` to datetime:

    df["date_added"] = pd.to_datetime(df["date_added"])
    print(df["date_added"].head())

Create a `year_added` column:

    df["year_added"] = df["date_added"].dt.year

----------------------------------------------------

## Step 6 – Explore Data (Core Tasks)

### 6.1 Movies vs TV Shows

    type_counts = df["type"].value_counts()
    print(type_counts)

### 6.2 Titles released in a specific year (e.g. 2020)

    titles_2020 = df[df["release_year"] == 2020]
    print("Titles released in 2020:", len(titles_2020))
    print(titles_2020[["title", "type"]].head(10))

### 6.3 Titles added per year (based on date_added)

    titles_per_year_added = df["year_added"].value_counts().sort_index()
    print(titles_per_year_added)

----------------------------------------------------

## Step 7 – Country Analysis

Extract main country:

    df["main_country"] = (
        df["country"]
        .str.split(",")
        .str[0]
        .str.strip()
    )

Top 10 countries:

    top_countries = df["main_country"].value_counts().head(10)
    print(top_countries)

----------------------------------------------------

## Step 8 – TV Shows with Many Seasons

Filter to TV Shows:

    tv = df[df["type"] == "TV Show"].copy()

Extract number of seasons from "3 Seasons" or "1 Season":

    tv["seasons"] = tv["duration"].str.extract(r"(\\d+)").astype(float)

Filter 5+ seasons:

    long_shows = tv[tv["seasons"] >= 5]
    print("TV shows with 5+ seasons:", len(long_shows))
    print(long_shows[["title", "seasons"]].head(10))

----------------------------------------------------

## Step 9 – Genre Filtering (listed_in)

Count “Comedies”:

    comedies = df[df["listed_in"].str.contains("Comedies", na=False)]
    print("Total comedies:", len(comedies))

Inspect some:

    print(comedies[["title", "type", "listed_in"]].head(10))

----------------------------------------------------

## Step 10 – Save Result

Movies released from 2015 onwards:

    recent_movies = df[
        (df["type"] == "Movie") &
        (df["release_year"] >= 2015)
    ]

    recent_movies.to_csv(
        "data/netflix/netflix_movies_2015_onwards.csv",
        index=False
    )

✅ **Core Project 2 complete**

----------------------------------------------------

## EXTRA TASKS – Netflix 🧠

### Task 1 – Top genres

The `listed_in` column contains comma-separated genres.

Goal: Find the most common genres overall.

Steps:

1. Split genres into a list.
2. Explode into separate rows.
3. Count frequencies.

Example:

    # Step 1 & 2: split and explode
    genres = (
        df["listed_in"]
        .str.split(",")
        .explode()
        .str.strip()
    )

    # Step 3: count
    genre_counts = genres.value_counts().head(20)
    print(genre_counts)

----------------------------------------------------

### Task 2 – Top 10 directors by number of titles

    # Drop missing directors first
    directors = df.dropna(subset=["director"])

    director_counts = directors["director"].value_counts().head(10)
    print(director_counts)

----------------------------------------------------

### Task 3 – Actor frequency (harder)

Goal: Find the actors with the most appearances.

Steps:

1. Drop rows with missing `cast`.
2. Split `cast` by comma.
3. Explode.
4. Strip whitespace.
5. Count.

Example:

    cast_series = (
        df["cast"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )

    top_actors = cast_series.value_counts().head(20)
    print(top_actors)

----------------------------------------------------

### Task 4 – Movies vs TV Shows per rating

Count how many Movies vs TV Shows exist in each `rating`:

    rating_type_counts = df.groupby(["rating", "type"])["show_id"].count()
    print(rating_type_counts)

Optionally, unstack for better readability:

    print(rating_type_counts.unstack(fill_value=0))

----------------------------------------------------

### Task 5 – Trend: how many titles added per year

We already created `year_added`. Now:

    titles_per_year = df["year_added"].value_counts().sort_index()
    print(titles_per_year)

Extra:

- Filter out rows where `year_added` is NaN:

      titles_per_year = df.dropna(subset=["year_added"])["year_added"].value_counts().sort_index()
      print(titles_per_year)

----------------------------------------------------

### Task 6 – Country + Type combination

Check which countries produce more Movies vs TV Shows:

    country_type = df.groupby(["main_country", "type"])["show_id"].count()
    print(country_type.sort_values(ascending=False).head(20))

----------------------------------------------------

### Task 7 – Export a curated dataset

Create a new DataFrame with only the following columns:

- show_id
- type
- title
- main_country
- release_year
- rating
- duration
- listed_in
- year_added

Then save it:

    export_columns = [
        "show_id", "type", "title",
        "main_country", "release_year",
        "rating", "duration",
        "listed_in", "year_added"
    ]

    df_export = df[export_columns].copy()

    df_export.to_csv(
        "data/netflix/netflix_curated.csv",
        index=False
    )

----------------------------------------------------
----------------------------------------------------

## What You Practiced Across Both Projects

- Reading & writing CSVs (`read_csv`, `to_csv`)
- Detecting & fixing missing data (`isna`, `fillna`, `dropna`)
- Dropping and selecting columns (`drop`, column lists)
- Row filtering with conditions (boolean indexing)
- Grouping & aggregation (`groupby`, `value_counts`)
- Working with categories (`astype("category")`)
- String operations (`str.split`, `str.contains`, `str.strip`, `str.extract`, `explode`)
- Date handling (`to_datetime`, `.dt.year`)
- Basic feature engineering (new columns like `FamilySize`, `IsAlone`, age bins, genre breakdown)

➡️ After finishing these projects (including some extra tasks), you’ll have a **solid practical foundation** for:
- Going deeper into **NumPy**
- Starting with **machine learning** (e.g., using `titanic_ml_ready.csv`)

Happy coding! 🚀
