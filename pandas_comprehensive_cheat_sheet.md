
# 🐼 Pandas Comprehensive Cheat Sheet  
### Important Functions, Explanations & Real‑World Usage

This document is a **broad and practical reference** to Pandas.  
It explains:
- ✅ What a function does  
- ✅ Why it is commonly used  
- ✅ Where it appears in real-world data work  

Save as **pandas_comprehensive_cheat_sheet.md**

---

## 1. Core Data Structures

### pd.Series
Creates a one‑dimensional labeled array.

```python
s = pd.Series([100, 200, 300], name="revenue")
```
**Used for:** Single metrics, time series, model outputs.

---

### pd.DataFrame
Two‑dimensional tabular data structure.

```python
df = pd.DataFrame({
    "product": ["A", "B"],
    "price": [10, 15]
})
```
**Used for:** Almost all real‑world datasets.

---

## 2. Reading & Writing Data

### Reading files
```python
pd.read_csv("data.csv")
pd.read_excel("file.xlsx")
pd.read_json("data.json")
pd.read_parquet("data.parquet")
```

**Use case:** Ingesting data from reports, storage, APIs.

---

### Writing files
```python
df.to_csv("out.csv", index=False)
df.to_excel("out.xlsx", index=False)
df.to_parquet("out.parquet")
```

**Use case:** Sharing results with business or pipelines.

---

## 3. Inspecting Data

```python
df.head()      # preview data
df.tail()
df.sample(5)
df.shape
df.columns
df.index
```

```python
df.info()      # datatypes & missing values
df.describe()  # summary statistics
```
**Use case:** First step of any analysis (EDA).

---

## 4. Selecting Data

### Column selection
```python
df["salary"]
df[["name", "salary"]]
```

### Row selection
```python
df.loc[df["salary"] > 50000, ["name", "department"]]
df.iloc[0:10, 0:3]
```

**Use case:** Feature selection, reporting, validation.

---

## 5. Filtering Rows

```python
df[df["age"] >= 18]
df[df["country"].isin(["US", "UK"])]
df[df["score"].between(60, 90)]
```

**Use case:** Customer filtering, cohort analysis.

---

## 6. Creating & Updating Columns

```python
df["revenue"] = df["price"] * df["quantity"]
df["log_salary"] = np.log(df["salary"])
```

### apply()
```python
df["tax"] = df["salary"].apply(lambda x: x * 0.3)
```

**Use case:** Business rules, feature engineering.

---

## 7. Dropping Data (VERY IMPORTANT)

### drop columns
```python
df.drop(columns=["temp_col", "unused"], inplace=True)
```

### drop rows
```python
df.drop(index=[0, 1])
df.dropna()
```

**Use case:** Cleaning noisy or invalid data.

---

## 8. Handling Missing Values

```python
df.isna().sum()
df.fillna(0)
df["age"].fillna(df["age"].median())
```

**Use case:** Required step before ML models.

---

## 9. Sorting & Ranking

```python
df.sort_values("sales", ascending=False)
df["rank"] = df["sales"].rank(method="dense")
```

**Use case:** Leaderboards, top‑N analyses.

---

## 10. Aggregation & GroupBy

```python
df.groupby("department")["salary"].mean()
```

```python
df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    count=("id", "count")
)
```

**Use case:** KPI dashboards, business metrics.

---

## 11. Correlation (Statistics)

### corr()
```python
df.corr(numeric_only=True)
```

### corrwith()
```python
df.corrwith(df["target"])
```

**Use case:** Feature selection, data science, finance.

---

## 12. Duplicates

```python
df.duplicated()
df.drop_duplicates()
df.drop_duplicates(subset=["email"])
```

**Use case:** Customer & transaction deduplication.

---

## 13. Merging & Concatenation

### concat()
```python
pd.concat([df1, df2], ignore_index=True)
```

### merge()
```python
pd.merge(df_orders, df_customers, on="customer_id", how="left")
```

**Use case:** SQL‑style joins between datasets.

---

## 14. Reshaping Data

### pivot_table
```python
pd.pivot_table(
    df,
    values="sales",
    index="region",
    columns="month",
    aggfunc="sum",
    fill_value=0
)
```

### melt
```python
df.melt(
    id_vars=["id"],
    value_vars=["q1", "q2"],
    var_name="quarter",
    value_name="sales"
)
```

**Use case:** Reporting & dashboards.

---

## 15. String Operations

```python
df["email"].str.contains("@gmail")
df["name"].str.upper()
df["city"].str.strip()
```

**Use case:** Text cleaning.

---

## 16. Date & Time Operations

```python
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df.set_index("date").resample("M")["sales"].sum()
```

**Use case:** Time‑series analysis.

---

## 17. Window / Rolling Functions

```python
df["rolling_avg"] = df["sales"].rolling(7).mean()
df["cumsum"] = df["sales"].cumsum()
```

**Use case:** Trend analysis, finance.

---

## 18. Value Counts & Frequency

```python
df["country"].value_counts(normalize=True)
```

**Use case:** Market & demographic analysis.

---

## 19. Performance Tips

```python
df.astype("category")
df.select_dtypes(include=["number"])
```

---

## 20. End‑to‑End Real‑World Example

### Sales Analytics Pipeline

```python
df = pd.read_csv("orders.csv")
df["date"] = pd.to_datetime(df["date"])

df["revenue"] = df["price"] * df["quantity"]

monthly_sales = (
    df
    .set_index("date")
    .resample("M")["revenue"]
    .sum()
)

top_products = (
    df.groupby("product")["revenue"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)
```

**Explanation:**
1. Load raw data
2. Clean & convert datatypes
3. Create business metrics
4. Aggregate results for reporting

---

## ✅ Summary

This file now includes:
- ✅ 50+ essential Pandas functions
- ✅ Clear explanations for each group
- ✅ Real‑world usage context
- ✅ Statistics, cleaning, joining & reshaping
- ✅ Production‑ready patterns

