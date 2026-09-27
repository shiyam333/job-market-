# Technologies & Libraries Used
## Job Market & Salary Intelligence Project

---

## 1. Core Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3 |
| Development Environment | Jupyter Notebook / Python script (.py) |
| Data Storage / Interchange | CSV files |
| Reporting | PDF (generated via ReportLab) |
| Version used to build this project | Python 3.12, pandas 3.0, scikit-learn 1.8 |

---

## 2. Python Libraries Used

| Library | Version (used) | Purpose in this project |
|---|---|---|
| **pandas** | 3.0.2 | Loading, cleaning, transforming, grouping and exporting tabular job-posting data (DataFrames) |
| **numpy** | 2.4.4 | Numerical operations — IQR outlier bounds, `np.clip()` capping, random sample data generation |
| **matplotlib** | 3.10.8 | Base plotting engine for bar charts, histograms, scatter plots, line charts, pie chart |
| **seaborn** | 0.13.2 | Statistical charts — box plots (outlier visualization) and heatmaps (missing values, correlation, pivot tables) |
| **scikit-learn** | 1.8.0 | Machine learning: `train_test_split`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `LinearRegression`, `RandomForestRegressor`, and evaluation metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`) |
| **os** (standard lib) | built-in | File/folder handling, checking existing files, managing output paths |
| **reportlab** | 4.4.10 | Generating the final project PDF report |

---

## 3. Machine Learning Techniques

| Technique | Where Used |
|---|---|
| **One-Hot Encoding** (`OneHotEncoder`) | Converting categorical columns (Job Title, Location, Skills, Employment Type) into numeric form |
| **ColumnTransformer + Pipeline** | Chaining preprocessing and model training into a single reusable object |
| **Linear Regression** | Primary salary-prediction model (interpretable baseline) |
| **Random Forest Regression** | Non-linear ensemble model used for comparison |
| **Train/Test Split (80/20)** | Standard supervised-learning evaluation setup |
| **IQR Method** | Statistical technique for detecting and capping salary outliers |
| **Evaluation Metrics** | MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R² (coefficient of determination) |

---

## 4. Data Engineering Techniques

- Duplicate detection & removal (`drop_duplicates`)
- Data-type conversion (`pd.to_numeric`, `pd.to_datetime`)
- Missing-value imputation (median for salaries, mode for location, constant fill for skills)
- Text standardization (`str.strip()`, `str.title()`)
- Data validation (negative values, Salary_Min > Salary_Max correction)
- Feature engineering (`Experience_Years` mapping, `Avg_Salary`, `Posted_Month`)
- Skill parsing via string split + `explode()` + one-hot skill flag columns
- Target-leakage avoidance (Salary_Min/Salary_Max excluded from model features since Avg_Salary is derived from them)

---

## 5. Why These Tools Were Chosen

- **pandas / numpy** — industry-standard for tabular data wrangling in Python; fast and well-documented.
- **matplotlib / seaborn** — together they cover every chart type needed (bar, pie, histogram, scatter, box plot, heatmap) without extra dependencies.
- **scikit-learn** — provides a consistent, production-style API (`Pipeline`, `ColumnTransformer`) so preprocessing and modeling stay reproducible and leak-free.
- **Linear Regression vs Random Forest** — comparing a simple linear (interpretable) model against a non-linear ensemble model shows whether salary is mostly linearly explained by experience/role/location, or needs non-linear interactions.
- **CSV format** — universally portable, easy to open in Excel/Google Sheets/Power BI for further reporting.
- **ReportLab** — lightweight, code-driven PDF generation, ideal for turning a data-science pipeline directly into a shareable report.
