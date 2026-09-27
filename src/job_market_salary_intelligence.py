# ================================================================
# JOB MARKET & SALARY INTELLIGENCE
# COMPLETE END-TO-END PYTHON PROJECT
# ================================================================
# This script:
#   1. Loads (or generates) job-postings data
#   2. Cleans it (duplicates, dtypes, missing values, validation)
#   3. Handles outliers (IQR capping)
#   4. Engineers features (Experience_Years, Avg_Salary, Posted_Month)
#   5. Extracts and analyzes skills
#   6. Produces EDA visualizations (saved as PNG files)
#   7. Builds Linear Regression + Random Forest salary-prediction models
#   8. Exports all cleaned data / summaries / charts to the outputs folder
#
# Folder layout expected (created automatically if missing):
#   outputs/data/     -> all CSV files
#   outputs/charts/    -> all PNG chart images
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
import os

# ----------------------------------------------------------------
# ENVIRONMENT DETECTION
# ----------------------------------------------------------------
# Detect whether we're running inside a notebook (Jupyter / Colab)
# or as a plain .py script. In a notebook we want charts to display
# inline in the cell output AND save to file. In a plain script
# (VS Code "Run", terminal, etc.) there is usually no display, so we
# force the non-interactive "Agg" backend so it never tries (and
# fails) to open a GUI window -- it just saves the file quietly.
try:
    ip = get_ipython()          # only exists inside Jupyter/Colab
except NameError:
    ip = None

IN_NOTEBOOK = ip is not None

if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

if IN_NOTEBOOK:
    try:
        ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# ----------------------------------------------------------------
# OUTPUT FOLDERS
# ----------------------------------------------------------------
# Works whether this runs as a saved .py script OR pasted into a
# notebook cell (Colab/Jupyter), where __file__ does not exist.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
except NameError:
    PROJECT_DIR = os.getcwd()  # e.g. /content in Colab

DATA_DIR = os.path.join(PROJECT_DIR, "outputs", "data")
CHART_DIR = os.path.join(PROJECT_DIR, "outputs", "charts")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


def save_chart(filename):
    """Save the current matplotlib figure to the charts folder.
    In a notebook it also displays inline; either way the PNG is saved."""
    path = os.path.join(CHART_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    plt.close()
    print("Chart saved:", filename)


print("Libraries imported successfully.")

# ================================================================
# 2. LOAD / GENERATE DATASET
# ================================================================

file_name = os.path.join(DATA_DIR, "Job_Market_Salary_Intelligence_1000_Raw.csv")

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
    print("Existing dataset loaded successfully.")
else:
    np.random.seed(42)

    job_titles = ["Data Analyst", "Data Scientist", "Business Analyst",
                  "Python Developer", "SQL Developer", "Power BI Developer"]
    companies = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Deloitte", "HCL"]
    locations = ["Chennai", "Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi"]
    experiences = ["0-2 years", "2-4 years", "4-6 years", "6-8 years", "8+ years"]
    employment_types = ["Full-time", "Contract", "Part-time"]
    skills_list = ["Python, SQL, Power BI", "Python, SQL", "SQL, Excel, Power BI",
                   "Python, Machine Learning", "SQL, Tableau", "Python, SQL, Excel",
                   "Python, SQL, Power BI, Excel"]

    n = 1000

    experience_base = {"0-2 years": 400000, "2-4 years": 600000, "4-6 years": 850000,
                        "6-8 years": 1100000, "8+ years": 1400000}
    job_base = {"Data Analyst": 0, "Data Scientist": 250000, "Business Analyst": 100000,
                "Python Developer": 150000, "SQL Developer": 50000, "Power BI Developer": 120000}
    location_base = {"Chennai": 0, "Bangalore": 150000, "Hyderabad": 80000,
                      "Pune": 70000, "Mumbai": 120000, "Delhi": 60000}
    employment_base = {"Full-time": 0, "Contract": -50000, "Part-time": -100000}

    records = []
    for i in range(n):
        job_title = np.random.choice(job_titles)
        company = np.random.choice(companies)
        location = np.random.choice(locations)
        experience = np.random.choice(experiences)
        skills = np.random.choice(skills_list)
        employment = np.random.choice(employment_types)

        base_salary = (experience_base[experience] + job_base[job_title]
                        + location_base[location] + employment_base[employment])

        salary_min = base_salary + np.random.randint(-50000, 50001)
        salary_max = salary_min + np.random.randint(100000, 400001)
        salary_min = max(salary_min, 200000)

        posted_date = pd.Timestamp("2026-01-01") + pd.to_timedelta(
            np.random.randint(0, 365), unit="D")

        records.append([f"J{i+1001}", job_title, company, location, salary_min,
                         salary_max, experience, skills, employment, posted_date])

    df = pd.DataFrame(records, columns=[
        "Job_ID", "Job_Title", "Company", "Location", "Salary_Min", "Salary_Max",
        "Experience", "Skills", "Employment_Type", "Posted_Date"])

    # Insert missing values
    salary_min_missing = np.random.choice(df.index, 25, replace=False)
    salary_max_missing = np.random.choice(df.index, 15, replace=False)
    skills_missing = np.random.choice(df.index, 20, replace=False)
    location_missing = np.random.choice(df.index, 15, replace=False)

    df.loc[salary_min_missing, "Salary_Min"] = np.nan
    df.loc[salary_max_missing, "Salary_Max"] = np.nan
    df.loc[skills_missing, "Skills"] = np.nan
    df.loc[location_missing, "Location"] = np.nan

    # Insert salary outliers
    outlier_indices = np.random.choice(df.index, 10, replace=False)
    for i, index in enumerate(outlier_indices):
        df.loc[index, "Salary_Max"] = 5000000 if i % 2 == 0 else 6000000

    df.to_csv(file_name, index=False)
    print("New sample dataset created successfully.")

# ================================================================
# 3-10. INSPECT, DEDUPLICATE, TYPE-CONVERT
# ================================================================

print("\nFIRST 5 ROWS\n", df.head())
print("\nDATASET SHAPE -> Rows:", df.shape[0], "Columns:", df.shape[1])
print("\nCOLUMN NAMES:", df.columns.tolist())
df.info()
print("\nDESCRIPTIVE STATISTICS\n", df.describe(include="all"))

print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))
print("Duplicate Job IDs:", df["Job_ID"].duplicated().sum())

df["Salary_Min"] = pd.to_numeric(df["Salary_Min"], errors="coerce")
df["Salary_Max"] = pd.to_numeric(df["Salary_Max"], errors="coerce")
df["Posted_Date"] = pd.to_datetime(df["Posted_Date"], errors="coerce")

# ================================================================
# 12-16. MISSING VALUES
# ================================================================

print("\nMISSING VALUES BEFORE CLEANING\n", df.isnull().sum())
missing_percentage = df.isnull().sum() / len(df) * 100
print("\nMISSING VALUE PERCENTAGE\n", missing_percentage)

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Before Cleaning")
save_chart("01_missing_values_before.png")

df["Salary_Min"] = df["Salary_Min"].fillna(df["Salary_Min"].median())
df["Salary_Max"] = df["Salary_Max"].fillna(df["Salary_Max"].median())
df["Skills"] = df["Skills"].fillna("Unknown")
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])

print("\nMISSING VALUES AFTER CLEANING\n", df.isnull().sum())

# ================================================================
# 17-19. STANDARDIZE TEXT
# ================================================================

text_columns = ["Job_Title", "Company", "Location", "Skills", "Experience", "Employment_Type"]
for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

df["Job_Title"] = df["Job_Title"].str.title()
df["Location"] = df["Location"].str.title()

# ================================================================
# 20-25. DATA VALIDATION
# ================================================================

print("\nMinimum Salary:", df["Salary_Min"].min())
print("Maximum Salary:", df["Salary_Max"].max())

negative_salary = df[(df["Salary_Min"] < 0) | (df["Salary_Max"] < 0)]
print("Negative salary records:", len(negative_salary))

invalid_salary = df[df["Salary_Min"] > df["Salary_Max"]]
print("Invalid salary records:", len(invalid_salary))

mask = df["Salary_Min"] > df["Salary_Max"]
df.loc[mask, ["Salary_Min", "Salary_Max"]] = df.loc[mask, ["Salary_Max", "Salary_Min"]].values
print("Invalid salary records after fixing:", (df["Salary_Min"] > df["Salary_Max"]).sum())

print("\nEXPERIENCE VALUES\n", df["Experience"].value_counts())

# ================================================================
# 26-31. OUTLIER HANDLING (IQR CAPPING)
# ================================================================

plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Salary_Max"])
plt.title("Salary Maximum - Before Outlier Handling")
save_chart("02_salary_boxplot_before.png")

Q1 = df["Salary_Max"].quantile(0.25)
Q3 = df["Salary_Max"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nQ1:", Q1, "Q3:", Q3, "IQR:", IQR)
print("Lower Bound:", lower_bound, "Upper Bound:", upper_bound)

outliers = df[(df["Salary_Max"] < lower_bound) | (df["Salary_Max"] > upper_bound)]
print("\nNumber of outliers:", len(outliers))
print(outliers[["Job_ID", "Job_Title", "Salary_Min", "Salary_Max"]])

df["Salary_Max"] = np.clip(df["Salary_Max"], lower_bound, upper_bound)

plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Salary_Max"])
plt.title("Salary Maximum - After Outlier Handling")
save_chart("03_salary_boxplot_after.png")

# ================================================================
# 32-35. FEATURE ENGINEERING
# ================================================================

experience_map = {"0-2 years": 1, "2-4 years": 3, "4-6 years": 5, "6-8 years": 7, "8+ years": 10}
df["Experience_Years"] = df["Experience"].map(experience_map)
df["Avg_Salary"] = (df["Salary_Min"] + df["Salary_Max"]) / 2
df["Posted_Month"] = df["Posted_Date"].dt.to_period("M").astype(str)

print("\nFEATURE ENGINEERING\n", df[["Experience", "Experience_Years", "Salary_Min",
                                      "Salary_Max", "Avg_Salary", "Posted_Date", "Posted_Month"]].head())

# ================================================================
# 36-40. SKILL EXTRACTION
# ================================================================

skills = df["Skills"].dropna().str.split(",").explode().str.strip()
skill_counts = skills.value_counts()
print("\nSKILL DEMAND\n", skill_counts)

top_10_skills = skill_counts.head(10)
print("\nTOP 10 SKILLS\n", top_10_skills)

skill_list = ["Python", "SQL", "Power BI", "Excel", "Tableau", "Machine Learning"]
for skill in skill_list:
    df[skill] = df["Skills"].str.contains(skill, case=False, na=False).astype(int)

print("\nSKILL COLUMNS\n", df[["Skills"] + skill_list].head(10))

# ================================================================
# 41-58. EXPLORATORY DATA ANALYSIS / VISUALIZATIONS
# ================================================================

job_title_count = df["Job_Title"].value_counts()
plt.figure(figsize=(10, 6))
job_title_count.plot(kind="bar")
plt.title("Job Demand by Job Title"); plt.xlabel("Job Title"); plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
save_chart("04_job_demand_by_title.png")

location_count = df["Location"].value_counts()
plt.figure(figsize=(10, 6))
location_count.plot(kind="bar")
plt.title("Job Postings by Location"); plt.xlabel("Location"); plt.ylabel("Number of Jobs")
plt.xticks(rotation=45)
save_chart("05_jobs_by_location.png")

employment_count = df["Employment_Type"].value_counts()
plt.figure(figsize=(7, 7))
employment_count.plot(kind="pie", autopct="%1.1f%%")
plt.title("Employment Type Distribution"); plt.ylabel("")
save_chart("06_employment_type_pie.png")

salary_by_job = df.groupby("Job_Title")["Avg_Salary"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
salary_by_job.plot(kind="bar")
plt.title("Average Salary by Job Title"); plt.xlabel("Job Title"); plt.ylabel("Average Salary")
plt.xticks(rotation=45)
save_chart("07_avg_salary_by_job_title.png")

salary_by_location = df.groupby("Location")["Avg_Salary"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
salary_by_location.plot(kind="bar")
plt.title("Average Salary by Location"); plt.xlabel("Location"); plt.ylabel("Average Salary")
plt.xticks(rotation=45)
save_chart("08_avg_salary_by_location.png")

salary_by_experience = df.groupby("Experience")["Avg_Salary"].mean()
plt.figure(figsize=(10, 6))
salary_by_experience.plot(kind="bar")
plt.title("Average Salary by Experience"); plt.xlabel("Experience"); plt.ylabel("Average Salary")
plt.xticks(rotation=45)
save_chart("09_avg_salary_by_experience.png")

plt.figure(figsize=(10, 6))
plt.hist(df["Avg_Salary"], bins=20)
plt.title("Salary Distribution"); plt.xlabel("Average Salary"); plt.ylabel("Number of Jobs")
save_chart("10_salary_distribution.png")

plt.figure(figsize=(8, 6))
plt.scatter(df["Experience_Years"], df["Avg_Salary"])
plt.title("Experience vs Average Salary"); plt.xlabel("Experience in Years"); plt.ylabel("Average Salary")
save_chart("11_experience_vs_salary.png")

plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Avg_Salary"])
plt.title("Average Salary Distribution")
save_chart("12_avg_salary_boxplot.png")

monthly_jobs = df.groupby("Posted_Month").size()
plt.figure(figsize=(12, 6))
monthly_jobs.plot(kind="line", marker="o")
plt.title("Job Postings Over Time"); plt.xlabel("Month"); plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
save_chart("13_job_postings_over_time.png")

plt.figure(figsize=(10, 6))
top_10_skills.plot(kind="bar")
plt.title("Top 10 In-Demand Skills"); plt.xlabel("Skill"); plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
save_chart("14_top_10_skills.png")

skill_demand = pd.DataFrame({"Skill": skill_counts.index, "Job_Postings": skill_counts.values})
print("\nSKILL DEMAND TABLE\n", skill_demand.head(10))

skill_salary = {}
for skill in skill_list:
    skill_data = df[df[skill] == 1]
    skill_salary[skill] = skill_data["Avg_Salary"].mean()

skill_salary_df = pd.DataFrame(list(skill_salary.items()), columns=["Skill", "Average_Salary"])
skill_salary_df = skill_salary_df.sort_values("Average_Salary", ascending=False)
print("\nAVERAGE SALARY BY SKILL\n", skill_salary_df)

plt.figure(figsize=(10, 6))
plt.bar(skill_salary_df["Skill"], skill_salary_df["Average_Salary"])
plt.title("Average Salary by Skill"); plt.xlabel("Skill"); plt.ylabel("Average Salary")
plt.xticks(rotation=45)
save_chart("15_avg_salary_by_skill.png")

numeric_columns = ["Salary_Min", "Salary_Max", "Experience_Years", "Avg_Salary"] + skill_list
correlation = df[numeric_columns].corr()
print("\nCORRELATION MATRIX\n", correlation)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_chart("16_correlation_heatmap.png")

job_location = pd.crosstab(df["Job_Title"], df["Location"])
plt.figure(figsize=(12, 7))
sns.heatmap(job_location, annot=True, fmt="d")
plt.title("Job Demand by Job Title and Location")
save_chart("17_job_title_location_heatmap.png")

salary_job_experience = pd.pivot_table(df, values="Avg_Salary", index="Job_Title",
                                        columns="Experience", aggfunc="mean")
plt.figure(figsize=(12, 7))
sns.heatmap(salary_job_experience, annot=True, fmt=".0f")
plt.title("Average Salary by Job Title and Experience")
save_chart("18_salary_job_title_experience_heatmap.png")

# ================================================================
# 68-88. MACHINE LEARNING: SALARY PREDICTION
# ================================================================
# NOTE: Salary_Min / Salary_Max are excluded from features because
# Avg_Salary is derived directly from them (target leakage).

X = df[["Experience_Years", "Job_Title", "Location", "Skills", "Employment_Type"]]
y = df["Avg_Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

categorical_columns = ["Job_Title", "Location", "Skills", "Employment_Type"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
    remainder="passthrough")

# --- Linear Regression ---
linear_model = Pipeline([("preprocessor", preprocessor), ("regression", LinearRegression())])
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)
print("\nLINEAR REGRESSION -> MAE:", round(mae_linear, 2), "RMSE:", round(rmse_linear, 2), "R2:", round(r2_linear, 4))

linear_result = pd.DataFrame({"Actual_Salary": y_test.values, "Predicted_Salary": y_pred_linear})
linear_result["Error"] = linear_result["Actual_Salary"] - linear_result["Predicted_Salary"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear)
plt.xlabel("Actual Salary"); plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary - Linear Regression")
save_chart("19_actual_vs_predicted_linear.png")

plt.figure(figsize=(10, 6))
plt.hist(linear_result["Error"], bins=20)
plt.title("Linear Regression Prediction Error"); plt.xlabel("Prediction Error"); plt.ylabel("Frequency")
save_chart("20_linear_regression_error.png")

# --- Random Forest Regression ---
random_forest_model = Pipeline([("preprocessor", preprocessor),
                                 ("regression", RandomForestRegressor(n_estimators=200, random_state=42))])
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print("RANDOM FOREST -> MAE:", round(mae_rf, 2), "RMSE:", round(rmse_rf, 2), "R2:", round(r2_rf, 4))

model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae_linear, mae_rf],
    "RMSE": [rmse_linear, rmse_rf],
    "R2": [r2_linear, r2_rf]})
print("\nMODEL COMPARISON\n", model_results)

plt.figure(figsize=(8, 5))
plt.bar(model_results["Model"], model_results["R2"])
plt.title("Model R2 Comparison"); plt.xlabel("Model"); plt.ylabel("R2 Score")
save_chart("21_model_r2_comparison.png")

plt.figure(figsize=(8, 5))
plt.bar(model_results["Model"], model_results["MAE"])
plt.title("Model MAE Comparison"); plt.xlabel("Model"); plt.ylabel("Mean Absolute Error")
save_chart("22_model_mae_comparison.png")

# ================================================================
# 89-102. BUSINESS SUMMARY & EXPORTS
# ================================================================

salary_by_job_df = df.groupby("Job_Title")["Avg_Salary"].mean().reset_index()
salary_by_job_df.columns = ["Job_Title", "Average_Salary"]

salary_by_location_df = df.groupby("Location")["Avg_Salary"].mean().reset_index()
salary_by_location_df.columns = ["Location", "Average_Salary"]

salary_by_experience_df = df.groupby("Experience")["Avg_Salary"].mean().reset_index()
salary_by_experience_df.columns = ["Experience", "Average_Salary"]

total_jobs = len(df)
average_salary = df["Avg_Salary"].mean()
median_salary = df["Avg_Salary"].median()
minimum_salary = df["Avg_Salary"].min()
maximum_salary = df["Avg_Salary"].max()
top_job = df["Job_Title"].value_counts().idxmax()
top_location = df["Location"].value_counts().idxmax()
top_skill = skill_counts.idxmax()

summary = pd.DataFrame({
    "Metric": ["Total Job Postings", "Average Salary", "Median Salary", "Minimum Average Salary",
               "Maximum Average Salary", "Most Common Job Title", "Most Common Location",
               "Most In-Demand Skill", "Linear Regression R2", "Random Forest R2"],
    "Value": [total_jobs, round(average_salary, 2), round(median_salary, 2), round(minimum_salary, 2),
              round(maximum_salary, 2), top_job, top_location, top_skill,
              round(r2_linear, 4), round(r2_rf, 4)]})

print("\n================================================")
print("JOB MARKET & SALARY INTELLIGENCE SUMMARY")
print("================================================")
print(summary.to_string(index=False))

# Exports
df.to_csv(os.path.join(DATA_DIR, "Job_Market_Salary_Intelligence_Cleaned.csv"), index=False)
skill_demand.to_csv(os.path.join(DATA_DIR, "Skill_Demand_Summary.csv"), index=False)
salary_by_job_df.to_csv(os.path.join(DATA_DIR, "Salary_By_Job_Title.csv"), index=False)
salary_by_location_df.to_csv(os.path.join(DATA_DIR, "Salary_By_Location.csv"), index=False)
salary_by_experience_df.to_csv(os.path.join(DATA_DIR, "Salary_By_Experience.csv"), index=False)
model_results.to_csv(os.path.join(DATA_DIR, "Model_Evaluation_Results.csv"), index=False)
summary.to_csv(os.path.join(DATA_DIR, "Job_Market_Executive_Summary.csv"), index=False)
linear_result.to_csv(os.path.join(DATA_DIR, "Actual_vs_Predicted_Salary.csv"), index=False)

print("\nAll CSV outputs exported to:", DATA_DIR)
print("All chart PNGs exported to:", CHART_DIR)

print("\n============================================================")
print("       JOB MARKET & SALARY INTELLIGENCE PROJECT - COMPLETED")
print("============================================================")
print("Total Job Postings   :", total_jobs)
print("Average Salary       :", round(average_salary, 2))
print("Median Salary        :", round(median_salary, 2))
print("Most Common Job Title:", top_job)
print("Most Common Location :", top_location)
print("Most In-Demand Skill :", top_skill)
print("Linear Regression R2 :", round(r2_linear, 4))
print("Random Forest R2     :", round(r2_rf, 4))
print("============================================================")
