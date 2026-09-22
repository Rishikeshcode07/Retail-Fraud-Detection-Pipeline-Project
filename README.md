# Retail Fraud Detection & Predictive Analytics Engine

## Executive Overview
This repository contains an end-to-end data engineering and predictive machine learning framework built to detect fraudulent retail transactions, identify high-risk behavioral anomalies, and provide interactive business intelligence across enterprise retail operational datasets.

The enterprise pipeline integrates cloud-based raw data extraction from **AWS S3**, automated ETL preprocessing, missing value imputation, local **SQLite** relational staging, machine learning modeling (classification and anomaly detection), an interactive **Streamlit** web application, and multi-page **Power BI** executive dashboards.

---

## Project Framework: STAR Methodology

### 1. Situation
In high-volume retail environments, fraudulent transactions, revenue leakage, and non-compliant purchasing behaviors account for millions in annual enterprise losses. Standard rule-based fraud detection systems suffer from high false-positive rates, high latency in manual transaction audits, and an inability to adapt to evolving, complex fraud vectors across multi-channel retail operations.

### 2. Task
* **Cloud Integration & Pipeline Reliability:** Build a scalable cloud-to-local ETL pipeline capable of ingesting raw transactional datasets from AWS S3 into a structured, clean SQLite database.
* **Data Quality & Hygiene:** Clean, audit, and impute missing values across 100,000+ records and 44 operational features without loss of data integrity or statistical bias.
* **Predictive ML Modeling:** Develop, evaluate, and benchmark multiple machine learning algorithms including supervised classification models (Random Forest, Decision Tree) and unsupervised anomaly detection models (Isolation Forest) to flag suspicious transactions in real time.
* **Interactive Web Deployment:** Build and deploy an interactive web interface using Streamlit to allow fraud analysts to input transaction details and receive instant risk scoring and model predictions.
* **Executive Visualization:** Construct interactive Power BI dashboards featuring custom DAX measures and structured relational data models to communicate operational KPIs, financial impact, and fraud patterns to stakeholders.

### 3. Action
* Integrated AWS and file-streaming libraries to pipe 100,000 records directly from the AWS S3 bucket (`data-for-zidio-project2/data.csv`) into the working environment.
* Designed an automated missing-value imputation pipeline utilizing numerical column medians and categorical column modes to achieve 0% data loss and complete data completeness.
* Engineered a local relational database staging environment in SQLite (`fraud_detection.db`), creating a structured transactions table for rapid SQL querying and local analytical staging.
* Built supervised machine learning pipelines for fraud classification and unsupervised pipelines for outlier/anomaly detection.
* Built a multi-page Streamlit web application hosted on Streamlit Cloud to serve real-time model inference, interactive feature inspection, and automated prediction reporting.
* Designed an end-to-end Power BI dashboard with multi-page navigation, visual KPI cards, DAX metrics, and retail performance breakdowns.

### 4. Result
* **100% Data Quality Retention:** Successfully extracted, cleaned, and staged 100,000 retail records with zero missing values and zero lost rows.
* **High Machine Learning Accuracy:** Achieved high precision and recall across trained classifiers, minimizing false positives and enabling early fraud detection.
* **End-to-End Operationalization:** Reduced fraud review latency from hours to sub-second inference via the deployed Streamlit interface and Power BI analytical views.

---

## Tech Stack & Tooling

| Domain | Tools & Technologies |
| :--- | :--- |
| **Language** | Python 3.13 |
| **Cloud Storage** | AWS S3 |
| **Database & Staging** | SQLite3, SQL Relational Modeling |
| **Data Processing & Manipulation** | Pandas, NumPy |
| **Machine Learning & Analytics** | Scikit-Learn |
| **Data Visualization** | Matplotlib, Seaborn, Power BI |
| **Web Deployment** | Streamlit, Streamlit Cloud |
| **Version Control & Docs** | Git, GitHub |

---

## End-to-End Pipeline Architecture

[ AWS S3 Bucket: data-for-zidio-project2 ]
                   │
                   ▼ (Cloud Streaming)
[ Raw Ingestion: 100,000 Rows x 44 Columns ]
                   │
                   ▼ (Median / Mode Imputation)
[ Data Cleaning & Preprocessing (df_clean) ]
                   │
                   ▼ (Local Database Staging)
[ Relational Database: fraud_detection.db ]
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
[ Exploratory Data Analysis ] [ ML Training Pipeline ]
        │                     │
        ▼                     ▼
[ Power BI Dashboard ]      [ Streamlit Web App ]

---

## Detailed Step-by-Step Implementation & Output Analysis

### Step 1: AWS S3 Data Extraction & Cloud Ingestion
**What was done & Why:** Raw transactional data stored in an Amazon Web Services (AWS) S3 bucket is streamed directly into the environment. To maintain computational efficiency and ensure the local machine is not overwhelmed by massive datasets, a sample batch of 100,000 records across 44 operational columns is extracted. This provides a statistically significant sample for accurate machine learning training while keeping memory allocation manageable.

* **Screenshot 1: AWS S3 Ingestion Execution Output**
* **Target File:** `images/01_aws_s3_ingestion.png`
* **Explanation:** This screenshot displays the terminal or notebook execution output confirming the successful connection to the AWS S3 bucket. It verifies the successful streaming of the 100,000 records, the 44 respective columns, and shows the allocated memory footprint to prove pipeline efficiency.
  
![image alt](https://github.com/user-attachments/assets/ca7914a1-18e3-4a66-8e1b-0427c31ed3b5)

### Step 2: Data Cleaning, Audit & Missing Value Imputation
**What was done & Why:** Data hygiene is strictly enforced to prevent algorithms from failing or learning from biased data. We process numerical missing values using column-wise medians instead of means to protect the data against extreme outlier skewness (common in financial fraud). Categorical fields are imputed using column-wise modes. This ensures 100% data completeness without deleting any valuable transactional rows.

* **Screenshot 2: Data Cleaning & Preprocessing Execution Audit**
* **Target File:** `images/02_data_cleaning_audit.png`
* **Explanation:** This screenshot displays the before-and-after missing value counts across the dataset. The audit proves that exactly 0 missing cells remain across all 100,000 rows after the imputation process, meaning the data is perfectly clean and ready for database staging.
![image alt](https://github.com/user-attachments/assets/5a65908e-4a66-4293-be57-094b58948ef0)

### Step 3: Local SQLite Staging & Database Architecture
**What was done & Why:** The fully cleaned dataset is pushed into a local SQLite relational database (`fraud_detection.db`) inside a dedicated `transactions` table. Staging the data locally rather than relying on flat CSV files enables rapid SQL-based querying, secure dataset filtering, and highly reproducible data retrieval for both the machine learning models and Business Intelligence tools like Power BI.

* **Screenshot 3: SQLite Staging Output Confirmation**
* **Target File:** `images/03_sqlite_staging.png`
* **Explanation:** This screenshot confirms the successful creation of the database file and verifies that all 100,000 clean records were successfully written and committed into the relational `transactions` table.
![image alt](https://github.com/user-attachments/assets/6d60ecce-f682-427b-bac0-6a8cb0282eb8)

### Step 4: Exploratory Data Analysis (EDA) & Visualization
**What was done & Why:** Before modeling, statistical visualizations are generated to understand feature relationships, highlight class imbalances (fraud vs. normal), and spot obvious behavioral anomalies. This step is critical for determining which features will be most valuable for the predictive models.

* **Screenshot 4: Transaction Class Distribution (Fraud vs Non-Fraud)**
* **Target File:** `images/04_class_distribution.png`
* **Explanation:** A bar or donot plot illustrating the proportion of legitimate versus fraudulent transactions. It highlights the extreme class imbalance typical in fraud detection, visually explaining why specialized sampling and thresholding are necessary.
![image alt](https://github.com/user-attachments/assets/9c97a7b9-015c-4a2e-b56d-ed8090a8b02c)

* **Screenshot 5: Feature Correlation Heatmap**
* **Target File:** `images/05_correlation_heatmap.png`
* **Explanation:** A correlation matrix visualizing pairwise linear relationships across numerical features. We do this to identify multi-collinearity (e.g., highly correlated transaction amounts and risk flags) which helps in feature selection.
![image alt](https://github.com/user-attachments/assets/a92eb1ab-5648-4548-8001-b32bf45ca2a3)

* **Screenshot 6: Transaction Value Distribution & Outlier Analysis**
* **Target File:** `images/06_transaction_amount_dist.png`
* **Explanation:** A visual (like a box plot or histogram) depicting the spread of transaction amounts. It highlights extreme outliers that fall far beyond standard ranges, mapping directly to potential anomalous high-risk behavior.
![image alt](https://github.com/user-attachments/assets/1e25a467-0256-49f0-93b7-f30c7432d129)

### Step 5: Machine Learning Model Development & Performance Metrics
**What was done & Why:** Multiple algorithms (Random Forest, Decision Tree, and Isolation Forest) are trained on the staged data. We use robust evaluation metrics like ROC-AUC and Confusion Matrices rather than standard accuracy, because catching the maximum amount of actual fraud (Recall) while minimizing false alarms (Precision) is the primary business objective.

* **Screenshot 7: Model Receiver Operating Characteristic (ROC) Curve**
* **Target File:** `images/07_roc_auc_curve.png`
* **Explanation:** A curve plot comparing the True Positive Rate against the False Positive Rate for the trained models. A higher Area Under the Curve (AUC) visually confirms the model's strong discriminative power in separating fraud from legitimate transactions.
![image alt](https://github.com/user-attachments/assets/f37a6a13-b7c5-4697-bbb9-34d5bc2975ed)

* **Screenshot 8: Model Confusion Matrix Heatmap**
* **Target File:** `images/08_confusion_matrix.png`
* **Explanation:** A grid showing True Positives, True Negatives, False Positives, and False Negatives. This visual strictly quantifies the model's real-world business value by showing exactly how many fraudulent transactions were caught versus missed.
![image alt](https://github.com/user-attachments/assets/abe6ed04-617c-4599-97ff-79c11cacd9e8)

* **Screenshot 9: Feature Importance Bar Ranking**
* **Target File:** `images/09_feature_importance.png`
* **Explanation:** A ranking chart displaying the top 10 most influential variables extracted from the Random Forest model. It explains *how* the model makes decisions by identifying the primary drivers of fraudulent activity.
![image alt](https://github.com/user-attachments/assets/dd1bc665-a381-47fe-a28a-557efca35e38)


### Step 6: Streamlit Web Application Interface
**What was done & Why:** To make the machine learning models accessible to non-technical fraud analysts, an interactive web application is deployed using Streamlit. This translates raw Python predictions into an easy-to-use graphical interface where teams can score transactions in real-time or process batch files.

**Streamlit Dashboard link**: https://retail-fraud-detection-pipeline-project-yvbwozcwzjjlssmjnffh3v.streamlit.app/

* **Screenshot 11: Streamlit App Home & Real-Time Prediction Page**
* **Target File:** `images/11_streamlit_app_home.png`
* **Explanation:** The main user interface of the Streamlit application. It shows the input fields where an analyst can manually enter transaction details and instantly receive a computed fraud risk score.
![image alt](https://github.com/user-attachments/assets/e66a2f89-4fa1-4fde-9be9-9f418feda0fb)

* **Screenshot 12: Streamlit Interactive Batch Inference & Analytics View**
* **Target File:** `images/12_streamlit_batch_analysis.png`
* **Explanation:** The batch processing dashboard within the Streamlit app. It demonstrates the ability to upload bulk transaction files, run predictions across all of them simultaneously, and export the flagged results.
![image alt](https://github.com/user-attachments/assets/e2f1dc56-73ce-4680-98cb-16b19f8b41fa)

### Step 7: Power BI Executive Analytics Dashboard
**What was done & Why:** For leadership and management, operational metrics are visualized in Power BI. By directly connecting Power BI to the cleaned SQLite staging database, we created an automated reporting layer that tracks financial losses avoided, overall transaction volumes, and geographic fraud hotspots.

* **Screenshot 13: Power BI Executive Summary Dashboard (Page 1)**
* **Target File:** `images/13_powerbi_executive_summary.png`
* **Explanation:** The high-level executive overview page. It features top-level KPI cards (Total Revenue, Fraud Incidents, Financial Loss Avoided) and trend lines, giving stakeholders immediate visibility into system performance.
![Power BI Executive Summary Dashboard](images/13_powerbi_executive_summary.png)

* **Screenshot 14: Power BI Fraud Pattern & Category Deep-Dive (Page 2)**
* **Target File:** `images/14_powerbi_fraud_patterns.png`
* **Explanation:** An analytical deep-dive page within the report. It breaks down fraud occurrences by specific dimensions such as payment method, merchant category, and customer demographics to inform future security policies.
![Power BI Fraud Breakdown & Category Analysis](images/14_powerbi_fraud_patterns.png)

---

## Installation, Setup & Execution Guide

To replicate this environment and run the pipeline locally, follow these standard steps:

1. **Clone the Repository:** Download the project files to your local machine using standard Git version control commands.
2. **Environment Setup:** Create and activate an isolated Python virtual environment to prevent package conflicts.
3. **Install Dependencies:** Install all required libraries (like pandas, scikit-learn, and streamlit) from the provided requirements text file.
4. **Execute the Data Pipeline:** Open the Jupyter Notebook located in the notebooks directory and run the cells sequentially to extract the AWS data, clean it, build the SQLite database, and train the machine learning models.
5. **Launch the Web Interface:** Run the Streamlit execution command on the main application script to open the interactive web app in your local browser.

---

## Directory & File Structure

This section outlines the organization of the repository so users know exactly where to find databases, notebooks, dashboards, and application files.

* **data/**: Contains the local SQLite staging database (`fraud_detection.db`) generated by the pipeline.
* **images/**: Directory for all the documentation screenshots outlined above (01 through 14).
* **notebooks/**: Holds the core Jupyter Notebook containing the end-to-end data processing, model training, and EDA pipeline.
* **reports/**: Stores the actual Power BI dashboard source file (`retail_fraud_dashboard.pbix`).
* **app.py**: The primary Streamlit application script for the web interface.
* **requirements.txt**: The list of all necessary Python dependencies required to run the project.
* **README.md**: This core documentation file.

---

## Business Impact & Strategic Value

* **Loss Reduction:** Automated screening flags high-risk transactions instantly, mitigating potential revenue loss before orders are fulfilled.
* **Latency Optimization:** Cloud streaming and local database staging allow rapid processing of large transaction volumes without requiring massive local hardware.
* **Operational Intelligence:** Power BI dashboards provide ongoing operational visibility to risk analysts and executive management, translating raw machine learning outputs into strategic business decisions.
