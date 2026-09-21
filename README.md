# Retail Transaction and Fraud Detection Pipeline

### Dashboard Link : [Insert Your Published Power BI Link Here]
### Streamlit App Link : https://retail-fraud-detection-pipeline-project-yvbwozcwzjjlssmjnffh3v.streamlit.app/

---

## Situation (Problem Statement)

The digital retail and e-commerce sector faces sophisticated fraud patterns hidden within massive, highly imbalanced transaction volumes. Standard analytical methods and baseline machine learning models often struggle to identify the minority fraud class, resulting in high false positive rates that disrupt legitimate customer experiences and cause severe revenue leakage. 

For the Zidio Work internship project, our team recognized a critical operational gap: data science teams were generating complex predictive models, but regional directors and business stakeholders lacked a centralized, unified view of this data. They struggled to answer complex multidimensional questions:
* *What is the actual net financial exposure to fraudulent transactions across different states and merchant categories?*
* *Are there specific times of day or demographic segments that are disproportionately targeted by fraudulent actors?*
* *How accurate is our machine learning model in real-world financial terms (Total Prevented Loss vs. False Positive friction)?*

Without a highly interactive, data-dense business intelligence dashboard integrated with the technical machine learning pipeline, decision-makers are forced to rely on fragmented reports. This leads to inefficient risk mitigation, delayed response to compromised merchant terminals, and a fundamental disconnect between the technical data processing and executive strategy.

## Task

The core objective was to engineer a comprehensive, end-to-end analytical ecosystem serving as both a technical diagnostic tool and an executive business intelligence command center. The specific technical and analytical goals included:

1. **Data Engineering & Predictive Modeling:** Develop a robust Python-based pipeline to clean raw transaction data and train machine learning classification algorithms (Random Forest, Decision Tree, Extra Tree, Isolation Forest) optimized for severe class imbalances.
2. **Real-Time Diagnostic Web Application:** Build an interactive Streamlit application to visualize algorithm performance, feature importance, and geospatial risk mapping using open-source libraries.
3. **Unified Metric Tracking in Power BI:** Develop dynamic KPIs for Total Transaction Volume, Total Fraud Loss, Average Risk Score, and Model Accuracy.
4. **Granular Risk Slicing:** Map out financial exposure across multiple dimensions: Temporal (Time/Day), Demographic (Age/Gender), and Structural (Merchant Category/Geography).
5. **Interactive UI/UX:** Create a highly navigational, 6-page Power BI interface strictly adhering to a high-contrast executive dark theme to match the Streamlit web application.

## Action (Steps Followed)

The project development was executed through a rigorous end-to-end data modeling, machine learning, and visual engineering workflow by the project team (Rishikesh Kashyap Ojha, Swapnil, Ashika, and Sridhar):

- **Step 1 : Data Extraction & Python Preprocessing:** Ingested raw retail datasets using Pandas. Handled missing values, standardized data types, and applied categorical encoding. Engineered a unified target variable (`is_fraud`) to prepare the dataset for supervised learning.
- **Step 2 : Machine Learning Implementation:** Trained and evaluated Scikit-Learn classification models. Due to the extreme class imbalance (e.g., 19,556 legitimate vs. ~100 fraud cases), models were tuned to prioritize recall and precision for the minority class, ensuring the algorithm could accurately flag risk without overwhelming the system with false positives.
- **Step 3 : Streamlit Application Engineering:** Developed a Python-based web app using Streamlit and Plotly. 
    * Engineered a row-normalized Confusion Matrix to ensure distinct color scaling across all true and predicted labels, fixing visual wash-out caused by the imbalanced data.
    * Deployed an interactive geographical scatter map to plot fraud coordinates without relying on paid API keys.
- **Step 4 : Power BI Data Import & Normalization:** Loaded the processed datasets into Power BI Desktop. Utilized Power Query Editor to validate column distribution, ensure correct data types (e.g., formatting Order Dates to DD/MM/YYYY), and isolate reference tables to build a structured relational data model (Star Schema).
- **Step 5 : UI/UX & Brand Theming:** Applied a custom dark executive background (`#0E1117`) across all Power BI canvas pages. Standardized visual containers with 10% transparency and 8px corner radii. Established a strict global data color palette: `#FF4D4D` (Coral Red) for Fraud/Risk and `#38BDF8` (Sky Blue) for Legitimate data.
- **Step 6 : DAX Measure Engineering:** Created a dedicated `MEASURE TABLE` to separate calculated metrics from raw data. Programmed robust DAX measures including:
    * `Total Transactions = COUNT(Fact_Transactions[transaction_id])`
    * `Fraudulent Transactions = CALCULATE([Total Transactions], Fact_Transactions[is_fraud] = 1)`
    * `Total Fraud Loss = CALCULATE(SUM(Fact_Transactions[amt]), Fact_Transactions[is_fraud] = 1)`
    * `Fraud Rate % = DIVIDE([Fraudulent Transactions], [Total Transactions], 0)`
- **Step 7 : Overview & Demographic Dashboards (Pages 1-2):** Constructed the primary entry points. Built line charts for volume over time, donut charts for legitimate vs. fraud splits, and demographic scatter plots cross-filtering customer age against transaction amounts.
- **Step 8 : Temporal & Merchant Risk Mapping (Pages 3-4):** Developed complex matrix heatmaps mapping Day of the Week against Hour of the Day to identify specific chronological vulnerabilities. Engineered decomposition trees and horizontal bar charts to rank the top 10 most compromised merchant categories.
- **Step 9 : Model Diagnostics Integration (Page 5):** Recreated the core Streamlit visual logic inside Power BI, building a matrix to represent the algorithmic Confusion Matrix and bar charts to display top predictive features (e.g., transaction amount, distance from home).
- **Step 10 : Executive Summary Build (Page 6):** Aggregated top-level financial exposure metrics into a single strategic view utilizing KPI scorecards, Key Influencer visuals, and transaction screening funnel charts for senior leadership review.

---

## Snapshot of Dashboard (Power BI Desktop)

### Page 1 — Executive Overview
![Page 1 Overview](https://github.com/user-attachments/assets/placeholder-link-for-page-1)

### Page 2 — Customer Demographics & Geographic Risk
![Page 2 Demographics](https://github.com/user-attachments/assets/placeholder-link-for-page-2)

### Streamlit Web Application — Model Diagnostics
![Streamlit App Diagnostics](https://github.com/user-attachments/assets/placeholder-link-for-streamlit)

---

## Result (Deep Insights & Data Inferences)

The resulting integrated ecosystem acts as a highly effective analytical tool bridging technical data science and business operations. Based on the aggregate data state, profound business inferences can be drawn:

### [1] Macro Financial Exposure & Imbalance Reality

* **The Volume Paradox:** The dataset encompasses a massive volume of total transactions, yet actual confirmed fraudulent events represent less than 1% of the total dataset. 
* **Disproportionate Financial Impact:** Despite their low frequency, these fraudulent transactions drive a severe financial impact. High-value transactions are disproportionately targeted, meaning a small cluster of compromised orders creates a massive spike in Total Fraud Loss.

### [2] Algorithmic Performance & The False Positive Challenge

* **Precision vs. Recall:** The row-normalized confusion matrix reveals the inherent trade-off in fraud detection. By tuning the Random Forest model to aggressively capture fraudulent events (high recall), the system inevitably flags a subset of legitimate transactions (false positives).
* **Business Translation:** The dashboard translates these technical metrics into operational reality. Every false positive represents a blocked legitimate customer, requiring the business to balance fraud prevention against customer friction.

### [3] Temporal Vulnerabilities

* **Chronological Heatmaps:** The matrix visual analyzing Day of Week vs. Hour of Day indicates distinct behavioral patterns among fraudulent actors. Fraud clusters tend to spike during specific off-peak hours (e.g., late night or early morning weekends) when manual transaction review teams are traditionally understaffed.
* **Velocity Interventions:** By isolating these time-based anomalies, operations teams can implement dynamic velocity checks, automatically increasing authentication friction during identified high-risk windows.

### [4] Merchant & Demographic Targeting

* **Category Exploitation:** The decomposition tree and horizontal bar charts explicitly highlight that not all merchants face equal risk. Specific high-liquidity categories (e.g., electronics, digital goods) account for the vast majority of the Total Fraud Loss.
* **Targeted Demographics:** Scatter plot clustering indicates specific demographic bands (based on age and transaction history) that are more susceptible to account takeover or targeted fraud campaigns, allowing for proactive, tailored security alerts.

---

## Data Model Structure

The Power BI data model was optimized for speed and analytical flexibility using a Star Schema structure:

* **Fact_Transactions (Central Table):** Contains all transactional event data (Transaction ID, Amount, Timestamp, Predicted Fraud Status, True Fraud Status).
* **Dim_Customer (Dimension Table):** Contains demographic profiles (Customer ID, Age, Gender, City, State).
* **Dim_Merchant (Dimension Table):** Contains vendor information (Merchant ID, Category, Location).
* **MEASURE TABLE 1:** A dedicated repository for all DAX calculations to keep the model clean, scalable, and easier to maintain.

---

## Tools & Technologies Used

| Category | Technology / Tool | Purpose |
| :--- | :--- | :--- |
| **Languages** | Python, SQL, DAX | Data manipulation, database querying, and dashboard calculations. |
| **Data Engineering** | Pandas, NumPy | Raw data ingestion, cleaning, scaling, and feature engineering. |
| **Machine Learning** | Scikit-Learn | Training classification models (Random Forest, Decision Tree, Extra Tree, Isolation Forest). |
| **Web Application** | Streamlit, Plotly | Building the interactive model diagnostic interface and geospatial maps. |
| **Business Intelligence** | Power BI Desktop, Power Query | Relational data modeling, advanced visualization, and executive dashboard design. |
| **Version Control** | Git, GitHub | Code repository management and technical documentation. |
