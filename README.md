# Ground Truth Tracker: Humanitarian Feedback ETL and Analytics Dashboard

## Project Summary

This project demonstrates a complete Data Engineering and Analytics pipeline designed to process raw beneficiary feedback data from a humanitarian survey. It executes a robust ETL (Extract, Transform, Load) cycle, performs core quantitative analysis (KPIs and Statistical Inference), and presents actionable findings through an interactive, executive-ready Streamlit web application.

## Key Features & Technical Skills

| Feature Area | Description | Technical Tools & Concepts |
| :--- | :--- | :--- |
| **Data Engineering (ETL)** | Robust ETL pipeline from CSV to a relational database. Includes schema definition, data type enforcement, and handling of null/missing values via imputation. | Python, Pandas, **MySQL/SQLAlchemy**, `dotenv`, Data Auditing (is_valid column). |
| **Data Quality & Cleaning** | Implements standardization for categorical data (e.g., gender, provider names) and enforces `NOT NULL` constraints on core score metrics via median imputation. | Pandas `replace()`, `fillna()`, Custom Transformation Functions. |
| **Core Analysis** | Calculates key performance indicators (KPIs) like the Net Satisfaction Score (NSS) and performs statistical deep dives to validate findings. | Python, **SciPy (Welch's t-test)**, Descriptive Statistics, KPI derivation. |
| **Advanced Visualization** | Generates high-impact, interactive charts to visualize multi-dimensional survey data. | **Plotly**, Diverging Stacked Bar Charts, Geospatial Mapping, Statistical Visuals. |
| **Web Application** | Deploys the analysis in a professional, interactive dashboard with dynamic filtering and data export functionality. | **Streamlit**, Cached data loading (`@st.cache_data`), Custom components, Responsive Layout. |

