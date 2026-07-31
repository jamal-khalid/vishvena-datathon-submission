# Vishvena AI – Data Studio (Streamlit)

An AI-powered educational assessment analytics dashboard built for the Vishvena Datathon. The application enables interactive analysis of student performance data through visualizations, competency analysis, predictive insights, and AI-generated recommendations.

---

# Features

* 📊 Interactive dashboards
* 📈 Performance trends and cohort analysis
* 👩‍🎓 Gender gap analysis
* 🎯 Competency and mastery analysis
* 🏆 Student and district rankings
* 🤖 AI-generated insights and action plans
* 🗺️ Karnataka district choropleth map
* 📑 Deep-dive report cards
* 📂 Excel, CSV and Parquet support

---


# Requirements

* Python 3.10 or above (Python 3.12 recommended)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/jamal-khalid/vishvena-datathon-submission.git
cd vishvena-datathon-submission
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Application

Run the dashboard from the repository root:

```bash
streamlit run streamlit_app/streamlit_app.py
```

The application will start at:

```
http://localhost:8501
```

---

# Input Dataset

# Use the Upload Dataset option in the left sidebar to provide the input data for analysis.

The application supports the following input formats:

📦 ZIP archive (.zip) containing one or more supported data files.
📊 Single Excel file (.xlsx or .xls).
📁 Multiple Excel files uploaded together.
📄 Single CSV file (.csv).
📁 Multiple CSV files uploaded together.

Expected columns include:

```
Year
Grade
Division
District
Block
Cluster
GP
Gender
Q1...Q20
Score
```

The application automatically detects compatible column names.

## Use the Question map option in the left sidebar to provide the (competency mapping with the question) file for analysis.


---

# Notes

* Ensure that all uploaded files follow the expected dataset structure and contain consistent column names for accurate processing.
* Keep the `streamlit_app` folder contents unchanged, as it contains required assets such as the Karnataka GeoJSON, adapter, and frontend resources.
* Ensure all dependencies are installed before running the application.
