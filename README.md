# Vishvena AI — Data Studio (Streamlit)

Education-assessment dashboard: upload the student dataset in the sidebar →
auto column detection, filters, hierarchy treemap/sunburst, trends & cohort
analysis, gender gap, mastery bands, deep-dive report cards, rankings,
forecasts, Karnataka choropleth with click-to-drill, item analysis, and the
AI insight / action-plan / brief tabs.

---

## 1. Setup (first time only)

Requires **Python 3.10+** (3.12 recommended).

### Windows (PowerShell or CMD)

    cd datathon_share\streamlit_app

    :: create a virtual environment
    python -m venv venv

    :: activate it  (PowerShell)
    venv\Scripts\Activate.ps1
    :: ...or (CMD)
    venv\Scripts\activate.bat

    :: install all dependencies
    pip install -r requirements.txt

> PowerShell blocks activation? Run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS / Linux

    cd datathon_share/streamlit_app
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

You know the venv is active when the prompt starts with `(venv)`.
Every later session: just re-run the **activate** line, nothing else.

---

## 2. Run

    streamlit run streamlit_app.py

Opens at **http://localhost:8501**.

Files that must sit in this folder (all included):

| File | Purpose |
|---|---|
| `streamlit_app.py` | the dashboard |
| `adapter.py` | data → analysis-layer bridge (**must match the app version — see Troubleshooting**) |
| `echarts.min.js` | offline smooth-animation charts |
| `karnataka_districts.geojson` | 31-district Karnataka map boundaries |
| `DATATHON_QUESTION_MAP.csv` | optional question → competency map (auto-loaded) |
| `.streamlit/config.toml` | dark theme, iframe flags, 2 GB upload limit |
| `../` (parent folder) | analysis layer: `analytics.py`, `insights.py`, `models_ml.py`, … |

---

## 3. Feed it data

**In the sidebar:**

1. **📂 Upload dataset** — `.xlsx`, `.csv`, or `.parquet`
   (expected shape: `Year | Grade | Division | District | Block | Cluster |
   GP | Gender | Q1..Q20 | Score`; the adapter auto-detects variations).
2. **🗺️ Question map (optional CSV)** — columns `question, competency`
   (difficulty optional). Groups Q1..Q20 into named skills (Numeracy,
   Algebra, …) across the whole dashboard **including** the AI tabs.
   *You usually don't need to upload it*: if `DATATHON_QUESTION_MAP.csv`
   sits next to the app it loads automatically — the sidebar shows
   "🗺️ Question map: 20 items → 6 competencies". Uploading a file
   overrides the local one. No map = per-question analysis (still works).
3. **🔬 Min students per group** (default 15) — charts hide groups smaller
   than this so tiny samples can't fake results. Drop to **5** when using
   the 1K sample file; leave at 15 for the real dataset.

**Big files:** convert once, upload the parquet — loads in seconds:

    python convert_to_parquet.py YOUR_BIG_FILE.xlsx

---

## 4. Test data

    python generate_sample_data.py

creates a sample dataset. A pre-generated 1K-row copy with realistic
planted patterns (weak districts, skill-specific gender gaps, grade
progression, declining blocks) is the recommended demo file. Remember to
set Min students → 5 for it.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `TypeError: build_agg() got an unexpected keyword argument 'qmap'` | Your `adapter.py` is older than `streamlit_app.py`. Replace `adapter.py` in **this** folder with the matching version, then restart Streamlit. They ship as a pair. |
| `ModuleNotFoundError: duckdb` (or plotly, sklearn, …) | The venv isn't active or requirements weren't installed **inside** it: activate the venv, then `pip install -r requirements.txt`. |
| Upload stuck / very slow | Excel parsing is slow at scale — use `convert_to_parquet.py` and upload the `.parquet`. Upload limit is already raised to 2 GB in config. |
| Map tab: "geojson not found" | `karnataka_districts.geojson` must sit next to `streamlit_app.py`, exact filename. |
| Charts look empty on the 1K sample | Sidebar → Min students per group → 5. |
| Everything one flat color / "no gap" everywhere | That's the data, not the app — uniform generated data has no group differences. Use the planted-pattern sample or real data. |
| Port already in use | `streamlit run streamlit_app.py --server.port 8502` |

---

## 6. requirements.txt (what's in it and why)

    streamlit>=1.37     # @st.fragment needs >=1.37
    pandas / numpy      # data handling
    plotly              # charts
    openpyxl            # .xlsx reading
    python-calamine     # 7-12x faster Excel reader (auto fallback to openpyxl)
    scikit-learn        # early-warning model, clustering
    duckdb              # streaming unpivot of 20 items at 2M-row scale
    pyarrow             # .parquet support