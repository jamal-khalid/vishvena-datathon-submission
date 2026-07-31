# Vishvena AI — Data Studio (Streamlit)

This is the analysis engine embedded at `/data-studio` in the React app
(login required). Upload Excel/CSV in its sidebar → auto column detection,
filters, trends, gender-gap and competency charts, and AI insights.

## Run it locally
    cd streamlit
    pip install -r requirements.txt
    streamlit run streamlit_app.py

It starts at http://localhost:8501 — the React app's Data Studio page will
detect it automatically and embed it. The `.streamlit/config.toml` here already
sets the Vishvena dark theme and the server flags needed for iframe embedding
(CORS/XSRF off — fine for the datathon; tighten for production).

## Pointing the React app somewhere else
The React side reads the URL from `src/config/streamlitConfig.js`
(default `http://localhost:8501`). Either edit that file or set in `.env`:

    REACT_APP_STREAMLIT_URL=https://your-server.com:8501

So today (backend on the server) → set the env var to the server URL;
later (backend local) → remove it and the default localhost kicks in.

## Test data
    python generate_sample_data.py
creates sample_education_data.xlsx (27,000 rows: grades 4-6, 3 years,
5 competencies, gender, Division>District>Block>Cluster + subjective columns).
A pre-generated copy is included.

## For the AI dev
Implement `get_insights(aggregates)` at the top of streamlit_app.py.
It receives a small dict of computed stats (never raw data) and returns
a list of insight strings. Debug expander in the AI tab shows the exact payload.