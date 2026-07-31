# Datathon 2026 — Analysis Pipeline + Dashboard

Reimagining Learning Outcomes Through Analytics — Akshara Foundation.

**No LLM or generative AI is used anywhere.** Every number comes from a
deterministic computation, so the same input always produces the same output.
That is a hard competition rule, not a preference.

---

## Run it

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run streamlit_app.py
```

Then either upload a dataset, or pick `DATATHON_TEST_1K.xlsx` from the
"load files already on disk" box.

> `DATATHON_TEST_1K.xlsx` is a 1,000-row sample so the app runs out of the box.
> It is **too small for real conclusions** — see "Group sizes" below.

---

## What is where

### Analysis layers (project root)

| File | What it does |
|---|---|
| `units.py` | **Read this first.** The response-vs-child rule. Getting it wrong inflates every headcount 20×. |
| `stats_tests.py` | z-test, Fisher's exact, Cohen's h. No scipy — pure `math`. |
| `verbalize.py` | Layer 1 — turns aggregated rows into sentences (templates, not a model). |
| `insights.py` | Layer 4 — 11 finding generators, each traced to a handbook question. |
| `competency.py` | Layer 5 — per-competency report (overview, gender, grade, geography, trend, risk). |
| `brief.py` | Layer 7 — role-based narrative briefs for block / district / policy readers. |
| `playbook.py` | Layer 8 — recommendation engine. 9 base actions × 6 modifiers = 576 combinations. |
| `models_ml.py` | Clustering, improvement benchmarks, What-If, and the early-warning model. |
| `charts.py` | Plotly figures shared by both apps. |
| `analytics.py` | DuckDB CSV→Parquet aggregation for the older standalone app. |
| `gen_data.py` | Generates synthetic test data so you can work without the real dataset. |
| `app.py` | Standalone explainer app — walks through each layer. The dashboard below is the real deliverable. |

### Dashboard (`streamlit_app/`)

| File | What it does |
|---|---|
| `streamlit_app.py` | The 17-tab dashboard. Also holds the multi-file loader. |
| `adapter.py` | Bridges any incoming file shape → the schema the layers expect. |
| `karnataka_districts.geojson` | Map boundaries. |
| `echarts.min.js` | Vendored so the app makes **zero network calls**. |
| `.streamlit/config.toml` | Theme. Keep it — the app is styled around it. |

---

## Three things that will bite you if you don't know them

### 1. `units.py` — responses vs children

When the 20 question columns become the competency dimension, **one child
produces 20 rows**. So `agg["n"].sum()` across competencies counts every child
20 times.

```python
n         -> assessment responses -> correct denominator for a PERCENTAGE
students  -> distinct children    -> correct denominator for a HEADCOUNT / Z-TEST
```

Percentages are unaffected (the factor cancels in a weighted mean). Headcounts
and significance tests are not. This bug was live and had the policy brief
reporting *"6,420 assessed children"* when there were **321**.

**Use `units.headcount(rows)` for any child count. Never `rows["n"].sum()`.**

### 2. Significance needs the right test

The 20 answers from one child are **one child measured 20 times**, not 20
independent observations. Feeding response counts to a z-test shrinks the
standard error by √20 and turns p=0.35 into p<0.0001.

Also, a z-test needs ~5 expected observations per cell. On small groups it is
invalid. `stats_tests.proportion_test()` picks z or Fisher's exact
automatically — **use it instead of calling `two_proportion_z` directly.**

Fixing this removed **94% of previously "significant" gender gaps** (208 → 12).
They were noise.

### 3. Group sizes decide whether anything is trustworthy

On the 1K sample, the median group is 2–6 students, so an 80-point "gender gap"
is three girls. The sidebar has **"Ignore groups smaller than"** — raise it on
real data. On the full 1.6M-row dataset the median group is ~2,280 students and
everything becomes reliable.

---

## Loading data

Excel caps a sheet at **1,048,576 rows**, so a full dataset arrives split. The
loader takes **multiple files or a `.zip`**, and reads `.xlsx / .xls / .csv /
.parquet`. Parts are concatenated *before* any analysis, so every layer sees the
whole dataset.

- Subfolders appear in the disk picker, and a whole folder can be selected at once
- A corrupt part is reported and skipped; the rest still load
- Parts with a missing optional column merge on the union of columns
- Genuine overlap between parts is detected and flagged
- The combined result is cached as Parquet next to the files

**Performance on the real 1.6M-row dataset (2 Excel parts):**

```
cold first run   49.3 s     (read 40.6s + aggregate 4.0s + layers 4.6s)
warm run          0.54 s    (Parquet cache, 75x)
```

`.xlsx` is read with `python-calamine` (Rust), ~8× faster than openpyxl with
byte-identical output. If that package is missing it falls back to openpyxl
automatically — slower, but it still runs.

> Don't add parallel reading. I measured it: 72s vs 54s sequential. Process
> spawn plus shipping frames back costs more than the parse saves.

---

## Verifying you haven't broken anything

The aggregation is cross-checked against a hand computation in plain pandas.
The invariant to preserve:

```
max |below_pct difference| = 0.0000
agg["n"].sum() == raw_rows × n_questions
units.headcount(year_rows) == distinct children that year
```

---

## Still open

| Item | Why it matters |
|---|---|
| **Map** — 0 of 31 districts match the GeoJSON | Needs a real Karnataka spelling map (Bengaluru/Bangalore, Kalaburagi/Gulbarga, Mysuru/Mysore, Vijayapura/Bijapur, Shivamogga/Shimoga, Tumakuru/Tumkur, plus Vijayanagara split from Ballari in 2021) |
| **Cross-dataset analysis** — not started | 15% of the total score |
| **`run_all.py` + `outputs/` + `manifest.yml` + `claims.json`** | Required submission shape; must rebuild offline in ~3 min |
| **Layer 4 scoring** mixes units | `yoy×5`, `z×10` etc. — ranking across finding *types* isn't meaningful |
| **Prediction tab** extrapolates from 2 points | Warned in-app. The early-warning model in `models_ml.py` is the sound one. |

## Also

- **Rotate the hardcoded Groq API key** in the other backend
  (`main/data_thon/app/utils.py:7`) — it is live and in source.
- `llm.py` was **deliberately left out**. It was dead code, but the rules ban
  LLMs including local ones and the repo is public during judging.
