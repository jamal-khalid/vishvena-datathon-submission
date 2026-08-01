"""
Layer 5 — visualisation (was skipped; it is 15% of the Datathon score).
Plotly only: the handbook bans Tableau/Power BI and requires code-based charts.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {"Critical": "#c0392b", "At-risk": "#e67e22", "Strong": "#27ae60"}


def gap_heatmap(agg, district, year=None):
    """Block x competency heatmap of below-grade %."""
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    piv = d.pivot_table(index="block", columns="competency",
                        values="below_pct", aggfunc="mean")
    fig = px.imshow(
        piv, text_auto=".0f", aspect="auto",
        color_continuous_scale="RdYlGn_r", zmin=0, zmax=100,
        labels=dict(color="% below grade"),
    )
    fig.update_layout(
        title=f"Learning gaps — {district} ({int(year)})",
        xaxis_title=None, yaxis_title=None, height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def trend_lines(agg, district):
    """Below-grade % over the years, one line per competency."""
    d = (agg[agg["district"] == district]
         .groupby(["year", "competency"])["below_pct"].mean().reset_index())
    fig = px.line(d, x="year", y="below_pct", color="competency", markers=True)
    fig.update_layout(
        title=f"Trend over time — {district}",
        yaxis_title="% below grade level", xaxis_title=None,
        height=340, margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(dtick=1)
    return fig


def gender_gap_bars(agg, district, year=None):
    """Diverging bars: positive = girls trail, negative = boys trail."""
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    t = d.groupby("competency")["gender_gap"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=t.values, y=t.index, orientation="h",
        marker_color=["#8e44ad" if v > 0 else "#2980b9" for v in t.values],
        text=[f"{v:+.1f}" for v in t.values], textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#888")
    fig.update_layout(
        title=f"Gender gap by competency — {district} "
              f"(right = girls trail, left = boys trail)",
        xaxis_title="percentage points", yaxis_title=None,
        height=320, margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ==================================================================
#  "What replaced the LLM" — visuals that show the MECHANISM
# ==================================================================

def insight_scores(items):
    """
    Layer 4 mechanism: every generator's finding, ranked by score.
    Shows that ranking is computed, not chosen by a model.
    """
    if not items:
        return None
    d = sorted(items, key=lambda x: x["score"])
    fig = go.Figure(go.Bar(
        x=[i["score"] for i in d],
        y=[i["category"] for i in d],
        orientation="h",
        marker_color="#2c6fbb",
        text=[f"{i['score']:.0f}" for i in d],
        textposition="outside",
        hovertext=[i["text"][:120] + "..." for i in d],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Layer 4 — each generator's finding, scored and ranked by code",
        xaxis_title="score (effect magnitude)", yaxis_title=None,
        height=330, margin=dict(l=10, r=60, t=50, b=10),
    )
    return fig


def significance_volcano(agg, district, year=None):
    """
    Layer 4 mechanism: the z-test filter, made visible.

    Every gender gap in the district is plotted. Only points that clear BOTH
    thresholds get reported. This is the step an LLM cannot do.
    """
    import math
    from stats_tests import proportion_test

    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]

    xs, ys, cols, txt = [], [], [], []
    for r in d.itertuples():
        if r.gender_gap is None or pd.isna(r.gender_gap):
            continue
        # Must use the SAME test insights.g_gender_gap uses, or the chart's
        # "red = reported" legend is false. Real girl/boy counts rather than an
        # assumed 50/50 split, and Fisher's exact when the cells are too small
        # for a z-test — which on block-level data is most of them.
        nf = int(getattr(r, "f_n", 0)) or max(int(r.n) // 2, 1)
        nm = int(getattr(r, "m_n", 0)) or max(int(r.n) - nf, 1)
        _, p, method = proportion_test(r.f_below, r.m_below, nf, nm)
        p = max(p, 1e-12)
        keep = abs(r.gender_gap) >= 4 and p < 0.05
        xs.append(r.gender_gap)
        ys.append(-math.log10(p))
        cols.append("#c0392b" if keep else "#bdc3c7")
        txt.append(f"{r.block} G{r.grade} {r.competency}<br>"
                   f"gap {r.gender_gap:+.1f} pts, p={p:.4f} "
                   f"({'Fisher exact' if method == 'fisher' else 'z-test'})<br>"
                   f"{nf} girls vs {nm} boys<br>"
                   f"{'REPORTED' if keep else 'suppressed as noise'}")

    if not xs:
        return None

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(color=cols, size=10, line=dict(width=0.5, color="#555")),
        hovertext=txt, hoverinfo="text",
    ))
    fig.add_hline(y=-math.log10(0.05), line_dash="dash", line_color="#e74c3c",
                  annotation_text="p = 0.05", annotation_position="right")
    fig.add_vline(x=4, line_dash="dot", line_color="#888")
    fig.add_vline(x=-4, line_dash="dot", line_color="#888",
                  annotation_text="±4 pts", annotation_position="top left")
    fig.update_layout(
        title="Layer 4 — gender-gap significance filter "
              "(red = reported, grey = suppressed as noise)",
        xaxis_title="gender gap (points).  right = girls trail",
        yaxis_title="−log₁₀(p)   higher = more certain",
        height=380, margin=dict(l=10, r=10, t=60, b=10), showlegend=False,
    )
    return fig


def competency_correlation(corr):
    """Which competencies fail together — points at shared root causes."""
    if corr is None:
        return None
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Blues", zmin=0, zmax=1)
    fig.update_layout(
        title="Do competencies fail together?  (r → 1 = same blocks weak in both)",
        xaxis_title=None, yaxis_title=None, height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def competency_block_bars(rep):
    """Block ranking for one competency."""
    t = rep["geography"]["table"].sort_values("below_pct")
    fig = go.Figure(go.Bar(
        x=t["below_pct"], y=t["block"], orientation="h",
        marker_color=["#c0392b" if v >= 60 else "#e67e22" if v >= 45 else "#27ae60"
                      for v in t["below_pct"]],
        text=[f"{v:.0f}%" for v in t["below_pct"]], textposition="outside",
    ))
    fig.add_vline(x=60, line_dash="dot", line_color="#c0392b",
                  annotation_text="Critical", annotation_position="top")
    fig.add_vline(x=45, line_dash="dot", line_color="#e67e22",
                  annotation_text="At-risk", annotation_position="bottom")
    fig.update_layout(
        title=f"{rep['competency']} — block ranking",
        xaxis_title="% below grade level", yaxis_title=None,
        height=320, margin=dict(l=10, r=60, t=50, b=10),
    )
    return fig


def playbook_grid(agg, district, year=None, min_n=30):
    """
    Layer 8 mechanism: the decision table itself, with real blocks placed in it.
    Every cell shows how many block x competency pairs landed there.
    """
    import playbook as pb

    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]

    # Classify EXACTLY as playbook.recommend() does, or the grid and the
    # recommendations disagree: an unweighted mean with no size floor counted
    # 48 cells for Bagalkot while the engine — correctly — issued none of them.
    # Same weighted mean, same evidence-gated bands, same min_n.
    def _w(s, col="below_pct"):
        wt = pd.to_numeric(s["n"], errors="coerce").fillna(0.0)
        v = pd.to_numeric(s[col], errors="coerce")
        ok = v.notna() & (wt > 0)
        return float((v[ok] * wt[ok]).sum() / wt[ok].sum()) if ok.any() else float("nan")

    g = (d.groupby(["block", "competency"])
           .apply(lambda s: pd.Series({"below_pct": _w(s),
                                       "prev_pct": _w(s, "prev_pct"),
                                       "n": s["n"].sum()}),
                  include_groups=False)
           .reset_index().dropna(subset=["below_pct"]))
    skipped = int((g["n"] < min_n).sum())
    g = g[g["n"] >= min_n]

    sev_order = ["Critical", "At-risk", "Strong"]
    traj_order = ["Declining", "Stagnant", "Improving"]
    counts = {(s, t): 0 for s in sev_order for t in traj_order}

    for r in g.itertuples():
        s = pb.severity(r.below_pct, r.n)
        t = pb.trajectory(r.below_pct, r.prev_pct, r.n)
        if (s, t) in counts:
            counts[(s, t)] += 1

    z, labels = [], []
    for s in sev_order:
        zrow, lrow = [], []
        for t in traj_order:
            c = counts[(s, t)]
            rule = pb._match(s, t)
            pri = rule["priority"].split(" — ")[0] if rule else "—"
            zrow.append(c)
            lrow.append(f"<b>{pri}</b><br>{c} cases" if rule else f"no rule<br>{c} cases")
        z.append(zrow); labels.append(lrow)

    fig = go.Figure(go.Heatmap(
        z=z, x=traj_order, y=sev_order,
        text=labels, texttemplate="%{text}",
        colorscale="Reds", showscale=False,
        xgap=3, ygap=3,
    ))
    _sub = (f"<br><sup>{int(g['n'].sum()):,} students across {len(g)} "
            f"block×competency pairs"
            + (f" · {skipped} pair(s) below the {min_n}-student minimum are "
               f"excluded, exactly as the engine excludes them" if skipped
               else "")
            + "</sup>")
    fig.update_layout(
        title=f"Layer 8 — the decision table, with {district}'s blocks "
              f"placed in it{_sub}",
        height=340, margin=dict(l=10, r=10, t=76, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def severity_scale_scatter(agg, district, year=None):
    """
    Severity vs scale — the chart that reframes priority.
    High % is not the same as high need.
    """
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    t = (d.groupby("block")
           .agg(below=("below_pct", "mean"), students=("n", "sum")).reset_index())
    t["affected"] = (t["students"] * t["below"] / 100).round()
    fig = px.scatter(
        t, x="below", y="affected", size="students", text="block",
        labels={"below": "% below grade level (severity)",
                "affected": "children below grade level (scale)"},
    )
    fig.update_traces(textposition="top center", marker_color="#c0392b", opacity=0.75)
    fig.update_layout(
        title=f"Severity vs scale — {district}",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Plain-language charts. The ones above answer "what does the analysis say?";
# these answer "what does that mean?" for a reader who does not work with data.
# ---------------------------------------------------------------------------

BAND_COLORS = {"below": "#e74c3c", "at": "#f4c542", "above": "#27ae60"}


def _largest_remainder(parts, total=100):
    """Round percentages to whole dots that still sum to exactly `total`."""
    raw = [max(p, 0.0) * total / 100.0 for p in parts]
    floors = [int(x) for x in raw]
    short = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in range(short):
        floors[order[i % len(order)]] += 1
    return floors


def hundred_children(below_pct, at_pct, above_pct, n_children=None):
    """
    "Out of every 100 children" — a 10x10 grid of dots, one per child.

    A percentage is an abstraction; a hundred dots you can count is not. This
    is how ASER and Pratham have reported learning levels in India for years,
    so the audience already reads it fluently — no legend-decoding, no axis.
    Unlike a pie chart it also supports counting ("about a third of the room"),
    and unlike a bar it makes the unit — one child — explicit.
    """
    nb, na, nv = _largest_remainder([below_pct, at_pct, above_pct])
    seq = (["below"] * nb) + (["at"] * na) + (["above"] * nv)
    labels = {"below": "Below grade level", "at": "At grade level",
              "above": "Above grade level"}

    fig = go.Figure()
    for key in ("below", "at", "above"):
        xs, ys = [], []
        for i, k in enumerate(seq):
            if k != key:
                continue
            xs.append(i % 10)
            ys.append(9 - i // 10)          # fill top-left downwards
        if not xs:
            continue
        cnt = len(xs)
        extra = ""
        if n_children:
            extra = f"<br>≈ {round(n_children * cnt / 100):,} of {n_children:,} children"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=f"{labels[key]} — {cnt}",
            marker=dict(size=20, color=BAND_COLORS[key], symbol="circle",
                        line=dict(color="rgba(255,255,255,.35)", width=1)),
            hovertemplate=f"<b>{labels[key]}</b><br>{cnt} in every 100{extra}"
                          "<extra></extra>"))
    fig.update_layout(
        height=360, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(visible=False, range=[-0.7, 9.7]),
        yaxis=dict(visible=False, range=[-0.7, 9.7],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)")
    return fig


def whatif_children(before_pct, students, scenarios):
    """
    What-If expressed in CHILDREN rather than percentage points.

    "7.6% below, improving 1.5 points" asks the reader to do arithmetic before
    they can care. "1,240 children below grade level; this plan moves 94 of
    them above it" does not. Each bar is one scenario: the red part is who is
    still behind, the green part is who moved.
    """
    if not students or not scenarios:
        return None
    now_below = round(students * float(before_pct) / 100.0)

    rows = [("Today", now_below, 0)]
    for s in scenarios:
        after = round(students * float(s["after_below_pct"]) / 100.0)
        rows.append((s["scenario"], after, max(now_below - after, 0)))

    names = [r[0] for r in rows][::-1]        # Today at the top
    still = [r[1] for r in rows][::-1]
    moved = [r[2] for r in rows][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=still, orientation="h", name="Still below grade level",
        marker_color="#e74c3c",
        text=[f"{v:,}" for v in still], textposition="inside",
        insidetextanchor="middle",
        hovertemplate="<b>%{y}</b><br>%{x:,} children still below"
                      "<extra></extra>"))
    fig.add_trace(go.Bar(
        y=names, x=moved, orientation="h", name="Moved above the line",
        marker_color="#27ae60",
        text=[f"+{v:,}" if v else "" for v in moved], textposition="inside",
        insidetextanchor="middle",
        hovertemplate="<b>%{y}</b><br>%{x:,} children moved above"
                      "<extra></extra>"))
    fig.update_layout(
        barmode="stack", height=90 + 52 * len(names),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="children below grade level today",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.35)
    return fig


# ---------------------------------------------------------------------------
# Cross-dataset charts. These exist to make the STATISTICS visible, not just
# the numbers — a reader who does not know what a p-value is can still see
# "the bar crosses zero" or "every bar is inside the grey band".
# ---------------------------------------------------------------------------

def effect_forest(table, crit=None, top=14, title=None):
    """
    Forest plot: each indicator's correlation with its 95% confidence interval.

    This is the whole argument in one picture:
      * the DOT is how strong the relationship looks
      * the BAR is how uncertain that is
      * the vertical line at 0 is "no relationship at all"
      * a bar crossing 0 means we cannot even be sure of the DIRECTION
      * the grey band is the detection floor — inside it, nothing can be
        distinguished from zero no matter what the dot says

    A plain bar chart of r hides all of that and invites over-reading.
    """
    if table is None or len(table) == 0:
        return None
    d = table.copy()
    d = d[d["r"].notna()]
    if d.empty:
        return None
    d = d.reindex(d["r"].abs().sort_values(ascending=True).index).tail(top)

    sig = d["verdict"].astype(str).str.startswith("significant") \
        if "verdict" in d.columns else pd.Series(False, index=d.index)

    fig = go.Figure()
    if crit and crit == crit:
        fig.add_vrect(x0=-crit, x1=crit, fillcolor="#8892b0", opacity=0.13,
                      line_width=0,
                      annotation_text=f"cannot be detected (|r| < {crit:.2f})",
                      annotation_position="top left",
                      annotation_font_size=11)
    for lbl, mask, colour in (("Real relationship", sig, "#27ae60"),
                              ("Not distinguishable from zero", ~sig, "#9aa4b2")):
        sub = d[mask]
        if sub.empty:
            continue
        xs, ys = [], []
        for r in sub.itertuples():
            xs += [r.ci_low, r.ci_high, None]
            ys += [r.variable, r.variable, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=lbl,
                                 line=dict(color=colour, width=6),
                                 opacity=0.45, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=sub["r"], y=sub["variable"], mode="markers", showlegend=False,
            marker=dict(color=colour, size=13,
                        line=dict(color="white", width=1.5)),
            customdata=np.stack([sub["ci_low"], sub["ci_high"],
                                 sub.get("p_adj", sub["r"] * 0)], axis=-1),
            hovertemplate="<b>%{y}</b><br>r = %{x:+.3f}"
                          "<br>95%% range %{customdata[0]:+.2f} to "
                          "%{customdata[1]:+.2f}"
                          "<br>corrected p = %{customdata[2]:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#e74c3c", line_dash="dash",
                  annotation_text="no relationship", annotation_position="top")
    fig.update_layout(
        title=title or "How strong is each link — and how sure are we?",
        xaxis_title="correlation with % below grade level  "
                    "(← more of it = fewer children behind)",
        yaxis_title="", height=max(340, 30 * len(d) + 140),
        margin=dict(t=60, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def relationship_scatter(df, xcol, ycol, label_col=None, r=None, p=None):
    """
    One indicator against the outcome, with the fitted line.

    The forest plot says how strong; this says what that strength LOOKS like.
    r = -0.34 is abstract until you see the cloud of districts and the line
    sloping through it.
    """
    d = df[[c for c in (xcol, ycol, label_col) if c]].dropna()
    if len(d) < 3:
        return None
    x = pd.to_numeric(d[xcol], errors="coerce")
    y = pd.to_numeric(d[ycol], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=13, color=y, colorscale="RdYlGn_r", showscale=False,
                    line=dict(color="white", width=1)),
        text=d[label_col] if label_col else None,
        hovertemplate=("<b>%{text}</b><br>" if label_col else "")
                      + f"{xcol}: %{{x:,.0f}}<br>{ycol}: %{{y:.1f}}%"
                        "<extra></extra>",
        name="districts"))
    if x.std() > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=b * xs + a, mode="lines",
                                 line=dict(color="#5eead4", width=3, dash="dot"),
                                 name="trend", hoverinfo="skip"))
    sub = ""
    if r is not None:
        sub = f"   (r = {r:+.2f}" + (f", p = {p:.3f})" if p is not None else ")")
    fig.update_layout(
        title=f"{xcol} vs children below grade level{sub}",
        xaxis_title=xcol, yaxis_title="% below grade level",
        height=420, margin=dict(t=60, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig
