"""
Statistical helpers — pure Python/math, no scipy needed.
Used to decide whether an observed gap is REAL or just noise.
"""
import math


def normal_cdf(x):
    """Standard normal cumulative distribution (via error function)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z(pct1, pct2, n1, n2):
    """
    Test whether two percentages differ more than chance would explain.

    pct1, pct2 : percentages (0-100), e.g. girls' below% vs boys' below%
    n1, n2     : how many students in each group

    Returns (z, p_value). p < 0.05 means the gap is statistically significant.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0

    p1, p2 = pct1 / 100.0, pct2 / 100.0
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)

    if pool <= 0 or pool >= 1:
        return 0.0, 1.0

    se = math.sqrt(pool * (1 - pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0

    z = (p1 - p2) / se
    p = 2 * (1 - normal_cdf(abs(z)))
    return z, p


def fisher_exact_2x2(a, b, c, d):
    """
    Two-sided Fisher's exact test on the table [[a, b], [c, d]].

    Exact, so it stays valid at the small cell counts where the normal
    approximation behind two_proportion_z breaks down. Pure math.comb, no scipy.
    Returns the p-value.
    """
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        return 1.0

    row1, col1 = a + b, a + c

    def pmf(k):
        return (math.comb(row1, k) * math.comb(n - row1, col1 - k)
                / math.comb(n, col1))

    observed = pmf(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    # Two-sided: every table at least as extreme as the one we saw.
    return min(1.0, sum(pmf(k) for k in range(lo, hi + 1)
                        if pmf(k) <= observed * (1 + 1e-9)))


def proportion_test(pct1, pct2, n1, n2):
    """
    Compare two percentages and pick a test that is actually VALID here.

    The normal approximation behind a z-test needs roughly 5 expected
    observations in every cell. Below that it is anti-conservative — on a
    3-girls-vs-5-boys split it returned p=0.02 for a difference Fisher puts at
    p=0.14, i.e. it called noise significant. So: z-test when the cells are big
    enough, Fisher's exact when they are not.

    Returns (statistic, p_value, method) where method is "z" or "fisher".
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0, "none"

    a = int(round(pct1 / 100.0 * n1))          # group 1, "below"
    b = int(n1) - a                            # group 1, "not below"
    c = int(round(pct2 / 100.0 * n2))          # group 2, "below"
    d = int(n2) - c

    if min(a, b, c, d) >= 5:
        z, p = two_proportion_z(pct1, pct2, n1, n2)
        return z, p, "z"
    return 0.0, fisher_exact_2x2(a, b, c, d), "fisher"


def cohens_h(pct1, pct2):
    """Effect size for two proportions. 0.2 small, 0.5 medium, 0.8 large."""
    p1, p2 = max(min(pct1 / 100.0, 1), 0), max(min(pct2 / 100.0, 1), 0)
    return abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def effect_label(h):
    if h >= 0.8:  return "large"
    if h >= 0.5:  return "medium"
    if h >= 0.2:  return "small"
    return "negligible"


def sig_marker(p):
    """Human-readable significance marker."""
    if p < 0.001: return "p<0.001"
    if p < 0.01:  return f"p={p:.3f}"
    if p < 0.05:  return f"p={p:.3f}"
    return f"p={p:.2f} (not significant)"
