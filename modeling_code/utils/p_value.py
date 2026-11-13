import numpy as np
from .V_calc import alt_v

# =============================================================================
# Breach probability: Option C (logit shift) from breach_probability_v3.ipynb
# =============================================================================
# The questionnaire/pentest result enters as a normalized safety score
# s in [0, 1] (higher = safer). The industry scalar enters *inside* the
# sigmoid as a bias term, so the result stays within the floor/ceiling.

# ---- Anchors: P(breach) at the extremes for a NEUTRAL industry ----
P_AT_ZERO = 0.98      # worst score
P_AT_PERFECT = 0.08   # best score
EPS = 0.01            # how far inside floor/ceiling to aim when solving sigmoid

# ---- Industry knobs ----
NEUTRAL_SCALAR = 0.50  # scalar that leaves the score-based P unchanged
BETA_C = 4.0           # strength of the industry logit shift

# ---- V -> normalized score ----
# alt_v.calculate_value returns V in [0, 20] without a pentest and
# [0, ~20.1] with one (higher = riskier). s = 1 - V / V_MAX flips and
# normalizes it, and works for both the questionnaire and PenTest paths.
V_MAX = 20.0


def get_industry_scalars():
    """NAICS 2-digit code -> targeting scalar. Higher = more attacked."""
    return {
        "11": 0.15,   # agriculture, forestry, fishing & hunting (A+)
        "21": 0.30,   # mining, quarrying, oil & gas extraction (A)
        "22": 0.45,   # utilities (B+)
        "23": 0.15,   # construction (A+)
        "31": 0.30,   # manufacturing (A)
        "32": 0.30,   # manufacturing (A)
        "33": 0.30,   # manufacturing (A)
        "41": 0.30,   # wholesale trade (A) - Canadian NAICS
        "42": 0.30,   # wholesale trade (A)
        "44": 0.30,   # retail trade (A)
        "45": 0.30,   # retail trade (A)
        "48": 0.30,   # transportation & warehousing (A)
        "49": 0.30,   # transportation & warehousing (A)
        "51": 0.58,   # information (IT & media) (B)
        "52": 0.750,  # finance and insurance (B-), adjusted from 0.85
        "53": 0.45,   # real estate & rental & leasing (B+)
        "54": 0.73,   # professional / scientific / technical services (B-)
        "55": 0.45,   # management of companies & enterprises (B+)
        "56": 0.30,   # administrative & support & waste mgmt & remediation (A)
        "61": 0.58,   # educational services (B)
        "62": 0.764,  # health care and social assistance (B-), adjusted from 0.88
        "71": 0.45,   # arts, entertainment, recreation (B+)
        "72": 0.45,   # accommodation & food services (B+)
        "81": 0.45,   # other services (except public administration) (B+)
        "91": 0.58,   # public administration (B) - Canadian NAICS
        "92": 0.58,   # public administration (B)
    }


def _solve_sigmoid(p_zero, p_perfect, eps):
    """Solve k, s0 for the floor/ceiling-rescaled sigmoid at NEUTRAL industry."""
    p_floor, p_ceil = p_perfect, p_zero
    width = p_ceil - p_floor
    sigma_0 = ((p_ceil - eps * width) - p_floor) / width
    sigma_1 = ((p_floor + eps * width) - p_floor) / width
    L0 = np.log((1 - sigma_0) / sigma_0)
    L1 = np.log((1 - sigma_1) / sigma_1)
    k = L1 - L0
    s0 = -L0 / k
    return k, s0, p_floor, width


# Solved once at import time (k ~= 9.19, s0 = 0.5 with the defaults above)
_K, _S0, _FLOOR, _WIDTH = _solve_sigmoid(P_AT_ZERO, P_AT_PERFECT, EPS)


def v_to_normalized_score(V):
    """Convert V (higher = riskier) to s in [0, 1] (higher = safer)."""
    return float(np.clip(1.0 - V / V_MAX, 0.0, 1.0))


def p_option_C(s, scalar):
    """Option C: industry scalar as a bias inside the sigmoid.

    Args:
        s: normalized security score in [0, 1], higher = safer
        scalar: industry targeting scalar
    """
    shift = BETA_C * (scalar - NEUTRAL_SCALAR)
    sigma = 1.0 / (1.0 + np.exp(_K * (s - _S0) - shift))
    return _FLOOR + _WIDTH * sigma


def get_p_value(company_size, security_data, assessment_data, business_data,
                industry_code=None, industry_benchmark=None):
    """
    Calculate breach probability using Option C (logit shift).

    industry_benchmark is no longer used in the calculation; it is kept in
    the signature so existing callers don't need to change.

    Returns:
        p_value, S_bar
    """
    V, S_bar = alt_v.v_calc(company_size, security_data, assessment_data, business_data)
    s = v_to_normalized_score(V)

    # Unknown or missing industry -> neutral scalar (no shift). A default of
    # 0 would apply a large negative shift and understate risk.
    scalar = NEUTRAL_SCALAR
    if industry_code:
        scalar = get_industry_scalars().get(str(industry_code), NEUTRAL_SCALAR)

    p_value = float(p_option_C(s, scalar))
    return p_value, S_bar
