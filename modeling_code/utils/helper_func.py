import math

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import scipy.stats as stats
import copy
import warnings
import logging
import os
import time
from datetime import date

# Suppress PyMC warnings and progress bars
# warnings.filterwarnings("ignore")
# logging.getLogger("pymc").setLevel(logging.ERROR)

#import pymc as pm
#import arviz as az

import json
from utils.pci_dss import calculate_pci_dss_assessment

logger = logging.getLogger(__name__)

################################### Read Json File Outputs ##################################

def convert_shorthand(value):
    value = value.strip()
    if value.endswith('K'):
        return float(value[:-1]) * 1_000
    elif value.endswith('M'):
        return float(value[:-1]) * 1_000_000
    else:
        return float(value)
    
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, bool):
        return bool(obj)
    return obj

def parse_number(number_str):
    number_str = number_str.replace('$', '').replace(',', '').strip()
    if number_str == '':
        return 1
    elif 'million' in number_str.lower():
        number = float(number_str.lower().replace('million', '').strip()) * 1_000_000
    elif 'billion' in number_str.lower():
        number = float(number_str.lower().replace('billion', '').strip()) * 1_000_000_000
    else:
        number = float(number_str)
    return number

def parse_revenue(revenue_str):
    revenue_str = revenue_str.replace('$', '').replace(',', '').strip()
    if 'Less than' in revenue_str:
        return 500_000  # Approximation for "Less than $1 million"
    elif 'More than' in revenue_str:
        return 1_500_000_000  # Approximation for "More than $1 billion"
    elif '-' in revenue_str:
        low_str, high_str = revenue_str.split('-')
        low = parse_number(low_str.strip())
        high = parse_number(high_str.strip())
        average_revenue = (low + high) / 2  
        return average_revenue
    else:
        return parse_number(revenue_str)

def parse_employee_count(employee_str):
    employee_str = employee_str.replace(',', '').strip()
    if '-' in employee_str:
        low_str, high_str = employee_str.split('-')
        low = int(low_str.replace(',', '').strip())
        high = int(high_str.replace(',', '').strip())
        average_employees = (low + high) / 2
        return average_employees
    elif '+' in employee_str:
        return 5500
    else:
        return int(employee_str)
    
def parse_number_of_records(records_str):
    records_str = records_str.replace(',', '').strip()  

    if records_str == 'None':
        return 0

    if '-' in records_str:
        low_str, high_str = records_str.split('-')
        low = int(low_str.replace(',', '').strip())
        high = int(high_str.replace(',', '').strip())
        average_records = (low + high) / 2
        return average_records
    elif 'More' in records_str:
        return 100000 
    else:
        return int(records_str.replace(',', ''))


DEFAULT_CURRENCY = "USD"
COUNTRY_CURRENCY_MAP = {
    "mexico": "MXN",
    "india": "INR",
    "canada": "CAD",
    "us": "USD",
    "usa": "USD",
    "united states": "USD",
    "united states of america": "USD",
}
STATIC_USD_RATES = {
    "USD": 1.0,
    "MXN": 17.0,
    "INR": 83.0,
    "CAD": 1.35,
}

UMA_DAILY_MXN = 113.14      # 2025 UMA, daily
UMA_ANNUAL_MXN = 41296.0 
FX_API_URL = os.environ.get("FX_API_URL", "https://open.er-api.com/v6/latest/USD")
FX_CACHE_TTL_SECONDS = int(os.environ.get("FX_CACHE_TTL_SECONDS", 6 * 60 * 60))
FX_REQUEST_TIMEOUT = 10

_fx_cache = {"rates": None, "fetched_at": 0.0}

CRORE_INR = 10_000_000.0

def get_crore_usd():
    """USD value of Rs. 1 crore at the current INR rate."""
    inr_per_usd = get_usd_to_target_rate("INR") or STATIC_USD_RATES["INR"]
    return CRORE_INR / inr_per_usd

def get_uma_usd():
    """Return (uma_daily_usd, uma_annual_usd) from the current MXN rate."""
    mxn_per_usd = get_usd_to_target_rate("MXN") or STATIC_USD_RATES["MXN"]
    return (UMA_DAILY_MXN / mxn_per_usd, UMA_ANNUAL_MXN / mxn_per_usd)


def detect_currency_from_country(country_value):
    country_text = str(country_value or "").strip().lower()
    return COUNTRY_CURRENCY_MAP.get(country_text, DEFAULT_CURRENCY)


def _normalize_regulation_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


# key -> (canonical regulation, flags)
REGULATION_KEY_MAP = {
    # India
    "IN_DPDPA_BASE":        ("DPDPA", ()),
    "IN_DPDPA_HEALTH":      ("DPDPA", ("health",)),
    "IN_DPDPA_BIOMETRIC":   ("DPDPA", ("biometrics",)),
    "IN_SEBI":              ("SEBI", ()),
    "IN_RBI":               ("RBI Bank Cyber", ()),
    "DPDPA_DISHA":          ("DPDPA", ("health",)),
    "DPDPA_UIDAI":          ("DPDPA", ("biometrics",)),
    # Canada
    "CA_PIPEDA_BASE":       ("OPC CA", ()),
    "CA_CSA_OSC":           ("CSA / OSC", ()),
    "CA_PHIPA":             ("PHIPA / Provincial Health Privacy Laws", ("health",)),
    "CA_OSFI":              ("OSFI", ()),
    "CA_QUEBEC":            ("Quebec Law 25", ()),
    "CA_PIPA_ABBC":         ("PIPA AB/BC", ()),
    "CSA":                  ("CSA / OSC", ()),
    "PHIPA":                ("PHIPA / Provincial Health Privacy Laws", ("health",)),
    "Law 25":               ("Quebec Law 25", ()),
    "PIPA":                 ("PIPA AB/BC", ()),
    # Mexico
    "MX_LFPDPPP_BASE":      ("LFPDPPP", ()),
    "MX_LFPDPPP_HEALTH":    ("LFPDPPP", ("health",)),
    "MX_LFPDPPP_BIOMETRIC": ("LFPDPPP", ("biometrics",)),
    "MX_LFPDPPP_AUTOMATED": ("LFPDPPP", ("automated",)),
    "MX_FINTECH":           ("Fintech Law", ()),
    "MX_Fintech":           ("Fintech Law", ()),
    "MX_CNBV_LMV":          ("CNBV / LMV", ()),
    "CNBV":                 ("CNBV / LMV", ()),
    "COFEPRIS":             ("COFEPRIS / NOM-024-SSA3", ("health",)),
    "LFPDPPP_Biometrics":   ("LFPDPPP", ("biometrics",)),
    "LFPDPPP_AI":           ("LFPDPPP", ("automated",)),
    # Shared
    "GDPR":                 ("GDPR", ()),
}

# Legacy free-text labels still in flight.
LEGACY_LABEL_ALIASES = {
    "FTC": {"FTC"},
    "SEC": {"SEC"},
    "HIPAA": {"HIPAA"},
    "OCC": {"OCC"},
    "CCPA": {"CCPA"},
    "BIPA": {"BIPA"},
    "NYDFS Part 500": {"NYDFS Part 500"},
    "UK ICO": {"UK ICO"},
    "CSA / OSC": {"CSA / OSC"},
    "Quebec Law 25": {"Quebec Law 25"},
    "PIPA AB/BC": {"PIPA AB/BC"},
    "OSFI": {"OSFI"},
    "CNBV / LMV": {"CNBV / LMV"},
    "COFEPRIS / NOM-024-SSA3": {"COFEPRIS / NOM-024-SSA3"},
    "Fintech Law": {"Fintech Law"},
    "DPDPA": {
        "DPDPA", "DPDPA 2023", "DPDPA 2023 / DPBI", "DPBI",
        "Digital Personal Data Protection Act 2023",
        "Data Protection Board of India", "India",
        "Biometrics / SPDI / DPDPA",
        "DPDPA 2023 (sensitive personal data) + DISHA / ABDM",
        "DPDPA 2023 + Aadhaar Act / UIDAI",
        "DPDPA 2023 / DPBI - Digital Personal Data Protection Act 2023 / "
        "Data Protection Board of India",
    },
    "LFPDPPP": {
        "LFPDPPP", "LFPDPPP 2025", "LFPDPPP 2025 / SABG",
        "LFPDPPP 2025 + COFEPRIS", "SABG", "Mexico",
    },
    "OPC CA": {"OPC CA", "PIPEDA", "PIPEDA / OPC CA", "Canada"},
    "RBI Fintech / PSSA": {"RBI", "RBI Bank Cyber", "RBI Fintech / PSSA"},
    "Health Data / ABDM / SPDI": {
        "Health Data / ABDM / SPDI", "DISHA / ABDM",
        "Ayushman Bharat Digital Mission framework",
    },
}

LEGACY_ALIAS_LOOKUP = {
    _normalize_regulation_text(alias): canonical
    for canonical, aliases in LEGACY_LABEL_ALIASES.items()
    for alias in aliases
}

LEGACY_LABEL_FLAGS = {
    _normalize_regulation_text("DPDPA 2023 (sensitive personal data) + DISHA / ABDM"): ("health",),
    _normalize_regulation_text("DPDPA 2023 + Aadhaar Act / UIDAI"): ("biometrics",),
    _normalize_regulation_text("DPDPA_DISHA"): ("health",),
    _normalize_regulation_text("DPDPA_UIDAI"): ("biometrics",),
    _normalize_regulation_text("LFPDPPP 2025 + COFEPRIS"): ("health",),
    _normalize_regulation_text("COFEPRIS"): ("health",),
    _normalize_regulation_text("Biometrics / SPDI / DPDPA"): ("biometrics",),
    _normalize_regulation_text("LFPDPPP_Biometrics"): ("biometrics",),
    _normalize_regulation_text("LFPDPPP_AI"): ("automated",),
}


def normalize_regulation_label(regulation):
    """Resolve an emitted key or legacy label to ONE canonical regulation name."""
    raw = str(regulation or "").strip()

    if raw in REGULATION_KEY_MAP:
        return REGULATION_KEY_MAP[raw][0]

    normalized = _normalize_regulation_text(raw)
    if normalized in LEGACY_ALIAS_LOOKUP:
        return LEGACY_ALIAS_LOOKUP[normalized]

    if raw:
        logger.warning("Unmapped regulation label: %r", raw)
    return raw


def get_regulation_flags(regulations):
    """Collect severity flags ({'biometrics', 'health', ...}) across a list."""
    flags = set()
    for reg in regulations or []:
        raw = str(reg or "").strip()
        if raw in REGULATION_KEY_MAP:
            flags.update(REGULATION_KEY_MAP[raw][1])
            continue
        flags.update(LEGACY_LABEL_FLAGS.get(_normalize_regulation_text(raw), ()))
    return flags


def get_effective_regulations(business_data):
    """Return normalized regulations, including doc-defined follow-up triggers."""
    effective = []

    def add(regulation):
        canonical = normalize_regulation_label(regulation)
        if canonical and canonical not in effective:
            effective.append(canonical)

    for regulation in business_data.get("regulations", []) or []:
        add(regulation)

    # US follow-up questions emitted by Regulation_Checklist_Payloads.docx.
    if str(business_data.get("FTC_CCPA", "")).strip().lower() == "yes":
        add("CCPA")
    if str(business_data.get("FTC_NYDFS", "")).strip().lower() == "yes":
        add("NYDFS Part 500")
    if str(business_data.get("FTC_BIPA", "")).strip().lower() == "yes":
        add("BIPA")
    if str(business_data.get("FTC_HIPAA", "")).strip().lower() == "yes":
        add("HIPAA")

    # Canada follow-up question for province-specific coverage.
    if str(business_data.get("PIPEDA_provinces", "")).strip().lower() == "yes":
        add("Quebec Law 25")
        add("PIPA AB/BC")

    return effective

def fetch_live_usd_rates(force_refresh=False):
    """Fetch USD-base rates, cached for FX_CACHE_TTL_SECONDS.

    Returns {} on failure; callers fall back to STATIC_USD_RATES.
    """
    import requests

    now = time.time()
    if (not force_refresh and _fx_cache["rates"]
            and (now - _fx_cache["fetched_at"]) < FX_CACHE_TTL_SECONDS):
        return _fx_cache["rates"]

    try:
        response = requests.get(FX_API_URL, timeout=FX_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("FX fetch failed (%s); using cached/static rates.", exc)
        return _fx_cache["rates"] or {}

    if not isinstance(payload, dict):
        logger.warning("FX response was not a JSON object; using fallback.")
        return _fx_cache["rates"] or {}

    if payload.get("result") not in (None, "success"):
        logger.warning("FX API returned result=%r.", payload.get("result"))
        return _fx_cache["rates"] or {}

    raw_rates = payload.get("rates") or payload.get("conversion_rates") or {}
    if not isinstance(raw_rates, dict) or not raw_rates:
        logger.warning("FX response contained no rates; using fallback.")
        return _fx_cache["rates"] or {}

    rates = {}
    for key, value in raw_rates.items():
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            rates[str(key).upper()] = rate

    if rates.get("USD", 1.0) != 1.0:
        logger.warning("FX base is not USD (USD=%s); using fallback.", rates.get("USD"))
        return _fx_cache["rates"] or {}

    _fx_cache["rates"] = rates
    _fx_cache["fetched_at"] = now
    logger.info("FX rates refreshed (%d currencies).", len(rates))
    return rates


def get_usd_to_target_rate(target_currency):
    target_currency = str(target_currency or DEFAULT_CURRENCY).upper()
    if target_currency == DEFAULT_CURRENCY:
        return 1.0

    live_rates = fetch_live_usd_rates()
    if target_currency in live_rates:
        return live_rates[target_currency]

    return STATIC_USD_RATES.get(target_currency, 1.0)


def _is_money_key(key_path):
    money_markers = (
        "annualrevenue",
        "revenue",
        "coverageamount",
        "deductibleamount",
        "incomeprofitcover",
        "incomeprofitdeductible",
        "contingentlosscover",
        "contingentlossdeductible",
        "continuingexpensecover",
        "continuingexpensedeductible",
        "regulatoryfinescover",
        "regulatoryfinesdeductible",
        "regulatorypenaltiescover",
        "regulatorypenaltiesdeductible",
        "legalexpensescover",
        "legalexpensesdeductible",
        "existingdeductibles",
        "loss",
        "coverage",
        "deductible",
        "amount",
        "fine",
        "fines",
        "penalty",
        "penalties",
        "cost",
        "costs",
        "expense",
        "expenses",
        "reduction_in_total_loss",
        "new_estimated_total_loss",
        "total_loss",
        "coverage_gap",
        "existing_coverage",
    )
    lowered = " ".join(str(part).lower() for part in key_path)
    if any(marker in lowered for marker in money_markers):
        return True
    if any("coverage" in str(part).lower() for part in key_path):
        return True
    return False


def _should_skip_currency_conversion(key_path):
    skip_markers = (
        "p_value",
        "confidence_rating",
        "rating",
        "company_size",
        "num_employees",
        "employeecount",
        "incidentcount",
        "numincident",
        "records",
        "recordsexposed",
        "probability",
        "likelihood",
        "score",
        "scores",
        "percentage",
        "percent",
    )
    lowered = " ".join(str(part).lower() for part in key_path)
    return any(marker in lowered for marker in skip_markers)


def _convert_money_value(value, usd_to_target_rate, to_target=True):
    if value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(value * usd_to_target_rate, 2) if to_target else round(value / usd_to_target_rate, 2)
    if isinstance(value, str):
        if not value.strip():
            return value   # keep blanks blank — downstream `if x else 0` guards handle them
        try:
            numeric_value = parse_number(value)
        except Exception:
            return value
        return round(numeric_value * usd_to_target_rate, 2) if to_target else round(numeric_value / usd_to_target_rate, 2)


def coerce_amount(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return parse_number(str(value))
    except Exception:
        return 0.0


NO_THREAT_DETECTION_VALUES = {
    "we do not have this",
    "we do not use security software to detect threats",
    "none",
    "no",
}

NO_ADMIN_ALERT_APP_VALUES = {
    "no",
    "none",
    "we do not use any application",
    "we don't use any application",
}

NO_BACKUP_VALUES = {
    "we do not backup our files",
    "we do not back up our data",
    "we do not back up our files",
    "we do not backup our data",
    "we don't back up our data",
}

NO_VENDOR_ACCESS_VALUES = {
    "we don't give vendors access to our systems or data",
    "we do not give vendors access to our systems or data",
    "i do not share data or systems access with other businesses",
}


def _normalize_text(value):
    if value is None:
        return ""
    return (
        str(value)
        .replace("’", "'")
        .replace("‘", "'")
        .strip()
        .lower()
    )


def _as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _first_non_empty(*values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def is_no_threat_detection(value):
    values = _as_list(value)
    if not values:
        return True

    normalized_values = {_normalize_text(item) for item in values}
    return any(value in NO_THREAT_DETECTION_VALUES for value in normalized_values)


def is_no_admin_alert_app(value):
    normalized_value = _normalize_text(value)
    return not normalized_value or normalized_value in NO_ADMIN_ALERT_APP_VALUES


def is_weak_or_missing_backup(value):
    normalized_value = _normalize_text(value)
    return normalized_value in NO_BACKUP_VALUES or normalized_value in {
        "once a year",
        "every 6 months",
        "every 3 months",
    }


def is_uncontrolled_vendor_access(value):
    normalized_value = _normalize_text(value)
    return normalized_value == "no"


def has_incident_response_plan(security_data):
    if security_data.get("noIncidentResponse") is True:
        return False                      # explicit denial wins
    if security_data.get("IncidentResponse") is True:
        return True
    for key in ("incidentFileUrl", "incidentResponsePlan",
                "incidentFile", "incidentResponseDocument"):
        if security_data.get(key) not in (None, "", [], False):
            return True
    return False                          # no artifact -> no IRP


def normalize_business_currency_fields(business_data, usd_to_target_rate):
    business_data = copy.deepcopy(business_data)

    def transform_money_value(value, rate, to_target=False):
        if isinstance(value, dict):
            return {k: transform_money_value(v, rate, to_target=to_target) for k, v in value.items()}
        if isinstance(value, list):
            return [transform_money_value(item, rate, to_target=to_target) for item in value]
        return _convert_money_value(value, rate, to_target=to_target)

    def transform(node, key_path=()):
        if isinstance(node, dict):
            transformed = {}
            for key, value in node.items():
                child_path = key_path + (key,)
                if _should_skip_currency_conversion(child_path):
                    transformed[key] = value
                elif _is_money_key(child_path):
                    transformed[key] = transform_money_value(value, usd_to_target_rate, to_target=False)
                else:
                    transformed[key] = transform(value, child_path)
            return transformed
        if isinstance(node, list):
            return [transform(item, key_path) for item in node]
        return node

    return transform(business_data)


def convert_outputs_to_target_currency(value, usd_to_target_rate, key_path=()):
    if isinstance(value, dict):
        converted = {}
        for key, child_value in value.items():
            child_path = key_path + (key,)
            converted[key] = convert_outputs_to_target_currency(child_value, usd_to_target_rate, child_path)
        return converted
    if isinstance(value, list):
        return [convert_outputs_to_target_currency(item, usd_to_target_rate, key_path) for item in value]
    if _should_skip_currency_conversion(key_path):
        return value
    if _is_money_key(key_path):
        return _convert_money_value(value, usd_to_target_rate, to_target=True)
    return value

################################### Numerical Calculations for P-values ##################################

def derive_E(incident_count):
    try:
        count = int(incident_count)
        return max(1, min(10, count))
    except:
        return 10.0 # Default value

def derive_T(data_exposed):
    if isinstance(data_exposed, list):
        return len(data_exposed) * 50  # Example scaling
    else:
        return 500  # Default value

def derive_M(budget):
    try:
        budget_value = parse_number(budget)
        if budget_value < 100_000:
            return 4
        elif budget_value < 500_000:
            return 7
        else:
            return 9
    except:
        return 7
    

################################### Load Historical and JSON data ##################################

def load_questionnaire_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_historical_data():
    affected_counts_df = pd.read_csv('./data/affected_count_range.csv')

    response_costs_df = pd.read_csv('./data/response_costs.csv')
    # Multiply the "Loss Amount (Response Costs)" column by 1000
    response_costs_df['Loss Amount (Response Costs)'] = (
        response_costs_df['Loss Amount (Response Costs)'].astype(float) * 1000
    )

    litigated_cases_df = pd.read_csv('./data/litigated_cases.csv')
    # Multiply the "Loss Amount (Litigated Cases)" column by 1000
    litigated_cases_df['Loss Amount (Litigated Cases)'] = (
        litigated_cases_df['Loss Amount (Litigated Cases)'].astype(float) * 1000
    )

    fines_penalties_df = pd.read_csv('./data/fines_penalties.csv')
    # Multiply the "Loss Amount (Fines & Penalties)" column by 1000
    fines_penalties_df['Loss Amount (Fines & Penalties)'] = (
        fines_penalties_df['Loss Amount (Fines & Penalties)'].astype(float) * 1000
    )

    economic_loss_df = pd.read_csv('./data/economic_loss.csv')
    # Multiply the "Loss Amount (Economic Loss)" column by 1000
    economic_loss_df['Loss Amount (Economic Loss)'] = (
        economic_loss_df['Loss Amount (Economic Loss)'].astype(float) * 1000
    )

    company_size_df = pd.read_csv('./data/count_by_company_size_and_type.csv')

    return affected_counts_df, response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df, company_size_df


def prepare_affected_counts(affected_counts_df):
    # Remove 'Total' row if present
    affected_counts_df = affected_counts_df[affected_counts_df['Affected Count Range'] != 'Total']

    # Calculate probabilities
    total_cases = affected_counts_df['Case Count'].sum()
    affected_counts_df['Probability'] = affected_counts_df['Case Count'] / total_cases

    # Prepare ranges and probabilities
    ranges = affected_counts_df['Affected Count Range'].values
    probabilities = affected_counts_df['Probability'].values

    # Parse ranges into numerical values
    range_tuples = []
    for r in ranges:
        if r.strip() == '(1M, )':
            low = 1_000_000
            high = 10_000_000  # Assume an upper bound
        else:
            r = r.strip('()[]')
            low_str, high_str = r.split(',')
            low = convert_shorthand(low_str.strip()) 
            high = convert_shorthand(high_str.strip()) 
        range_tuples.append((low, high))

    return range_tuples, probabilities

def get_insurability_rating(p_value, automatic_failure=False):
    ranges = [11.10, 22.21, 33.32, 44.43, 55.54, 66.65, 100.0]
    grades = ["A+", "A", "A-", "B+", "B", "B-", "U"]

    p_value_pct = np.round(float(p_value) * 100, 2)
    rating = "U"

    for limit, grade_label in zip(ranges, grades):
        if p_value_pct <= limit:
            rating = grade_label
            break

    if automatic_failure:
        rating = "U"

    return rating, p_value_pct
################################### Numerical Calculations ##################################

def calculate_lambda(p_value):
    p_value = np.clip(p_value, 0, 1 - 1e-10)
    lambda_value = -np.log(1 - p_value)
    return lambda_value

def simulate_num_records_lost():
    # Set seed for reproducible record simulation
    np.random.seed(42)
    
    # Define the ranges and probabilities based on the data
    ranges = [
        (0, 10),
        (10, 100),
        (100, 1000),
        (1000, 10000),
        (10000, 100000),
        (100000, 1000000),
        (1000000, 1500000)
    ]
    case_counts = [58269, 14872, 20437, 14143, 6651, 2660, 1726]
    total_cases = sum(case_counts)
    probabilities = [count / total_cases for count in case_counts]

    # Choose a range based on probabilities
    selected_range = np.random.choice(len(ranges), p=probabilities)
    low, high = ranges[selected_range]
    
    # Simulate a number within the selected range
    num_records = np.random.uniform(low, high)
    return num_records

# def calculate_expected_loss_total():
#     """
#     Loads all_loss_amounts array from .npy file and returns company_size_df.
#     This replaces the CSV-dependent version with precomputed values.
#     """
#     # Load all_loss_amounts from .npy file
#     all_loss_amounts = np.load('./data/all_loss_amounts.npy')
    
#     # Hardcoded company size data
#     company_size_data = {
#         'Company Size': ['Small', 'Medium', 'Large', 'Very Large', 'Giant', 'Mega', 'Colossal', 'Unknown'],
#         'PRV': [15902, 19387, 22430, 12727, 7663, 1666, 1283, 15890],
#         'PUB': [485, 1436, 2490, 4717, 7206, 10306, 3667, 6664],
#         'GOV': [307, 675, 2428, 1498, 1819, 7842, 2067, 3020],
#         'NPR': [710.0, 620.0, 953.0, 435.0, 550.0, 33.0, np.nan, 552.0],
#         'Total': [17404, 22118, 28301, 19377, 17238, 19847, 7017, 26126]
#     }
#     company_size_df = pd.DataFrame(company_size_data)
    
#     return all_loss_amounts, company_size_df

def calculate_expected_loss_total():
    """
    Combine all response_costs, litigated_cases, fines_penalties, and economic_loss
    into one distribution (all_loss_amounts).
    """
    (affected_counts_df, response_costs_df, litigated_cases_df,
     fines_penalties_df, economic_loss_df, company_size_df
    ) = load_historical_data()

    # Filter out 'Total' rows
    rc_df = response_costs_df[response_costs_df['Case Type'] != 'Total'].copy()
    lc_df = litigated_cases_df[litigated_cases_df['Case Type'] != 'Total'].copy()
    fp_df = fines_penalties_df[fines_penalties_df['Case Type'] != 'Total'].copy()
    el_df = economic_loss_df[economic_loss_df['Case Type'] != 'Total'].copy()

    # Rename columns for consistency
    rc_df.rename(columns={
        'Loss Amount (Response Costs)': 'response_loss',
        'Case Count (Response Costs)': 'response_count'
    }, inplace=True)
    lc_df.rename(columns={
        'Loss Amount (Litigated Cases)': 'litigated_loss',
        'Case Count (Litigated Cases)': 'litigated_count'
    }, inplace=True)
    fp_df.rename(columns={
        'Loss Amount (Fines & Penalties)': 'fines_loss',
        'Case Count (Fines & Penalties)': 'fines_count'
    }, inplace=True)
    el_df.rename(columns={
        'Loss Amount (Economic Loss)': 'economic_loss',
        'Case Count (Economic Loss)': 'economic_count'
    }, inplace=True)

    # Merge all on 'Case Type'
    merged_1 = pd.merge(
        rc_df[['Case Type','response_loss','response_count']],
        lc_df[['Case Type','litigated_loss','litigated_count']],
        on='Case Type', how='outer'
    )
    merged_2 = pd.merge(
        merged_1,
        fp_df[['Case Type','fines_loss','fines_count']],
        on='Case Type', how='outer'
    )
    merged_all = pd.merge(
        merged_2,
        el_df[['Case Type','economic_loss','economic_count']],
        on='Case Type', how='outer'
    )

    # Fill NaNs with 0
    num_cols = ['response_loss','litigated_loss','fines_loss','economic_loss']
    count_cols = ['response_count','litigated_count','fines_count','economic_count']
    for c in num_cols:
        merged_all[c] = merged_all[c].fillna(0).astype(float)
    for c in count_cols:
        merged_all[c] = merged_all[c].fillna(0).astype(int)

    # total_loss = sum of all cost columns, total_cases = sum of all case count columns
    merged_all['total_loss'] = (
        merged_all['response_loss']
        + merged_all['litigated_loss']
        + merged_all['fines_loss']
        + merged_all['economic_loss']
    )
    merged_all['total_cases'] = (
        merged_all['response_count']
        + merged_all['litigated_count']
        + merged_all['fines_count']
        + merged_all['economic_count']
    )

    # Build a big array of individual losses
    all_loss_amounts = []
    for _, row in merged_all.iterrows():
        if row['total_cases'] > 0:
            avg_loss = row['total_loss'] / row['total_cases']
            # replicate 'avg_loss' row['total_cases'] times
            all_loss_amounts.extend([avg_loss]*row['total_cases'])

    all_loss_amounts = np.array(all_loss_amounts)
    all_loss_amounts = all_loss_amounts[all_loss_amounts > 0]
    return all_loss_amounts, company_size_df

def create_size_weighted_distribution(all_loss_amounts, company_size_df, user_size):
    """
    Scales and weights the loss distribution based on the company's size.

    Args:
        all_loss_amounts (array-like): Global distribution of losses (e.g., response costs, fines, etc.).
        company_size_df (DataFrame): Historical data containing case counts by company size.
        user_size (str): The size of the company (e.g., 'micro', 'small', 'medium', 'large').

    Returns:
        np.ndarray: A scaled and resampled loss distribution adjusted for the company's size.
    """
    # Normalize size proportions
    size_totals = company_size_df.groupby('Company Size')['Total'].sum()
    grand_total = size_totals.sum()
    size_proportion = size_totals / grand_total

    # Extract the user size proportion
    user_size_proportion = size_proportion.get(user_size.capitalize(), 1.0)

    # Define scaling factors based on company size
    scaling_factors = {
        'Micro': 0.2,   # Scale to 20% of global losses
        'Small': 0.4,   # Scale to 40% of global losses
        'Medium': 0.7,  # Scale to 70% of global losses
        'Large': 1.0    # Retain full loss values
    }
    scaling_factor = scaling_factors.get(user_size.capitalize(), 1.0)

    # Scale the global loss amounts
    scaled_loss_amounts = np.array(all_loss_amounts) * scaling_factor

    # Determine the number of samples to draw based on size proportion
    n_samples = int(len(all_loss_amounts) * user_size_proportion)

    # Avoid under-sampling by ensuring a minimum of 1 sample
    n_samples = max(n_samples, 1)

    # Sample from the scaled loss amounts
    size_weighted_distribution = np.random.choice(scaled_loss_amounts, n_samples, replace=True)

    return size_weighted_distribution


def compute_size_fractions(csv_path='./data/count_by_company_size_and_type.csv'):
    """
    Reads the 'company_size.csv' file, computes a fraction for each size
    = (size_total / grand_total).
    Returns a dictionary: { 'Small': fraction_small, 'Medium': fraction_medium, ... }.
    """

    df = pd.read_csv(csv_path)
    # Convert 'Total' to float (in case it's read as string or has missing data)
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
    
    # Drop rows with NaN in Total
    df = df.dropna(subset=['Total'])
    
    # Compute grand total
    grand_total = df['Total'].sum()
    
    # Build a dictionary of fractions
    fractions = {}
    for _, row in df.iterrows():
        size_name = row['Company Size']
        size_total = row['Total']
        fraction = size_total / grand_total if grand_total > 0 else 0
        fractions[size_name] = fraction

    return fractions

def scale_loss_distribution(all_loss_amounts, user_size):
    """
    Scale all loss events by a given fraction.
    E.g., if fraction=0.20, all events become 20% of original.
    """
    # Hardcoded size fractions based on company_size_df data
    size_fractions = {
        'Small': 0.14135081629513488,
        'Medium': 0.1414995680345013,
        'Large': 0.18093159609120522,
        'Very Large': 0.12392218523947553,
        'Giant': 0.11021205956678701,
        'Mega': 0.12694233576642331,
        'Colossal': 0.04489525477707602,
        'Unknown': 0.16706183805029767
    }
    
    if user_size == 'small':
        user_size = 'Small'
        fraction = size_fractions[user_size]
    elif user_size == 'medium':
        user_size = 'Medium'
        fraction = size_fractions[user_size]
    elif user_size == 'micro':
        fraction = 0.08
    else:
        fraction = 1.0  # Default for other sizes
    
    return all_loss_amounts * fraction

# def bayesian_estimate_mu_sigma(all_loss_amounts, annual_revenue, base_revenue=1_000_000):
#     """Estimate mu and sigma using Bayesian inference."""
#     # Filter zeros and take logarithm
#     all_loss_amounts = all_loss_amounts[all_loss_amounts > 0]
#     log_loss_amounts = np.log(all_loss_amounts)
#     shift_mu = np.log(annual_revenue / base_revenue)

#     # Explicit coordinate declaration
#     coords = {"observation": np.arange(len(log_loss_amounts))}
    
#     with pm.Model(coords=coords) as model:
#         # Priors with proper dimension alignment
#         empirical_mu = np.mean(log_loss_amounts)
#         empirical_sigma = np.std(log_loss_amounts)
        
#         mu = pm.StudentT('mu', nu=4, mu=empirical_mu + shift_mu,
#                         sigma=pm.Exponential('mu_scale', 1/2.0))
        
#         sigma = pm.HalfStudentT('sigma', nu=4, sigma=empirical_sigma)
        
#         # Observed with explicit dims
#         pm.StudentT('obs', nu=4, mu=mu, sigma=sigma,
#                    observed=log_loss_amounts, dims=("observation"))
        
#         # Sampling with chain dimension handling
#         trace = pm.sample(10000, tune=2000, target_accept=0.9)
        
#         # Posterior predictive with dimension preservation
#         post_pred = pm.sample_posterior_predictive(
#             trace, 
#             var_names=["obs"],
#             extend_inferencedata=True
#         )
    
#     return trace

# def calculate_bayesian_params(num_samples, user_size, annual_revenue):
#     """Calculate posterior samples for mu and sigma."""
#     all_loss_amounts, _ = calculate_expected_loss_total()
#     all_loss_amounts = scale_loss_distribution(all_loss_amounts, user_size)
    
#     trace = bayesian_estimate_mu_sigma(all_loss_amounts, annual_revenue)
    
#     # Proper parameter extraction using xarray
#     mu_samples = trace.posterior["mu"].stack(sample=("chain", "draw"))[:num_samples].values
#     sigma_samples = trace.posterior["sigma"].stack(sample=("chain", "draw"))[:num_samples].values
    
#     return mu_samples, sigma_samples


def estimate_mu_sigma_mle(all_loss_amounts, annual_revenue, base_revenue=1_000_000):
    # Set seed for reproducible fitting
    np.random.seed(42)
    
    # Fit the lognormal distribution using MLE
    shape, loc, scale = lognorm.fit(all_loss_amounts, floc=0)
    sigma_mle = shape  # 'shape' parameter is sigma in lognorm

    shift_mu = np.log(annual_revenue / base_revenue)
    mu_mle = np.log(scale)
    mu_mle += shift_mu
    return mu_mle, sigma_mle

def calculate_lognormal_params(num_samples, user_size, annual_revenue):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Collect all loss amounts
    all_loss_amounts, company_size_df = calculate_expected_loss_total()
    all_loss_amounts = scale_loss_distribution(all_loss_amounts, user_size)
    mu_mle, sigma_mle = estimate_mu_sigma_mle(all_loss_amounts, annual_revenue)
    
    # Use explicit arrays instead of np.full to avoid any potential variance
    mu_samples = np.array([mu_mle] * num_samples)
    sigma_samples = np.array([sigma_mle] * num_samples)
    
    # Return the posterior samples
    return mu_samples, sigma_samples


def calculate_subcategory_proportions(response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df):
    # Sum loss amounts for each subcategory
    response_cost_total = response_costs_df.loc[response_costs_df['Case Type'] == 'Total', 'Loss Amount (Response Costs)'].values[0]
    litigated_cases_total = litigated_cases_df.loc[litigated_cases_df['Case Type'] == 'Total', 'Loss Amount (Litigated Cases)'].values[0]
    fines_penalties_total = fines_penalties_df.loc[fines_penalties_df['Case Type'] == 'Total', 'Loss Amount (Fines & Penalties)'].values[0]
    economic_loss_total = economic_loss_df.loc[economic_loss_df['Case Type'] == 'Total', 'Loss Amount (Economic Loss)'].values[0]

    # Convert to floats
    response_cost_total = float(response_cost_total)
    litigated_cases_total = float(litigated_cases_total)
    fines_penalties_total = float(fines_penalties_total)
    economic_loss_total = float(economic_loss_total)

    # Calculate total loss
    total_loss = response_cost_total + litigated_cases_total +  economic_loss_total + fines_penalties_total

    # Calculate proportions
    subcategory_totals = {
        'Incident Response Costs': response_cost_total,
        'Litigation Costs': litigated_cases_total,
        'Privacy Liability Coverage': fines_penalties_total,
        'Business Interruption Loss': economic_loss_total
    }


    subcategory_proportions = {k: v / total_loss for k, v in subcategory_totals.items()}

    adjusted_totals = {
        k: v for k, v in subcategory_totals.items() if k != 'Privacy Liability Coverage'
    }
    adjusted_total_loss = sum(adjusted_totals.values())
    adjusted_proportions = {k: v / adjusted_total_loss for k, v in adjusted_totals.items()}

    adjusted_proportions = {'Incident Response Costs': 0.11451240272969528, 'Litigation Costs': 0.2725245111719511, 'Business Interruption Loss': 0.6129630860983536}

    return adjusted_proportions

def calculate_business_interruption_subcategory_losses(coverage_gap_df, result_df):
    lossNetIncome = 0.6
    continueOperate = 0.3
    CB1 = 0.1

    BI_loss = result_df['expected_subcategory_losses']['Business Interruption Loss']

    coverage_gap_df["Business Interruption Loss SubCats"] = {}
    coverage_gap_df["Business Interruption Loss SubCats"]["Loss Net Income"] = lossNetIncome * BI_loss
    coverage_gap_df["Business Interruption Loss SubCats"]["Continuing Expenses"] = continueOperate * BI_loss
    coverage_gap_df["Business Interruption Loss SubCats"]["Contingent Business Interruption"] = CB1 * BI_loss

    return coverage_gap_df


def get_industry_map(total_loss, industry_sector_code):

    # Financial sector is the baseline (scalar = 1.00)
    industry_scalars = {
    "11":                                            0.44,   # Agriculture, forestry, fishing and hunting   
    "21":                                            0.81,   # Mining, quarrying, and oil and gas extraction 
    "22":                                            0.81,   # Utilities
    "23":                                            0.80,   # Construction
    "31":                                            0.82,   # Manufacturing
    "32":                                            0.82,   # Manufacturing
    "33":                                            0.82,   # Manufacturing
    "42":                                            0.66,   # Wholesale trade
    "44":                                            0.65,   # Retail trade
    "45":                                            0.64,   # Retail trade
    "48":                                            0.71,   # Transportation and warehousing
    "49":                                            0.71,   # Transportation and warehousing
    "51":                                            0.79,   # Information
    "52":                                            1.00,   # baseline
    "53":                                            0.63,   # Real estate and rental and leasing
    "54":                                            0.76,   # Professional, scientific, and technical services
    "55":                                            0.50,   # Management of companies and enterprises
    "56":                                            0.62,   # Administrative and support and waste management and remediation services
    "61":                                            0.62,   # Educational services
    "62":                                            1.85,   # Health care and social assistance
    "71":                                            0.61,   # Arts, entertainment, and recreation
    "72":                                            0.57,   # Accommodation and food services
    "81":                                            0.50,   # Other services (except public administration)
    "92":                                            0.44,   # Public administration
}
    scaled_loss = total_loss * industry_scalars.get(industry_sector_code, 1.0)
    return scaled_loss

def get_industry_benchmark(industry_sector_code):
    # Switch these to NAICS codes for US
    get_industry_benchmarks = {
    "11":                                            "A+",   # Agriculture, forestry, fishing and hunting   
    "21":                                            "A",   # Mining, quarrying, and oil and gas extraction 
    "22":                                            "B+",   # Utilities
    "23":                                            "A+",   # Construction
    "31":                                            "A",   # Manufacturing
    "32":                                            "A",   # Manufacturing
    "33":                                            "A",   # Manufacturing
    "42":                                            "A",   # Wholesale trade
    "44":                                            "B-",   # Retail trade
    "45":                                            "B-",   # Retail trade
    "48":                                            "A",   # Transportation and warehousing
    "49":                                            "A",   # Transportation and warehousing
    "51":                                            "B",   # Information
    "52":                                            "B-",   # baseline
    "53":                                            "B+",   # Real estate and rental and leasing
    "54":                                            "B-",   # Professional, scientific, and technical services
    "55":                                            "B+",   # Management of companies and enterprises
    "56":                                            "A",   # Administrative and support and waste management and remediation services
    "61":                                            "B",   # Educational services
    "62":                                            "B-",   # Health care and social assistance
    "71":                                            "B+",   # Arts, entertainment, and recreation
    "72":                                            "B+",   # Accommodation and food services
    "81":                                            "B+",   # Other services (except public administration)
    "92":                                            "B",   # Public administration
}

    # Convert input to lowercase for case-insensitive matching
    benchmark_value = get_industry_benchmarks.get(industry_sector_code, "B-")
    return benchmark_value

def convert_code_to_naics(code, label):
    label = label.strip().lower()
    if label == 'naics':
        return code
    elif label == 'nic':

        convert = {
            "011": "11",
            "012": "11",
            "013": "11",
            "014": "11",
            "015": "11",
            "016": "11",
            "017": "11",
            "021": "11",
            "022": "11",
            "023": "11",
            "024": "11",
            "031": "11",
            "032": "11",
            "051": "21",
            "052": "21",
            "061": "21",
            "062": "21",
            "071": "21",
            "072": "21",
            "081": "21",
            "089": "21",
            "091": "21",
            "099": "21",
            "101": "31",
            "102": "31",
            "103": "31",
            "104": "31",
            "105": "31",
            "106": "31",
            "107": "31",
            "108": "31",
            "110": "31",
            "120": "31",
            "131": "31",
            "139": "31",
            "141": "31",
            "142": "31",
            "143": "31",
            "151": "31",
            "152": "31",
            "161": "32",
            "162": "32",
            "170": "32",
            "181": "32",
            "182": "32",
            "191": "32",
            "192": "32",
            "201": "32",
            "202": "32",
            "203": "32",
            "210": "32",
            "221": "32",
            "222": "32",
            "231": "32",
            "239": "32",
            "241": "33",
            "242": "33",
            "243": "33",
            "251": "33",
            "252": "33",
            "259": "33",
            "261": "33",
            "262": "33",
            "263": "33",
            "264": "33",
            "265": "33",
            "266": "33",
            "267": "33",
            "268": "33",
            "271": "33",
            "272": "33",
            "273": "33",
            "274": "33",
            "275": "33",
            "279": "33",
            "281": "33",
            "282": "33",
            "291": "33",
            "292": "33",
            "293": "33",
            "301": "33",
            "302": "33",
            "303": "33",
            "304": "33",
            "309": "33",
            "310": "33",
            "321": "33",
            "322": "33",
            "323": "33",
            "324": "33",
            "325": "33",
            "329": "33",
            "331": "33",
            "332": "33",
            "351": "22",
            "352": "22",
            "353": "22",
            "360": "21",
            "370": "21",
            "381": "21",
            "382": "21",
            "383": "21",
            "390": "21",
            "410": "23",
            "421": "23",
            "422": "23",
            "429": "23",
            "431": "23",
            "432": "23",
            "433": "23",
            "439": "23",
            "451": "44",
            "452": "44",
            "453": "44",
            "454": "44",
            "461": "42",
            "462": "42",
            "463": "42",
            "464": "42",
            "465": "42",
            "466": "42",
            "469": "42",
            "471": "45",
            "472": "45",
            "473": "45",
            "474": "45",
            "475": "45",
            "476": "45",
            "477": "45",
            "478": "45",
            "479": "45",
            "491": "48",
            "492": "48",
            "493": "48",
            "501": "48",
            "502": "48",
            "511": "48",
            "512": "48",
            "521": "49",
            "522": "49",
            "531": "49",
            "532": "49",
            "551": "72",
            "552": "72",
            "559": "72",
            "561": "72",
            "562": "72",
            "563": "72",
            "581": "51",
            "582": "51",
            "591": "51",
            "592": "51",
            "601": "51",
            "602": "51",
            "611": "51",
            "612": "51",
            "613": "51",
            "619": "51",
            "620": "51",
            "631": "51",
            "639": "51",
            "641": "52",
            "642": "52",
            "643": "52",
            "649": "52",
            "651": "52",
            "652": "52",
            "653": "52",
            "661": "52",
            "662": "52",
            "663": "52",
            "681": "53",
            "682": "53",
            "691": "54",
            "692": "54",
            "701": "54",
            "702": "54",
            "711": "54",
            "712": "54",
            "721": "54",
            "722": "54",
            "731": "54",
            "732": "54",
            "741": "54",
            "742": "54",
            "749": "54",
            "750": "54",
            "771": "56",
            "772": "56",
            "773": "56",
            "774": "56",
            "781": "56",
            "782": "56",
            "783": "56",
            "791": "56",
            "799": "56",
            "801": "56",
            "802": "56",
            "803": "56",
            "811": "56",
            "812": "56",
            "813": "56",
            "821": "56",
            "822": "56",
            "823": "56",
            "829": "56",
            "841": "92",
            "842": "92",
            "843": "92",
            "851": "61",
            "852": "61",
            "853": "61",
            "854": "61",
            "855": "61",
            "861": "62",
            "862": "62",
            "869": "62",
            "871": "62",
            "872": "62",
            "873": "62",
            "879": "62",
            "881": "62",
            "889": "62",
            "900": "71",
            "910": "71",
            "920": "71",
            "931": "71",
            "932": "71",
            "941": "81",
            "942": "81",
            "949": "81",
            "951": "81",
            "952": "81",
            "960": "81",
            "970": "81",
            "981": "81",
            "982": "81",
            "990": "92",
        }
    elif label == 'scian':
        #only those that are different from naics
        convert = {
            "43": "42",
            "46": "45",
            "93": "92",

        }
    elif label == "naics_canada":
        convert = {
            "41": "42",
            "91": "92"
        }

    else:
        return code  # Unknown label, return as is

    
    return convert.get(code, code)  # Return converted code or original if not found
        

def get_industry_name(code, country = 'US'):
    code = str(code).strip()
    naics_to_name = {
        "11": "agriculture, forestry, fishing and hunting",
        "21": "mining, quarrying, and oil and gas extraction",
        "31": "Food, textile, apparel, leather",
        "32": "Wood, paper, printing, petroleum, chemical, plastics, nonmetallic mineral, rubber, nonmetallic minerals",
        "33": "Primary metal, fabricated metal, machinery, computer/electronic, electrical equipment, transportation equipment, furniture, and miscellaneous manufacturing",
        "22": "utilities",
        "23": "construction",
        "42": "wholesale trade",
        "44": "Motor vehicle/parts dealers, furniture/home furnishings, electronics/appliances, building materials/garden supplies, food/beverage stores, health/personal care stores",
        "45": "Gasoline stations, clothing/accessories, sporting goods/hobby/book/music, general merchandise, miscellaneous store retailers, nonstore retailers",
        "48": "Air, rail, water, truck, transit/ground passenger, pipeline, scenic/sightseeing, and support activities for transportation",
        "49": "Postal service, couriers/messengers, warehousing/storage",
        "51": "information",
        "52": "finance and insurance",
        "53": "real estate and rental and leasing",
        "54": "professional, scientific, and technical services",
        "55": "management of companies and enterprises",
        "56": "administrative and support and waste management and remediation services",
        "61": "educational services",
        "62": "health care and social assistance",
        "71": "arts, entertainment, and recreation",
        "72": "accommodation and food services",
        "81": "other services (except public administration)",
        "92": "public administration",
    }

    nic_to_name = {
        "011": "Growing of non-perennial crops",
        "012": "Growing of perennial crops",
        "013": "Plant propagation",
        "014": "Animal production",
        "015": "Mixed farming",
        "016": "Support activities to agriculture and post-harvest crop activities",
        "017": "Hunting, trapping and related service activities",
        "021": "Silviculture and other forestry activities",
        "022": "Logging",
        "023": "Gathering of non-wood forest products",
        "024": "Support services to forestry",
        "031": "Fishing",
        "032": "Aquaculture",
        "051": "Mining of hard coal",
        "052": "Mining of lignite",
        "061": "Extraction of crude petroleum",
        "062": "Extraction of natural gas",
        "071": "Mining of iron ores",
        "072": "Mining of non-ferrous metal ores",
        "081": "Quarrying of stone, sand and clay",
        "089": "Mining and quarrying n.e.c.",
        "091": "Support activities for petroleum and natural gas mining",
        "099": "Support activities for other mining and quarrying",
        "101": "Processing and preserving of meat",
        "102": "Processing and preserving of fish, crustaceans and molluscs",
        "103": "Processing and preserving of fruit and vegetables",
        "104": "Manufacture of vegetable and animal oils and fats",
        "105": "Manufacture of dairy products",
        "106": "Manufacture of grain mill products, starches and starch products",
        "107": "Manufacture of other food products",
        "108": "Manufacture of prepared animal feeds",
        "110": "Manufacture of beverages",
        "120": "Manufacture of tobacco products",
        "131": "Spinning, weaving and finishing of textiles",
        "139": "Manufacture of other textiles",
        "141": "Manufacture of wearing apparel, except fur apparel",
        "142": "Manufacture of articles of fur",
        "143": "Manufacture of knitted and crocheted apparel",
        "151": "Tanning and dressing of leather; manufacture of luggage, handbags, saddlery and harness; dressing and dyeing of fur",
        "152": "Manufacture of footwear",
        "161": "Sawmilling and planing of wood",
        "162": "Manufacture of products of wood, cork, straw and plaiting materials",
        "170": "Manufacture of paper and paper products",
        "181": "Printing and service activities related to printing",
        "182": "Reproduction of recorded media",
        "191": "Manufacture of coke oven products",
        "192": "Manufacture of refined petroleum products",
        "201": "Manufacture of basic chemicals, fertilizer and nitrogen compounds, plastics and synthetic rubber in primary forms",
        "202": "Manufacture of other chemical products",
        "203": "Manufacture of man-made fibres",
        "210": "Manufacture of pharmaceuticals, medicinal chemical and botanical products",
        "221": "Manufacture of rubber products",
        "222": "Manufacture of plastics products",
        "231": "Manufacture of glass and glass products",
        "239": "Manufacture of non-metallic mineral products n.e.c.",
        "241": "Manufacture of basic iron and steel",
        "242": "Manufacture of basic precious and other non-ferrous metals",
        "243": "Casting of metals",
        "251": "Manufacture of structural metal products, tanks, reservoirs and steam generators",
        "252": "Manufacture of weapons and ammunition",
        "259": "Manufacture of other fabricated metal products; metalworking service activities",
        "261": "Manufacture of electronic components",
        "262": "Manufacture of computers and peripheral equipment",
        "263": "Manufacture of communication equipment",
        "264": "Manufacture of consumer electronics",
        "265": "Manufacture of measuring, testing, navigating and control equipment; watches and clocks",
        "266": "Manufacture of irradiation, electromedical and electrotherapeutic equipment",
        "267": "Manufacture of optical instruments and equipment",
        "268": "Manufacture of magnetic and optical media",
        "271": "Manufacture of electric motors, generators, transformers and electricity distribution and control apparatus",
        "272": "Manufacture of batteries and accumulators",
        "273": "Manufacture of wiring and wiring devices",
        "274": "Manufacture of electric lighting equipment",
        "275": "Manufacture of domestic appliances",
        "279": "Manufacture of other electrical equipment",
        "281": "Manufacture of general purpose machinery",
        "282": "Manufacture of special-purpose machinery",
        "291": "Manufacture of motor vehicles",
        "292": "Manufacture of bodies (coachwork) for motor vehicles; manufacture of trailers and semi-trailers",
        "293": "Manufacture of parts and accessories for motor vehicles",
        "301": "Building of ships and boats",
        "302": "Manufacture of railway locomotives and rolling stock",
        "303": "Manufacture of air and spacecraft and related machinery",
        "304": "Manufacture of military fighting vehicles",
        "309": "Manufacture of transport equipment n.e.c.",
        "310": "Manufacture of furniture",
        "321": "Manufacture of jewellery, bijouterie and related articles",
        "322": "Manufacture of musical instruments",
        "323": "Manufacture of sports goods",
        "324": "Manufacture of games and toys",
        "325": "Manufacture of medical and dental instruments and supplies",
        "329": "Other manufacturing n.e.c.",
        "331": "Repair of fabricated metal products, machinery and equipment",
        "332": "Installation of industrial machinery and equipment",
        "351": "Electric power generation, transmission and distribution",
        "352": "Manufacture of gas; distribution of gaseous fuels through mains",
        "353": "Steam and air conditioning supply",
        "360": "Water collection, treatment and supply",
        "370": "Sewerage",
        "381": "Waste collection",
        "382": "Waste treatment and disposal",
        "383": "Materials recovery",
        "390": "Remediation activities and other waste management services",
        "410": "Construction of buildings",
        "421": "Construction of roads and railways",
        "422": "Construction of utility projects",
        "429": "Construction of other civil engineering projects",
        "431": "Demolition and site preparation",
        "432": "Electrical, plumbing and other construction installation activities",
        "433": "Building completion and finishing",
        "439": "Other specialized construction activities",
        "451": "Sale of motor vehicles",
        "452": "Maintenance and repair of motor vehicles",
        "453": "Sale of motor vehicle parts and accessories",
        "454": "Sale, maintenance and repair of motorcycles and related parts and accessories",
        "461": "Wholesale on a fee or contract basis",
        "462": "Wholesale of agricultural raw materials and live animals",
        "463": "Wholesale of food, beverages and tobacco",
        "464": "Wholesale of household goods",
        "465": "Wholesale of machinery, equipment and supplies",
        "466": "Other specialized wholesale",
        "469": "Non-specialized wholesale trade",
        "471": "Retail sale in non-specialized stores",
        "472": "Retail sale of food, beverages and tobacco in specialized stores",
        "473": "Retail sale of automotive fuel in specialized stores",
        "474": "Retail sale of information and communications equipment in specialized stores",
        "475": "Retail sale of other household equipment in specialized stores",
        "476": "Retail sale of cultural and recreation goods in specialized stores",
        "477": "Retail sale of other goods in specialized stores",
        "478": "Retail sale via stalls and markets",
        "479": "Retail trade not in stores, stalls or markets",
        "491": "Transport via railways",
        "492": "Other land transport",
        "493": "Transport via pipeline",
        "501": "Sea and coastal water transport",
        "502": "Inland water transport",
        "511": "Passenger air transport",
        "512": "Freight air transport",
        "521": "Warehousing and storage",
        "522": "Support activities for transportation",
        "531": "Postal activities",
        "532": "Courier activities",
        "551": "Short term accommodation activities",
        "552": "Camping grounds, recreational vehicle parks and trailer parks",
        "559": "Other accommodation",
        "561": "Restaurants and mobile food service activities",
        "562": "Event catering and other food service activities",
        "563": "Beverage serving activities",
        "581": "Publishing of books, periodicals and other publishing activities",
        "582": "Software publishing",
        "591": "Motion picture, video and television programme activities",
        "592": "Sound recording and music publishing activities",
        "601": "Radio broadcasting",
        "602": "Television programming and broadcasting activities",
        "611": "Wired telecommunications activities",
        "612": "Wireless telecommunications activities",
        "613": "Satellite telecommunications activities",
        "619": "Other telecommunications activities",
        "620": "Computer programming, consultancy and related activities",
        "631": "Data processing, hosting and related activities; web portals",
        "639": "Other information service activities",
        "641": "Monetary intermediation",
        "642": "Activities of holding companies",
        "643": "Trusts, funds and other financial vehicles",
        "649": "Other financial service activities, except insurance and pension funding activities",
        "651": "Insurance",
        "652": "Reinsurance",
        "653": "Pension funding",
        "661": "Activities auxiliary to financial service activities, except insurance and pension funding",
        "662": "Activities auxiliary to insurance and pension funding",
        "663": "Fund management activities",
        "681": "Real estate activities with own or leased property",
        "682": "Real estate activities on a fee or contract basis",
        "691": "Legal activities",
        "692": "Accounting, bookkeeping and auditing activities; tax consultancy",
        "701": "Activities of head offices",
        "702": "Management consultancy activities",
        "711": "Architectural and engineering activities and related technical consultancy",
        "712": "Technical testing and analysis",
        "721": "Research and experimental development on natural sciences and engineering",
        "722": "Research and experimental development on social sciences and humanities",
        "731": "Advertising",
        "732": "Market research and public opinion polling",
        "741": "Specialized design activities",
        "742": "Photographic activities",
        "749": "Other professional, scientific and technical activities n.e.c.",
        "750": "Veterinary activities",
        "771": "Renting and leasing of motor vehicles",
        "772": "Renting and leasing of personal and household goods",
        "773": "Renting and leasing of other machinery, equipment and tangible goods n.e.c.",
        "774": "Leasing of nonfinancial intangible assets",
        "781": "Activities of employment placement agencies",
        "782": "Temporary employment agency activities",
        "783": "Human resources provision and management of human resources functions",
        "791": "Travel agency and tour operator activities",
        "799": "Other reservation service activities",
        "801": "Private security activities",
        "802": "Security systems service activities",
        "803": "Investigation activities",
        "811": "Combined facilities support activities",
        "812": "Cleaning activities",
        "813": "Landscape care and maintenance service activities",
        "821": "Office administrative and support activities",
        "822": "Activities of call centres",
        "823": "Organization of conventions and trade shows",
        "829": "Business support service activities n.e.c.",
        "841": "Administration of the State and the economic and social policy of the community",
        "842": "Provision of services to the community as a whole",
        "843": "Compulsory social security activities",
        "851": "Primary education",
        "852": "Secondary education",
        "853": "Higher education",
        "854": "Other education",
        "855": "Educational support services",
        "861": "Hospital activities",
        "862": "Medical and dental practice activities",
        "869": "Other human health activities",
        "871": "Nursing care facilities",
        "872": "Residential care activities for mental retardation, mental health and substance abuse",
        "873": "Residential care activities for the elderly and disabled",
        "879": "Other residential care activities n.e.c.",
        "881": "Social work activities without accommodation for the elderly and disabled",
        "889": "Other social work activities without accommodation n.e.c.",
        "900": "Creative, arts and entertainment activities",
        "910": "Libraries, archives, museums and other cultural activities",
        "920": "Gambling and betting activities",
        "931": "Sports activities",
        "932": "Other amusement and recreation activities",
        "941": "Activities of business, employers and professional membership organizations",
        "942": "Activities of trade unions",
        "949": "Activities of other membership organizations",
        "951": "Repair of computers and communication equipment",
        "952": "Repair of personal and household goods",
        "960": "Other personal service activities",
        "970": "Activities of households as employers of domestic personnel",
        "981": "Undifferentiated goods-producing activities of private households for own use",
        "982": "Undifferentiated service-producing activities of private households for own use",
        "990": "Activities of extraterritorial organizations and bodies",
    }

    scian_to_name = {
        "11": "Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza",
        "21": "Minería",
        "22": "Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final",
        "23": "Construcción",
        "31-33": " Industrias manufactureras",
        "31": "Food, textile, apparel, leather and allied products",
        "32": "Wood products, paper, printing and related support activities",
        "33": "Primary metal, fabricated metal, machinery, computer/electronic, electrical equipment, transportation equipment, furniture, and miscellaneous manufacturing",
        "43": "Wholesale Trade",
        "46": "Retail Trade",
        "48-49": "Transportation and Warehousing",
        "48": "Air, rail, water, truck, transit/ground passenger, pipeline, scenic/sightseeing, and support activities for transportation", #Air, rail, water, truck, transit and ground passenger, pipeline, scenic and sightseeing transportation, support activities for transportation
        "49": "Postal service, couriers/messengers, warehousing/storage", #Postal service, couriers and messengers, warehousing and storage
        "51": "Información en medios masivos",
        "52": "Servicios financieros y de seguros",
        "53": "Servicios inmobiliarios y de alquiler de bienes muebles e intangibles",
        "54": "Servicios profesionales, científicos y técnicos",
        "55": "Dirección y administración de grupos empresariales o corporativos",
        "56": "Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación",
        "61": "Servicios educativos",
        "62": "Servicios de salud y de asistencia social",
        "71": "Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos",
        "72": "Servicios de alojamiento temporal y de preparación de alimentos y bebidas",
        "81": "Otros servicios excepto actividades gubernamentales",
        "93": "Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales",
    }

    naics_canada_to_name = {
        "11": "Agriculture, Forestry, Fishing and Hunting",
        "21": "Mining, Quarrying, and Oil and Gas Extraction",
        "22": "Utilities",
        "23": "Construction",
        "31-33": "Manufacturing",
        "31": "Food, textile, apparel, leather and allied products",
        "32": "Wood products, paper, printing and related support activities",
        "33": "Primary metal, fabricated metal, machinery, computer/electronic, electrical equipment, transportation equipment, furniture, and miscellaneous manufacturing",
        "41": "Wholesale Trade",
        "44": "Motor vehicle/parts dealers, furniture/home furnishings, electronics/appliances, building materials/garden supplies, food/beverage stores, health/personal care stores",
        "45": "Gasoline stations, clothing/accessories, sporting goods/hobby/book/music, general merchandise, miscellaneous store retailers, nonstore retailers",
        "48-49": "Transportation and Warehousing",
        "48": "Air, rail, water, truck, transit and ground passenger, pipeline, scenic and sightseeing transportation, support activities for transportation", #Air, rail, water, truck, transit and ground passenger, pipeline, scenic and sightseeing transportation, support activities for transportation
        "49": "Postal service, couriers and messengers, warehousing and storage", #Postal service, couriers and messengers, warehousing and storage
        "51": "Information and cultural industries",
        "52": "Finance and Insurance",
        "53": "Real Estate and Rental and Leasing",
        "54": "Professional, Scientific, and Technical Services",
        "55": "Management of Companies and Enterprises",
        "56": "Administrative and Support and Waste Management and Remediation Services",
        "61": "Educational Services",
        "62": "Health Care and Social Assistance",
        "71": "Arts, Entertainment, and Recreation",
        "72": "Accommodation and Food Services",
        "81": "Other Services (except Public Administration)",
        "91": "Public Administration",
    }

    code_str = str(code).upper()
    label = None
    # Check for numeric NAICS code
    if country == 'US':
        label ='naics'
        return naics_to_name[code_str], code_str, label
    # Check for SCIAN code
    elif country == "Mexico":
        label = 'scian'
        return scian_to_name[code_str], code_str, label
    # Check for Canada NAICS code
    elif country == "Canada":
        label = 'naics_canada'
        return naics_canada_to_name[code_str], code_str, label
    # Check for letter NIC code
    elif country == "India":
        label = 'nic'
        return nic_to_name[code_str], code_str, label
    else:
        
        return None




# ----------------------*******----------------------
# New regulatory loss model
# PCI 
# Refers to data associated with credit and debit card transactions, 
# including card numbers, cardholder names, expiration dates, and security codes.


# PII
# PII is any information that can be used to identify an individual, 
# such as names, addresses, Social Security numbers, Passport, identification, biometric, phone and email addresses.


# PHI
# HI includes any health information that can be used to identify an individual, 
# such as medical records, health status.
# Section 0.3 — Breach exposure rate by NAICS (Verizon DBIR 2024)
BREACH_EXPOSURE_RATES = {
    '62': 0.31,                          # Healthcare
    '52': 0.22,                          # Finance & Insurance
    '51': 0.44,                          # Information / Tech
    '44': 0.38, '45': 0.38, '46': 0.38,  # Retail & Wholesale
    '31': 0.29, '32': 0.29, '33': 0.29,  # Manufacturing
    '61': 0.35,                          # Education
}
DEFAULT_BREACH_EXPOSURE_RATE = 0.34
# ---------------------------------------------------------------------------
# Regulatory constants (2025/2026 inflation-adjusted)
# ---------------------------------------------------------------------------

# FTC -- 16 C.F.R. 1.98(d). Bounded per-practice, per-day civil penalty.
FTC_DAILY_PENALTY_USD = 53_088

# SEC -- 17 C.F.R. 201.1001, Tier amounts for "any other person" (entities).
SEC_TIER_PENALTY_USD = {1: 118_225, 2: 591_127, 3: 1_182_251}
SEC_VIOLATION_COUNT = 2      # late 8-K + materially misleading 10-K

# HIPAA -- 45 C.F.R. 160.404, 2025 inflation-adjusted.
# r_min/r_max are per-violation-per-day; annual_cap is PER PROVISION CATEGORY.
HIPAA_TIERS = {
    1: {'r_min': 146,    'r_max': 73_011, 'annual_cap': 36_500},
    2: {'r_min': 1_460,  'r_max': 73_011, 'annual_cap': 146_000},
    3: {'r_min': 14_602, 'r_max': 73_011, 'annual_cap': 366_000},
    4: {'r_min': 73_011, 'r_max': 73_011, 'annual_cap': 2_190_294},
}
HIPAA_OBSERVED_MAX_USD = 16_000_000     # Anthem, 2018 -- largest OCR resolution

# CCPA -- Cal. Civ. Code 1798.140(d) "business" revenue threshold.
CCPA_REVENUE_THRESHOLD_USD = 26_625_000

# BIPA -- 740 ILCS 14/20 liquidated damages.
BIPA_NEGLIGENT_PER_RECORD = 1_000
BIPA_RECKLESS_PER_RECORD = 5_000

# NYDFS Part 500.
NYDFS_ALPHA_0_USD = 500_000             # median penalty at A_ref, average posture
NYDFS_ASSET_REF_USD = 50_000_000
NYDFS_BETA_ASSETS = 0.31
NYDFS_BETA_SECURITY = 1.8
NYDFS_CALIBRATION_S = 0.50              # posture the intercept was calibrated at
NYDFS_OBSERVED_MAX_USD = 50_000_000     # CNA Financial, 2021
NYDFS_EXEMPT_CAP_USD = 500_000
NYDFS_EXEMPT_EMPLOYEES = 20
NYDFS_EXEMPT_NY_REVENUE_USD = 7_500_000
NYDFS_EXEMPT_ASSETS_USD = 15_000_000
NYDFS_CLASS_A_EMPLOYEES = 2_000
NYDFS_CLASS_A_GLOBAL_REVENUE_USD = 1_000_000_000
NYDFS_CLASS_A_NY_REVENUE_USD = 20_000_000

# India DPDPA 2023 -- penalty provisions not fully enforceable until May 2027.
DPDPA_ENFORCEMENT_DATE = date(2027, 5, 30)

# Canada -- these were referenced but never defined (NameError on any Canadian
# applicant selecting Quebec Law 25 or FINTRAC).
QC_REALIZATION_FACTOR = 0.15            # CAI settles far below statutory ceiling
FINTRAC_REPORTABLE_FRACTION = 0.05      # share of revenue in reportable cash txns
FINTRAC_RATE_MIN_CAD = 1_000
FINTRAC_RATE_MAX_CAD = 500_000
FINTRAC_AMP_CEILING_CAD = 20_000_000

# Culpability levels.
NEGLIGENCE_NONE = 'none'
NEGLIGENCE_NEGLIGENT = 'negligent'
NEGLIGENCE_RECKLESS = 'reckless'


def count_critical_findings(assessment_data):
    """Number of Critical/High pen-test findings (CVSS >= 7.0)."""
    if not assessment_data:
        return 0
    count = 0
    for item in (assessment_data.get('vulnerabilityData') or []):
        if not isinstance(item, dict):
            continue
        level = _normalize_text(item.get('Risk Score/Level'))
        try:
            cvss = float(item.get('CVSS Score') or 0)
        except (TypeError, ValueError):
            cvss = 0.0
        if level in ('critical', 'high') or cvss >= 7.0:
            count += 1
    return count


def assess_negligence(security_data, assessment_data, k_provisions,
                      recent_breach, has_irp):
    """Classify culpability from the questionnaire and the pen test.

    Most privacy statutes only impose a monetary penalty where negligence or
    reckless disregard is established, so this gates every fine in the model.
    Recklessness is expressed through statutory rate selection (CCPA $7,500 vs
    $2,500, BIPA $5,000 vs $1,000), so the multiplier is only ever 0.0 or 1.0.

    Returns (level, multiplier).
    """
    security_data = security_data or {}
    critical_findings = count_critical_findings(assessment_data)
    had_breach = str(recent_breach).strip().lower() == 'yes'

    resolved = _normalize_text(security_data.get('pentestCriticalResolved'))
    unresolved_critical = critical_findings > 0 and resolved not in ('yes', 'y')
    ignores_known_issues = _normalize_text(security_data.get('fixProblems')) in ('no', 'n')
    repeat_without_irp = had_breach and not has_irp

    # Reckless: the applicant KNEW (pen test / prior incident) and did not act.
    if unresolved_critical or (ignores_known_issues and critical_findings > 0) \
            or repeat_without_irp:
        return NEGLIGENCE_RECKLESS, 1.0

    # Negligent: control gaps exist, but no evidence of knowing disregard.
    if k_provisions > 0 or critical_findings > 0 or ignores_known_issues:
        return NEGLIGENCE_NEGLIGENT, 1.0

    # No negligence found -> no monetary penalty exposure.
    return NEGLIGENCE_NONE, 0.0

def get_breach_exposure_rate(naics_code):
    return BREACH_EXPOSURE_RATES.get(str(naics_code)[:2], DEFAULT_BREACH_EXPOSURE_RATE)

def get_cad_usd():
    """USD value of 1 CAD at the current rate (replaces the hardcoded 0.74)."""
    cad_per_usd = get_usd_to_target_rate("CAD") or STATIC_USD_RATES["CAD"]
    return 1.0 / cad_per_usd

def saturating_loss(exposure, capacity):
    """Smoothly bound `exposure` by `capacity` (both USD).

    exposure << capacity -> ~exposure (no distortion at small values)
    exposure >> capacity -> asymptotes to capacity
    Monotonic in exposure at every scale, so unlike min() it never returns
    the identical figure for companies with very different exposure.
    """
    if capacity <= 0 or exposure <= 0:
        return 0.0
    return capacity * (1.0 - math.exp(-exposure / capacity))

def calculate_privacy_liability_coverage_loss(
        regulations,
        pci_records,  # PCI (Payment Card Information)
        pii_records,  # PII (Personally Identifiable Information)
        phi_records,  # PHI (Protected Health Information)
        annual_revenue,
        regulation_subquestions,
        s_bar,
        naics_code,
        security_data,
        recent_breach,
        num_employees,  # Required for NYDFS sizing tiers
        p_value,
        assessment_data=None,
        business_data=None,
        company_size='medium',
):
    # Ensure security_data is a dict to avoid NoneType errors
    security_data = security_data or {}
    
    LEGAL_COST_CAPS = {
        'GDPR': {'pct': 0.0020, 'abs': 30_000_000, 'currency': 'EUR'},
        'HIPAA': {'pct': 0.0020, 'abs': 150_000_000, 'currency': 'USD'},
        'FTC': {'pct': 0.12, 'abs': 750_000_000, 'currency': 'USD'},
        'SEC': {'pct': 0.05, 'abs': 200_000_000, 'currency': 'USD'},
        'BIPA': {'pct': 0.008, 'abs': 700_000_000, 'currency': 'USD'},
        'UK ICO': {'pct': 0.0020, 'abs': 50_000_000, 'currency': 'GBP'},
        'OPC CA': {'pct': 0.006, 'abs': 250_000_000, 'currency': 'CAD'},
        'NYDFS Part 500': {'pct': 0.0, 'abs': 0, 'currency': 'USD'},
        'CCPA': {'pct': 0.005, 'abs': 400_000_000, 'currency': 'USD'},
    }
    BIPA_REALIZATION_AT_10K = 0.20
    BIPA_CLASS_DECAY = -0.167          # solves the two anchors above
    BIPA_OBSERVED_CEILING = 650_000_000   # largest BIPA settlement on record
    
    # --- NORMALIZED USD CONSTANTS ---
    uma_daily_usd, uma_annual_usd = get_uma_usd()
    crore_inr_usd = get_crore_usd()
    cad_to_usd = get_cad_usd()
    eur_to_usd = 1.0 / (get_usd_to_target_rate("EUR") or 0.92)
    gbp_to_usd = 1.0 / (get_usd_to_target_rate("GBP") or 0.79)
    
    # --- LEXI Universal Inputs ---
    try:
        security_score = max(0.0, min(1.0, float(s_bar)))
    except (ValueError, TypeError):
        security_score = 0.50 
    
    print(s_bar)
    naics_str = str(naics_code)
    naics_prefix = naics_str[:2]
    
    # UPDATED: Dwell Time calculation based on new LEXI Table
    dwell_times = {
        '62': 236, '52': 168, '31': 199, '32': 199, '33': 199
        # Retail (44-46) and Education (61) now fallback to the 194 default
    }
    base_dwell = dwell_times.get(naics_prefix, 194)

    # Universal Asset Estimation (Needed for OCC and NYDFS)
    mu_a = 8.5 if naics_prefix == '52' else (0.9 if naics_prefix == '54' else 1.4)
    total_assets = annual_revenue * mu_a

    # --- Section 0.3: n_exposed vs n_total -------------------------------
    had_breach = str(recent_breach).strip().lower() == "yes"
    rho = get_breach_exposure_rate(naics_str)
    kappa_r = 2.0 if had_breach else 1.0   # global recidivism multiplier

    def _reported_exposed(data_type_key):
        if not business_data:
            return None
        block = business_data.get(data_type_key) or {}
        try:
            return int(float(block.get('recordsExposed', 0)))
        except (TypeError, ValueError):
            return None

    def n_exposed(n_total, data_type_key=None):
        """Q11a = Yes -> Records Exposed; Q11a = No -> n_total * rho."""
        try:
            n_total = max(0.0, float(n_total or 0))
        except (TypeError, ValueError):
            return 0.0
        if had_breach:
            reported = _reported_exposed(data_type_key) if data_type_key else None
            if reported:
                return float(min(reported, n_total))
            return n_total
        return n_total * rho

    PII_KEY = 'Personally Identifiable Information (PII)'
    PCI_KEY = 'Payment Card Information (PCI)'
    PHI_KEY = 'Protected Health Information (PHI)'

    pii_exposed = n_exposed(pii_records, PII_KEY)
    pci_exposed = n_exposed(pci_records, PCI_KEY)
    phi_exposed = n_exposed(phi_records, PHI_KEY)
    all_exposed = pii_exposed + pci_exposed + phi_exposed

    # --- SPECIFIC SECURITY DATA EXTRACTION ---
    has_irp = has_incident_response_plan(security_data)
    
    td_val = _first_non_empty(security_data.get("edrToolsUsed"), security_data.get("edrTools"))
    no_detection = is_no_threat_detection(td_val)
    
        # --- Pen-test derived signals ----------------------------------------
    critical_findings = count_critical_findings(assessment_data)
    _vuln_text = " ".join(
        _normalize_text((v or {}).get('Vulnerability name'))
        for v in ((assessment_data or {}).get('vulnerabilityData') or [])
        if isinstance(v, dict)
    )
    pentest_cleartext = any(t in _vuln_text for t in
                            ('cleartext', 'plaintext', 'unencrypted', 'plain text'))
    pentest_data_in_logs = any(t in _vuln_text for t in
                               ('information disclosure', 'directory listing',
                                'log file', 'exported share'))

    # --- HIPAA provision counter (K) --------------------------------------
    # 45 C.F.R. Part 164. Each poorly-answered control maps to one provision
    # CATEGORY; the annual cap in 160.404(b) applies per category, per year.
    hipaa_provisions = []
    mfa_answer = _first_non_empty(security_data.get("useMFA"), security_data.get("mfaTypes"))
    if not mfa_answer or any(_normalize_text(item) in ("nobody", "none")
                             for item in _as_list(mfa_answer)):
        hipaa_provisions.append('164.312(d) Person Authentication')
    if security_data.get("lockAccount") != "Yes":
        hipaa_provisions.append('164.312(a)(2)(iii) Automatic Logoff')
    if is_no_admin_alert_app(security_data.get("ReceiveAlertApp")):
        hipaa_provisions.append('164.312(b) Audit Controls')
    if _normalize_text(security_data.get("securityTraining")) in (
            "", "we don't provide cybersecurity training", "once a year"):
        hipaa_provisions.append('164.308(a)(5) Security Awareness')
    if security_data.get("fixProblems") != "Yes":
        hipaa_provisions.append('164.308(a)(5)(ii)(B) Malicious Software')
    if security_data.get("dataEncryption") != "Yes":
        hipaa_provisions.append('164.312(a)(2)(iv) Encryption at Rest')
    if is_weak_or_missing_backup(security_data.get("dataBacking")):
        hipaa_provisions.append('164.308(a)(7) Contingency Plan')
    if is_uncontrolled_vendor_access(security_data.get("thirdPartyControls")):
        hipaa_provisions.append('164.308(b)(1) Business Associate Contracts')
    if no_detection:
        hipaa_provisions.append('164.308(a)(1) Security Management Process')
    if not has_irp:
        hipaa_provisions.append('164.308(a)(6) Security Incident Procedures')
    # NEW questionnaire items. Only counted when explicitly answered "No", so K
    # does not inflate for applicants who answered before these questions shipped.
    if _normalize_text(security_data.get("riskAnalysis")) in ("no", "n"):
        hipaa_provisions.append('164.308(a)(1)(ii)(A) Risk Analysis')
    if _normalize_text(security_data.get("securityOfficial")) in ("no", "n"):
        hipaa_provisions.append('164.308(a)(2) Assigned Responsibility')
    # PEN TEST derived provisions.
    if pentest_cleartext:
        hipaa_provisions.append('164.312(e)(2)(ii) Transmission Security')
    if pentest_data_in_logs:
        hipaa_provisions.append('164.514(b) Minimum Necessary')
    # No floor of 1: a clean questionnaire means no provision was violated.
    k_provisions = len(hipaa_provisions)

    # --- Culpability gate --------------------------------------------------
    negligence_level, negligence_multiplier = assess_negligence(
        security_data, assessment_data, k_provisions, recent_breach, has_irp)
    is_reckless = negligence_level == NEGLIGENCE_RECKLESS
    no_negligence = negligence_level == NEGLIGENCE_NONE

    fines = []
    regulatory_loss_and_fines = {}

    # =========================================================================
    # CONSOLIDATED PREPROCESSING
    # ==========================================
    expanded_regulations = []
    normalized_regulations = []
    for reg in regulations:
        canonical_reg = normalize_regulation_label(reg)
        if canonical_reg and canonical_reg not in normalized_regulations:
            normalized_regulations.append(canonical_reg)

    is_finance_sector = naics_prefix == '52'

    def is_yes(key):
        return str(regulation_subquestions.get(key, "No")).strip().lower() == "yes"

        # --- Entity-type gates -------------------------------------------------
    # Several regimes only reach a specific kind of registrant. Without these
    # checks the model charged SEC / OCC / NYDFS penalties to private,
    # non-financial applicants who can never be subject to them.
    def _flag(*values):
        raw = _first_non_empty(*values)
        if raw is None:
            return None
        return _normalize_text(raw) in ('yes', 'y', 'true', 'public')

    _bd = business_data or {}
    is_public_company = _flag(_bd.get('isPublicCompany'), _bd.get('publicCompany'),
                              regulation_subquestions.get('SEC_public'))
    if is_public_company is None:
        is_public_company = False        # no question asked -> assume private

    is_occ_supervised = _flag(_bd.get('isBank'), _bd.get('occSupervised'),
                              regulation_subquestions.get('OCC_bank'))
    if is_occ_supervised is None:
        is_occ_supervised = is_finance_sector

    is_nydfs_licensed = _flag(_bd.get('isNYDFSLicensed'),
                              regulation_subquestions.get('NYDFS_licensed'))
    if is_nydfs_licensed is None:
        is_nydfs_licensed = is_finance_sector

    for reg in normalized_regulations:
        if reg == 'DPDPA':
            if 'DPDPA' not in expanded_regulations:
                expanded_regulations.append('DPDPA')
            # if is_yes('DPDPA_processor') and 'IT Act / SPDI Rules' not in expanded_regulations:
            #     expanded_regulations.append('IT Act / SPDI Rules')
            if is_yes('DPDPA_payment') and is_finance_sector and 'RBI Fintech / PSSA' not in expanded_regulations:
                expanded_regulations.append('RBI Fintech / PSSA')

        elif reg == 'LFPDPPP':
            if 'LFPDPPP' not in expanded_regulations:
                expanded_regulations.append('LFPDPPP')
            if (is_yes('MX_fintech_sandbox') or is_yes('MX_crypto')) and 'Fintech Law' not in expanded_regulations:
                expanded_regulations.append('Fintech Law')
                expanded_regulations.append('CNBV / CONDUSEF / Banxico')
            if is_yes('MX_website') and 'PROFECO / LFPC' not in expanded_regulations:
                expanded_regulations.append('PROFECO / LFPC')
                
        elif 'PROFECO' in reg:
            if 'PROFECO / LFPC' not in expanded_regulations:
                expanded_regulations.append('PROFECO / LFPC')
                
        elif reg == 'OPC CA':
            if 'OPC_CA' in regulation_subquestions:
                try:
                    regulation_subquestions['OPC_CA'] = int(regulation_subquestions['OPC_CA'])
                except (ValueError, TypeError):
                    pass
            if reg not in expanded_regulations:
                expanded_regulations.append(reg)
                
        else:
            if reg not in expanded_regulations:
                expanded_regulations.append(reg)

    # =========================================================================
    # CORE CALCULATION LOOP (Outputs in USD)
    # =========================================================================
    for regulation in expanded_regulations:
        fine = 0
        legal_cost = 0

        # ==========================================
        # EUROPE & UK (Converted to USD)
        # ==========================================
        if regulation == 'GDPR' and regulation_subquestions.get('GDPR', 0) != 0:
            relevant_records = regulation_subquestions['GDPR']
            lower_limit = 30 * relevant_records
            revenue_cap = 0.04 * annual_revenue
            # Convert 20M EUR cap to USD
            upper_limit = 20000000 * eur_to_usd
            fine = min(lower_limit, upper_limit, revenue_cap)
            #legal_cost = 100 * relevant_records
            
            cap_info = LEGAL_COST_CAPS.get(regulation, {'pct': 0.0, 'abs': 0})
            # legal_cost = min(legal_cost, cap_info['pct'] * annual_revenue, cap_info['abs'] * eur_to_usd)

        elif regulation == 'UK ICO' and regulation_subquestions.get('UK_ICO', 0) != 0:
            relevant_records = regulation_subquestions['UK_ICO']
            lower_limit = 200 * relevant_records
            # Convert 17.5M GBP cap to USD
            upper_limit = 17500000 * gbp_to_usd
            revenue_cap = 0.04 * annual_revenue
            fine = min(lower_limit, upper_limit, revenue_cap)
            #legal_cost = 200 * relevant_records
            
            cap_info = LEGAL_COST_CAPS.get(regulation, {'pct': 0.0, 'abs': 0})
            #legal_cost = min(legal_cost, cap_info['pct'] * annual_revenue, cap_info['abs'] * gbp_to_usd)

        # ==========================================
        # UNITED STATES (Native USD)
        # ==========================================
        elif regulation == 'FTC':
            # s.45(m)(1)(A): $51,744 per violation, each day of continuance a
            # separate violation. Days ARE the violation count - not multiplied
            # by affected-consumer count.
            r_ftc = FTC_DAILY_PENALTY_USD
            d_ftc = base_dwell                      # Section 0.2, no grace period
            fine = 0 if no_negligence else r_ftc * d_ftc

            cap_info = LEGAL_COST_CAPS.get(regulation, {'pct': 0.12, 'abs': 750_000_000})
            fine = min(fine, cap_info['pct'] * annual_revenue,
                       cap_info['abs'], annual_revenue)

            #legal_cost = 30 * pii_records
            
        elif regulation == 'SEC':
            # 8-K / 10-K disclosure duties attach to public reporting companies only
            if not is_public_company:
                logger.info("SEC selected but applicant is not a public reporting company; fine=0")
                fine = 0
            elif no_negligence:
                fine = 0
            else:
                # 15 U.S.C. § 78u(d)(3) statutory tier penalties (entity rates)
                if is_reckless and had_breach:
                    fine = 1_182_251  # Tier 3 — Fraud + Substantial Losses
                elif is_reckless:
                    fine = 591_127    # Tier 2 — Fraud / Reckless
                else:
                    fine = 118_225

        elif regulation == 'HIPAA':
            # HIPAA only reaches entities holding PHI
            if phi_records <= 0:
                logger.info("HIPAA selected but no PHI records held; fine=0")
                fine = 0
            elif no_negligence or k_provisions == 0:
                fine = 0
            else:
                # 45 C.F.R. § 160.404 statutory tier mapping:
                # Tier 1: $73,011 | Tier 2: $73,011 | Tier 3: $73,011 | Tier 4: $2,190,294
                HIPAA_FLAT_TIERS = {
                    1: 73_011,
                    2: 73_011,
                    3: 73_011,
                    4: 2_190_294,
                }

                # Evaluate remediation only if critical findings existed from a pen test
                corrected = False
                if critical_findings > 0:
                    corrected = (
                        _normalize_text(security_data.get('pentestCriticalResolved')) in ('yes', 'y')
                        or _normalize_text(security_data.get('pentestCritical30Days')) in ('yes', 'y')
                    )

                if is_reckless and critical_findings > 0 and not corrected:
                    tier = 4            # Tier 4 — Willful Neglect, Not Corrected
                elif is_reckless:
                    tier = 3            # Tier 3 — Willful Neglect, Corrected (or reckless without pen test)
                elif security_score < 0.80:
                    tier = 2            # Tier 2 — Reasonable Cause
                else:
                    tier = 1            # Tier 1 — Did Not Know

                # The annual cap applies per violation category (k_provisions)
                fine = HIPAA_FLAT_TIERS[tier]
            
        elif regulation == 'OCC':
            # 12 U.S.C. 1818(i)(2) reaches national banks and federal savings
            # associations only.
            if not is_occ_supervised:
                logger.info("OCC selected but applicant is not an OCC-supervised "
                            "institution; fine=0")
                fine = 0
            elif no_negligence:
                fine = 0
            else:
                # The statutory figures are daily MAXIMA, not floors.
                if security_score >= 0.70:
                    r_occ = 5_000
                elif security_score >= 0.40:
                    r_occ = 25_000
                else:
                    # "the lesser of $1,000,000 or 1% of total assets" per day,
                    # but never below the Tier 2 ceiling -- otherwise the worst
                    # posture produced the smallest fine.
                    r_occ = max(25_000, min(1_000_000, (0.01 * total_assets) / 365.0))

                d_occ = base_dwell + 30
                fine = min(r_occ * d_occ, 0.25 * annual_revenue)

        elif regulation == 'CCPA':
            n_ca = _safe_int(regulation_subquestions.get('CCPA', 0))
            # Cal. Civ. Code 1798.140(d): the CCPA only reaches a "business" that
            # clears the revenue threshold AND holds California resident records.
            # Both conditions, or the exposure is zero.
            if annual_revenue < CCPA_REVENUE_THRESHOLD_USD or n_ca <= 0:
                logger.info("CCPA thresholds not met (revenue=%s, CA records=%s); fine=0",
                            annual_revenue, n_ca)
                fine = 0
            elif no_negligence:
                fine = 0
            else:
                n_ca_exp = n_exposed(n_ca)
                # Track 1 -- CPPA administrative penalty, 1798.155(b).
                p_intent = 7_500 if is_reckless else 2_500
                # Track 2 -- private right of action, 1798.150(a)(1).
                r_b = 100 + 650 * (1 - security_score)
                size_factor = {
                    'micro': 0.25,
                    'small': 0.50,
                    'medium': 0.75,
                    'large': 1.00,
                }.get(str(company_size).lower(), 0.75)
                statutory_exposure = n_ca_exp * (p_intent + r_b) * size_factor
                cap_info = LEGAL_COST_CAPS.get(regulation, {'pct': 0.005, 'abs': 400_000_000})
                fine = min(statutory_exposure,
                           cap_info['pct'] * annual_revenue,
                           cap_info['abs'],
                           annual_revenue)
            

        elif regulation == 'BIPA':
            n_il = _safe_int(regulation_subquestions.get('BIPA', 0))
            if not n_il:
                logger.warning("BIPA selected but no Illinois record count; fine=0")
                fine = 0
            elif no_negligence:
                fine = 0
            else:
                # 740 ILCS 14/20:
                # $5,000 per violation for reckless/intentional conduct
                # $1,000 per violation for negligent / unaware conduct
                rate_per_record = 5_000 if is_reckless else 1_000
                fine = n_il * rate_per_record
            
        elif regulation == 'NYDFS Part 500':
            # 23 NYCRR 500 reaches DFS-licensed entities only.
            if not is_nydfs_licensed:
                logger.info("NYDFS selected but applicant is not a DFS-licensed "
                            "entity; fine=0")
                fine = 0
            elif no_negligence:
                fine = 0
            else:
                _nydfs = regulation_subquestions.get('NYDFS_Part_500') or []
                ny_revenue = annual_revenue      # R_NY proxy where not collected

                # 500.19(a) limited exemption: <20 employees OR <$7.5M NY revenue
                # OR <$15M year-end total assets -> capped exposure.
                is_exempt = (num_employees < NYDFS_EXEMPT_EMPLOYEES
                             or ny_revenue < NYDFS_EXEMPT_NY_REVENUE_USD
                             or total_assets < NYDFS_EXEMPT_ASSETS_USD)

                # 500.1(d) Class A: >=$20M NY gross revenue in EACH of the last
                # two fiscal years AND (>2,000 employees globally OR >$1B global
                # gross revenue). The size arm is OR, not AND.
                ny_rev_2yr = (bool(_nydfs[2]) if len(_nydfs) > 2
                              else ny_revenue >= NYDFS_CLASS_A_NY_REVENUE_USD)
                emp_flag = (bool(_nydfs[1]) if len(_nydfs) > 1
                            else num_employees > NYDFS_CLASS_A_EMPLOYEES)
                is_class_a = ny_rev_2yr and (
                    emp_flag
                    or num_employees > NYDFS_CLASS_A_EMPLOYEES
                    or annual_revenue > NYDFS_CLASS_A_GLOBAL_REVENUE_USD
                )

                alpha_0 = NYDFS_ALPHA_0_USD * (2.0 if is_class_a else 1.0)
                kappa_coop = 0.6 if has_irp else 1.0

                # F = a0 * (A/A_ref)^b1 * [(1-s)^b2 / (1-s_cal)^b2] * k_coop
                # Assets AND posture are normalised, so a0 keeps its calibrated
                # meaning: $500K at $50M assets and average posture. The previous
                # form multiplied $500K by assets^0.31 in raw dollars, which
                # returned ~$100M for a $50M firm.
                asset_ratio = max(total_assets, 1.0) / NYDFS_ASSET_REF_USD
                posture_ratio = ((1 - security_score) ** NYDFS_BETA_SECURITY) / \
                                ((1 - NYDFS_CALIBRATION_S) ** NYDFS_BETA_SECURITY)
                fine = (alpha_0 * (asset_ratio ** NYDFS_BETA_ASSETS)
                        * posture_ratio * kappa_coop)

                # C_NYDFS: observed enforcement maximum (CNA Financial, 2021).
                fine = min(fine, NYDFS_OBSERVED_MAX_USD)
                if is_exempt:
                    fine = min(fine, NYDFS_EXEMPT_CAP_USD)
            legal_cost = 0

        # ==========================================
        # MEXICO (Converted to USD)
        # ==========================================
        elif regulation == 'LFPDPPP':
            is_small = is_yes('LFPDPPP_small') or any('small' in r.lower() for r in regulations)
            has_biometrics = any('Biometrics' in r for r in regulations)

            # Tier logic per LEXI PDF Section 10:
            # Very Serious: s̃ < 0.10 OR phi_records > 0
            # Serious:      s̃ in [0.10, 0.40)
            # Minor:        s̃ >= 0.40
            # Small business exemption overrides all tiers per original bug fix note.

            if is_small:
                n_min, n_max = 100, 10000

            elif security_score < 0.10 or phi_records > 0 or has_biometrics:  # Very Serious
                n_min, n_max = 160000, 320000

            elif security_score < 0.40:  # Serious
                n_min, n_max = 10000, 160000

            else:  # Minor
                n_min, n_max = 100, 10000

            kappa_r = 2 if str(recent_breach).strip().lower() == "yes" else 1
            fine = uma_daily_usd * (n_min + (n_max - n_min) * (1 - security_score)) * kappa_r
        elif regulation == 'PROFECO / LFPC': 
            is_serious = (phi_records > 0 or security_score < 0.40 or 
                          any('Serious Case' in r for r in regulations))
            
            if is_serious:
                n_min, n_max = 1000, 100000
                legal_cost = 30 * pii_records
            else:
                n_min, n_max = 200, 10000
                legal_cost = 20 * pii_records
                
            fine = uma_daily_usd  * (n_min + (n_max - n_min) * (1 - security_score))

        elif regulation == 'CNBV / LMV': 
            n_min = 1000
            n_max = 100000
            fine = uma_daily_usd * (n_min + (n_max - n_min) * (1 - security_score))
            #legal_cost = 30 * pii_records

        elif regulation == 'CNBV / CONDUSEF / Banxico': 
            n_min = 1000
            n_max = 100000
            fine = uma_annual_usd * (n_min + (n_max - n_min) * (1 - security_score))
            #legal_cost = 40 * (pci_records + pii_records + phi_records)

        elif regulation == 'Fintech Law': 
            if no_detection and naics_prefix == '52':
                n_min, n_max = 10000, 500000
            else:
                n_min, n_max = 200, 10000
            fine = uma_annual_usd * (n_min + (n_max - n_min) * (1 - security_score))
            #legal_cost = 50 * (pci_records + pii_records)

        elif regulation == 'Fintech Law - API / Data Misuse': 
            if no_detection and naics_prefix == '52':
                n_min, n_max = 10000, 500000
            else:
                n_min, n_max = 200, 10000
            fine = uma_annual_usd * (n_min + (n_max - n_min) * (1 - security_score))
            #legal_cost = 60 * (pci_records + pii_records)

        elif regulation == 'COFEPRIS / NOM-024-SSA3': 
            lower_limit = 1000 * uma_daily_usd
            upper_limit = 50000 * uma_daily_usd
            record_based_estimate = 50 * phi_records
            fine = min(max(lower_limit, record_based_estimate), upper_limit)
            #legal_cost = 100 * phi_records

        # ==========================================
        # INDIA (Converted to USD)
        # ==========================================
        elif regulation == 'DPDPA' and date.today() < DPDPA_ENFORCEMENT_DATE:
            # DPDPA 2023 penalty provisions are not fully enforceable until the
            # DPDP Rules transition period closes on 30 May 2027. Until then the
            # modelled exposure is zero.
            logger.info("DPDPA penalties not enforceable before %s; fine=0",
                        DPDPA_ENFORCEMENT_DATE)
            fine = 0
            legal_cost = 0
        elif regulation == 'DPDPA': 
            possible_caps = []
            possible_legal_costs = []
            
            # 1. Align caps with actual DPDPA Schedule (converted to USD)
            if is_yes('DPDPA_minors') or any('Children Data' in r for r in regulations):
                possible_caps.append(200 * crore_inr_usd)
                possible_legal_costs.append(0.50 * (pii_records + phi_records)) # Scaled down per-record cost
                
            if not has_irp or any('Breach Notification' in r for r in regulations):
                possible_caps.append(200 * crore_inr_usd)
                possible_legal_costs.append(0.25 * (pci_records + pii_records + phi_records))
                
            if any('Biometrics' in r for r in regulations) or phi_records > 0 or is_yes('DPDPA_payment') or any('Significant' in r for r in regulations):
                possible_caps.append(250 * crore_inr_usd)
                possible_legal_costs.append(0.75 * pii_records)
                
            if is_yes('DPDPA_processor') and not possible_caps:
                possible_caps.append(50 * crore_inr_usd)
                possible_legal_costs.append(0.15 * (pci_records + pii_records + phi_records))
            
            # Correct default baseline obligation to 50 Crore instead of 250 Crore
            if not possible_caps:
                possible_caps.append(50 * crore_inr_usd)
                possible_legal_costs.append(0.20 * (pci_records + pii_records + phi_records))
            
            # 2. Prevent legal costs from exceeding a realistic company threshold (e.g., 10% of revenue)
            #revenue_usd = 5000000 # Dynamically pass company revenue here
            #legal_cost = min(max(possible_legal_costs), revenue_usd * 0.10)
            
            #Calculate fine by scaling breach intensity instead of reducing the maximum cap
            max_cap = max(possible_caps)
            breach_intensity = (1 - security_score) ** 0.7
            
            # Base fine scales realistically to the company size, but mathematically cannot exceed statutory max cap
            base_fine = (annual_revenue * 0.04) * breach_intensity 
            fine = min(base_fine, max_cap)

        # elif regulation == 'IT Act / SPDI Rules': 
        #     ITACT_ADJUDICATION_CAP_INR = 50_000_000.0   # Rs 5 crore, IT Act s.46 adjudicating officer limit
        #     inr_per_usd = 1e7 / crore_inr_usd
        #     r_spdi_inr = 250 + 4750 * ((1 - security_score) ** 0.7)
        #     affected = pci_records + pii_records + phi_records
        #     exposure = (affected * r_spdi_inr) / inr_per_usd

        #     # s.46 caps adjudicating-officer jurisdiction at Rs 5 crore.
        #     bounded = min(exposure, ITACT_ADJUDICATION_CAP_INR / inr_per_usd)
        #     fine = saturating_loss(bounded, total_assets)

        elif regulation == 'CERT-In': 
            c_cert = 1 * crore_inr_usd 
            p_noncomply = 0.62 * (1 - security_score) if not has_irp else 0.10
            fine = p_noncomply * c_cert
            #legal_cost = 10 * (pci_records + pii_records + phi_records)

        elif regulation == 'RBI Bank Cyber' or regulation == 'RBI Fintech / PSSA':
            if regulation == 'RBI Fintech / PSSA' and not is_finance_sector:
                continue
            # BR Act s.47A(1)(b): base penalty PLUS a per-day continuation amount.
            # The tier figures are the BASE. s.47A caps continuation at Rs 1 lakh/day.
            inr_per_usd = CRORE_INR / crore_inr_usd
            lakh_usd = 100_000 / inr_per_usd
            if security_score >= 0.70:   base_cr, pd_lakh = 0.5, 0.25
            elif security_score >= 0.40: base_cr, pd_lakh = 1.0, 0.5
            else:                        base_cr, pd_lakh = 2.0, 1.0
            base_penalty = min(base_cr * crore_inr_usd, annual_revenue * 0.02)
            fine = base_penalty + (pd_lakh * lakh_usd * base_dwell)
            #legal_cost = 30 * (pci_records + pii_records + phi_records)

        elif regulation == 'SEBI': 
            if security_score >= 0.70:
                r_j = 0.5 * crore_inr_usd 
            elif security_score >= 0.40:
                r_j = 12.5 * crore_inr_usd 
            else:
                r_j = 25 * crore_inr_usd
            fine = r_j 
            #legal_cost = 20 * pii_records

        elif regulation == 'Consumer Protection Act / CCPA India': 
            # Assuming these base heuristic numbers represented local currency initially or were undefined. 
            # Treating them as USD base if they were part of your original logic.
            lower_limit = 1 * pii_records
            upper_limit = 1000000
            fine = min(lower_limit, upper_limit)
            #legal_cost = 20 * pii_records

        elif regulation == 'Health Data / ABDM / SPDI': 
            lower_limit = 125 * phi_records
            upper_limit = 2500000000  # Note: if this was INR, it should be converted. Kept as original heuristic.
            fine = min(lower_limit, upper_limit)
            #legal_cost = 75 * phi_records

        # ==========================================
        # CANADA (Converted to USD)
        # ==========================================
        # ==========================================
        # CANADA (Part 2 of the LEXI reference)
        # ==========================================
        elif regulation == 'OPC CA' or regulation == 'OPC / PIPEDA':
            # PIPEDA carries NO privacy fine (s.28 obstruction only, $100k CAD).
            # Exposure is notification cost: C_notify = n_CA-res * rho * c_notify
            n_ca_res = regulation_subquestions.get('OPC_CA', 0) or (
                pci_records + pii_records + phi_records)
            c_notify_cad = 25.0            # OPC 2023 midpoint of $18–$32 CAD
            fine = 0.0
            legal_cost = n_exposed(n_ca_res) * c_notify_cad * cad_to_usd

            # Optional: s.28 obstruction penalty only where there is no IRP.
            if not has_irp:
                fine = 100_000 * cad_to_usd

        elif regulation == 'Quebec Law 25':
            alpha = 1.0
            beta = QC_REALIZATION_FACTOR      # see note below
            r_global = annual_revenue         # R_global = R_annual fallback

            # Tier 1: third-party disclosure / credential theft (Q11c) OR s < 0.30
            cause = _normalize_text(
                (business_data or {}).get('breachCause', '')) if business_data else ''
            tier1_cause = any(t in cause for t in ('third party', 'vendor', 'credential'))
            is_tier_1 = (security_score < 0.30) or tier1_cause

            p_t = 0.04 if is_tier_1 else 0.02
            f_stat_cad = 25_000_000 if is_tier_1 else 10_000_000
            fine = max(p_t * r_global, f_stat_cad * cad_to_usd) * alpha * beta

        elif regulation == 'OSFI':
            # F = mu_0 * (R_annual / R_ref)^0.4 * phi(s),  phi(s) = 2 - s
            mu_0_usd = 500_000 * cad_to_usd
            r_ref_usd = 50_000_000 * cad_to_usd
            phi_s = 2.0 - security_score
            fine = mu_0_usd * ((max(annual_revenue, 1.0) / r_ref_usd) ** 0.4) * phi_s

        elif regulation == 'FINTRAC':
            d_fintrac = base_dwell                      # no grace period per doc
            r_thresh_usd = 10_000 * cad_to_usd

            # Reportable volume, not gross revenue.
            reportable_usd = annual_revenue * FINTRAC_REPORTABLE_FRACTION
            reports_per_day = reportable_usd / (r_thresh_usd * 365.0)

            # Only a fraction are missed; posture drives the miss rate.
            miss_rate = (1 - security_score) ** 1.5
            v_fin = d_fintrac * reports_per_day * miss_rate

            r_fin_usd = (FINTRAC_RATE_MIN_CAD
                         + (FINTRAC_RATE_MAX_CAD - FINTRAC_RATE_MIN_CAD)
                         * miss_rate) * cad_to_usd

            fine = min(v_fin * r_fin_usd, FINTRAC_AMP_CEILING_CAD * cad_to_usd)

        elif regulation == 'PIPA AB/BC':
            n_ca_res = regulation_subquestions.get('OPC_CA', 0) or (
                pci_records + pii_records + phi_records)
            # PIPA AB s.59 / BC s.56: $100k CAD organisations
            fine = min(n_exposed(n_ca_res) * 20 * cad_to_usd, 100_000 * cad_to_usd)

        elif regulation == 'PHIPA / Provincial Health Privacy Laws':
            # PHIPA (ON) s.72(2): $1M CAD organisations
            fine = min(n_exposed(phi_records, PHI_KEY) * 500 * cad_to_usd,
                       1_000_000 * cad_to_usd)

        elif regulation == 'CSA / OSC':
            fine = min(n_exposed(pci_records + pii_records) * 50 * cad_to_usd,
                       5_000_000 * cad_to_usd)

        elif regulation == 'OPC / PIPEDA - Biometrics':
            fine = min(n_exposed(pii_records, PII_KEY) * 200 * cad_to_usd,
                       100_000 * cad_to_usd)

        elif regulation == 'CRTC / CASL':
            fine = min(n_exposed(pii_records, PII_KEY) * 50 * cad_to_usd,
                       10_000_000 * cad_to_usd)

        elif regulation == 'Competition Bureau':
            fine = min(n_exposed(pii_records, PII_KEY) * 50 * cad_to_usd,
                       10_000_000 * cad_to_usd)

        # ==========================================
        # UNKNOWN REGULATION
        # ==========================================
        else:
            fine = 0
            legal_cost = 0

        both_loss = int(fine + legal_cost)
        regulatory_loss_and_fines[regulation] = both_loss

    # --- PCI DSS ---------------------------------------------------------
    # Triggered by holding card data, not by the regulations list.
    if pci_records and int(pci_records) > 0:
        records_exposed = 0
        if business_data:
            records_exposed = (business_data.get('Payment Card Information (PCI)', {}) or {}).get('recordsExposed', 0)

        pci_total, pci_breakdown = calculate_pci_dss_assessment(
            pci_records=pci_records,
            annual_revenue=annual_revenue,
            p_value=p_value,
            recent_breach=recent_breach,
            records_exposed=records_exposed,
            security_data=security_data,
            assessment_data=assessment_data,
        )

        regulatory_loss_and_fines['PCI DSS'] = int(pci_total)
        _pci_detail = pci_breakdown
    else:
        _pci_detail = {}

    return fines, regulatory_loss_and_fines, _pci_detail

def _safe_int(value, default=0):
    """Coerce form values (which may be '', None, or '1,234') to int safely.

    Regulation record fields arrive from the form as strings and are often left
    blank, which previously caused `int('')` -> ValueError and crashed the whole
    report generation. Blank/invalid values fall back to `default`.
    """
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return default

def parse_regulations(business_data):

    reg_list = get_effective_regulations(business_data)
    subquestions = {}
    for reg in reg_list:
        reg = normalize_regulation_label(reg)
        if reg == "NYDFS Part 500":
            reg = reg.replace(" ", "_")
            records = business_data.get(reg, {})
            emp = business_data.get("NYDFS_emp", "")
            emp = 1 if emp == "Yes" else 0
            revenue = business_data.get("NYDFS_revenue", "")
            revenue = 1 if revenue == "Yes" else 0
            subquestions[reg] = [_safe_int(records), emp, revenue]
            # subquestions[reg].append(records)
            # subquestions[reg].append()
            # subquestions[reg].append(records)
            
            # subquestions[reg].append(int(answers[0]))
            # if answers[1] == 'Yes':
            #     subquestions[reg].append(1)
            # else:
            #     subquestions[reg].append(0)
        elif reg == "FTC":
            for key in ["FTC_CCPA", "FTC_NYDFS", "FTC_BIPA", "FTC_HIPAA", "FTC_employees"]:
                subquestions[key] = business_data.get(key, "")
        elif reg == "DPDPA":
            for key in [
                "DPDPA_processor", "DPDPA_payment", "DPDPA_employees",
                "DPDPA_minors", "DPDPA_outsideIndia", "DPDPA_DISHA",
                "DPDPA_UIDAI"
            ]:
                subquestions[key] = business_data.get(key, "")
        elif reg == "LFPDPPP":
            for key in [
                "LFPDPPP_small", "LFPDPPP_Biometrics", "LFPDPPP_AI",
                "MX_fintech_sandbox", "MX_crypto", "MX_website",
                "MX_outsideMexico", "MX_employees", "MX_Fintech", "CNBV",
                "COFEPRIS"
            ]:
                subquestions[key] = business_data.get(key, "")
        elif reg == "Fintech Law":
            subquestions["MX_Fintech"] = business_data.get("MX_Fintech", "")
            subquestions["MX_fintech_sandbox"] = business_data.get("MX_fintech_sandbox", "")
        elif reg == "CNBV / LMV":
            subquestions["CNBV"] = business_data.get("CNBV", "")
        elif reg == "COFEPRIS / NOM-024-SSA3":
            subquestions["COFEPRIS"] = business_data.get("COFEPRIS", "")
        elif reg == "OPC CA":
            subquestions["OPC_CA"] = _safe_int(business_data.get("OPC_CA", 0))
            for key in ["PIPEDA_personaldata", "PIPEDA_outsideCanada", "PIPEDA_employees", "PIPEDA_provinces"]:
                subquestions[key] = business_data.get(key, "")
        elif reg in ["CCPA", "GDPR", "BIPA", "UK ICO", "OPC CA"]:
            reg = reg.replace(" ", "_")
            answers = business_data.get(reg, {})
            subquestions[reg] = _safe_int(answers)
    return subquestions
# ----------------------*******----------------------

def determine_company_size(num_employees, current_revenue, prev_year_revenue1, prev_year_revenue2):

    revenue_vals = np.array([prev_year_revenue2, prev_year_revenue1, current_revenue])

    if num_employees < 0:
        raise ValueError('Number of employees must be a positive integer.')
    if any(revenue_vals < 0):
        raise ValueError('Revenue values must be non-negative.')

    # Previous-year revenue can legitimately be 0. Avoid dividing by zero and
    # fall back to current revenue when growth cannot be estimated safely.
    if prev_year_revenue1 == 0:
        predicted_revenue = current_revenue
    elif prev_year_revenue2 == 0:
        growth_rate_2 = (current_revenue - prev_year_revenue1) / prev_year_revenue1
        predicted_revenue = current_revenue * (1 + growth_rate_2)
    else:
        growth_rate_1 = (prev_year_revenue1 - prev_year_revenue2) / prev_year_revenue2
        growth_rate_2 = (current_revenue - prev_year_revenue1) / prev_year_revenue1
        average_growth_rate = (1 + growth_rate_1) * (1 + growth_rate_2) ** 0.5 - 1
        predicted_revenue = current_revenue * (1 + average_growth_rate)

    #use a linear regression model to predict the revenue for the next year
    #use the predicted revenue to determine the company size
    
    if (10_000_000 < predicted_revenue):
        return 'medium'
    elif (250000 < predicted_revenue <= 27_000_000):
        return 'small'
    elif predicted_revenue <= 250000:
        return 'micro'
    else:
        return 'large'
    

SIMPLE_COVERAGE_ALIASES = {
    'Bricking': 'Bricking Coverage',
    'PCI DSS': 'PCI DSS Coverage',
    'Malicious System Failure': 'Malicious System Failure Coverage',
    'Ransomware': 'Ransomware Coverage',
}

def get_coverage_gap(calculated_loss, questionnaire):

    coverage_gaps = {}
    total_loss = 0
    total_loss_with_coverage = 0
    for category, loss in calculated_loss.items():

        if category == 'Incident Response Costs':
            loss = calculated_loss[category]['Total']

        # Reset per-category so values never leak between iterations.
        # Defaulting the gap to the full loss means an unhandled category
        # reports itself as fully uncovered instead of borrowing a neighbor's gap.
        requested_coverage = 0
        existing_coverage = 0
        deductible = 0
        coverage_gap = loss

        # Translate loss-category name -> questionnaire key (only for lookups;
        # `category` itself stays unchanged for branching and output keys)
        questionnaire_key = SIMPLE_COVERAGE_ALIASES.get(category, category)

        if questionnaire_key in questionnaire:
            q = questionnaire[questionnaire_key]

            if category == 'Business Interruption Loss':
                if questionnaire['hasInsurance'] == "Yes":
                    lostRevenueCover = float(q['incomeProfitCover']) if q['incomeProfitCover'] else 0
                    contingentLossCover = float(q["contingentLossCover"]) if q["contingentLossCover"] else 0
                    continuingExpenseCover = float(q["continuingExpenseCover"]) if q["continuingExpenseCover"] else 0

                    lostRevenueDeductible = float(q["incomeProfitDeductible"]) if q["incomeProfitDeductible"] else 0
                    contingentLossDeductible = float(q['contingentLossDeductible']) if q['contingentLossDeductible'] else 0
                    continuingExpenseDeductible = float(q['continuingExpenseDeductible']) if q['continuingExpenseDeductible'] else 0

                    existing_coverage = lostRevenueCover + contingentLossCover + continuingExpenseCover
                    deductible = lostRevenueDeductible + contingentLossDeductible + continuingExpenseDeductible
                    coverage_gap = loss - existing_coverage
                else:
                    requested_coverage = float(q['coverageAmount']) if q['coverageAmount'] else 0
                    deductible = 0
                    coverage_gap = loss

            elif category == 'Privacy Liability Coverage':
                if questionnaire['hasInsurance'] == "Yes":
                    regulatoryFinesCover = float(q['regulatoryFinesCover']) if q['regulatoryFinesCover'] else 0
                    regulatoryPenaltiesCover = float(q['regulatoryPenaltiesCover']) if q['regulatoryPenaltiesCover'] else 0
                    legalExpensesCover = float(q['legalExpensesCover']) if q['legalExpensesCover'] else 0

                    existing_coverage = regulatoryFinesCover + regulatoryPenaltiesCover + legalExpensesCover
                    deductible = float(q['deductibleAmount']) if q['deductibleAmount'] else 0
                    coverage_gap = loss - existing_coverage
                else:
                    requested_coverage = float(q['coverageAmount']) if q['coverageAmount'] else 0
                    deductible = 0
                    coverage_gap = loss

            elif category in ('Incident Response Costs', 'Bricking', 'PCI DSS', 'Malicious System Failure', 'Ransomware'):
                # Simple coverageAmount / deductibleAmount pattern
                if questionnaire['hasInsurance'] == "Yes":
                    existing_coverage = float(q['coverageAmount']) if q.get('coverageAmount') else 0
                    deductible = float(q['deductibleAmount']) if q.get('deductibleAmount') else 0
                    coverage_gap = loss - existing_coverage
                else:
                    requested_coverage = float(q['coverageAmount']) if q.get('coverageAmount') else 0
                    deductible = 0
                    coverage_gap = loss

            if coverage_gap < 0:
                coverage_gap = 0

            coverage_gaps[category] = {
                'loss': loss,
                'coverage': requested_coverage,
                'deductible': deductible,
                'existing_coverage': existing_coverage,
                'coverage_gap': round(coverage_gap, 2)
            }

        else:
            coverage_gaps[category] = {
                'loss': loss,
                'coverage': 0,
                'deductible': 0,
                'existing_coverage': 0,
                'coverage_gap': loss
            }
        total_loss += loss
        total_loss_with_coverage += coverage_gap

    # Add total loss and total loss with coverage to the results
    coverage_gaps['Total'] = {
        'total_loss': total_loss,
        'total_loss_with_coverage': round(total_loss_with_coverage, 2)
    }
    return coverage_gaps


def calculate_incident_response_costs_subcategory_losses(company_data, business_data, security_data, total_ir_loss):
    """
    Allocate incident response loss across subcategories using breach exposure,
    regulation pressure, and control weakness instead of a fixed residual split.
    """
    personal_records_total = 0.0
    exposed_records_total = 0.0

    for data_type in company_data.get('personalRecords', []):
        record_info = business_data.get(data_type, {})
        try:
            personal_records_total += float(record_info.get('records', 0) or 0)
        except (TypeError, ValueError):
            pass

    for data_type in company_data.get('dataExposed', []):
        record_info = business_data.get(data_type, {})
        try:
            exposed_records_total += float(record_info.get('recordsExposed', 0) or 0)
        except (TypeError, ValueError):
            pass

    exposure_ratio = exposed_records_total / max(personal_records_total, 1.0)
    exposure_ratio = min(max(exposure_ratio, 0.0), 1.0)

    breach_flag = 1.0 if business_data.get('recentBreach') == 'Yes' else 0.0

    high_risk_regulations = {
        'FTC', 'HIPAA', 'GDPR', 'CCPA', 'OPC CA', 'UK ICO', 'NYDFS Part 500', 'BIPA',
        'DPDPA', 'LFPDPPP', 'RBI Fintech / PSSA', 'Health Data / ABDM / SPDI', 'SEBI'
    }
    active_regulations = [
        normalize_regulation_label(reg)
        for reg in business_data.get('regulations', [])
    ]
    active_regulations = [reg for reg in active_regulations if reg in high_risk_regulations]
    regulation_pressure = min(len(active_regulations) / 3.0, 1.0)

    control_deficit = 0.0
    if security_data.get('dataEncryption') != 'Yes':
        control_deficit += 0.20

    data_backing = security_data.get('dataBacking', '')
    if _normalize_text(data_backing) == 'weekly' or is_weak_or_missing_backup(data_backing):
        control_deficit += 0.15

    if security_data.get('fixProblems') != 'Yes' and security_data.get('updatesIn30Days') != 'Yes':
        control_deficit += 0.15

    if is_uncontrolled_vendor_access(security_data.get('thirdPartyControls')):
        control_deficit += 0.10

    edr_tools = _first_non_empty(security_data.get('edrToolsUsed'), security_data.get('edrTools'))
    if is_no_threat_detection(edr_tools):
        control_deficit += 0.20

    if security_data.get('lockAccount') != 'Yes':
        control_deficit += 0.10

    if not has_incident_response_plan(security_data):
        control_deficit += 0.10

    control_deficit = min(control_deficit, 1.0)

    raw_weights = {
        'Notification': 1.0 + 6.0 * exposure_ratio + 1.8 * breach_flag + 1.0 * regulation_pressure,
        'Forensics': 1.2 + 4.0 * control_deficit + 1.0 * exposure_ratio,
        'Legal': 1.5 + 4.5 * regulation_pressure + 2.0 * breach_flag + 1.5 * exposure_ratio,
        'Technical Response': 1.5 + 5.0 * control_deficit + 0.8 * breach_flag,
        'Crisis Management': 0.8 + 1.2 * breach_flag + 1.2 * control_deficit,
    }

    total_weight = sum(raw_weights.values()) or 1.0
    allocations = {key: (total_ir_loss * weight / total_weight) for key, weight in raw_weights.items()}

    notification = round(allocations['Notification'], 2)
    forensics = round(allocations['Forensics'], 2)
    legal = round(allocations['Legal'], 2)
    technical = round(allocations['Technical Response'], 2)
    crisis = round(total_ir_loss - (notification + forensics + legal + technical), 2)

    return {
        'Notification': notification,
        'Forensics': forensics,
        'Legal': legal,
        'Technical Response': technical,
        'Crisis Management': crisis,
    }


def count_num_records(company_data, business_data, data_multipliers, p_value):
    total_records = 0
    records_exposed = 0
    multiplers_used = []
    forensics_multipliers_used = []
    notification_costs = 0
    forensics_costs = 0
    for i in company_data['personalRecords']:
        total_records += float(business_data[i]['records'])
        multiplers_used.append(data_multipliers[i]['Notification'])
        forensics_multipliers_used.append(data_multipliers[i]['Forensics'])
    
    #if business_data['cyberIncident'] == 'Yes': 
    if 'recentBreach' in list(business_data.keys()) and business_data['recentBreach'] == 'Yes':
        company_data['incidentCount'] = business_data['incidentCount']      
        for data_type in company_data['dataExposed']:
            notification_costs += data_multipliers[data_type]['Notification'] * 0.73 * float(business_data[data_type]['recordsExposed'])
            forensics_costs += data_multipliers[data_type]['Forensics'] * 0.73 * float(business_data[data_type]['recordsExposed'])
    else:
        for i in range(len(multiplers_used)):
            notification_costs += multiplers_used[i] * 0.73 * total_records * p_value
            forensics_costs += forensics_multipliers_used[i] * 0.73 * total_records * p_value


    if "Payment Card Information (PCI)" in list(business_data.keys()):
        pci_records = int(business_data["Payment Card Information (PCI)"]['records'])
    else:
        pci_records = 0

    if "Personally Identifiable Information (PII)" in list(business_data.keys()):
        pii_records = int(business_data["Personally Identifiable Information (PII)"]['records'])
    else:
        pii_records = 0

    if "Protected Health Information (PHI)" in list(business_data.keys()):
        phi_records = int(business_data["Protected Health Information (PHI)"]['records'])
    else:
        phi_records = 0
    
    return total_records, notification_costs, forensics_costs, pci_records, pii_records, phi_records




    

################################### Visualizations ##################################

def plot_loss_exceedance_curve(total_losses):
    sorted_losses = np.sort(total_losses)[::-1]
    exceedance_probabilities = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

    plt.figure(figsize=(10,6))
    plt.plot(sorted_losses, exceedance_probabilities)
    plt.xlabel('Total Loss Over Simulation Period')
    plt.ylabel('Exceedance Probability')
    plt.title('Loss Exceedance Curve')
    plt.show()
