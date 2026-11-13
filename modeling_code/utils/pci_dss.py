"""
pci_dss.py

PCI DSS fines & assessments, as a key inside `regulatory_loss_and_fines`.

Called from calculate_privacy_liability_coverage_loss(); adds a 'PCI DSS' entry which then
flows into subcategory_losses_list['Privacy Liability Coverage'] (summed with
np.sum, so genuinely additive) and renders as a line item in
serializable_results['Privacy Liability Coverage Subcat'].

No double-count: the Advisen 'Fines & Penalties' bucket is discarded wholesale by
the override on line 267 of financial_services_finloss.py, so nothing in the
existing pipeline already contains card-brand assessments.

=============================================================================
THIS DOES NOT FOLLOW THE CONVENTION OF ITS NEIGHBOURS -- BY DESIGN
=============================================================================
Every other regulation in calculate_privacy_liability_coverage_loss is computed at FULL
record counts (e.g. `50 * (pci_records + pii_records)`), i.e. as a
conditional-on-breach severity. PCI DSS here is probability-weighted by p_value,
i.e. an EXPECTED annual loss.

That is a deliberate divergence. It means the 'PCI DSS' key is not on the same
footing as 'FTC' or 'OPC CA' and the two should not be compared line-for-line.

=============================================================================
STOCHASTICITY: EXPECTED VALUES, NOT DRAWS
=============================================================================
calculate_privacy_liability_coverage_loss runs ONCE, outside the Monte Carlo. Drawing a Beta
or a Poisson here would not produce a distribution -- it would produce a single
arbitrary sample, making the output seed-dependent while yielding no tail
information. So the Beta and Poisson below are collapsed to their means:

    fraud realization rate   Beta(a, b)   -> a / (a + b)
    remediation timeframe    Poisson(k)   -> k

Mathematically identical to what a one-draw version approximates, and
reproducible.

=============================================================================
UNSOURCED PRIORS -- every constant below
=============================================================================
REISSUANCE_FEE_PER_CARD, FRAUD_REALIZATION_A/B, AVG_FRAUDULENT_CHARGE,
CASE_MANAGEMENT_FEE_PER_CARD, MONTHLY_NONCOMPLIANCE_FINE, PFI_COST,
and the merchant-tier thresholds.

These are NOT sourced. Advisen does not separate card-brand assessments from
statutory fines, so it cannot support them. They are plausible figures, not
benchmarked ones. They must be verified against actual card-brand schedules
(Visa GCAR / Mastercard ADC) before this informs pricing.

MERCHANT TIER is a particularly weak inference: card-brand levels are defined on
ANNUAL TRANSACTION COUNT, which the questionnaire does not collect. It is proxied
here from annual_revenue and pci_records (cards STORED, which is not the same
thing). Treat the tier as indicative.
"""

import numpy as np


# --- Merchant tier proxy (see caveat in header) ------------------------------
def derive_merchant_tier(annual_revenue, pci_records):
    rev = float(annual_revenue or 0)
    cards = float(pci_records or 0)
    if rev >= 500_000_000 or cards >= 6_000_000:
        return 1
    if rev >= 50_000_000 or cards >= 1_000_000:
        return 2
    if rev >= 2_500_000 or cards >= 20_000:
        return 3
    return 4


# --- Unsourced priors --------------------------------------------------------
REISSUANCE_FEE_PER_CARD = 4.50
CASE_MANAGEMENT_FEE_PER_CARD = 5.00

# Fraud realization: fraction of breached cards that see an actual fraudulent
# charge. Heavily right-skewed -- most breached cards are never monetized.
FRAUD_REALIZATION_A = 1.2
FRAUD_REALIZATION_B = 30.0
E_FRAUD_REALIZATION = FRAUD_REALIZATION_A / (FRAUD_REALIZATION_A + FRAUD_REALIZATION_B)  # 0.0385

AVG_FRAUDULENT_CHARGE = 260.0

MONTHLY_NONCOMPLIANCE_FINE = {1: 50_000, 2: 25_000, 3: 10_000, 4: 5_000}
PFI_COST = {1: 200_000, 2: 110_000, 3: 45_000, 4: 18_000}

# Remediation timeframe, months. Poisson mean, scaled by how far from compliant
# the merchant is. A validated-compliant merchant is not fined at all.
REMEDIATION_MONTHS_BASE = 1.0
REMEDIATION_MONTHS_MAX_EXTRA = 6.0


# =============================================================================
# COMPLIANCE POSTURE -- derived from questions the form ALREADY asks
# =============================================================================

# -----------------------------------------------------------------------------
# PCI DSS requirements, mapped onto the CANONICAL security questions.
#
# COVERAGE IS A SUBSET. PCI DSS has 12 top-level requirements. The questionnaire
# touches 10 of the sub-requirements below and says NOTHING about several others:
#   Req 1/2  - firewalls, network segmentation, vendor defaults
#   Req 4    - encryption of cardholder data in transit
#   Req 7    - restrict access by business need-to-know
#   Req 9    - physical access to cardholder data
# `compliance_ratio` is therefore NOT "PCI DSS compliance". It is compliance across
# the requirements this form happens to observe. Do not present it as an AOC
# substitute, and do not let it imply a merchant is compliant overall.
#
# Enums below are taken from parse_security.py (parse_security_data_with_pentest /
# _no_pentest and mapping_values), NOT guessed. Where a question is not asked at
# all, the requirement returns None = NOT ASSESSED and drops out of the
# denominator rather than counting as a failure.
# -----------------------------------------------------------------------------

def _pentest_submitted(assessment_data):
    """
    Was a pentest actually submitted?

    Deliberately does NOT key on `is_valid_assessment` or `assessmentType`. The
    former is absent from the real payload, and model() (lines 94-99) OVERWRITES
    assessmentType to None whenever it is absent -- so both are unreliable by the
    time this runs. The vulnerability list and the file URL are the artifacts that
    actually survive. An empty vulnerabilityData list is a CLEAN pentest, not a
    missing one, so presence of the key matters, not its length.
    """
    a = assessment_data or {}
    return bool(a.get('assessmentfileUrl')) or ('vulnerabilityData' in a)


MFA_ALL_EMPLOYEES = 'All employees'
NO_TRAINING = "We don't provide cybersecurity training"
TRAINING_CADENCES = {'Once a year', 'Twice a year', 'Every 3 months'}
NO_VENDOR_ACCESS = "We don't give vendors access to our systems or data"
NO_EDR_MARKERS = {
    'We do not use security software to detect threats',
    'We do not have this', 'None',
}


def _req3_stored_data(s, a):
    """Req 3 - protect stored account data. Proxy: dataEncryption."""
    v = s.get('dataEncryption')
    if v is None:
        return None
    return v == 'Yes'


def _req5_antimalware(s, a):
    """Req 5 - anti-malware. Uses the edrTools boolean, falling back to edrToolsUsed."""
    if s.get('edrTools') in ('Yes', 'No'):
        return s.get('edrTools') == 'Yes'
    used = s.get('edrToolsUsed')
    if used is None:
        return None
    if isinstance(used, list):
        if not used:
            return False
        return not any(str(x).strip() in NO_EDR_MARKERS for x in used)
    return str(used).strip() not in NO_EDR_MARKERS


def _req6_patching(s, a):
    """
    Req 6.3.3 - critical patches within one month.
    `updatesIn30Days` is an exact match for PCI's one-month critical-patch window,
    so it is a PASS in its own right, not merely partial credit.
    """
    fix, thirty = s.get('fixProblems'), s.get('updatesIn30Days')
    if fix is None and thirty is None:
        return None
    return fix == 'Yes' or thirty == 'Yes'


def _req8_3_lockout(s, a):
    """
    Req 8.3 - account lockout, with particular weight on privileged accounts.

    `lockAccount` (Yes/No) and `specialAccess` (which privileged groups are
    covered) are the SAME control, not two. parse_security.py makes this explicit:
    its comment reads "# 3) Special Access - lockAccount", and special_access_points
    is scored off lockAccount. specialAccess is the which-groups sub-question.

    Scoring: lockAccount is the gate. specialAccess corroborates it -- lockout
    enabled but covering no privileged group is not a pass, since PCI cares
    specifically about administrative access.
    """
    lock = s.get('lockAccount')
    if lock is None:
        return None
    if lock != 'Yes':
        return False
    groups = s.get('specialAccess')
    if isinstance(groups, list) and len(groups) == 0:
        return False          # lockout on, but no privileged group covered
    return True


def _req8_4_mfa(s, a):
    """
    Req 8.4 - MFA for ALL access to the CDE.

    Canonical enum: 'All employees' / 'Admins or high-access users (...)' /
    'Leadership and security teams (...)' / 'Nobody'.
    Only 'All employees' satisfies 8.4. Admin-only MFA is a FAIL.
    """
    mfa = s.get('useMFA') or s.get('mfaTypes')
    if not mfa:
        return None if mfa is None else False
    if isinstance(mfa, str):
        mfa = [mfa]
    return any(str(x).strip() == MFA_ALL_EMPLOYEES for x in mfa)


def _req10_logging(s, a):
    """
    Req 10 - logging & monitoring.

    Deliberately NOT edrToolsUsed -- that is Req 5, and scoring one answer as two
    requirements double-weights it. The alerting questions (ReceiveAlertApp /
    securityAlerts) are the right signal. They are absent from the sample payload,
    in which case this returns None and drops out of the denominator.
    """
    for key in ('ReceiveAlertApp', 'securityAlerts'):
        if key in s and s[key] is not None:
            v = s[key]
            return bool(v) if isinstance(v, bool) else (str(v).strip() == 'Yes')
    return None


def _req11_3_pentest(s, a):
    """
    Req 11.3 - penetration testing AND remediation of findings.

    Uses the questions actually asked (pentestCriticalResolved /
    pentestCritical30Days), not a reconstruction from vulnerabilityData.

      no pentest submitted                 -> None  (NOT ASSESSED)
      pentestCriticalResolved == 'Yes'     -> True
      otherwise                            -> False

    A missing pentestCriticalResolved WITH a pentest present is scored False, to
    match parse_security_data_with_pentest(), which awards 0 pentest_points in
    exactly that case.
    """
    if not _pentest_submitted(a):
        return None
    return s.get('pentestCriticalResolved') == 'Yes'


def _req12_6_training(s, a):
    """Req 12.6 - security awareness training, at least annually."""
    v = s.get('securityTraining')
    if v is None:
        return None
    return str(v).strip() in TRAINING_CADENCES


def _req12_8_tpsp(s, a):
    """
    Req 12.8 - third-party service provider management.

    'We don't give vendors access to our systems or data' means the requirement
    does not apply -> NOT ASSESSED, not a failure. My earlier version failed this
    case, penalising merchants for having no vendors.
    """
    v = s.get('thirdPartyControls')
    if v is None:
        return None
    v = str(v).strip()
    if v == NO_VENDOR_ACCESS:
        return None
    return v == 'Yes'


def _req12_10_irp(s, a):
    """Req 12.10 - incident response plan."""
    if 'IncidentResponse' not in s:
        return None
    return bool(s.get('IncidentResponse'))


PCI_REQUIREMENTS = {
    'Req 3 - Protect stored account data':                 _req3_stored_data,
    'Req 5 - Anti-malware':                                _req5_antimalware,
    'Req 6.3.3 - Critical patches within one month':       _req6_patching,
    'Req 8.3 - Account lockout (incl. privileged accounts)': _req8_3_lockout,
    'Req 8.4 - MFA for all CDE access':                    _req8_4_mfa,
    'Req 10 - Logging & monitoring':                       _req10_logging,
    'Req 11.3 - Pentest performed & findings remediated':  _req11_3_pentest,
    'Req 12.6 - Security awareness training':              _req12_6_training,
    'Req 12.8 - Third-party service provider management':  _req12_8_tpsp,
    'Req 12.10 - Incident response plan':                  _req12_10_irp,
}


def assess_pci_compliance(security_data, assessment_data):
    """
    Returns (compliance_ratio, failed_requirements, not_assessed_requirements).

    Three-state scoring. A predicate returning None is NOT ASSESSED: dropped from
    numerator AND denominator rather than counted as a failure. The ratio is over
    the requirements the questionnaire can actually observe.
    """
    s = security_data or {}
    a = assessment_data or {}

    failed, not_assessed, passed = [], [], 0
    for name, test in PCI_REQUIREMENTS.items():
        result = test(s, a)
        if result is None:
            not_assessed.append(name)
        elif result:
            passed += 1
        else:
            failed.append(name)

    assessable = len(PCI_REQUIREMENTS) - len(not_assessed)
    ratio = (passed / assessable) if assessable else 0.0
    return ratio, failed, not_assessed


# =============================================================================
# THE THREE PENALTY VECTORS
# =============================================================================

def calculate_pci_dss_assessment(
    pci_records,
    annual_revenue,
    p_value,
    recent_breach,
    records_exposed,
    security_data,
    assessment_data,
):
    """
    Expected annual PCI DSS fines & assessments (USD).

    Returns (total, breakdown_dict).

    Exposed-card count follows the count_num_records() convention:
      recentBreach == 'Yes'  -> use the actual recordsExposed
      otherwise              -> pci_records * p_value
    """
    cards = float(pci_records or 0)
    if cards <= 0:
        return 0.0, {}

    if str(recent_breach).strip() == 'Yes':
        exposed = float(records_exposed or 0)
        breach_weight = 1.0
    else:
        exposed = cards * float(p_value)
        breach_weight = float(p_value)

    if exposed <= 0:
        return 0.0, {}

    tier = derive_merchant_tier(annual_revenue, cards)
    compliance_ratio, failed_reqs, not_assessed = assess_pci_compliance(
        security_data, assessment_data)

    # -- Vector 1: card replacement (deterministic) ---------------------------
    reissuance = exposed * REISSUANCE_FEE_PER_CARD

    # -- Vector 2: fraud recovery (Beta collapsed to its mean) ----------------
    fraud_recovery = exposed * E_FRAUD_REALIZATION * AVG_FRAUDULENT_CHARGE

    # -- Vector 3: non-compliance x remediation (Poisson collapsed to its mean)
    # Brands generally waive assessments against merchants validated compliant at
    # the time of breach, and escalate against those found non-compliant.
    if compliance_ratio >= 0.95:
        remediation_months = 0.0
    else:
        remediation_months = (
            REMEDIATION_MONTHS_BASE
            + REMEDIATION_MONTHS_MAX_EXTRA * (1.0 - compliance_ratio)
        )
    noncompliance = (
        MONTHLY_NONCOMPLIANCE_FINE[tier] * remediation_months * breach_weight
    )

    # -- Case management + PFI ------------------------------------------------
    # Case management scales with non-compliance; PFI is a fixed engagement that
    # only fires on a confirmed compromise.
    compliance_mult = 1.50 - 1.25 * compliance_ratio   # 0.25x compliant -> 1.5x not
    case_management = exposed * CASE_MANAGEMENT_FEE_PER_CARD * compliance_mult
    pfi = PFI_COST[tier] * breach_weight

    breakdown = {
        'Card Reissuance': round(reissuance, 2),
        'Fraud Recovery': round(fraud_recovery, 2),
        'Non-Compliance Assessments': round(noncompliance, 2),
        'Case Management': round(case_management, 2),
        'PFI Forensic Investigation': round(pfi, 2),
    }
    total = round(sum(breakdown.values()), 2)

    meta = {
        'merchant_tier': tier,
        'compliance_ratio': round(compliance_ratio, 3),
        'requirements_assessed': len(PCI_REQUIREMENTS) - len(not_assessed),
        'failed_requirements': failed_reqs,
        'not_assessed_requirements': not_assessed,
        'exposed_cards': round(exposed, 1),
        'remediation_months': round(remediation_months, 2),
    }
    return total, {'components': breakdown, 'meta': meta}
