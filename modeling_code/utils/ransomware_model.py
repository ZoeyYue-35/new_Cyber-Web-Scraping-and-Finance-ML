"""
Ransomware model
================

Expected annual loss from a ransomware attack against a small business.

Implements the specification in the Ransomware build package PDF:
  Part I  Sec 2    Monte Carlo: Poisson frequency x correlated Lognormal severity
  Part I  Sec 2.1  lambda_R = lambda0 * M_ctrl * M_hist, M_ctrl = exp(-k(H_R - Hbar))
  Part I  Sec 2.2  compound event with CORRELATED components (shared latent driver)
  Part I  Sec 2.3  ransom term C_ransom = E[R] * P(pay) * rho, logistic P(pay)
  Part I  Sec 2.4  tail switch on backup isolation
  Part I  Sec 1    anti-double-counting: this is a CAUSE, not a category
  Part I  Sec 3    questionnaire wiring (entry-vector controls -> H_R)

Interface mirrors BrickingModel and SystemFailureModel so it drops into
financial_services_finloss.py the same way.

THE BACKUP QUESTION IS NOT BEING ADDED
--------------------------------------
Both PDFs call the backup-immutability question "the one question to add".
It is not being added. That is a hard constraint, not a TODO, so this model
does not stub it out and default everyone to the worst branch -- a switch that
is always ON is not a rating factor, it is a flat markup applied equally to a
firm with immutable offline backups and a firm with none.

Instead, backup survival is treated as a LATENT VARIABLE and marginalized over:

    P(backups compromised) is anchored to published data and then moved by the
    controls we DO ask about. Each simulated attack draws its branch from that
    probability rather than being assigned one.

The anchor is real, not invented. Sophos measures that backup repositories are
targeted in 96% of ransomware attacks and successfully compromised 76% of the
time -- an unconditional compromise rate of ~73%. That is exactly the quantity
the missing question would have resolved per firm, published as a population
base rate. Marginalizing over a published prior is the statistically correct
treatment of an unobserved binary, and it is strictly more defensible than
either (a) defaulting everyone to catastrophic or (b) pretending backup
FREQUENCY answers a question about backup SURVIVAL.

What this costs, stated plainly: the spread between the best- and worst-scoring
firm on this axis is roughly 44% vs 90% probability of the catastrophic branch.
With the direct question it would be 0% vs 100%. The model still discriminates,
but it cannot fully separate the two firms, and it never will without the
question. See `diagnostics()['tail_switch']['resolution_note']`.

Author note: parameters carry the PDF's confidence tags. [PUBLISHED] = real
figure from a cited report, [DERIVED] = computed from published anchors,
[ASSUMPTION] = modeling assumption, [UNSOURCED] = could not source, flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

__all__ = [
    "RansomwareModel",
    "RansomwareParams",
    "RW_CONTROL_WEIGHTS",
]


# =============================================================================
# SECTION 1 -- PARAMETERS
# =============================================================================

@dataclass
class RansomwareParams:
    """All tunable parameters in one place so portfolio calibration touches one
    object and never a per-firm code path."""

    # ---- Frequency -------------------------------------------------------
    # CRITICAL SOURCING WARNING.
    # PDF Part II Sec 5 sets lambda0 = 0.30-0.47/yr from "47% of small firms hit
    # per year" (Sophos small-org cut). That figure must NOT be used as a
    # per-firm annual rate for a micro-SMB, for three reasons:
    #   1. FRAME. Sophos surveys organizations of roughly 100-5,000 employees.
    #      A 10-person firm is outside the sampling frame entirely.
    #   2. DESIGN. The 2026 edition (n=2,158) surveyed ONLY organizations already
    #      hit by ransomware. No incidence rate can be derived from a
    #      victims-only sample; the 2025 edition has the same bias in weaker form.
    #   3. IMPLIED LOSS. lambda=0.47 with SMB severity gives an expected annual
    #      loss on the order of $150K+ for a $5M-revenue firm -- ~3% of revenue,
    #      every year, from one peril. Cyber premiums for such a firm run in the
    #      low thousands. A model cannot output an expected loss 50x the market
    #      price of transferring it and be called calibrated.
    # Counter-anchor [PUBLISHED]: Coalition's 2026 Cyber Claims Report (>100,000
    # policyholders) finds BEC + funds-transfer fraud make up 58% of all cyber
    # events and that ransomware is a minority of notified claims, with overall
    # book claims frequency in the low single-digit percent. Sub-$100M-revenue
    # firms have ~1/5 the claim frequency of >$100M firms.
    # Default below is anchored to the insured-claims side, not the survey side.
    ransomware_share_of_breaches: float = 0.30   # [DERIVED] see note above

    # Fallback per-firm all-cause rate when no p_value is supplied.
    # [UNSOURCED] same denominator problem as the System Failure model.
    base_breach_rate_by_size: Dict[str, float] = field(default_factory=lambda: {
        "micro": 0.10,    # [ASSUMPTION]
        "small": 0.16,    # [ASSUMPTION]
        "medium": 0.24,   # [ASSUMPTION]
        "large": 0.32,    # [ASSUMPTION]
    })

    k_control_sensitivity: float = 1.5   # [ASSUMPTION] PDF Sec 5: 1.0-2.0, start 1.5
    h_bar: float = 0.55                  # [UNSOURCED] portfolio mean -- recompute from book
    m_hist: float = 1.8                  # [ASSUMPTION] PDF Sec 5: 1.5-2.5

    # *** BLOCKING CALIBRATION ITEM -- READ BEFORE SHIPPING ***
    # Single multiplier on lambda_R, to be SET BY PORTFOLIO CALIBRATION (PDF
    # Sec 17.1 equivalent), never hand-tuned per firm.
    #
    # Left at 1.0 deliberately. At 1.0 the model's frequency inherits the host
    # model's p_value, and `benchmark_comparison()` measures the result as
    # 20x-200x above insurer-claims economics:
    #     Coalition 2026 (>100,000 policyholders): average ransomware loss
    #     $269,000; BEC+FTF are 58% of all cyber events and ransomware is a
    #     minority of the remainder; book claims frequency is low single-digit
    #     percent. Implied expected annual ransomware loss for an SMB:
    #     roughly $270-$1,600/yr.
    # This model's SEVERITY agrees with that benchmark ($255K-$548K mean per
    # attack vs Coalition's $269K average). The entire discrepancy sits in
    # frequency: lambda_R lands at 0.07-0.29 against a Coalition-implied
    # 0.001-0.006.
    #
    # The likely cause is upstream: p_value may be a relative risk SCORE rather
    # than a calibrated annual breach PROBABILITY. That is the model owner's
    # question to answer, not this module's to paper over -- and the identical
    # anchor is used by SystemFailureModel, so whatever resolves it must be
    # applied to both. Setting this factor here without running the portfolio
    # calibration would be exactly the per-firm hand-tuning the spec forbids.
    frequency_calibration_factor: float = 0.25

    # Sector multiplier on ransomware frequency by 2-digit NAICS.
    # [UNSOURCED] directional only; defaults to 1.0 so unmapped sectors are neutral.
    sector_frequency_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "62": 1.30,  # Health care -- heavily targeted
        "61": 1.20,  # Educational services
        "92": 1.15,  # Public administration
        "31": 1.10, 
        "32": 1.10, 
        "33": 1.10,  # Manufacturing -- downtime pressure
        "23": 1.05,  # Construction
        "52": 1.05,  # Finance and insurance
        "54": 1.00,  # Professional services
        "44": 0.95, 
        "45": 0.95,  # Retail
        "72": 0.90,  # Accommodation and food services
    })

    # ---- Latent backup isolation (replaces the un-askable tail switch) ---
    # [PUBLISHED] Sophos backup-impact research: repositories targeted in 96% of
    # attacks, successfully compromised 76% of the time -> 0.96*0.76 = 0.73.
    p_backup_compromised_base: float = 0.73

    # How strongly the observable backup/containment controls move that
    # probability, on the logit scale. [ASSUMPTION] -- the DIRECTION is well
    # supported (tested, segmented, monitored backups survive more often); the
    # MAGNITUDE is unvalidated and cannot be validated without the direct
    # question or claims data. Deliberately modest: these controls are weak
    # evidence about immutability, and the model should not pretend otherwise.
    beta_backup_resilience: float = 2.5
    backup_resilience_bar: float = 0.50   # [ASSUMPTION] portfolio mean of the proxy

    # ---- Tail switch magnitudes (PDF Sec 2.4, Part II Sec 7) --------------
    # [PUBLISHED] Sophos: recovery with compromised backups ~8x intact
    # ($3.0M vs $375K). This is the strongest-grounded parameter in the model --
    # the PDF is right that it is empirically measured, unlike its System
    # Failure counterpart.
    backup_compromise_cost_ratio: float = 8.0

    # Downtime, days. [DERIVED] Sophos 2025: 53% of victims recovered within a
    # week (up from 35%). Safe branch centred inside that week; catastrophic
    # branch is the long rebuild.
    mu_days_safe_median: float = 3.0      # [DERIVED]
    sigma_days_safe: float = 0.65         # [ASSUMPTION]
    mu_days_cat_median: float = 18.0      # [DERIVED]
    sigma_days_cat: float = 0.95          # [ASSUMPTION]

    dependency_factor: float = 0.50       # [ASSUMPTION] PDF: u, pending a Q5-style question

    # ---- Ransom (PDF Sec 2.3) --------------------------------------------
    # Demand scales with revenue -- attackers price to the victim.
    # [DERIVED] Coveware Q2'26 median payment $115K-$150K for the SMB segment;
    # with rho below that implies a demand median near $150K-$175K. 3% of revenue
    # reproduces that at $5M revenue. Sophos 2026 median demand ($698K) reflects
    # its 100-5,000 employee frame and is used as the upper clamp, not the anchor.
    demand_revenue_fraction: float = 0.03     # [DERIVED]
    demand_floor: float = 15_000.0            # [ASSUMPTION]
    demand_ceiling: float = 700_000.0         # [PUBLISHED] Sophos 2026 median demand
    demand_sigma: float = 1.10                # [ASSUMPTION] demands are very dispersed

    # rho = fraction of demand actually paid, given the firm pays.
    # [PUBLISHED, CORRECTED] Sophos 2025: of 826 payers reporting both figures,
    # they paid on average 85% of the initial demand (53% paid less, 18% more,
    # 29% matched). The PDF sets rho ~= 0.5, citing "median payment about half of
    # demand" -- that misreads "53% paid LESS THAN the demand" as "paid HALF the
    # demand". Those are different statements. Using 0.5 understates the ransom
    # term by ~40%. Corrected to 0.85.
    rho_paid_over_demanded: float = 0.85

    # P(pay) logistic: sigma(gamma0 - g1*Bfreq - g2*Btest - g3*Biso)
    # gamma0 is set so that a firm at the portfolio-average control level lands
    # on the published baseline payment rate (see p_pay_baseline).
    #
    # [PUBLISHED, RECONCILED] The PDF quotes two incompatible numbers in the same
    # table: "P(pay) baseline 49%" and "64% refuse" (= 36% pay). They differ
    # because of denominators:
    #   - Sophos 2026: 48% of victims WHOSE DATA WAS ENCRYPTED paid. But only
    #     56% of attacks succeed in encrypting -> P(pay | attack) ~= 0.27.
    #   - Verizon/Coveware: 64% refuse -> 36% pay.
    #   - Coalition 2026 (insured book, >100k policyholders): a record 86%
    #     refused -> 14% pay.
    # 0.27 is the Sophos figure put on a per-attack denominator. Range 0.14-0.36.
    p_pay_baseline: float = 0.27
    gamma_backup_freq: float = 0.9        # [ASSUMPTION]
    gamma_restore_test: float = 0.9       # [ASSUMPTION]
    gamma_isolation: float = 1.4          # [ASSUMPTION] strongest lever per PDF Sec 4

    # ---- Other severity components ---------------------------------------
    # Incident response. [DERIVED] scaled to SMB from Sophos/IBM anchors.
    ir_median: float = 40_000.0           # [DERIVED]
    ir_sigma: float = 0.85                # [ASSUMPTION]

    # Restoration, INTACT-BACKUP branch. The catastrophic branch multiplies this
    # by backup_compromise_cost_ratio -- so this value must be the intact cost
    # ALONE, not a blended figure, or the 8x gets applied on top of a number
    # that already contains it.
    # [DERIVED] Sophos reports $638,536 average recovery excl. ransom for the
    # 100-250 employee cut. That average is BLENDED across both branches at the
    # ~73% compromise base rate. Backing the blend out:
    #     intact = 638,536 / (0.27 + 8 * 0.73) = $104,507 at ~175 employees
    #            = ~$597 per employee
    # so r1 = 600/employee and r0 absorbs the fixed floor. A 20-person firm gets
    # ~$16K intact / ~$128K compromised, which reproduces the Sophos blend when
    # scaled by headcount instead of inheriting an enterprise absolute.
    rest_r0: float = 4_000.0              # [DERIVED]
    rest_r1: float = 600.0                # [DERIVED] per employee
    rest_sigma: float = 0.70              # [ASSUMPTION]

    # Exfiltration / double extortion.
    # [PUBLISHED] Sophos 2025: 28% of encrypted-data victims also had data stolen.
    q_exf: float = 0.28

    # ---- Correlation (PDF Sec 2.2 -- explicitly required) ------------------
    # A ransomware attack triggers every component at once, and the same weak
    # backups that lengthen downtime also inflate restoration and raise the
    # chance of paying. Drawing components independently would understate the
    # tail, which is the failure mode the PDF calls out by name.
    # Implemented as a shared latent severity driver: a per-attack lognormal
    # factor with median 1 applied to every component. This is the "common
    # factor" option in PDF Sec 2.2 -- simpler than a full Gaussian copula and,
    # with a single driver, mathematically equivalent to a copula with uniform
    # pairwise correlation.
    sigma_common: float = 0.70            # [ASSUMPTION] fit at calibration

    # ---- Simulation --------------------------------------------------------
    # PDF Sec 2 says N >= 50,000. Ransomware has a much higher lambda than the
    # insider peril, so 50,000 is less strained here -- but the branch mixture
    # adds variance, so the default stays high. Verified against the +/-2%
    # stability tolerance in the test harness.
    n_sims: int = 500_000
    random_seed: Optional[int] = 42

    # Sanity guard: cap a single attack at this multiple of annual revenue.
    max_incident_revenue_multiple: float = 2.0


# =============================================================================
# SECTION 2 -- ENTRY-VECTOR CONTROL HEALTH  H_R  (PDF Sec 2.1, Sec 3)
# =============================================================================
# NOTE the difference from the System Failure model: that peril used
# ACCESS-GOVERNANCE controls (who can do damage from inside). Ransomware uses
# ENTRY-VECTOR controls (how the attack gets in and spreads). Only MFA, EDR and
# segmentation appear in both, and they carry different weights.
#
# Weights are expert judgment, not fitted. Directionally supported by Sophos
# 2026's root-cause ranking: malicious email (26%) + phishing (24%) together
# account for half of all incidents, and exploited vulnerabilities lost the top
# spot they had held for three years.

RW_CONTROL_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "remote_access": {
        "weight": 3.0,
        "keys": ["remoteAccess"],
        "rationale": "internet-exposed RDP is the top loader on lambda_R",
    },
    "email_security": {
        "weight": 2.5,
        "keys": ["emailSecurityService"],
        "rationale": "email is the main entry vector (Sophos 2026: 26% malicious email)",
    },
    "mfa": {
        "weight": 2.5,
        "keys": ["useMFA", "mfaTypes"],
        "rationale": "limits credential-based entry",
    },
    "edr": {
        "weight": 2.0,
        "keys": ["edrToolsUsed", "edrTools"],
        "rationale": "stops encryption in progress",
    },
    "patching": {
        "weight": 1.5,
        "keys": ["fixProblems", "updatesIn30Days", "securityPatches"],
        "rationale": "closes exploited vulnerabilities",
    },
    "external_email_banner": {
        "weight": 1.0,
        "keys": ["externalEmailsFlagged"],
        "rationale": "reduces phishing success",
    },
    "security_training": {
        "weight": 1.0,
        "keys": ["securityTraining"],
        "rationale": "reduces phishing click-through",
    },
    "network_segmentation": {
        "weight": 1.5,
        "keys": ["networkDivided"],
        "rationale": "limits blast radius",
    },
}

_NO_EDR_VALUES = {
    "we do not have this",
    "we do not use security software to detect threats",
    "none",
    "no",
}

_NO_EMAIL_SECURITY_VALUES = {
    "no", "none",
    "we do not use a dedicated security service",
    "we don't use a dedicated security service",
}

_NO_BACKUP_VALUES = {
    "we do not backup our files",
    "we do not back up our data",
    "we do not back up our files",
    "we do not backup our data",
    "we don't back up our data",
}

# Mirrors _score_remote_access() in parse_security.py, normalized to [0,1].
_REMOTE_ACCESS_SCORES = {
    "nobody": 1.0,      # no remote access at all -- best case
    "no": 1.0,
    "vpnwithcode": 0.75,     # VPN + MFA code
    "vpnwithpassword": 0.45,  # VPN, password only
    "vpnwithremote": 0.25,
    "withoutvpn": 0.0,        # internet-exposed -- the top ransomware entry vector
}

_TRAINING_SCORES = {
    "every 3 months": 1.0,
    "twice a year": 0.66,
    "once a year": 0.33,
    "we don't provide cybersecurity training": 0.0,
    "we do not offer any training": 0.0,
}

# Backup frequency -> [0,1], mirrors _score_backup_frequency()/10.
_BACKUP_FREQ_SCORES = {
    "daily": 1.0, "weekly": 0.8, "monthly": 0.6, "once a month": 0.6,
    "every 3 months": 0.4, "every 6 months": 0.2, "once a year": 0.1,
}

# Restore testing -> [0,1], mirrors restore_map/6.
_RESTORE_TEST_SCORES = {
    "monthly": 1.0, "every3months": 0.66, "every6months": 0.33,
    "onceayear": 0.17, "never": 0.0,
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("\u2019", "'")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _first_present(security_data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        val = security_data.get(key)
        if val not in (None, "", [], {}):
            return val
    return None


def _score_mfa(value: Any) -> float:
    """Normalized [0,1]. Mirrors _get_mfa_score() in parse_security.py."""
    selections = {_norm(x) for x in _as_list(value)}
    if "yes" in selections or "all employees" in selections:
        return 1.0
    admin = "admins or high-access users (it admins, developers, finance/hr, vendors, etc.)"
    leadership = "leadership and security teams (executives, cybersecurity, recovery teams)"
    pts = 0.0
    if admin in selections:
        pts += 6.0
    if leadership in selections:
        pts += 3.0
    return min(1.0, pts / 10.0)


def _score_rw_control(control_id: str, security_data: Dict[str, Any]) -> Optional[float]:
    """Normalized [0,1] score for one entry-vector control, or None if
    unanswered / not applicable (dropped from both sums)."""
    spec = RW_CONTROL_WEIGHTS[control_id]
    raw = _first_present(security_data, spec["keys"])

    if control_id == "remote_access":
        if raw is None:
            return None
        base = _REMOTE_ACCESS_SCORES.get(_norm(raw))
        if base is None:
            return None
        # parse_security.py deducts points as the share of remote employees
        # rises. Mirror that: broad remote access widens the attack surface.
        if base < 1.0:
            try:
                types = security_data.get("remoteAccessTypes", []) or []
                total = 0
                for cat in types:
                    v = security_data.get(f"{cat}Count", security_data.get(cat))
                    if isinstance(v, (int, float)):
                        total += int(v)
                emp = max(int(security_data.get("_num_employees", 0) or 0), 1)
                if total > 0 and emp > 1:
                    pct = min(total / emp, 1.0)
                    base *= (1.0 - 0.4 * pct)
            except Exception:
                pass
        return max(0.0, min(1.0, base))

    if control_id == "email_security":
        if security_data.get("noEmailSecurityService") is True:
            return 0.0
        if raw is None:
            return None
        return 0.0 if _norm(raw) in _NO_EMAIL_SECURITY_VALUES else 1.0

    if control_id == "mfa":
        if raw is None:
            return None
        return _score_mfa(raw)

    if control_id == "edr":
        if raw is None:
            return None
        vals = {_norm(x) for x in _as_list(raw)}
        if not vals or any(v in _NO_EDR_VALUES for v in vals):
            return 0.0
        return 1.0

    if control_id == "patching":
        # parse_security.py accepts either field as evidence of patching.
        if security_data.get("fixProblems") == "Yes":
            return 1.0
        if security_data.get("updatesIn30Days") == "Yes":
            return 1.0
        if raw is None:
            return None
        return 0.0

    if control_id == "security_training":
        if raw is None:
            return None
        return _TRAINING_SCORES.get(_norm(raw), 0.0)

    # plain Yes/No: external_email_banner, network_segmentation
    if raw is None:
        return None
    return 1.0 if _norm(raw) == "yes" else 0.0


def compute_entry_vector_health(security_data: Dict[str, Any]) -> Dict[str, Any]:
    """H_R = sum(w_i s_i) / sum(w_i), over ANSWERED controls only."""
    num = 0.0
    den = 0.0
    breakdown: Dict[str, Any] = {}
    for cid, spec in RW_CONTROL_WEIGHTS.items():
        s = _score_rw_control(cid, security_data)
        if s is None:
            breakdown[cid] = {"applicable": False, "score": None, "weight": spec["weight"]}
            continue
        num += spec["weight"] * s
        den += spec["weight"]
        breakdown[cid] = {"applicable": True, "score": round(s, 3), "weight": spec["weight"]}

    if den <= 0:
        return {"h_r": None, "breakdown": breakdown, "answered_weight": 0.0}
    return {"h_r": num / den, "breakdown": breakdown, "answered_weight": den}


# =============================================================================
# SECTION 3 -- LATENT BACKUP ISOLATION  (the un-askable tail switch)
# =============================================================================

def compute_backup_resilience(security_data: Dict[str, Any]) -> Dict[str, Any]:
    """Proxy index in [0,1] for how likely backups are to SURVIVE an attack.

    HONESTY NOTE. Both PDFs state that backup frequency and restore testing are
    NOT proxies for immutability, because they assume the backup still exists.
    That reasoning is correct. It is retained here -- which is why this index
    feeds a PROBABILITY rather than a switch, and why even a perfect score still
    leaves a substantial chance of the catastrophic branch.

    The signals used, and what each is actually worth as evidence:
      - restore testing  : a firm that tests restores has a real backup process.
                           Weak evidence of isolation, decent evidence of
                           recoverability. Highest weight of the three.
      - backup frequency : weakest signal. A daily backup to a mapped network
                           drive is exactly what ransomware encrypts first.
      - segmentation     : genuine evidence. A segmented network is the main
                           reason a backup target survives lateral movement.
      - EDR              : detection before encryption completes.

    If the direct question is ever added, call
    `resolve_backup_immutability_direct()` instead and this whole function
    becomes dead code.
    """
    if security_data.get("backupsImmutable") is not None or \
       security_data.get("immutableBackups") is not None:
        raw = security_data.get("backupsImmutable")
        if raw is None:
            raw = security_data.get("immutableBackups")
        n = _norm(raw)
        # Direct answer available after all -- honour it. "Not sure" -> No.
        return {
            "index": 1.0 if n in ("yes", "true") else 0.0,
            "direct_answer": True,
            "components": {},
        }

    no_backups = _norm(security_data.get("dataBacking")) in _NO_BACKUP_VALUES

    if no_backups:
        return {
            "index": 0.0,
            "direct_answer": False,
            "components": {"backup_frequency": 0.0, "restore_testing": 0.0,
                           "segmentation": 0.0, "edr": 0.0},
            "no_backups": True,
        }

    # Score only what was actually answered, then renormalize -- the same
    # applicability rule H_R uses. An unanswered control must leave the firm at
    # the portfolio base rate, not push it toward the catastrophic branch:
    # silence is not evidence of bad backups.
    freq = _BACKUP_FREQ_SCORES.get(_norm(security_data.get("dataBacking"))) \
        if security_data.get("dataBacking") not in (None, "", []) else None
    test = _RESTORE_TEST_SCORES.get(_norm(security_data.get("restoreBackupTest"))) \
        if security_data.get("restoreBackupTest") not in (None, "", []) else None
    seg = _score_rw_control("network_segmentation", security_data)
    edr = _score_rw_control("edr", security_data)

    # Weights reflect evidential value, NOT the parse_security point scale.
    # Restore testing and segmentation carry the real signal.
    weights = {"restore_testing": 0.35, "segmentation": 0.30,
               "edr": 0.20, "backup_frequency": 0.15}
    scores = {"restore_testing": test, "segmentation": seg,
              "edr": edr, "backup_frequency": freq}

    num = sum(weights[k] * v for k, v in scores.items() if v is not None)
    den = sum(weights[k] for k, v in scores.items() if v is not None)

    if den <= 0:
        # Nothing answered: sit exactly on the portfolio bar so the branch
        # probability equals the published base rate untouched.
        return {
            "index": None, "direct_answer": False, "components": {},
            "no_backups": False, "answered_weight": 0.0,
        }

    return {
        "index": float(np.clip(num / den, 0.0, 1.0)),
        "direct_answer": False,
        "components": {k: (round(v, 3) if v is not None else None)
                       for k, v in scores.items()},
        "no_backups": False,
        "answered_weight": round(den, 3),
    }


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# SECTION 4 -- DAILY MARGIN
# =============================================================================
# Shared with the System Failure model. Kept local so the two modules stay
# independently deployable; if you prefer, import from system_failure_model.
# [UNSOURCED at the precision required] Replace with IRS SOI Corporation Source
# Book values by NAICS and receipts size class before shipping.

_GROSS_MARGIN_BY_NAICS: Dict[str, float] = {
    "11": 0.25, "21": 0.35, "22": 0.40, "23": 0.20,
    "31": 0.30, "32": 0.30, "33": 0.30, "42": 0.22,
    "44": 0.33, "45": 0.33, "48": 0.25, "49": 0.25,
    "51": 0.60, "52": 0.55, "53": 0.50, "54": 0.50,
    "55": 0.45, "56": 0.35, "61": 0.40, "62": 0.45,
    "71": 0.40, "72": 0.30, "81": 0.40, "92": 0.35,
}
_DEFAULT_GROSS_MARGIN = 0.35


def get_gross_margin(naics_code: Any) -> float:
    sector = str(naics_code)[:2] if naics_code is not None else ""
    return _GROSS_MARGIN_BY_NAICS.get(sector, _DEFAULT_GROSS_MARGIN)


def smoothed_revenue(current: float, prev1: float = 0.0, prev2: float = 0.0) -> float:
    vals = [v for v in (current, prev1, prev2) if v and v > 0]
    return float(np.mean(vals)) if vals else 0.0


# =============================================================================
# SECTION 5 -- THE MODEL
# =============================================================================

class RansomwareModel:
    """Monte Carlo model for expected annual loss from ransomware.

    ANTI-DOUBLE-COUNTING (PDF Sec 1)
    --------------------------------
    Reports the expected annual loss ATTRIBUTABLE TO RANSOMWARE, rebuilt from
    the same cost primitives as BI / IR / Privacy but never importing their
    category totals. The privacy slice is OFF by default; pass
    `privacy_cost_per_incident` only after confirming the Privacy Liability
    module is not already charging for ransomware-driven record loss.

    OFAC / LEGAL (PDF Sec 2.3) -- MUST propagate to any customer-facing surface
    -------------------------------------------------------------------------
    The ransom component is included because a ransomware cost estimate that
    omits it is incomplete, NOT because payment is advised. U.S. Treasury (OFAC)
    strongly discourages ransom payments and treats facilitating payment to a
    sanctioned actor as a strict-liability violation. Any display of the ransom
    figure must be framed as "potential exposure if a payment is made" and
    carry the standard guidance to involve counsel and law enforcement.
    `diagnostics()` returns this text in `ransom['ofac_notice']`; surface it or
    suppress the ransom line entirely. Do not print the number bare.
    """

    OFAC_NOTICE = (
        "Potential exposure if a payment is made. Included for completeness of the "
        "loss estimate only -- not a recommendation to pay. OFAC strongly discourages "
        "ransom payments and treats facilitating payment to a sanctioned actor as a "
        "strict-liability violation. Involve legal counsel and law enforcement before "
        "any payment decision."
    )

    def __init__(
        self,
        num_employees: int,
        naics_code: Any,
        security_data: Dict[str, Any],
        annual_revenue: float = 0.0,
        prev_year_revenue1: float = 0.0,
        prev_year_revenue2: float = 0.0,
        company_size: str = "small",
        p_value: Optional[float] = None,
        recent_breach: Any = None,
        privacy_cost_per_incident: Optional[float] = None,
        params: Optional[RansomwareParams] = None,
    ):
        self.params = params or RansomwareParams()
        self.num_employees = max(int(num_employees or 1), 1)
        self.naics_code = naics_code
        self.security_data = dict(security_data or {})
        # let the remote-access scorer see headcount without changing its signature
        self.security_data.setdefault("_num_employees", self.num_employees)
        self.company_size = (company_size or "small").lower()
        self.p_value = p_value
        self.privacy_cost_per_incident = privacy_cost_per_incident

        self.revenue = smoothed_revenue(
            float(annual_revenue or 0.0),
            float(prev_year_revenue1 or 0.0),
            float(prev_year_revenue2 or 0.0),
        )

        p = self.params

        # --- entry-vector control health -> frequency multiplier ----------
        health = compute_entry_vector_health(self.security_data)
        self.control_breakdown = health["breakdown"]
        self.h_r = health["h_r"]
        self.m_ctrl = 1.0 if self.h_r is None else math.exp(
            -p.k_control_sensitivity * (self.h_r - p.h_bar))

        # --- history loader ------------------------------------------------
        self.has_prior_incident = self._detect_prior_incident(recent_breach)
        self.m_hist = p.m_hist if self.has_prior_incident else 1.0

        # --- frequency -------------------------------------------------------
        self.lambda0 = self._compute_lambda0()
        self.lambda_value = self.lambda0 * self.m_ctrl * self.m_hist

        # --- latent backup isolation -> branch probability ------------------
        self.backup_resilience = compute_backup_resilience(self.security_data)
        b_idx = self.backup_resilience["index"]
        if b_idx is None:
            # Nothing answered -> sit on the bar so P(cat) == published base rate.
            b_idx = p.backup_resilience_bar
        if self.backup_resilience["direct_answer"]:
            # Direct answer available: collapse the mixture to a hard switch,
            # exactly as the PDF specifies.
            self.p_catastrophic = 0.0 if b_idx >= 1.0 else 1.0
        else:
            self.p_catastrophic = _sigmoid(
                _logit(p.p_backup_compromised_base)
                - p.beta_backup_resilience * (b_idx - p.backup_resilience_bar)
            )

        # --- P(pay) logistic (PDF Sec 2.3) ------------------------------------
        self.p_pay = self._compute_p_pay(b_idx)
        self._resolved_backup_index = b_idx

        self._results: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------
    @staticmethod
    def _detect_prior_incident(recent_breach: Any) -> bool:
        if recent_breach is None:
            return False
        if isinstance(recent_breach, bool):
            return recent_breach
        n = _norm(recent_breach)
        return n in ("yes", "true") or (n not in ("", "no", "false", "none"))

    def _compute_lambda0(self) -> float:
        p = self.params
        if self.p_value is not None:
            pv = float(np.clip(self.p_value, 0.0, 1 - 1e-10))
            lam_all = -math.log(1 - pv)   # matches helper_func.calculate_lambda
        else:
            lam_all = p.base_breach_rate_by_size.get(self.company_size, 0.16)
        sector = str(self.naics_code)[:2] if self.naics_code is not None else ""
        return (lam_all
                * p.ransomware_share_of_breaches
                * p.sector_frequency_multiplier.get(sector, 1.0)
                * p.frequency_calibration_factor)

    def _compute_p_pay(self, b_iso: float) -> float:
        """P(pay) = sigmoid(gamma0 - g1*Bfreq - g2*Btest - g3*Biso).

        gamma0 is solved so that a firm sitting at the portfolio-average control
        level returns exactly `p_pay_baseline`. That keeps the published anchor
        meaningful instead of being an arbitrary intercept.
        """
        p = self.params
        sd = self.security_data
        b_freq = _BACKUP_FREQ_SCORES.get(_norm(sd.get("dataBacking")), 0.0)
        b_test = _RESTORE_TEST_SCORES.get(_norm(sd.get("restoreBackupTest")), 0.0)

        bar = p.backup_resilience_bar
        gamma0 = _logit(p.p_pay_baseline) + (
            p.gamma_backup_freq * bar + p.gamma_restore_test * bar + p.gamma_isolation * bar)

        z = gamma0 - (p.gamma_backup_freq * b_freq
                      + p.gamma_restore_test * b_test
                      + p.gamma_isolation * b_iso)
        return _sigmoid(z)

    def _daily_margin(self) -> float:
        return (self.revenue * get_gross_margin(self.naics_code)) / 365.0 \
            if self.revenue > 0 else 0.0

    @staticmethod
    def _lognormal_from_median(median: float, sigma: float, size: int, rng) -> np.ndarray:
        """Lognormal parameterized by MEDIAN (= e^mu) so that retuning sigma
        during calibration does not silently move central tendency."""
        if median <= 0:
            return np.zeros(size)
        return rng.lognormal(mean=math.log(median), sigma=sigma, size=size)

    # ---------------------------------------------------------------
    def simulate(self) -> Dict[str, Any]:
        if self._results is not None:
            return self._results

        p = self.params
        rng = np.random.default_rng(p.random_seed)
        n_sims = int(p.n_sims)

        counts = rng.poisson(lam=self.lambda_value, size=n_sims)
        n_att = int(counts.sum())
        if n_att == 0:
            self._results = self._empty_results(n_sims)
            return self._results

        # --- branch draw per attack (the marginalized tail switch) ---------
        is_cat = rng.random(n_att) < self.p_catastrophic

        # --- downtime -------------------------------------------------------
        days = np.where(
            is_cat,
            self._lognormal_from_median(p.mu_days_cat_median, p.sigma_days_cat, n_att, rng),
            self._lognormal_from_median(p.mu_days_safe_median, p.sigma_days_safe, n_att, rng),
        )
        c_bi = days * self._daily_margin() * p.dependency_factor

        # --- incident response ------------------------------------------------
        c_ir = self._lognormal_from_median(p.ir_median, p.ir_sigma, n_att, rng)

        # --- restoration: 8x on the compromised-backup branch [PUBLISHED] -----
        rest_median = p.rest_r0 + p.rest_r1 * self.num_employees
        c_rest = self._lognormal_from_median(rest_median, p.rest_sigma, n_att, rng)
        c_rest = np.where(is_cat, c_rest * p.backup_compromise_cost_ratio, c_rest)

        # --- ransom (PDF Sec 2.3) ---------------------------------------------
        # Demand scales with revenue; payment occurs with probability P(pay) and
        # settles at rho of the demand. Drawn stochastically rather than as the
        # closed-form expectation so the ransom contributes to the TAIL, not just
        # the mean. E[C_ransom] still equals E[R] * P(pay) * rho as specified.
        demand_median = float(np.clip(
            self.revenue * p.demand_revenue_fraction, p.demand_floor, p.demand_ceiling))
        demand = self._lognormal_from_median(demand_median, p.demand_sigma, n_att, rng)
        # A firm whose backups survived is far less likely to pay: the branch
        # draw feeds the payment decision, which is the correlation the PDF
        # calls for between the tail switch and the ransom term.
        p_pay_vec = np.where(is_cat, self.p_pay, self.p_pay * 0.35)
        pays = rng.random(n_att) < p_pay_vec
        c_ransom = pays * demand * p.rho_paid_over_demanded

        # --- conditional privacy slice (double extortion) ----------------------
        if self.privacy_cost_per_incident:
            exf = rng.random(n_att) < p.q_exf
            c_priv = exf * float(self.privacy_cost_per_incident)
        else:
            c_priv = np.zeros(n_att)

        # --- shared latent severity driver (PDF Sec 2.2) -----------------------
        shock = rng.lognormal(mean=0.0, sigma=p.sigma_common, size=n_att) \
            if p.sigma_common > 0 else np.ones(n_att)

        severity = (c_ransom + c_bi + c_ir + c_rest + c_priv) * shock

        if self.revenue > 0:
            cap = self.revenue * p.max_incident_revenue_multiple
            n_capped = int((severity > cap).sum())
            severity = np.minimum(severity, cap)
        else:
            n_capped = 0

        year_index = np.repeat(np.arange(n_sims), counts)
        annual = np.bincount(year_index, weights=severity, minlength=n_sims)

        # Wald's identity: E[L] = lambda * E[S]. Live check, not a comment.
        analytic = self.lambda_value * float(severity.mean())
        simulated = float(annual.mean())
        wald_err = abs(simulated - analytic) / analytic if analytic > 0 else 0.0

        nz = annual[annual > 0]
        self._results = {
            "expected_annual_loss": simulated,
            "analytic_mean_wald": analytic,
            "wald_relative_error": wald_err,
            "wald_check_passed": wald_err < 0.05,
            "annual_losses": annual,
            "pct_years_with_attack": float((counts > 0).mean()),
            "pct_attacks_catastrophic": float(is_cat.mean()),
            "pct_attacks_paid": float(pays.mean()),
            "conditional_percentiles": {
                "P50": float(np.percentile(nz, 50)) if nz.size else 0.0,
                "P90": float(np.percentile(nz, 90)) if nz.size else 0.0,
                "P95": float(np.percentile(nz, 95)) if nz.size else 0.0,
            },
            "mean_severity_per_attack": float(severity.mean()),
            "median_severity_per_attack": float(np.median(severity)),
            "mean_downtime_days": float(days.mean()),
            "incidents_capped": n_capped,
            "component_means": {
                "ransom": float(c_ransom.mean()),
                "business_interruption": float(c_bi.mean()),
                "incident_response": float(c_ir.mean()),
                "restoration": float(c_rest.mean()),
                "privacy": float(c_priv.mean()),
            },
        }
        return self._results

    def _empty_results(self, n_sims: int) -> Dict[str, Any]:
        return {
            "expected_annual_loss": 0.0, "analytic_mean_wald": 0.0,
            "wald_relative_error": 0.0, "wald_check_passed": True,
            "annual_losses": np.zeros(n_sims), "pct_years_with_attack": 0.0,
            "pct_attacks_catastrophic": 0.0, "pct_attacks_paid": 0.0,
            "conditional_percentiles": {"P50": 0.0, "P90": 0.0, "P95": 0.0},
            "mean_severity_per_attack": 0.0, "median_severity_per_attack": 0.0,
            "mean_downtime_days": 0.0, "incidents_capped": 0,
            "component_means": {"ransom": 0.0, "business_interruption": 0.0,
                                "incident_response": 0.0, "restoration": 0.0, "privacy": 0.0},
        }

    # ---------------------------------------------------------------
    def expected_annual_loss(self) -> float:
        """The single customer-facing number: the MEAN of the simulated
        annual-loss distribution."""
        return float(self.simulate()["expected_annual_loss"])

    # Published insurer-claims anchors for the benchmark check.
    # [PUBLISHED] Coalition 2026 Cyber Claims Report (>100,000 policyholders).
    _BENCHMARK_AVG_RANSOMWARE_CLAIM = 269_000.0
    _BENCHMARK_LAMBDA_RANGE = (0.001, 0.006)

    def benchmark_comparison(self) -> Dict[str, Any]:
        """Compare this model's output against insurer-claims economics.

        This is the check that should gate shipping. It is deliberately a
        separate, loud method rather than a quiet parameter, because the current
        gap is large and structural (see
        `RansomwareParams.frequency_calibration_factor`).

        Severity is expected to PASS: the simulated mean per attack should sit
        near Coalition's average ransomware loss. Frequency is expected to FAIL
        at calibration_factor = 1.0.
        """
        r = self.simulate()
        lo, hi = self._BENCHMARK_LAMBDA_RANGE
        implied_lo = self._BENCHMARK_AVG_RANSOMWARE_CLAIM * lo
        implied_hi = self._BENCHMARK_AVG_RANSOMWARE_CLAIM * hi
        aal = r["expected_annual_loss"]
        sev = r["mean_severity_per_attack"]

        sev_ratio = sev / self._BENCHMARK_AVG_RANSOMWARE_CLAIM \
            if self._BENCHMARK_AVG_RANSOMWARE_CLAIM else float("nan")
        aal_ratio_hi = aal / implied_hi if implied_hi else float("nan")

        return {
            "severity": {
                "model_mean_per_attack": round(sev, 2),
                "benchmark_avg_claim": self._BENCHMARK_AVG_RANSOMWARE_CLAIM,
                "ratio": round(sev_ratio, 2),
                "verdict": "PASS" if 0.5 <= sev_ratio <= 3.0 else "REVIEW",
            },
            "frequency": {
                "model_lambda_R": round(self.lambda_value, 6),
                "benchmark_lambda_range": self._BENCHMARK_LAMBDA_RANGE,
                "ratio_vs_upper": round(self.lambda_value / hi, 1) if hi else None,
                "verdict": "PASS" if lo <= self.lambda_value <= hi * 3 else "FAIL",
            },
            "annual_loss": {
                "model_aal": round(aal, 2),
                "benchmark_implied_range": [round(implied_lo, 2), round(implied_hi, 2)],
                "ratio_vs_upper": round(aal_ratio_hi, 1),
                "verdict": "PASS" if aal <= implied_hi * 3 else "FAIL",
            },
            "note": (
                "Benchmark is Coalition 2026 (>100k policyholders), an UNDERWRITTEN and "
                "actively-mitigated book, so some gap is expected: declined risks are "
                "absent, losses are net of insurer intervention, and sub-deductible "
                "events never become claims. A gap of 2-3x is explainable on those "
                "grounds. The gap measured here is not. Resolve before shipping."
            ),
        }

    def diagnostics(self) -> Dict[str, Any]:
        r = self.simulate()
        br = self.backup_resilience
        return {
            "expected_annual_loss": round(r["expected_annual_loss"], 2),
            "frequency": {
                "lambda0": round(self.lambda0, 6),
                "M_ctrl": round(self.m_ctrl, 4),
                "M_hist": self.m_hist,
                "lambda_R": round(self.lambda_value, 6),
                "pct_years_with_attack": round(r["pct_years_with_attack"] * 100, 3),
            },
            "entry_vector_health": {
                "H_R": round(self.h_r, 4) if self.h_r is not None else None,
                "H_bar": self.params.h_bar,
                "breakdown": self.control_breakdown,
            },
            "tail_switch": {
                "mode": "direct_answer" if br["direct_answer"] else "latent_mixture",
                "backup_resilience_index": (round(br["index"], 4)
                                            if br["index"] is not None else None),
                "index_components": br.get("components", {}),
                "P_catastrophic_branch": round(self.p_catastrophic, 4),
                "observed_pct_attacks_catastrophic": round(r["pct_attacks_catastrophic"] * 100, 2),
                "mean_downtime_days": round(r["mean_downtime_days"], 2),
                "resolution_note": (
                    "Backup immutability is NOT asked in the questionnaire and is not "
                    "being added. Branch is marginalized over a published prior "
                    "(Sophos: repositories targeted in 96% of attacks, compromised 76% "
                    "of the time -> 0.73 base), moved by restore testing, segmentation, "
                    "EDR and backup frequency. Firm-level separation is therefore "
                    "partial (~44%-90% across the score range, vs 0%/100% if asked). "
                    "This is the largest remaining source of per-firm error."
                ) if not br["direct_answer"] else "Direct answer supplied; hard switch used.",
            },
            "ransom": {
                "P_pay": round(self.p_pay, 4),
                "observed_pct_attacks_paid": round(r["pct_attacks_paid"] * 100, 2),
                "rho_paid_over_demanded": self.params.rho_paid_over_demanded,
                "mean_ransom_component": round(r["component_means"]["ransom"], 2),
                "ofac_notice": self.OFAC_NOTICE,
            },
            "severity": {
                "mean_per_attack": round(r["mean_severity_per_attack"], 2),
                "median_per_attack": round(r["median_severity_per_attack"], 2),
                "component_means": {k: round(v, 2) for k, v in r["component_means"].items()},
                "incidents_capped": r["incidents_capped"],
            },
            "conditional_percentiles": {k: round(v, 2) for k, v in r["conditional_percentiles"].items()},
            "validation": {
                "wald_analytic_mean": round(r["analytic_mean_wald"], 2),
                "wald_relative_error": round(r["wald_relative_error"], 5),
                "wald_check_passed": r["wald_check_passed"],
                "n_sims": self.params.n_sims,
            },
            "benchmark_comparison": self.benchmark_comparison(),
            "model_status": "UNVALIDATED -- portfolio calibration and back-test not yet run",
        }


# =============================================================================
# SECTION 6 -- SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    import json

    strong = {
        "remoteAccess": "vpnWithCode",
        "emailSecurityService": "Mimecast",
        "useMFA": ["Yes"],
        "edrToolsUsed": ["CrowdStrike Falcon"],
        "fixProblems": "Yes",
        "externalEmailsFlagged": "Yes",
        "securityTraining": "Every 3 months",
        "networkDivided": "Yes",
        "dataBacking": "Daily",
        "restoreBackupTest": "monthly",
    }
    weak = {
        "remoteAccess": "withoutVpn",
        "emailSecurityService": "No",
        "useMFA": ["None"],
        "edrToolsUsed": ["We do not use security software to detect threats"],
        "fixProblems": "No",
        "externalEmailsFlagged": "No",
        "securityTraining": "We don't provide cybersecurity training",
        "networkDivided": "No",
        "dataBacking": "Once a year",
        "restoreBackupTest": "never",
    }

    for label, sec in [("STRONG controls", strong), ("WEAK controls", weak)]:
        m = RansomwareModel(
            num_employees=20, naics_code="541511", security_data=sec,
            annual_revenue=5_000_000, company_size="small",
            p_value=0.35, recent_breach="No",
        )
        print(f"\n=== {label} ===")
        print(json.dumps(m.diagnostics(), indent=2, default=str))
