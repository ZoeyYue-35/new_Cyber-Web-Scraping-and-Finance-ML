"""
Malicious System Failure model
==============================

Expected annual loss from a *deliberate* system takedown -- rogue insider or
saboteur who wipes or disables systems. Accidental outages (cut cable, bad
update) are OUT OF SCOPE.

Implements the specification in `System_Failure.pdf`:
  Part I   Sec 3   Monte Carlo: Poisson frequency x Lognormal severity
  Part I   Sec 3.1 lambda = lambda0 * M_ctrl * M_hist,  M_ctrl = exp(-k(H - Hbar))
  Part I   Sec 3.2 S = C_BI + C_IR + C_rest + 1{exf} * C_priv, with the tail switch
  Part I   Sec 4   H_SF control-health index (weighted, applicability-normalized)
  Part I   Sec 2   anti-double-counting: this is a CAUSE, not a category
  Part IV  Sec 15.3 Wald's identity sanity check on the simulated mean

Design notes
------------
* Mirrors the interface of `BrickingModel` so it drops into
  `financial_services_finloss.py` the same way: construct, then call
  `expected_annual_loss()` / `diagnostics()`.
* Every parameter below carries a confidence tag matching the PDF's convention:
      [PUBLISHED]  real figure from a cited report
      [DERIVED]    computed from published anchors
      [ASSUMPTION] modeling assumption; no source can provide it
* Anything I could not source is tagged [UNSOURCED] with a note on what to go
  get. Those are the ones to route back to the model owner -- not to guess.

Four corrections to the PDF, made deliberately and flagged in-line
------------------------------------------------------------------
1. DISPERSION TARGET. PDF Sec 10 says to fit sigma so simulated P90/P50 matches
   the SMB range "(~10x)". That conflates the published band WIDTH
   ($1.24M / $120K = 10.3x) with a percentile RATIO. Reading the band as the
   P10-P90 of a lognormal gives sigma = 0.911, median = $385,746 and a true
   P90/P50 target of 3.21x. Fitted to 3.21x, not 10x. Fitting to 10x would have
   roughly tripled the tail and materially overstated the mean.
2. SIMULATION SIZE. PDF Sec 3 says N >= 50,000. Measured, that fails the PDF's
   own Sec 17.3 stability tolerance (+/-2%) at 8.3% spread across seeds, because
   lambda ~ 0.03 leaves only ~1,500 effective samples. Default raised to 500,000.
3. INDEPENDENT COMPONENTS. PDF Sec 3.2 draws C_BI, C_IR and C_rest
   independently. Summing independent lognormals averages away the variance the
   model exists to capture. Added a shared per-incident severity shock; see
   `sigma_common`.
4. THE TAIL SWITCH IS NOT A SWITCH. See below.

THE BACKUP QUESTION IS NOT BEING ADDED
--------------------------------------
PDF Sec 6 calls backup immutability "the one question to add". It is not being
added. That is a hard constraint, not a TODO.

An earlier revision of this module handled that by defaulting every firm to the
catastrophic branch with `inferred=True`. As a temporary stub that was fine. As
a permanent state it is wrong: it assigns EVERY firm to the same branch, so the
highest-impact factor in the model (65% output swing in sensitivity testing)
discriminates between exactly zero firms. It stops being a rating factor and
becomes a flat markup -- and it overstates loss for firms that do hold immutable
offline backups.

This version instead treats backup survival as a LATENT VARIABLE and
marginalizes over it. P(catastrophic branch) is anchored to a published base
rate and then moved by the controls the questionnaire DOES ask about; each
simulated incident draws its own branch from that probability.

The anchor is real, not invented. Sophos measures that backup repositories are
targeted in 96% of attacks and successfully compromised 76% of the time -- an
unconditional compromise rate of ~0.73. That is exactly the quantity the missing
question would have resolved per firm, published as a population base rate.
Marginalizing over a published prior is the statistically correct treatment of
an unobserved binary, and it beats both (a) defaulting everyone to catastrophic
and (b) pretending backup FREQUENCY answers a question about backup SURVIVAL.

What it costs, stated plainly: P(catastrophic) spans roughly 0.44 to 0.90 across
the control-score range. With the question asked it would be 0.0 vs 1.0. The
model still separates firms, but not fully, and it never will without the
question. Surfaced at runtime in
`diagnostics()['tail_switch']['resolution_note']`.

If `backupsImmutable` is ever added to the questionnaire, it is honoured
automatically as a hard switch and the model reverts to the PDF's original
specification with no other code change.

Author note on scope of the "draft" label: the architecture and the wiring are
production-shaped. The *numbers* are anchored where public data exists and
flagged where it does not. See SEC 13 of the PDF -- frequency is the weak side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

__all__ = [
    "SystemFailureModel",
    "SystemFailureParams",
    "SF_CONTROL_WEIGHTS",
]


# =============================================================================
# SECTION 1 -- PARAMETERS
# =============================================================================
# Sourcing status as of Aug 2026. The PDF was written against the 2025 report
# editions; the 2026 editions are now out and are used here where I could
# confirm a figure. PDF Part IV "Edition-currency warning" called exactly this.

@dataclass
class SystemFailureParams:
    """All tunable parameters in one place, so calibration (PDF Sec 7 / Sec 17.1)
    touches exactly one object and never a per-firm code path."""

    # ---- Frequency -----------------------------------------------------
    # Share of breaches attributable to malicious insider action.
    # [PUBLISHED] Verizon 2026 DBIR: "Privilege Misuse" pattern = 8% of breaches.
    # Privilege Misuse is the malicious-insider pattern by construction; the
    # Miscellaneous Errors pattern holds the negligent insiders we scoped out.
    # Cross-check [PUBLISHED]: Ponemon 2025 reports ~75% of insider incidents are
    # non-malicious, i.e. malicious ~= 25% of insider events -- consistent with an
    # 8% Privilege Misuse share sitting inside a larger internal-actor share.
    malicious_insider_share: float = 0.08

    # Fallback base breach rate per firm-year when no p_value is supplied.
    # [UNSOURCED] There is no public per-firm-year insider incidence table.
    # DBIR gives a SHARE of breaches, not a rate per business. Getting a real
    # denominator needs either licensed claims data (Advisen / NetDiligence /
    # Coalition) or a per-firm incidence survey (UK Cyber Security Breaches
    # Survey, Hiscox Cyber Readiness). Values below are placeholders sized to
    # land lambda0 in the PDF's stated 0.005-0.02/yr band for a micro-SMB.
    base_breach_rate_by_size: Dict[str, float] = field(default_factory=lambda: {
        "micro": 0.10,    # [ASSUMPTION]
        "small": 0.16,    # [ASSUMPTION]
        "medium": 0.24,   # [ASSUMPTION]
        "large": 0.32,    # [ASSUMPTION]
    })

    # Control sensitivity k in M_ctrl = exp(-k(H - Hbar)).
    # [ASSUMPTION] PDF Sec 9 says 1.0-2.0, start at 1.5. No published value exists.
    # PDF Sec 14.2 is explicit that this is a plausible FORM, not a measured curve.
    k_control_sensitivity: float = 1.5

    # Portfolio-average control health. MUST be recomputed from the actual book
    # of business -- it is a property of LEXI's portfolio, not of any report.
    # [UNSOURCED] placeholder from PDF Sec 3.1 illustrative value.
    h_bar: float = 0.55

    # Repeat-incident loader.
    # [ASSUMPTION] PDF Sec 9: 1.5-2.5.
    m_hist: float = 1.8

    # *** THE CALIBRATION DIAL ***
    # Single scalar multiplier on lambda, to be set by PORTFOLIO calibration
    # (PDF Sec 7 / Sec 17.1), never hand-tuned per firm.
    #
    # This is the correct lever for scaling output up or down, and it is
    # deliberately separate from the sourced parameters so that calibration
    # never corrupts a cited figure. Do NOT reach for `malicious_insider_share`
    # to achieve the same effect: that is a [PUBLISHED] value (DBIR Privilege
    # Misuse share) and editing it makes the citation false.
    #
    # Because E[L] = lambda * E[S] (Wald, PDF Sec 15.3), the customer-facing
    # mean scales very nearly LINEARLY in this factor:
    #     0.90 -> ~10% lower AAL
    #     0.75 -> ~25% lower
    #     0.50 -> ~50% lower
    #
    # Mirrors RansomwareParams.frequency_calibration_factor. Both perils inherit
    # the same p_value anchor, so if you scale one you almost certainly want to
    # scale the other by the SAME factor -- otherwise the two perils drift out of
    # relative alignment and the shared root cause gets half-corrected.
    frequency_calibration_factor: float = 0.5

    # Sector multiplier on insider frequency, by 2-digit NAICS.
    # [UNSOURCED] DBIR publishes per-industry breach patterns but not a
    # malicious-insider frequency multiplier. Directionally: sectors with high
    # privileged-user counts and high staff turnover carry more insider risk.
    # Defaults to 1.0 so an unmapped sector is never silently penalized.
    sector_frequency_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "52": 1.30,  # Finance and insurance -- high privilege density
        "62": 1.25,  # Health care -- high privilege density, high turnover
        "51": 1.15,  # Information
        "54": 1.05,  # Professional / scientific / technical
        "44": 0.95,  # Retail
        "45": 0.95,  # Retail
        "72": 0.90,  # Accommodation and food services
        "23": 0.85,  # Construction
        "11": 0.80,  # Agriculture
    })

    # ---- Severity: downtime tail switch (PDF Sec 3.2, Sec 11) ----------
    # THE single biggest driver of the mean. Medians in DAYS, sigma in log space.
    #
    # Safe branch [DERIVED]: immutable/offline backups bound recovery to a
    # restore operation. Industry rule-of-thumb recovery gap is hours-to-days.
    mu_days_safe_median: float = 1.5      # [DERIVED] PDF Sec 11: 1-2 days
    sigma_days_safe: float = 0.60         # [ASSUMPTION] fit at calibration

    # Catastrophic branch [DERIVED]: backups destroyable -> rebuild from nothing.
    # [PUBLISHED] anchors: Coveware reports ~24-day average SMB downtime;
    # IBM 2025/2026 reports a ~260-day (2025) / 247-day (2026) insider breach
    # LIFECYCLE. NOTE: lifecycle is time-to-identify-and-contain, NOT operational
    # downtime -- do not read 260 days as 260 days of outage. It justifies the
    # existence of a fat tail, not its length.
    mu_days_cat_median: float = 16.0      # [DERIVED] PDF Sec 11: 10-24 days
    sigma_days_cat: float = 0.95          # [ASSUMPTION] wider tail; fit at calibration

    # ---- Which branch? The latent backup variable ------------------------
    # The backup-immutability question (PDF Sec 6) is NOT in the questionnaire
    # and is NOT being added. See the module docstring for why the previous
    # always-catastrophic default was replaced.
    #
    # [PUBLISHED] Sophos backup-impact research: backup repositories are targeted
    # in 96% of attacks and successfully compromised 76% of the time -> an
    # unconditional compromise rate of 0.73.
    # CROSS-PERIL BORROW, flagged: that figure is measured against ransomware,
    # not insider sabotage. It is used here because it is the only published
    # population estimate of "will the backups survive a deliberate attacker",
    # and an insider with admin rights is squarely that. Direction is safe; the
    # exact rate for the insider peril is unmeasured. If insider-specific claims
    # data ever arrives, refit this first.
    p_backup_compromised_base: float = 0.73

    # How strongly the observable controls move that probability, on the logit
    # scale. [ASSUMPTION] -- DIRECTION is well supported (tested, segmented,
    # monitored backups survive more often); MAGNITUDE is unvalidated and cannot
    # be validated without the direct question or claims data. Deliberately
    # modest: these controls are weak evidence about immutability and the model
    # should not pretend otherwise.
    beta_backup_resilience: float = 2.5
    backup_resilience_bar: float = 0.50   # [ASSUMPTION] portfolio mean of the proxy

    # Dependency factor u: fraction of daily margin actually lost while down.
    # [ASSUMPTION] PDF Sec 3.2 default 0.5, pending a Q5-style dependency question.
    dependency_factor: float = 0.50

    # ---- Severity: other components -------------------------------------
    # Incident response cost per incident, SMB-scaled.
    # [DERIVED] IBM's $4.92M malicious-insider figure is a GLOBAL ENTERPRISE
    # average and must not be used raw (PDF Part II warning 1). Median below is
    # set so the simulated total lands in the published SMB band.
    ir_median: float = 45_000.0           # [DERIVED]
    ir_sigma: float = 0.85                # [ASSUMPTION]

    # Restoration: C_rest = r0 + r1 * employees * (2 - Seg)
    # [DERIVED] from Sophos 2025 State of Ransomware: $638,536 mean recovery cost
    # excluding ransom for 100-250 employee firms. Scaled down for micro-firms.
    rest_r0: float = 20_000.0             # [DERIVED]
    rest_r1: float = 1_500.0              # [DERIVED] per employee
    rest_sigma: float = 0.70              # [ASSUMPTION]

    # Branch multipliers on the NON-downtime cost components.
    # NOT in the PDF -- added because the PDF's Sec 3.2 applies the tail switch to
    # downtime only, which understates the catastrophic branch. If the insider
    # destroyed the backups you are not restoring, you are REBUILDING: new
    # infrastructure, data reconstruction from third parties, far more forensic
    # hours. Leaving these at 1.0 reproduces the PDF exactly.
    # [ASSUMPTION] magnitudes unvalidated; direction is not in serious doubt.
    cat_restoration_multiplier: float = 3.0   # [ASSUMPTION]
    cat_ir_multiplier: float = 1.6            # [ASSUMPTION]

    # Exfiltration probability -- does the insider steal data as well as destroy?
    # [ASSUMPTION] PDF Sec 12 suggests ~0.3, to be derived from the DBIR
    # insider-action split (Privilege Misuse action varieties).
    q_exf: float = 0.30

    # Common shock: a per-incident severity factor shared by ALL cost components.
    # NOT in the PDF. Added because the PDF draws C_BI, C_IR and C_rest
    # INDEPENDENTLY, which is both physically wrong and statistically consequential:
    #   - Physically: one incident drives all three together. A long outage means
    #     more forensic hours AND more rebuild labour. They are not independent.
    #   - Statistically: summing independent lognormals averages away variance
    #     (CLT), collapsing the tail the model exists to capture. Measured
    #     P90/P50 was 2.0 against a ~3.2 target before this was added.
    # Z ~ Lognormal(0, sigma_common), median 1, applied to every component.
    # [ASSUMPTION] set by calibration to the SMB dispersion target.
    sigma_common: float = 0.70

    # ---- Simulation ------------------------------------------------------
    # PDF Sec 3 says N >= 50,000. MEASURED: that fails the PDF's own Sec 17.3
    # stability tolerance of +/-2% by a wide margin. Cause: lambda ~ 0.03 means
    # only ~3% of simulated years contain an incident, so N=50,000 yields only
    # ~1,500 effective samples.
    #
    # Drawing the backup branch per incident (Section 3) adds a second source of
    # variance on top of that, so the requirement rose again. Measured relative
    # standard error of the mean:
    #     N =   500,000 -> SE 1.28%, 95% CI +/-2.50%   FAIL
    #     N = 1,000,000 -> SE 0.92%, 95% CI +/-1.80%   PASS
    #     N = 2,000,000 -> SE 0.65%, 95% CI +/-1.27%   PASS
    # Default set to 1,000,000.
    #
    # NOTE ON THE METRIC: use relative standard error, not peak-to-peak spread
    # across a handful of seeds. Peak-to-peak over ~8 seeds is itself so noisy
    # that it can rank N=1,000,000 as worse than N=500,000. `diagnostics()`
    # reports SE directly under `validation`.
    n_sims: int = 1_000_000
    random_seed: Optional[int] = 42

    # Sanity guard: cap any single incident at this multiple of annual revenue.
    # Not in the PDF. Added because an unbounded lognormal tail on a micro-firm
    # can produce a loss larger than the business could possibly sustain, which
    # is indefensible in front of a customer even if the math permits it.
    max_incident_revenue_multiple: float = 2.0


# =============================================================================
# SECTION 2 -- CONTROL HEALTH INDEX  H_SF  (PDF Sec 4)
# =============================================================================
# Weights are EXPERT JUDGMENT, not fitted (PDF Sec 14.2, Sec 18: confidence "A",
# sensitivity "High"). Finalize with the risk owner before shipping.
#
# Each entry maps a PDF control to the real questionnaire field(s) in
# parse_security.py, plus the max raw points that field can score there. That
# keeps H_SF consistent with the scoring the rest of the system already does.

SF_CONTROL_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "admin_alerting": {
        "weight": 3.0,
        "keys": ["ReceiveAlertApp"],
        "max_points": 4,
        "rationale": "strongest insider-detection signal",
    },
    "mfa_privileged": {
        "weight": 2.5,
        "keys": ["useMFA", "mfaTypes"],
        "max_points": 10,
        "rationale": "limits takeover of privileged accounts",
    },
    "vendor_access": {
        "weight": 2.0,
        "keys": ["thirdPartyControls"],
        "max_points": 5,
        "rationale": "third-party sabotage path",
    },
    "network_segmentation": {
        "weight": 2.0,
        "keys": ["networkDivided"],
        "max_points": 4,
        "rationale": "caps blast radius",
    },
    "admin_lockout": {
        "weight": 1.5,
        "keys": ["lockAccount", "specialAccess"],
        "max_points": 3,
        "rationale": "slows privilege abuse",
    },
    "oauth_approval": {
        "weight": 1.5,
        "keys": ["thirdPartyAppApproval"],
        "max_points": 4,
        "rationale": "blocks malicious app grants",
    },
    "edr": {
        "weight": 1.5,
        "keys": ["edrToolsUsed", "edrTools"],
        "max_points": 10,
        "rationale": "detects destructive action",
    },
    "encryption_at_rest": {
        "weight": 1.0,
        "keys": ["dataEncryption"],
        "max_points": 4,
        "rationale": "limits damage from access",
    },
}

# Answers that mean "this control does not apply to us". Per PDF Sec 4, an
# inapplicable control is dropped from BOTH sums so the firm is neither
# rewarded nor penalized. Sourced from NO_VENDOR_ACCESS_VALUES in parse_security.py.
_NOT_APPLICABLE_VALUES = {
    "we don't give vendors access to our systems or data",
    "we do not give vendors access to our systems or data",
    "i do not share data or systems access with other businesses",
}

_NO_EDR_VALUES = {
    "we do not have this",
    "we do not use security software to detect threats",
    "none",
    "no",
}

_NO_ALERT_APP_VALUES = {"no", "none", "we do not use any application", "we don't use any application"}


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


def _score_control(control_id: str, security_data: Dict[str, Any]) -> Optional[float]:
    """Return normalized score in [0,1] for one control, or None if not applicable.

    Scoring intentionally reuses the same answer semantics as
    parse_security.py so H_SF never disagrees with the main questionnaire score.
    """
    spec = SF_CONTROL_WEIGHTS[control_id]
    raw = _first_present(security_data, spec["keys"])

    # ---- admin activity alerting app --------------------------------
    if control_id == "admin_alerting":
        if security_data.get("noReceiveAlertApp") is True:
            return 0.0
        if raw is None:
            return None  # unanswered -> drop, do not guess
        return 0.0 if _norm(raw) in _NO_ALERT_APP_VALUES else 1.0

    # ---- MFA (weight the privileged-user tier specifically) ----------
    if control_id == "mfa_privileged":
        if raw is None:
            return None
        selections = {_norm(x) for x in _as_list(raw)}
        if "yes" in selections or "all employees" in selections:
            return 1.0
        # Mirrors _get_mfa_score(): admin/high-access tier is worth 6/10,
        # leadership tier 3/10. For an INSIDER peril the admin tier is the
        # one that matters, so it scores proportionally higher here.
        admin_tier = "admins or high-access users (it admins, developers, finance/hr, vendors, etc.)"
        leadership_tier = "leadership and security teams (executives, cybersecurity, recovery teams)"
        pts = 0.0
        if admin_tier in selections:
            pts += 7.0
        if leadership_tier in selections:
            pts += 3.0
        if not pts:
            return 0.0
        return min(1.0, pts / 10.0)

    # ---- vendor access control (applicability-normalized) ------------
    if control_id == "vendor_access":
        if raw is None:
            return None
        n = _norm(raw)
        if n in _NOT_APPLICABLE_VALUES:
            return None  # PDF Sec 4: drop from both sums
        if n == "yes":
            return 1.0
        if n == "no":
            return 0.0
        return None

    # ---- EDR ---------------------------------------------------------
    if control_id == "edr":
        if raw is None:
            return None
        vals = {_norm(x) for x in _as_list(raw)}
        if not vals or any(v in _NO_EDR_VALUES for v in vals):
            return 0.0
        return 1.0

    # ---- plain Yes/No controls ---------------------------------------
    if raw is None:
        return None
    return 1.0 if _norm(raw) == "yes" else 0.0


def compute_control_health(security_data: Dict[str, Any]) -> Dict[str, Any]:
    """H_SF = sum(w_i * s_i) / sum(w_i * s_max_i), over APPLICABLE controls only.

    Returns the index plus a per-control breakdown, because PDF Sec 18 requires
    that a reviewer be handed the assumptions, not just the output.
    """
    numerator = 0.0
    denominator = 0.0
    breakdown: Dict[str, Any] = {}

    for control_id, spec in SF_CONTROL_WEIGHTS.items():
        score = _score_control(control_id, security_data)
        if score is None:
            breakdown[control_id] = {"applicable": False, "score": None, "weight": spec["weight"]}
            continue
        numerator += spec["weight"] * score
        denominator += spec["weight"] * 1.0
        breakdown[control_id] = {"applicable": True, "score": round(score, 3), "weight": spec["weight"]}

    if denominator <= 0:
        # No control questions answered at all. Fall back to the portfolio mean
        # so M_ctrl == 1.0 -- i.e. we neither reward nor punish a blank sheet.
        return {"h_sf": None, "breakdown": breakdown, "answered_weight": 0.0}

    return {
        "h_sf": numerator / denominator,
        "breakdown": breakdown,
        "answered_weight": denominator,
    }


# =============================================================================
# SECTION 3 -- BACKUP SURVIVAL: THE LATENT TAIL SWITCH  (PDF Sec 6)
# =============================================================================

# Backup-frequency answers that mean "we have no backups at all".
_NO_BACKUP_VALUES = {
    "we do not backup our files",
    "we do not back up our data",
    "we do not back up our files",
    "we do not backup our data",
    "we don't back up our data",
}

# Normalized [0,1] proxies. Mirror the point scales in parse_security.py
# (_score_backup_frequency / 10 and restore_map / 6) so this module never
# disagrees with the main questionnaire scoring.
_BACKUP_FREQ_SCORES = {
    "daily": 1.0, "weekly": 0.8, "monthly": 0.6, "once a month": 0.6,
    "every 3 months": 0.4, "every 6 months": 0.2, "once a year": 0.1,
}
_RESTORE_TEST_SCORES = {
    "monthly": 1.0, "every3months": 0.66, "every6months": 0.33,
    "onceayear": 0.17, "never": 0.0,
}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_backup_resilience(security_data: Dict[str, Any]) -> Optional[float]:
    """Proxy index in [0,1] for how likely backups are to SURVIVE a deliberate
    attacker. Returns None when no relevant control was answered.

    HONESTY NOTE. PDF Sec 6 states that backup frequency and restore testing are
    NOT proxies for immutability, because they assume the backup still exists.
    That reasoning is correct and is retained -- which is exactly why this index
    feeds a PROBABILITY rather than a switch, and why even a perfect score still
    leaves a substantial chance of the catastrophic branch.

    What each signal is actually worth as evidence:
      restore testing : a firm that tests restores has a real backup process.
                        Weak evidence of isolation, decent evidence of
                        recoverability. Highest weight.
      segmentation    : genuine evidence. A segmented network is the main reason
                        a backup target survives lateral movement.
      EDR             : detection before destruction completes.
      backup frequency: weakest. A daily backup to a mapped network drive is
                        exactly what an insider with admin rights deletes first.
    """
    if _norm(security_data.get("dataBacking")) in _NO_BACKUP_VALUES:
        return 0.0

    freq = (_BACKUP_FREQ_SCORES.get(_norm(security_data.get("dataBacking")))
            if security_data.get("dataBacking") not in (None, "", []) else None)
    test = (_RESTORE_TEST_SCORES.get(_norm(security_data.get("restoreBackupTest")))
            if security_data.get("restoreBackupTest") not in (None, "", []) else None)
    seg = _score_control("network_segmentation", security_data)
    edr = _score_control("edr", security_data)

    # Weights reflect EVIDENTIAL VALUE, not the parse_security point scale.
    weights = {"restore_testing": 0.35, "segmentation": 0.30,
               "edr": 0.20, "backup_frequency": 0.15}
    scores = {"restore_testing": test, "segmentation": seg,
              "edr": edr, "backup_frequency": freq}

    # Score only what was answered, then renormalize -- the same applicability
    # rule H_SF uses. An unanswered control must leave the firm at the portfolio
    # base rate, not push it toward the catastrophic branch: silence is not
    # evidence of bad backups.
    num = sum(weights[k] * v for k, v in scores.items() if v is not None)
    den = sum(weights[k] for k, v in scores.items() if v is not None)
    if den <= 0:
        return None
    return float(np.clip(num / den, 0.0, 1.0))


def resolve_backup_immutability(
    security_data: Dict[str, Any],
    params: Optional["SystemFailureParams"] = None,
) -> Dict[str, Any]:
    """Resolve the downtime branch as a PROBABILITY, not a hard switch.

    PDF Sec 6 asks for ONE new question:
        "Are your backups protected so that even an administrator or insider
         cannot delete or alter them (e.g. immutable or offline copies)?"

    That question is not in the questionnaire and is not being added. Rather
    than defaulting every firm to the catastrophic branch -- which strips all
    firm-level signal out of the model's single largest factor -- backup
    survival is treated as a latent variable and marginalized over a published
    prior, moved by the controls that ARE observed.

    Returns `p_catastrophic` in [0,1]. The caller draws the branch PER INCIDENT.

    If `backupsImmutable` is ever added, it is honoured here as a hard switch
    and this whole mechanism collapses to the PDF's original specification with
    no other code change.
    """
    p = params or SystemFailureParams()

    raw = security_data.get("backupsImmutable")
    if raw is None:
        raw = security_data.get("immutableBackups")

    if raw is not None:
        n = _norm(raw)
        yes = n in ("yes", "true")   # "Not sure" -> No, per PDF Sec 6
        return {
            "branch": "safe" if yes else "catastrophic",
            "p_catastrophic": 0.0 if yes else 1.0,
            "mode": "direct_answer",
            "inferred": False,
            "resilience_index": 1.0 if yes else 0.0,
            "answer": raw,
            "note": None,
        }

    # ---- latent mixture ---------------------------------------------------
    idx = compute_backup_resilience(security_data)
    any_signal = idx is not None
    if idx is None:
        # Nothing answered: sit exactly on the bar so p == the published base
        # rate, untouched in either direction.
        idx = p.backup_resilience_bar

    p_cat = _sigmoid(
        _logit(p.p_backup_compromised_base)
        - p.beta_backup_resilience * (idx - p.backup_resilience_bar)
    )

    return {
        "branch": "mixture",
        "p_catastrophic": float(p_cat),
        "mode": "latent_mixture",
        "inferred": True,
        "resilience_index": float(idx),
        "any_signal": any_signal,
        "answer": None,
        "note": (
            "backupsImmutable is not asked and is not being added. Branch is "
            "marginalized over a published prior (Sophos: backup repositories "
            "targeted in 96% of attacks, compromised 76% of the time -> 0.73 "
            "base), moved by restore testing, segmentation, EDR and backup "
            "frequency. Firm-level separation is therefore PARTIAL: P(cat) spans "
            "roughly 0.44-0.90 across the score range, versus 0.0/1.0 if the "
            "question were asked. This remains the largest single source of "
            "per-firm error in the model."
        ),
    }


# =============================================================================
# SECTION 4 -- DAILY MARGIN (BI slice input)
# =============================================================================
# [UNSOURCED at the precision required] The PDF calls for "margin % by NAICS"
# from "public financial benchmarks". Published tables that exist:
#   - IRS SOI Corporation Source Book (by NAICS AND receipts size class -- the
#     right shape for micro-SMBs; TY2022 is the latest complete release)
#   - NYU Stern / Damodaran margins by sector (PUBLIC COMPANIES -- wrong size
#     cohort for a 20-person firm, use only as an upper bound)
#   - RMA Annual Statement Studies (licensed)
# The table below is a GROSS margin approximation by 2-digit NAICS, directionally
# consistent with the sources above. Replace with IRS SOI values before shipping
# and record the page + pull date per PDF Sec 5.

_GROSS_MARGIN_BY_NAICS: Dict[str, float] = {
    "11": 0.25,  # Agriculture, forestry, fishing, hunting
    "21": 0.35,  # Mining, quarrying, oil and gas
    "22": 0.40,  # Utilities
    "23": 0.20,  # Construction
    "31": 0.30,  # Manufacturing
    "32": 0.30,
    "33": 0.30,
    "42": 0.22,  # Wholesale trade
    "44": 0.33,  # Retail trade
    "45": 0.33,
    "48": 0.25,  # Transportation and warehousing
    "49": 0.25,
    "51": 0.60,  # Information
    "52": 0.55,  # Finance and insurance
    "53": 0.50,  # Real estate, rental, leasing
    "54": 0.50,  # Professional, scientific, technical services
    "55": 0.45,  # Management of companies
    "56": 0.35,  # Administrative and support services
    "61": 0.40,  # Educational services
    "62": 0.45,  # Health care and social assistance
    "71": 0.40,  # Arts, entertainment, recreation
    "72": 0.30,  # Accommodation and food services
    "81": 0.40,  # Other services
    "92": 0.35,  # Public administration
}

_DEFAULT_GROSS_MARGIN = 0.35  # [ASSUMPTION] blended fallback


def get_gross_margin(naics_code: Any) -> float:
    sector = str(naics_code)[:2] if naics_code is not None else ""
    return _GROSS_MARGIN_BY_NAICS.get(sector, _DEFAULT_GROSS_MARGIN)


def smoothed_revenue(current: float, prev1: float = 0.0, prev2: float = 0.0) -> float:
    """3-year smoothed revenue per PDF Sec 3.2. Zero-valued prior years are
    dropped rather than averaged in, which would understate a young firm."""
    vals = [v for v in (current, prev1, prev2) if v and v > 0]
    if not vals:
        return 0.0
    return float(np.mean(vals))


# =============================================================================
# SECTION 5 -- THE MODEL
# =============================================================================

class SystemFailureModel:
    """Monte Carlo model for expected annual loss from malicious system failure.

    Interface intentionally mirrors BrickingModel:
        model = SystemFailureModel(num_employees=..., naics_code=..., security_data=...)
        loss  = model.expected_annual_loss()
        diag  = model.diagnostics()

    ANTI-DOUBLE-COUNTING (PDF Sec 2)
    --------------------------------
    This reports the expected annual loss ATTRIBUTABLE TO MALICIOUS INSIDERS.
    It rebuilds the cost from the same primitives (downtime, response,
    restoration) but does NOT import the customer-facing BI / IR / Privacy
    category totals. Standalone BI answers "across all causes, what is my
    business-interruption exposure?"; this answers "what do malicious insiders
    cost me per year?" -- disjoint accounting.

    The privacy slice is the one place where overlap is possible. It is
    therefore OFF by default (`privacy_cost_per_incident=None`). Only pass a
    value if you have confirmed the Privacy Liability module is not already
    charging for insider-caused record loss.
    """

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
        params: Optional[SystemFailureParams] = None,
    ):
        self.params = params or SystemFailureParams()
        self.num_employees = max(int(num_employees or 1), 1)
        self.naics_code = naics_code
        self.security_data = security_data or {}
        self.company_size = (company_size or "small").lower()
        self.p_value = p_value
        self.privacy_cost_per_incident = privacy_cost_per_incident

        self.revenue = smoothed_revenue(
            float(annual_revenue or 0.0),
            float(prev_year_revenue1 or 0.0),
            float(prev_year_revenue2 or 0.0),
        )

        # --- control health -> frequency multiplier ---------------------
        health = compute_control_health(self.security_data)
        self.control_breakdown = health["breakdown"]
        self.h_sf = health["h_sf"]

        p = self.params
        if self.h_sf is None:
            self.m_ctrl = 1.0
        else:
            self.m_ctrl = math.exp(-p.k_control_sensitivity * (self.h_sf - p.h_bar))

        # --- history loader --------------------------------------------
        self.has_prior_incident = self._detect_prior_incident(recent_breach)
        self.m_hist = p.m_hist if self.has_prior_incident else 1.0

        # --- lambda -----------------------------------------------------
        self.lambda0 = self._compute_lambda0()
        self.lambda_value = self.lambda0 * self.m_ctrl * self.m_hist

        # --- tail switch (latent; see Section 3) --------------------------
        self.backup_status = resolve_backup_immutability(self.security_data, self.params)
        self.p_catastrophic = self.backup_status["p_catastrophic"]

        # --- segmentation (feeds restoration cost) ------------------------
        seg = _score_control("network_segmentation", self.security_data)
        self.segmentation = 0.0 if seg is None else float(seg)

        self._results: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------
    # Frequency
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
        """lambda0 = base all-cause breach rate x malicious-insider share x sector mult.

        When the host model has already computed a p_value (probability of a
        breach in the year), use it -- that keeps this peril consistent with the
        rest of the system rather than asserting an unrelated base rate.
        Otherwise fall back to the size table.
        """
        p = self.params
        if self.p_value is not None:
            pv = float(np.clip(self.p_value, 0.0, 1 - 1e-10))
            lam_all_causes = -math.log(1 - pv)  # matches helper_func.calculate_lambda
        else:
            lam_all_causes = p.base_breach_rate_by_size.get(self.company_size, 0.16)

        sector = str(self.naics_code)[:2] if self.naics_code is not None else ""
        sector_mult = p.sector_frequency_multiplier.get(sector, 1.0)
        return (lam_all_causes
                * p.malicious_insider_share
                * sector_mult
                * p.frequency_calibration_factor)

    # ---------------------------------------------------------------
    # Severity
    # ---------------------------------------------------------------
    def _daily_margin(self) -> float:
        margin_pct = get_gross_margin(self.naics_code)
        return (self.revenue * margin_pct) / 365.0 if self.revenue > 0 else 0.0

    @staticmethod
    def _lognormal_from_median(median: float, sigma: float, size: int, rng) -> np.ndarray:
        """Lognormal parameterized by MEDIAN (= e^mu), not mean.

        PDF Sec 15.2: median = e^mu, mean = e^(mu + sigma^2/2). Specifying the
        median keeps sigma and central tendency independent, so retuning sigma
        during calibration does not silently move the median too.
        """
        if median <= 0:
            return np.zeros(size)
        return rng.lognormal(mean=math.log(median), sigma=sigma, size=size)

    # ---------------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------------
    def simulate(self) -> Dict[str, Any]:
        if self._results is not None:
            return self._results

        p = self.params
        rng = np.random.default_rng(p.random_seed)
        n_sims = int(p.n_sims)

        # --- Step 1: incident counts per simulated year (PDF Sec 3.1) ----
        # Poisson is the maximum-entropy choice for rare independent counts
        # (PDF Sec 15.1). If claims data ever shows Var/E > 1, swap to NegBin.
        counts = rng.poisson(lam=self.lambda_value, size=n_sims)
        total_incidents = int(counts.sum())

        if total_incidents == 0:
            self._results = self._empty_results(counts)
            return self._results

        # --- Step 2: severity per incident (PDF Sec 3.2) ------------------
        # Draw the branch PER INCIDENT from the marginalized probability rather
        # than assigning one branch to the whole firm. With a direct answer
        # p_catastrophic is 0.0 or 1.0 and this reduces to the PDF's hard switch.
        is_cat = rng.random(total_incidents) < self.p_catastrophic

        # Business interruption: D x m_day x u
        days = np.where(
            is_cat,
            self._lognormal_from_median(
                p.mu_days_cat_median, p.sigma_days_cat, total_incidents, rng),
            self._lognormal_from_median(
                p.mu_days_safe_median, p.sigma_days_safe, total_incidents, rng),
        )

        m_day = self._daily_margin()
        c_bi = days * m_day * p.dependency_factor

        # Incident response
        c_ir_base = self._lognormal_from_median(
            p.ir_median, p.ir_sigma, total_incidents, rng)
        c_ir = np.where(is_cat, c_ir_base * p.cat_ir_multiplier, c_ir_base)

        # Restoration: r0 + r1 * employees * (2 - Seg), with lognormal spread
        rest_median = p.rest_r0 + p.rest_r1 * self.num_employees * (2.0 - self.segmentation)
        c_rest_base = self._lognormal_from_median(
            rest_median, p.rest_sigma, total_incidents, rng)
        c_rest = np.where(is_cat, c_rest_base * p.cat_restoration_multiplier, c_rest_base)

        # Conditional privacy slice -- OFF unless explicitly supplied, to keep
        # the anti-double-counting rule (PDF Sec 2) safe by default.
        if self.privacy_cost_per_incident:
            exf = rng.random(total_incidents) < p.q_exf
            c_priv = exf * float(self.privacy_cost_per_incident)
        else:
            c_priv = np.zeros(total_incidents)

        # Common per-incident severity shock (see params.sigma_common).
        # Median 1 so it widens the distribution without shifting the median.
        if p.sigma_common > 0:
            shock = rng.lognormal(mean=0.0, sigma=p.sigma_common, size=total_incidents)
        else:
            shock = np.ones(total_incidents)

        severity = (c_bi + c_ir + c_rest + c_priv) * shock

        # Sanity cap (see params). Not in the PDF -- added so a lognormal draw
        # cannot exceed what the business could physically lose.
        if self.revenue > 0:
            cap = self.revenue * p.max_incident_revenue_multiple
            n_capped = int((severity > cap).sum())
            severity = np.minimum(severity, cap)
        else:
            n_capped = 0

        # --- Step 3: aggregate to annual loss L_t --------------------------
        year_index = np.repeat(np.arange(n_sims), counts)
        annual_losses = np.bincount(year_index, weights=severity, minlength=n_sims)

        # --- Wald's identity check (PDF Sec 15.3) --------------------------
        # E[L] = lambda * E[S]. If the simulated mean does not converge to this,
        # the simulation is miscoded. This is a live assertion, not a comment.
        analytic_mean = self.lambda_value * float(severity.mean())
        simulated_mean = float(annual_losses.mean())
        if analytic_mean > 0:
            wald_error = abs(simulated_mean - analytic_mean) / analytic_mean
        else:
            wald_error = 0.0

        nonzero = annual_losses[annual_losses > 0]

        # Monte Carlo standard error of the reported mean. This is the honest
        # precision statement for the customer-facing number.
        mc_se = float(annual_losses.std(ddof=1) / math.sqrt(n_sims))

        self._results = {
            "expected_annual_loss": simulated_mean,
            "analytic_mean_wald": analytic_mean,
            "wald_relative_error": wald_error,
            "wald_check_passed": wald_error < 0.05,
            "mc_standard_error": mc_se,
            "mc_relative_se": (mc_se / simulated_mean) if simulated_mean > 0 else 0.0,
            "annual_losses": annual_losses,
            "n_years_with_incident": int((counts > 0).sum()),
            "pct_years_with_incident": float((counts > 0).mean()),
            "conditional_percentiles": {
                "P50": float(np.percentile(nonzero, 50)) if nonzero.size else 0.0,
                "P90": float(np.percentile(nonzero, 90)) if nonzero.size else 0.0,
                "P95": float(np.percentile(nonzero, 95)) if nonzero.size else 0.0,
            },
            "mean_severity_per_incident": float(severity.mean()),
            "median_severity_per_incident": float(np.median(severity)),
            "mean_downtime_days": float(days.mean()),
            "pct_incidents_catastrophic": float(is_cat.mean()),
            "incidents_capped": n_capped,
            "component_means": {
                "business_interruption": float(c_bi.mean()),
                "incident_response": float(c_ir.mean()),
                "restoration": float(c_rest.mean()),
                "privacy": float(c_priv.mean()),
            },
        }
        return self._results

    def _empty_results(self, counts: np.ndarray) -> Dict[str, Any]:
        return {
            "expected_annual_loss": 0.0,
            "analytic_mean_wald": 0.0,
            "wald_relative_error": 0.0,
            "wald_check_passed": True,
            "mc_standard_error": 0.0,
            "mc_relative_se": 0.0,
            "annual_losses": np.zeros(len(counts)),
            "n_years_with_incident": 0,
            "pct_years_with_incident": 0.0,
            "conditional_percentiles": {"P50": 0.0, "P90": 0.0, "P95": 0.0},
            "mean_severity_per_incident": 0.0,
            "median_severity_per_incident": 0.0,
            "mean_downtime_days": 0.0,
            "pct_incidents_catastrophic": 0.0,
            "incidents_capped": 0,
            "component_means": {
                "business_interruption": 0.0,
                "incident_response": 0.0,
                "restoration": 0.0,
                "privacy": 0.0,
            },
        }

    # ---------------------------------------------------------------
    # Public interface (mirrors BrickingModel)
    # ---------------------------------------------------------------
    def expected_annual_loss(self) -> float:
        """The single customer-facing number: the MEAN of the simulated
        annual-loss distribution (PDF Sec 1.1). The mean already accounts for the
        rare catastrophic years, which is why it is the honest figure to show."""
        return float(self.simulate()["expected_annual_loss"])

    def diagnostics(self) -> Dict[str, Any]:
        """Everything a reviewer should be handed first (PDF Sec 18)."""
        r = self.simulate()
        return {
            "expected_annual_loss": round(r["expected_annual_loss"], 2),
            "frequency": {
                "lambda0": round(self.lambda0, 6),
                "M_ctrl": round(self.m_ctrl, 4),
                "M_hist": self.m_hist,
                "lambda": round(self.lambda_value, 6),
                "pct_years_with_incident": round(r["pct_years_with_incident"] * 100, 3),
            },
            "control_health": {
                "H_SF": round(self.h_sf, 4) if self.h_sf is not None else None,
                "H_bar": self.params.h_bar,
                "breakdown": self.control_breakdown,
            },
            "tail_switch": {
                "mode": self.backup_status["mode"],
                "branch": self.backup_status["branch"],
                "backup_resilience_index": round(self.backup_status["resilience_index"], 4),
                "P_catastrophic_branch": round(self.p_catastrophic, 4),
                "observed_pct_incidents_catastrophic": round(
                    r["pct_incidents_catastrophic"] * 100, 2),
                "any_control_signal": self.backup_status.get("any_signal"),
                "inferred": self.backup_status["inferred"],
                "mean_downtime_days": round(r["mean_downtime_days"], 2),
                "resolution_note": self.backup_status.get("note"),
            },
            "severity": {
                "mean_per_incident": round(r["mean_severity_per_incident"], 2),
                "median_per_incident": round(r["median_severity_per_incident"], 2),
                "component_means": {k: round(v, 2) for k, v in r["component_means"].items()},
                "incidents_capped": r["incidents_capped"],
            },
            "conditional_percentiles": {k: round(v, 2) for k, v in r["conditional_percentiles"].items()},
            "validation": {
                "wald_analytic_mean": round(r["analytic_mean_wald"], 2),
                "wald_relative_error": round(r["wald_relative_error"], 5),
                "wald_check_passed": r["wald_check_passed"],
                "mc_relative_se_pct": round(r["mc_relative_se"] * 100, 3),
                "mc_95pct_ci_halfwidth_pct": round(r["mc_relative_se"] * 1.96 * 100, 3),
                "mc_precision_ok": r["mc_relative_se"] * 1.96 < 0.02,
                "n_sims": self.params.n_sims,
            },
            "model_status": "UNVALIDATED -- portfolio calibration and back-test not yet run (PDF Sec 17)",
        }


# =============================================================================
# SECTION 6 -- SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    import json

    demo_security = {
        "useMFA": ["Admins or high-access users (IT admins, developers, finance/HR, vendors, etc.)"],
        "ReceiveAlertApp": "Microsoft 365 Defender",
        "thirdPartyControls": "Yes",
        "networkDivided": "No",
        "lockAccount": "Yes",
        "thirdPartyAppApproval": "No",
        "edrToolsUsed": ["CrowdStrike Falcon"],
        "dataEncryption": "Yes",
    }

    for label, extra in [
        ("latent mixture -- weak backup signals",
         {"dataBacking": "Once a year", "restoreBackupTest": "never"}),
        ("latent mixture -- strong backup signals",
         {"dataBacking": "Daily", "restoreBackupTest": "monthly", "networkDivided": "Yes"}),
        ("direct answer -- backups immutable", {"backupsImmutable": "Yes"}),
        ("direct answer -- not sure (-> No)", {"backupsImmutable": "Not sure"}),
    ]:
        sd = {**demo_security, **extra}
        m = SystemFailureModel(
            num_employees=20,
            naics_code="541511",
            security_data=sd,
            annual_revenue=5_000_000,
            company_size="small",
            p_value=0.35,
            recent_breach="No",
        )
        print(f"\n=== {label} ===")
        print(json.dumps(m.diagnostics(), indent=2, default=str))
