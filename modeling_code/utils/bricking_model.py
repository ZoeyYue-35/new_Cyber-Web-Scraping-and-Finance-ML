"""
BRICKING MODEL -- device loss/theft followed by an authorized remote wipe.

SCOPE (settled on the 2026-08-06 call with Cam)
-----------------------------------------------
Covered peril: a company-issued mobile device is lost or stolen, and the
organization sends an authorized signal that renders it unusable to protect the
data on it. The device is not expected back, so it is a total loss and gets
replaced.

IN SCOPE:  laptops, tablets, phones.
OUT OF SCOPE: servers, network gear, OT/IoT. You do not remote-wipe a rack.

NOT THIS MODEL: destructive wiper malware spreading from a breach (NotPetya
class). Cam's guidance was that being bricked as part of a ransomware event is
a SYSTEM FAILURE and belongs in that model. The previous version of this file
modeled that peril; it has been moved out. Ransomware proper is a third model
again -- its dominant costs are BI, extortion and IR labour, not hardware.

DESIGN
------
Cam's prescription, deliberately followed without embellishment:

    number of devices
      x how often one is lost or stolen
      x what fraction of those get bricked
      x cost of the underlying asset

He warned explicitly that the more complex the scenario, the less defendable it
becomes and the more points of error get introduced, and that this should be a
simple likelihood-and-impact Monte Carlo. The prior version of this file grew
five multiplicative unsourced parameters and produced numbers nobody could
defend in either direction. Resist re-adding them here.

THE CONTROL LOGIC IS INVERTED, ON PURPOSE
-----------------------------------------
Everywhere else in this pipeline, better security posture means lower loss.
Not here. You can only brick a device you can reach, so good endpoint
management INCREASES the number of bricking events. It also converts what would
have been an uncontrolled data breach into a bounded hardware cost -- which is a
good trade, and shows up as a lower privacy/notification loss in those models,
not here. Do not "fix" this by reusing control_deficit.

DOUBLE-COUNT BOUNDARY
---------------------
A lost device holding personal data also generates notification and privacy
liability. Those live in calculate_privacy_liability_coverage_loss() and the
notification costs. THIS MODEL PRICES HARDWARE ONLY. Do not add a data-exposure
term here.

BASIS -- READ BEFORE INTEGRATING
--------------------------------
Device theft has its OWN frequency, unrelated to the cyber-breach lambda used
elsewhere in this pipeline. Laptops get stolen whether or not the company is
breached. Therefore:

    expected_annual_loss()  is the primary and correct output.

There is no meaningful "per breach event" version of this peril, and multiplying
by lambda_value would be wrong -- it would tie laptop theft to breach frequency.
See the integration note supplied with this file.
"""

import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================

ASSET_CLASSES = ('laptops', 'tablets', 'phones')

# --- Fleet: mobile devices per employee by NAICS 2-digit prefix --------------
# UNSOURCED PRIOR. Field/mobile industries carry more phones and rugged tablets;
# desk-bound industries carry more laptops. Better: ask on the questionnaire.
# "How many company-owned laptops, tablets and phones do you issue?" is a
# question a broker can actually answer, unlike network segmentation.
MOBILE_DENSITY = {
    #              laptops, tablets, phones
    '21': (0.45, 0.20, 0.85),   # mining / oil & gas -- field heavy
    '22': (0.50, 0.18, 0.80),   # utilities
    '23': (0.40, 0.25, 0.90),   # construction -- site crews, rugged tablets
    '31': (0.45, 0.12, 0.55),   # manufacturing
    '32': (0.45, 0.12, 0.55),
    '33': (0.45, 0.12, 0.55),
    '42': (0.70, 0.10, 0.70),   # wholesale -- reps on the road
    '44': (0.35, 0.20, 0.45),   # retail
    '45': (0.35, 0.20, 0.45),
    '48': (0.35, 0.22, 0.85),   # transportation -- drivers
    '49': (0.35, 0.22, 0.85),
    '51': (1.05, 0.15, 0.80),   # information / IT
    '52': (1.00, 0.12, 0.75),   # finance & insurance
    '54': (1.10, 0.15, 0.80),   # professional / technical -- consultants travel
    '56': (0.60, 0.10, 0.55),   # admin & support
    '61': (0.80, 0.20, 0.45),   # educational services
    '62': (0.55, 0.25, 0.50),   # health care -- clinical tablets
    '71': (0.50, 0.15, 0.50),   # arts / recreation
    '72': (0.30, 0.15, 0.45),   # accommodation & food
    '81': (0.50, 0.10, 0.60),   # other services
    '92': (0.70, 0.10, 0.55),   # public administration
}
DEFAULT_MOBILE_DENSITY = (0.75, 0.15, 0.65)

# --- Annual loss/theft rate per device ---------------------------------------
# UNSOURCED PRIOR and the weakest input in the model. Published rates come
# largely from vendors selling anti-theft products and run hot; these are set
# below the headline survey figures deliberately. Phones are lost far more often
# than laptops; tablets sit between.
#
# This is the parameter to replace first with real claims data, and the one
# Cam's Bayesian self-correction loop should be updating.
THEFT_LOSS_RATE = {
    'laptops': 0.020,
    'tablets': 0.025,
    'phones':  0.045,
}

# --- Replacement cost per unit, USD ------------------------------------------
# Hardware plus provisioning labour. These are price-checkable -- Cam's advice
# was to pull standard cost rates for a Dell/Lenovo laptop, an Android handset
# and an iPad, and use the spread to gauge a range. Do that; this is the one
# input in the file that does not have to stay a prior.
REPLACEMENT_COST = {
    'laptops': 1_400 + 350,
    'tablets':   700 + 150,
    'phones':    900 + 150,
}

# Lognormal spread on unit cost, capturing fleet mix (ThinkPads vs MacBook Pros).
COST_SIGMA = 0.35

# --- Re-provisioning cost when a bricked device IS recovered -----------------
# Cam noted mechanisms exist to rebuild a bricked device if it comes back. Then
# the loss is labour, not hardware.
REPROVISION_COST = 220.0

# Probability a bricked device is recovered and rebuilt. Cam's view: if you pull
# the trigger you are probably not expecting it back, so keep this low.
RECOVERY_RATE = 0.12

# --- P(wipe | device lost) ---------------------------------------------------
# You can only brick what you can reach. Floor is the fraction of firms that
# manage to wipe with no formal endpoint tooling (carrier kill, Find My, manual
# account revocation); ceiling is a fully managed fleet.
WIPE_RATE_FLOOR = 0.25
WIPE_RATE_CEILING = 0.95

# Mobility uplift on theft rates, from the share of the workforce with remote
# access. Devices that leave the building get lost; devices in a locked office
# do not. Capped so it cannot dominate the base rate.
MOBILITY_MAX_UPLIFT = 0.60


# =============================================================================
# QUESTIONNAIRE READS
# =============================================================================
# Field names and value vocabularies follow parse_security.py. Keep them in
# sync -- these are read directly off securityQuestions.

NO_THREAT_DETECTION_VALUES = {
    'we do not have this',
    'we do not use security software to detect threats',
    'none',
    'no',
}


def _normalize_text(value):
    if value is None:
        return ''
    return str(value).replace('\u2019', "'").replace('\u2018', "'").strip().lower()


def _as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ''):
        return []
    return [value]


def _parse_positive_int(value):
    """
    Tolerant int parse. parse_security.py's version uses int(str(value)) which
    raises on floats -- num_employees arrives here as a float, so that version
    silently returned 0 and drove the employee denominator to 1, pinning
    mobility at its 1.0 ceiling for every assessment. Parse via float.
    """
    try:
        return int(float(str(value).strip().replace(',', '')))
    except (TypeError, ValueError, AttributeError):
        return 0


def _has_endpoint_tooling(security_data):
    """
    EDR presence as a proxy for managed endpoints. The questionnaire does not
    ask about MDM/EMM directly -- it should, and that is the single highest-value
    question to add for this model. Until then, a firm running CrowdStrike or
    Defender across its estate is far more likely to be able to reach and wipe a
    lost device than one running nothing.
    """
    s = security_data or {}
    values = _as_list(s.get('edrToolsUsed') or s.get('edrTools'))
    if not values:
        return False
    return not any(_normalize_text(v) in NO_THREAT_DETECTION_VALUES for v in values)


def compute_wipe_capability(security_data):
    """
    0.0 (cannot reach a lost device) -> 1.0 (fully managed fleet).

    NOTE THE DIRECTION: this rises with better controls, and it MULTIPLIES the
    loss. Strong posture means more devices get deliberately bricked. That is
    the correct behaviour for this peril -- see the module docstring.

    Signals used, all present in securityQuestions:
      edrToolsUsed      -- managed endpoint tooling  (0.50)
      dataEncryption    -- disk encryption implies device management  (0.25)
      lockAccount       -- account lockout / revocation discipline  (0.15)
      IncidentResponse  -- someone knows to pull the trigger, and fast  (0.10)
    """
    s = security_data or {}
    c = 0.0
    if _has_endpoint_tooling(s):
        c += 0.50
    if s.get('dataEncryption') == 'Yes':
        c += 0.25
    if s.get('lockAccount') == 'Yes':
        c += 0.15

    has_ir = s.get('IncidentResponse') is True or s.get('noIncidentResponse') is False
    if not has_ir:
        for key in ('incidentResponsePlan', 'incidentFile', 'incidentResponseDocument'):
            if s.get(key) not in (None, '', [], False):
                has_ir = True
                break
    if has_ir:
        c += 0.10

    return float(np.clip(c, 0.0, 1.0))


def compute_mobility(security_data, num_employees):
    """
    Share of the workforce with remote access, as a proxy for how much of the
    fleet leaves a controlled site. Reads remoteAccessTypes and the per-category
    counts exactly as _score_remote_access() in parse_security.py does.

    Falls back to the remoteAccess method answer when counts are absent:
    'nobody' means a static fleet, anything else implies some mobility.
    """
    s = security_data or {}
    employees = max(_parse_positive_int(num_employees), 1)

    categories = s.get('remoteAccessTypes') or []
    total_remote = 0
    saw_count = False
    for category in categories:
        raw = s.get(f'{category}_count', s.get(category))
        if raw not in (None, ''):
            saw_count = True
            total_remote += _parse_positive_int(raw)

    if saw_count:
        return float(np.clip(total_remote / employees, 0.0, 1.0))

    if s.get('remoteAccess') in ('nobody', 'No'):
        return 0.0
    if categories:
        return 0.50
    return 0.35


# =============================================================================
# THE MODEL
# =============================================================================

class BrickingModel:
    """
        bm = BrickingModel(num_employees, naics_code, security_data)

        bm.expected_annual_loss()          # headline -- use this
        bm.simulate_annual(n_sims)         # distribution for percentiles
        bm.diagnostics()

    All figures USD.
    """

    def __init__(self, num_employees, naics_code, security_data, seed=None):
        self.num_employees = max(float(num_employees or 1), 1.0)
        self.naics_prefix = str(naics_code)[:2]
        self.security_data = security_data or {}
        self.rng = np.random.default_rng(seed)

        self.wipe_capability = compute_wipe_capability(self.security_data)
        self.mobility = compute_mobility(self.security_data, self.num_employees)

        self.fleet = self._derive_fleet()
        self.wipe_rate = self._wipe_rate()
        self.theft_rates = self._theft_rates()

    # -- parameterization ----------------------------------------------------

    def _derive_fleet(self):
        """
        Device counts. If the questionnaire ever collects real device counts,
        override here -- a stated count should always beat a density prior.
        """
        density = MOBILE_DENSITY.get(self.naics_prefix, DEFAULT_MOBILE_DENSITY)
        return {cls: self.num_employees * coeff
                for cls, coeff in zip(ASSET_CLASSES, density)}

    def _wipe_rate(self):
        """P(device is bricked | device lost). Rises with control posture."""
        r = WIPE_RATE_FLOOR + (WIPE_RATE_CEILING - WIPE_RATE_FLOOR) * self.wipe_capability
        return float(np.clip(r, WIPE_RATE_FLOOR, WIPE_RATE_CEILING))

    def _theft_rates(self):
        """Per-device annual loss/theft probability, uplifted by mobility."""
        uplift = 1.0 + MOBILITY_MAX_UPLIFT * self.mobility
        return {c: float(np.clip(THEFT_LOSS_RATE[c] * uplift, 0.0, 1.0))
                for c in ASSET_CLASSES}

    # -- analytic ------------------------------------------------------------

    def expected_devices_bricked(self):
        """Expected count of devices bricked per year, by class."""
        return {c: self.fleet[c] * self.theft_rates[c] * self.wipe_rate
                for c in ASSET_CLASSES}

    def expected_annual_loss(self):
        """
        HEADLINE. E[replacement cost per year].

        E = SUM_c  fleet_c x theft_rate_c x wipe_rate
                   x [ (1 - recovery) x replacement_c + recovery x reprovision ]

        Not conditioned on a breach. Device theft runs on its own clock.
        """
        bricked = self.expected_devices_bricked()
        return float(sum(
            bricked[c] * ((1.0 - RECOVERY_RATE) * REPLACEMENT_COST[c]
                          + RECOVERY_RATE * REPROVISION_COST)
            for c in ASSET_CLASSES
        ))

    def expected_cost_per_bricking_event(self):
        """
        E[cost of ONE bricked device], averaged over the class mix. Useful for
        reporting ("each incident costs about X") but NOT the number to add to
        an annual total.
        """
        bricked = self.expected_devices_bricked()
        n = sum(bricked.values())
        if n <= 0:
            return 0.0
        return self.expected_annual_loss() / n

    # -- simulation ----------------------------------------------------------

    def simulate_annual(self, num_sims=200_000):
        """
        Monte Carlo over annual bricking cost, as Cam suggested: likelihood of
        occurrence and a range of cost.

        Per sim and per class:
            lost    ~ Binomial(fleet, theft_rate)
            bricked ~ Binomial(lost, wipe_rate)
            back    ~ Binomial(bricked, recovery_rate)
            cost    = lognormal unit costs for (bricked - back) replacements
                      + flat reprovisioning for the recovered ones

        Returns an array of annual dollar losses. Naturally zero-inflated: a
        small firm with a managed fleet will have many loss-free years.
        """
        num_sims = int(num_sims)
        total = np.zeros(num_sims, dtype=float)

        for c in ASSET_CLASSES:
            n_units = int(round(self.fleet[c]))
            if n_units <= 0:
                continue

            lost = self.rng.binomial(n_units, self.theft_rates[c], size=num_sims)
            bricked = self.rng.binomial(lost, self.wipe_rate)
            recovered = self.rng.binomial(bricked, RECOVERY_RATE)
            replaced = bricked - recovered

            total += recovered * REPROVISION_COST

            n_replaced = int(replaced.sum())
            if n_replaced == 0:
                continue

            # Lognormal on unit cost, median-corrected so E[unit] == REPLACEMENT_COST.
            mu = np.log(REPLACEMENT_COST[c]) - 0.5 * COST_SIGMA ** 2
            unit_costs = self.rng.lognormal(mu, COST_SIGMA, size=n_replaced)
            idx = np.repeat(np.arange(num_sims), replaced)
            np.add.at(total, idx, unit_costs)

        return total

    # -- diagnostics ---------------------------------------------------------

    def diagnostics(self, num_sims=100_000):
        bricked = self.expected_devices_bricked()
        sims = self.simulate_annual(num_sims)
        return {
            'fleet': {c: round(v, 1) for c, v in self.fleet.items()},
            'total_devices': round(sum(self.fleet.values()), 1),
            'gross_fleet_value': round(
                sum(self.fleet[c] * REPLACEMENT_COST[c] for c in ASSET_CLASSES), 2),
            'wipe_capability': round(self.wipe_capability, 3),
            'wipe_rate': round(self.wipe_rate, 3),
            'mobility': round(self.mobility, 3),
            'theft_rates': {c: round(v, 4) for c, v in self.theft_rates.items()},
            'expected_devices_bricked_per_year': {
                c: round(v, 3) for c, v in bricked.items()},
            'expected_devices_bricked_total': round(sum(bricked.values()), 3),
            'expected_cost_per_bricking_event': round(
                self.expected_cost_per_bricking_event(), 2),
            'expected_annual_loss': round(self.expected_annual_loss(), 2),
            'simulated_annual_mean': round(float(np.mean(sims)), 2),
            'simulated_p50': round(float(np.percentile(sims, 50)), 2),
            'simulated_p95': round(float(np.percentile(sims, 95)), 2),
            'simulated_p99': round(float(np.percentile(sims, 99)), 2),
            'loss_free_year_probability': round(float(np.mean(sims == 0)), 4),
        }
