import openai
import os
import json
from .parse_security_LLM import get_security_scores
from .parse_security import check_automatic_denial
from anthropic import Anthropic
from math import floor
import heapq
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY must be set in the environment")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai.api_key = OPENAI_API_KEY

questions_noPentest = {
    "pentestCriticalResolved": "Have you fixed all critical or high-risk issues from your penetration test?",
    "pentestCritical30Days": "Will you fix them within 60 days and can provide proof if required?",
    "useMFA": "Who is required to use a verification code, such as a text message or app, to log into your systems?",
    "remoteAccess": "How do people log in to your systems when they are not in the office?",
    "remoteAccessTypes": "Which of these employees log in from outside the office, and how many employees have access in each corresponding group?",
    "lockAccount": "Are admin accounts automatically locked after three failed login attempts?",
    "receiveAlert": "Provide the name of any application you use to receive alerts on unusual activity on administrator accounts.",
    "emailSecurityService": "What dedicated email security service filters spam/phishing and scans links and attachments?",
    "externalEmailsFlagged": "Are emails from outside your company visually flagged so staff can spot impersonation?",
    "verifyMoneyRequests": "How do you verify requests to send money or change bank details?",
    "dualAuthorization": "Do payments above a set dollar amount require approval from a second person?",
    "dataEncryption": "Is customer and business data protected via encryption so it can't be read if stolen, both when stored and when sent?",
    "dataBacking": "How often do you back up your data?",
    "restoreBackupTest": "How often do you test restoring data from a backup?",
    "storeCustomersCardNumber": "Do you store customers' full card numbers on your own systems?",
    "fixProblems": "Does your company apply security updates to all work devices and software within 60 days of release?",
    "updatesIn30Days": "Will you start applying updates within 60 days going forward?",
    "networkDivided": "Is your network divided so guest Wi-Fi, payment systems, and office devices are kept separate?",
    "thirdPartyAppApproval": "Does an admin approve third-party apps before employees connect them to Microsoft or Google accounts?",
    "edrToolsUsed": "Do you use security software to detect threats on your network?",
    "securityTraining": "How often do employees receive formal cybersecurity training?",
    "thirdPartyControls": "Do you control or limit how outside vendors access your systems?",
    "noIncidentResponse": "Upload your 'incident response plan' for responding to security incidents.",
}

QUESTION_MAX_SCORES = {
    "pentestCriticalResolved": 10,
    "pentestCritical30Days": 4,
    "useMFA": 10,
    "remoteAccess": 5,
    "lockAccount": 3,
    "receiveAlert": 4,
    "emailSecurityService": 4,
    "externalEmailsFlagged": 4,
    "verifyMoneyRequests": 4,
    "dualAuthorization": 4,
    "dataEncryption": 4,
    "dataBacking": 10,
    "restoreBackupTest": 6,
    "storeCustomersCardNumber": 3,
    "fixProblems": 5,
    "updatesIn30Days": 5,
    "networkDivided": 4,
    "thirdPartyAppApproval": 4,
    "edrToolsUsed": 10,
    "securityTraining": 3,
    "thirdPartyControls": 5,
    "noIncidentResponse": 8,
}

# These documented payload fields are alternate representations of controls
# already listed above. They are deliberately excluded from
# `questions_noPentest` so they cannot produce duplicate recommendations.
# Their scores are aliases and must not be added together with the canonical
# control score.
AUXILIARY_QUESTION_MAX_SCORES = {
    "noReceiveAlertApp": 4,
    "noEmailSecurityService": 4,
    "incidentFileUrl": 8,
    "incidentFile": 8,
}

MANDATORY_PRIORITY_RULES = [
    {
        "id": "no_mfa",
        "keys": ["useMFA", "mfaTypes"],
        "answers": {"nobody", "none"},
        "vulnerability": "No multi-factor authentication for system logins",
        "recommendation": "Require MFA for all users, starting with administrators, executives, finance, IT, and remote access users.",
    },
    {
        "id": "no_backups",
        "keys": ["dataBacking"],
        "answers": {
            "we do not backup our files",
            "we do not back up our files",
            "we do not backup our data",
            "we do not back up our data",
            "we don't back up our data",
            "we don t back up our data",
            "we don’t back up our data",
        },
        "vulnerability": "No data backup program",
        "recommendation": "Implement automated, encrypted backups with defined retention and restore testing.",
    },
    {
        "id": "no_threat_detection",
        "keys": ["edrToolsUsed", "edrTools"],
        "answers": {
            "we do not have this",
            "we do not use security software to detect threats",
            "none",
            "no",
        },
        "vulnerability": "No network or endpoint threat detection software",
        "recommendation": "Deploy managed endpoint and network threat detection with alert triage and escalation.",
    },
]

# Rating hierarchy for improvements
RATING_ORDER = ["U", "B-", "B", "B+", "A-", "A", "A+"]

# Rating to probability mapping
RATING_PROBABILITY_MAP = {
    "A+": (2.0, 11.10),
    "A": (11.11, 22.21),
    "A-": (22.22, 33.32),
    "B+": (33.33, 44.43),
    "B": (44.44, 55.54),
    "B-": (55.55, 66.65),
    "U": (66.66, 98.0)
}


def normalize_rating(rating):
    """Normalize rating to standard format"""
    if not rating:
        return "B"
    
    rating = str(rating).strip().upper()
    
    if rating in ["UNINSURABLE", "U (UNINSURABLE)"]:
        return "U"
    
    if rating in RATING_ORDER:
        return rating
    
    return "B"


def get_rating_for_likelihood(likelihood_pct):
    """Rating implied by breach likelihood alone, ignoring automatic-denial overrides."""
    value = extract_numeric_value(likelihood_pct)

    for rating in reversed(RATING_ORDER):
        _, target_max = RATING_PROBABILITY_MAP[rating]
        if value <= target_max:
            return rating

    return "U"


def get_improvement_tiers(current_rating, current_likelihood=None):
    """
    Get all improvement tiers above the current rating through A+.
    Returns: (normalized_current_rating, list_of_improvement_tiers)
    """
    normalized_current = normalize_rating(current_rating)
    
    try:
        current_idx = RATING_ORDER.index(normalized_current)
    except ValueError:
        normalized_current = "B"
        current_idx = RATING_ORDER.index(normalized_current)

    # An automatic questionnaire failure forces the rating to U even when breach
    # likelihood already sits in a better band. Starting from the rating alone would
    # make the first tier target a likelihood worse than the company's current one,
    # which yields a tier with no achievable loss reduction.
    start_idx = current_idx
    if current_likelihood is not None:
        likelihood_idx = RATING_ORDER.index(get_rating_for_likelihood(current_likelihood))
        start_idx = max(start_idx, likelihood_idx)

    if start_idx >= len(RATING_ORDER) - 1:
        return normalized_current, []

    improvement_tiers = RATING_ORDER[start_idx + 1:]
    
    return normalized_current, improvement_tiers


# Automatic-denial triggers from check_automatic_denial, and what clears them.
AUTOMATIC_DENIAL_BLOCKERS = {
    "no_mfa": {
        "question_key": "useMFA",
        "gap": "No multi-factor authentication is required for system logins",
        "required_action": (
            "Require MFA for all users, starting with administrators, executives, finance, "
            "IT, and remote access users."
        ),
    },
    "no_edr": {
        "question_key": "edrToolsUsed",
        "gap": "No security software is deployed to detect threats on the network",
        "required_action": (
            "Deploy an endpoint detection and response tool across all endpoints and servers."
        ),
    },
    "no_backups": {
        "question_key": "dataBacking",
        "gap": "Company data is not backed up",
        "required_action": (
            "Establish scheduled backups of critical systems with offsite or immutable copies."
        ),
    },
}


def get_insurability_blockers(security_questions):
    """
    Automatic-denial triggers that hold the rating at U regardless of breach likelihood.

    Clearing these is what makes the company insurable, so they carry no likelihood
    or loss impact and are reported separately from the graded improvement tiers.
    """
    triggered = check_automatic_denial(security_questions or {})

    return [
        dict(AUTOMATIC_DENIAL_BLOCKERS[trigger])
        for trigger in triggered
        if trigger in AUTOMATIC_DENIAL_BLOCKERS
    ]


def extract_numeric_value(value_str):
    """Extract numeric value from string (handles $, %, commas, etc.)"""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return 0.0
    
    cleaned = value_str.replace('$', '').replace(',', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _normalize_answer(value):
    return " ".join(str(value or "").replace("'", " ").replace("’", " ").strip().lower().split())


def _answer_matches(value, expected_answers):
    if isinstance(value, list):
        values = value
    elif value in (None, ""):
        values = []
    else:
        values = [value]

    normalized_expected = {_normalize_answer(answer) for answer in expected_answers}
    return any(_normalize_answer(item) in normalized_expected for item in values)


def get_mandatory_priority_gaps(security_questions):
    gaps = []
    seen = set()
    for rule in MANDATORY_PRIORITY_RULES:
        for key in rule["keys"]:
            if key in security_questions and _answer_matches(security_questions.get(key), rule["answers"]):
                if rule["id"] not in seen:
                    gaps.append({
                        "question_key": rule["keys"][0],
                        "vulnerability": rule["vulnerability"],
                        "recommendation": rule["recommendation"],
                    })
                    seen.add(rule["id"])
                break
    return gaps


def format_mandatory_priority_gaps(mandatory_priority_gaps):
    if not mandatory_priority_gaps:
        return "No mandatory priority questionnaire gaps were triggered."

    return "\n".join(
        f"{index}. Vulnerability: {gap['vulnerability']}\n"
        f"   Required recommendation focus: {gap['recommendation']}"
        for index, gap in enumerate(mandatory_priority_gaps, start=1)
    )


def _selected_values(value):
    if isinstance(value, list):
        return [_normalize_answer(item) for item in value]
    if value in (None, ""):
        return []
    return [_normalize_answer(value)]


def _first_non_blank(security_questions, keys):
    for key in keys:
        value = security_questions.get(key)
        if value not in (None, "", [], False):
            return value
    return None


def _get_admin_alert_answer(security_questions):
    answer = _first_non_blank(
        security_questions,
        ("receiveAlert", "ReceiveAlertApp"),
    )
    if answer is not None:
        return answer
    if security_questions.get("noReceiveAlertApp") is True:
        return "No application is used to monitor unusual administrator activity"
    return None


def _get_email_security_answer(security_questions):
    answer = _first_non_blank(security_questions, ("emailSecurityService",))
    if answer is not None:
        return answer
    if security_questions.get("noEmailSecurityService") is True:
        return "No dedicated email security service is used"
    return None


def _get_incident_response_answer(security_questions):
    answer = _first_non_blank(
        security_questions,
        (
            "incidentFileUrl",
            "incidentFile",
            "incidentResponsePlan",
            "incidentResponseDocument",
        ),
    )
    if answer is not None:
        return answer
    if security_questions.get("noIncidentResponse") is True:
        return "No incident response plan uploaded"
    if security_questions.get("noIncidentResponse") is False:
        return "Incident response plan uploaded"
    return None


def add_auxiliary_question_scores(security_questions, scores):
    """Attach non-additive scores for documented auxiliary payload fields."""
    admin_alert_answer = _first_non_blank(
        security_questions,
        ("receiveAlert", "ReceiveAlertApp"),
    )
    admin_alert_score = (
        AUXILIARY_QUESTION_MAX_SCORES["noReceiveAlertApp"]
        if admin_alert_answer is not None
        and security_questions.get("noReceiveAlertApp") is not True
        else 0
    )

    email_security_answer = _first_non_blank(
        security_questions,
        ("emailSecurityService",),
    )
    email_security_score = (
        AUXILIARY_QUESTION_MAX_SCORES["noEmailSecurityService"]
        if email_security_answer is not None
        and security_questions.get("noEmailSecurityService") is not True
        else 0
    )

    incident_response_artifact = _first_non_blank(
        security_questions,
        (
            "incidentFileUrl",
            "incidentFile",
            "incidentResponsePlan",
            "incidentResponseDocument",
        ),
    )
    incident_response_score = (
        AUXILIARY_QUESTION_MAX_SCORES["incidentFileUrl"]
        if (
            incident_response_artifact is not None
            or security_questions.get("noIncidentResponse") is False
        )
        and security_questions.get("noIncidentResponse") is not True
        else 0
    )

    scores["receiveAlert"] = admin_alert_score
    scores["emailSecurityService"] = email_security_score
    scores["noIncidentResponse"] = incident_response_score
    scores["noReceiveAlertApp"] = admin_alert_score
    scores["noEmailSecurityService"] = email_security_score
    scores["incidentFileUrl"] = (
        incident_response_score
        if security_questions.get("incidentFileUrl") not in (None, "", [], False)
        else 0
    )
    scores["incidentFile"] = (
        incident_response_score
        if security_questions.get("incidentFile") not in (None, "", [], False)
        else 0
    )
    return scores


# remoteAccessTypes is headcount context. It is not a scored control and cannot
# be used to justify a recommendation.
CONTEXT_ONLY_QUESTION_KEYS = frozenset({"remoteAccessTypes"})

# These pairs describe one control asked as a follow-up. A recommendation sourced
# to either key is about that same area.
SAME_CONTROL_GROUPS = (
    frozenset({"fixProblems", "updatesIn30Days"}),
)

QUESTION_RECOMMENDATION_TERMS = {
    "useMFA": (
        "mfa", "multi factor", "multi-factor", "verification code",
        "2fa", "two-factor", "two factor", "authenticator", "passkey", "fido",
        "phishing-resistant", "phishing resistant", "hardware key", "number matching",
    ),
    "remoteAccess": (
        "remote access", "remote login", "vpn", "rdp", "ztna",
        "device posture", "posture check", "compliant device", "device compliance",
        "zero trust", "conditional access", "remote desktop",
    ),
    "lockAccount": (
        "account lockout", "failed login", "failed attempt", "brute force",
        "lockout", "locked out",
    ),
    "receiveAlert": ("administrator alert", "admin alert", "administrative activity alert"),
    "emailSecurityService": (
        "email security", "spam filter", "phishing filter", "secure email gateway",
        "email filtering", "anti-phishing", "attachment sandboxing", "link scanning",
    ),
    "externalEmailsFlagged": ("external email", "email banner", "impersonation banner"),
    "verifyMoneyRequests": (
        "bank detail", "money transfer", "payment request", "callback verification",
        "call back", "out-of-band", "out of band", "bank change",
    ),
    "dualAuthorization": ("dual approval", "dual authorization", "two person approval", "second approval"),
    # Avoid bare "encrypted": it also matches "encrypted backups".
    "dataEncryption": (
        "encryption", "data encryption", "encrypt data", "encrypted data", "unencrypted",
        "cryptographic", "tls", "at rest", "in transit", "key management",
    ),
    "storeCustomersCardNumber": ("card number", "card storage", "payment card", "pci"),
    "fixProblems": ("patch", "security update", "updates within 60", "vulnerability remediation"),
    "updatesIn30Days": ("patch", "security update", "updates within 60", "vulnerability remediation"),
    "networkDivided": (
        "network segment", "segmentation", "flat network", "network divided",
        "vlan", "microsegmentation", "guest network", "guest wi-fi", "guest wifi",
    ),
    "thirdPartyAppApproval": ("third party app", "third-party app", "oauth", "connected app"),
    "edrToolsUsed": (
        "edr", "endpoint detection", "endpoint protection", "threat detection",
        "threat hunting", "security software",
        "xdr", "mdr", "managed detection", "antivirus", "anti-malware",
    ),
    "securityTraining": (
        "security training", "cybersecurity training", "security awareness", "phishing simulation",
        "awareness training", "phishing test", "phishing exercise",
    ),
    "thirdPartyControls": (
        "vendor access", "third party access", "third-party access",
        "third-party vendor", "third parties", "supplier",
        "contractor access", "partner access",
        "vendor risk", "vendor management", "third-party risk", "third party risk",
    ),
    "noIncidentResponse": (
        "incident response plan", "incident response document", "ir plan",
        "incident response", "tabletop", "ir playbook",
    ),
    # Bare "backup" is intentionally absent so a restore-testing recommendation
    # is not treated as a new backup program.
    "dataBacking": (
        "immutable", "offsite", "off-site", "3-2-1",
        "backup frequency", "backup cadence", "backup schedule",
        "immutable backup", "offsite backup", "implement backup", "establish backup",
    ),
    "restoreBackupTest": (
        "restore test", "restore testing", "recovery point", "recovery test", "recovery drill",
    ),
}


def _third_party_controls_are_best_available(answer):
    """Yes and the no-vendor answers are the best options this question offers."""
    normalized = _normalize_answer(answer)
    return normalized == "yes" or normalized.startswith("we don t give vendors access")


def _card_storage_is_best_available(answer):
    """No means the payment processor stores the cards. That is the best option."""
    normalized = _normalize_answer(answer)
    if normalized == "no":
        return True
    return normalized.startswith("no") and "payment processor" in normalized


def get_max_scored_question_keys(security_responses):
    """Return answered questionnaire keys whose earned score meets their maximum.

    thirdPartyControls is included for Yes and for answers that start with
    "we don't give vendors access" even when the points stay below 5.
    storeCustomersCardNumber is included for No and for the payment-processor
    answer; those answers also print as the best available status.
    """
    keys = {
        response["key"]
        for response in security_responses
        if response.get("max_score") is not None
        and response.get("score") is not None
        and float(response["score"]) >= float(response["max_score"])
    }
    for response in security_responses:
        if (
            response.get("key") == "thirdPartyControls"
            and _third_party_controls_are_best_available(response.get("answer"))
        ):
            keys.add("thirdPartyControls")
        if (
            response.get("key") == "storeCustomersCardNumber"
            and _card_storage_is_best_available(response.get("answer"))
        ):
            keys.add("storeCustomersCardNumber")
    return keys


def format_max_scored_controls(security_responses, max_scored_keys):
    controls = [
        f"- {response['key']}: {response['question']} (answer: {response['answer']})"
        for response in security_responses
        if response["key"] in max_scored_keys
    ]
    return "\n".join(controls) or "No questionnaire controls earned maximum points."


def _controls_are_same_topic(left, right):
    if not left or not right:
        return False
    if left == right:
        return True
    return any(left in group and right in group for group in SAME_CONTROL_GROUPS)


def infer_questionnaire_keys_from_recommendation(recommendation):
    """Best-effort topic guard for recommendations mislabeled as general."""
    text = _normalize_answer(
        " ".join(
            str(recommendation.get(field, ""))
            for field in ("vulnerability", "recommendation", "example")
        )
    )
    matched = {
        key
        for key, terms in QUESTION_RECOMMENDATION_TERMS.items()
        if any(_normalize_answer(term) in text for term in terms)
    }

    if "backup" in text:
        if "restore" in text or "recovery test" in text or "recovery drill" in text:
            matched.add("restoreBackupTest")
        if any(term in text for term in (
            "backup frequency", "backup cadence", "backup schedule", "daily backup",
            "weekly backup", "implement backup", "establish backup", "immutable backup",
            "offsite backup", "3-2-1", "3 2 1",
        )):
            matched.add("dataBacking")

    return matched


def _blocked_topics_for_recommendation(recommendation, ineligible_keys, source_question_key=None):
    """Max-score or best-answer topics this text discusses, excluding its own source."""
    topics = infer_questionnaire_keys_from_recommendation(recommendation) & set(ineligible_keys)
    return {
        topic
        for topic in topics
        if not _controls_are_same_topic(source_question_key, topic)
    }


def validate_recommendation_sources(
    recommendation,
    max_scored_keys,
    questionnaire_keys,
    pentest_finding_names,
):
    """Validate provenance and prevent recommendations for ineligible controls.

    `max_scored_keys` includes numeric maximums and best-answer controls that are
    off limits even when their point value is below the printed maximum, such as
    VPN-with-code remote access.
    """
    source_type = recommendation.get("source_type")
    source_question_key = recommendation.get("source_question_key")
    source_pentest_finding = recommendation.get("source_pentest_finding")
    ineligible_keys = set(max_scored_keys or [])

    if source_type not in {"questionnaire", "pentest", "general"}:
        return "source_type must be questionnaire, pentest, or general"

    if source_question_key in CONTEXT_ONLY_QUESTION_KEYS:
        return f"{source_question_key} is context only and is not a valid source key"

    if source_type == "questionnaire":
        if source_question_key not in questionnaire_keys:
            return "questionnaire source requires a valid source_question_key"
        if source_question_key in ineligible_keys:
            return f"questionnaire control {source_question_key} is not eligible for a recommendation"
    elif source_type == "pentest":
        normalized_finding = _normalize_answer(source_pentest_finding)
        if not normalized_finding or normalized_finding not in pentest_finding_names:
            return "pentest source requires an exact provided vulnerability name"
        finding_topics = infer_questionnaire_keys_from_recommendation({
            "vulnerability": source_pentest_finding,
            "recommendation": "",
        })
        unrelated = {
            topic
            for topic in _blocked_topics_for_recommendation(recommendation, ineligible_keys)
            if topic not in finding_topics
            and not any(_controls_are_same_topic(topic, finding_topic) for finding_topic in finding_topics)
        }
        if unrelated:
            return (
                "pentest finding does not support a recommendation about ineligible control(s): "
                + ", ".join(sorted(unrelated))
            )
    elif source_question_key not in (None, ""):
        return "general source must not set source_question_key"

    # A pentest finding may discuss an ineligible control only when the finding
    # itself is about that control. That case is handled above.
    if source_type != "pentest":
        blocked_topics = _blocked_topics_for_recommendation(
            recommendation,
            ineligible_keys,
            source_question_key if source_type == "questionnaire" else None,
        )
        if blocked_topics:
            return (
                "recommendation concerns ineligible questionnaire control(s): "
                + ", ".join(sorted(blocked_topics))
            )

    return None


def strip_recommendation_source_metadata(parsed):
    """Remove internal provenance fields from the public response."""
    for tier in parsed.get("grade_improvement_recommendations", []):
        recommendations = tier.get("recommendations_improved_rating", {}).get("recommendations", [])
        for recommendation in recommendations:
            recommendation.pop("source_type", None)
            recommendation.pop("source_question_key", None)
            recommendation.pop("source_pentest_finding", None)
    return parsed


def _has_selected_value(security_questions, keys, expected_values):
    normalized_expected = {_normalize_answer(value) for value in expected_values}
    for key in keys:
        if any(value in normalized_expected for value in _selected_values(security_questions.get(key))):
            return True
    return False


def _remote_access_is_in_office(answer):
    """True when the stored answer says people do not log in from outside the office."""
    normalized = _normalize_answer(answer)
    if normalized in {"nobody", "no"} or normalized.startswith("nobody"):
        return True
    full_label = _normalize_answer(
        "Nobody works outside the office (such as from home, a coffee shop, or while traveling)"
    )
    return normalized == full_label


def _remote_method_was_not_asked(security_questions):
    """Pentest flow stores Yes/No. A blank method with remote headcount is the same case."""
    answer = _normalize_answer(security_questions.get("remoteAccess"))
    if answer == "yes":
        return True
    remote_missing = security_questions.get("remoteAccess") in (None, "", [])
    remote_types = security_questions.get("remoteAccessTypes")
    return remote_missing and remote_types not in (None, "", [])


def get_recommendation_blocked_keys(security_questions, max_scored_keys):
    """Controls that must not be recommended, including best answers below the point maximum.

    If either patching question is already a best answer, both are ineligible.
    They are one control asked as a follow-up.
    """
    blocked = set(max_scored_keys or [])
    answer = _normalize_answer(security_questions.get("remoteAccess"))
    if (
        answer == "vpnwithcode"
        or _remote_access_is_in_office(security_questions.get("remoteAccess"))
        or _remote_method_was_not_asked(security_questions)
    ):
        blocked.add("remoteAccess")
    for group in SAME_CONTROL_GROUPS:
        if blocked & group:
            blocked |= set(group)
    return blocked


def get_protected_control_facts(security_questions, scores):
    facts = []

    remote_answer = _normalize_answer(security_questions.get("remoteAccess"))
    if _remote_access_is_in_office(security_questions.get("remoteAccess")):
        facts.append(
            "The company states nobody works outside the office or no key employees "
            "log in remotely. Do not recommend VPN deployment, remote-access "
            "hardening, RDP controls, or remote-login monitoring as if remote access "
            "is currently used."
        )
    elif remote_answer == "vpnwithcode":
        facts.append(
            "Remote access is through a VPN with a password and a verification code. "
            "This is the strongest remote-access option this questionnaire assesses. "
            "Do not describe remote access as weak, incomplete, or lacking any control. "
            "Do not recommend remote-access hardening, device posture or compliance "
            "checks, ZTNA, VPN replacement, or remote-login monitoring."
        )
    elif _remote_method_was_not_asked(security_questions):
        facts.append(
            "The questionnaire did not ask how remote access is secured. Do not make "
            "any statement about how remote access is protected, and do not recommend "
            "remote-access changes unless a provided pentest finding names a VPN, "
            "remote desktop, or remote-access issue."
        )

    edr_answer = security_questions.get("edrToolsUsed", security_questions.get("edrTools"))
    if scores.get("edrToolsUsed") == QUESTION_MAX_SCORES["edrToolsUsed"]:
        facts.append(
            "Threat detection/EDR is implemented. The selected tools are "
            f"{edr_answer}. SentinelOne, Sentinel One, Microsoft Defender, "
            "CrowdStrike Falcon, and Other all count as implemented threat "
            "detection/EDR for this questionnaire. Do not describe EDR, endpoint "
            "detection, threat detection, or security software as absent, missing, "
            "not deployed, or lacking."
        )

    mfa_values = set(_selected_values(security_questions.get("useMFA", security_questions.get("mfaTypes"))))
    leadership_group = _normalize_answer(
        "Leadership and security teams (executives, cybersecurity, recovery teams)"
    )
    admin_group = _normalize_answer(
        "Admins or high-access users (IT admins, developers, finance/HR, vendors, etc.)"
    )
    all_employees_group = _normalize_answer("All Employees")
    if all_employees_group in mfa_values:
        facts.append(
            "MFA is already required for all employees, including for remote logins. "
            "Do not recommend implementing or expanding MFA, and do not describe MFA "
            "as missing or incomplete anywhere in the response."
        )
    if leadership_group in mfa_values:
        facts.append(
            "MFA already covers executives, cybersecurity professionals, and "
            "emergency/recovery teams. Do not recommend MFA expansion to those groups."
        )
    if admin_group in mfa_values:
        facts.append(
            "MFA already covers IT admins, developers, finance/HR, vendors, and other "
            "high-access users. Do not recommend MFA expansion to those groups."
        )
    if leadership_group in mfa_values and admin_group in mfa_values:
        facts.append(
            "The only valid MFA recommendation for this response is expansion from key "
            "and high-access users to all employees, if MFA is recommended at all. Do "
            "not call current MFA coverage incomplete for key employees."
        )

    no_vendor_access_values = {
        "we don t give vendors access to our systems or data",
        "i do not share data or systems access with other businesses",
    }
    if _has_selected_value(security_questions, ["thirdPartyControls"], no_vendor_access_values):
        facts.append(
            "The company explicitly states it does not give vendors access to systems "
            "or data. Do not recommend vendor access management, conditional access "
            "for vendors, vendor activity logging, vendor portals, or vendor access "
            "reviews as if vendors currently have access."
        )
    if _normalize_answer(security_questions.get("thirdPartyControls")) == "yes":
        facts.append(
            "The company controls and limits how outside vendors access its systems. "
            "Do not make any recommendation about vendors or third parties, including "
            "vendor access, vendor risk management, vendor risk assessments, or vendor documentation."
        )

    backup_score = scores.get("dataBacking", 0)
    restore_score = scores.get("restoreBackupTest", 0)
    if backup_score > 0:
        facts.append(
            "The company has a data backup cadence. If restore testing is weak, "
            "recommend backup restore testing, not implementing backups from scratch."
        )
    if backup_score == QUESTION_MAX_SCORES["dataBacking"]:
        facts.append(
            "Daily backups are already implemented. Do not recommend implementing "
            "or increasing backup frequency; only recommend restore testing if that "
            "specific answer is not at maximum."
        )
    if restore_score == QUESTION_MAX_SCORES["restoreBackupTest"]:
        facts.append(
            "Backup restore testing is already at the maximum score. Do not recommend "
            "testing restores more frequently."
        )

    if _card_storage_is_best_available(security_questions.get("storeCustomersCardNumber")):
        facts.append(
            "The company does not store full card numbers; its payment processor stores them. "
            "The company may keep only tokens or the last four digits, which is already the "
            "recommended practice. Do not describe the company as storing card data, partially "
            "or otherwise. Do not recommend tokenizing, truncating, removing, minimizing, or "
            "documenting card data, or reducing PCI scope."
        )

    if scores.get("emailSecurityService") == QUESTION_MAX_SCORES["emailSecurityService"]:
        facts.append(
            "Dedicated email security is implemented. Do not recommend deploying "
            "email security filtering as a missing control."
        )
    admin_alert_score = scores.get("receiveAlert", scores.get("ReceiveAlertApp"))
    if admin_alert_score == QUESTION_MAX_SCORES["receiveAlert"]:
        facts.append(
            "Administrator unusual-activity alerting is implemented. Do not recommend "
            "adding an alerting application as a missing control."
        )
    if scores.get("externalEmailsFlagged") == QUESTION_MAX_SCORES["externalEmailsFlagged"]:
        facts.append(
            "External email flagging is implemented. Do not recommend adding external "
            "email banners as a missing control."
        )
    if scores.get("networkDivided") == QUESTION_MAX_SCORES["networkDivided"]:
        facts.append(
            "Network segmentation is implemented. Do not recommend segmenting the "
            "network as if it is flat or unsegmented."
        )
    if scores.get("thirdPartyAppApproval") == QUESTION_MAX_SCORES["thirdPartyAppApproval"]:
        facts.append(
            "Admin approval for third-party apps is implemented. Do not recommend "
            "adding third-party app approval as a missing control."
        )
    incident_response_score = scores.get(
        "noIncidentResponse",
        scores.get("incidentResponsePlan")
    )
    if incident_response_score == QUESTION_MAX_SCORES["noIncidentResponse"]:
        facts.append(
            "An incident response plan is uploaded. Do not recommend creating or "
            "uploading an incident response plan as if one does not exist."
        )
    if scores.get("fixProblems") == QUESTION_MAX_SCORES["fixProblems"]:
        facts.append(
            "Security updates are handled within the required 60-day window or the "
            "company committed to doing so. Do not recommend starting 60-day patching "
            "as if it is absent."
        )
    if security_questions.get("lockAccount") == "No":
        facts.append(
            "The admin lockout questionnaire asks whether admin accounts are locked "
            "after three failed login attempts. If recommending account lockout, use "
            "the same three-failed-attempt threshold. Do not recommend five failed "
            "attempts as the target control because that does not satisfy the specific "
            "questionnaire gap."
        )
    if scores.get("lockAccount") == QUESTION_MAX_SCORES["lockAccount"]:
        facts.append(
            "Admin account lockout is implemented. Do not recommend adding automatic "
            "admin lockout as if it is missing."
        )

    if not facts:
        return "No protected control facts were triggered."

    return "\n".join(f"- {fact}" for fact in facts)


def build_mandatory_priority_recommendations(mandatory_priority_gaps, company_data):
    industry = company_data.get("primary_industry_name") or "the organization"
    examples = {
        "No multi-factor authentication for system logins": (
            f"For {industry}, enforce MFA through the identity provider for email, VPN, cloud apps, "
            "admin consoles, and privileged service accounts. Start with admins and executives, then "
            "require all employees to enroll in an authenticator app or hardware key."
        ),
        "No data backup program": (
            f"For {industry}, schedule daily encrypted backups for production systems, databases, "
            "documents, and critical SaaS exports. Store at least one immutable or offline copy and "
            "run restore tests on a documented cadence."
        ),
        "No network or endpoint threat detection software": (
            f"For {industry}, deploy EDR or managed detection across servers, laptops, and key network "
            "segments. Route alerts to an owner or MDR provider with escalation steps for malware, "
            "credential theft, and suspicious remote access."
        ),
    }

    recommendations = []
    for gap in mandatory_priority_gaps:
        recommendations.append({
            "vulnerability": gap["vulnerability"],
            "recommendation": gap["recommendation"],
            "example": examples.get(
                gap["vulnerability"],
                f"For {industry}, implement this control with documented ownership, rollout milestones, and verification."
            ),
            "impact_contribution": 100,
            "source_type": "questionnaire",
            "source_question_key": gap["question_key"],
            "source_pentest_finding": None,
        })
    return recommendations


def enforce_mandatory_priority_recommendations(parsed, mandatory_priority_gaps, company_data):
    if not mandatory_priority_gaps:
        return parsed

    tiers = parsed.get("grade_improvement_recommendations", [])
    if not tiers:
        return parsed

    first_rating_data = tiers[0].get("recommendations_improved_rating", {})
    existing_recommendations = first_rating_data.get("recommendations", [])
    mandatory_recommendations = build_mandatory_priority_recommendations(
        mandatory_priority_gaps,
        company_data
    )
    mandatory_vulnerabilities = {
        _normalize_answer(rec["vulnerability"])
        for rec in mandatory_recommendations
    }

    remaining_recommendations = [
        rec for rec in existing_recommendations
        if _normalize_answer(rec.get("vulnerability")) not in mandatory_vulnerabilities
    ]

    first_rating_data["recommendations"] = (
        mandatory_recommendations + remaining_recommendations
    )[:5]
    return parsed


def _recommendation_identity(recommendation):
    return _normalize_answer(
        " ".join(
            str(recommendation.get(field, ""))
            for field in ("vulnerability", "recommendation")
        )
    )


def dedupe_recommendations(parsed):
    """Keep the first copy of each recommendation across every tier.

    Tiers are not padded back to five. A later tier that only repeated earlier
    advice is left shorter rather than filled with unassessed governance text.
    """
    tiers = parsed.get("grade_improvement_recommendations", [])
    seen = set()

    for tier in tiers:
        rating_data = tier.get("recommendations_improved_rating", {})
        unique_recommendations = []
        for rec in rating_data.get("recommendations", []):
            key = _recommendation_identity(rec)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_recommendations.append(rec)
        rating_data["recommendations"] = unique_recommendations

    return parsed


def attach_recommendation_basis(parsed):
    """Public label for whether a line comes from the client's evidence or is general practice."""
    for tier in parsed.get("grade_improvement_recommendations", []):
        recommendations = tier.get("recommendations_improved_rating", {}).get("recommendations", [])
        for recommendation in recommendations:
            if recommendation.get("source_type") in {"questionnaire", "pentest"}:
                recommendation["basis"] = "Based on your answers"
            else:
                recommendation["basis"] = "Recommended practice"
    return parsed


def sanitize_protected_control_contradictions(parsed, security_questions, scores):
    tiers = parsed.get("grade_improvement_recommendations", [])
    if not tiers:
        return parsed

    edr_is_implemented = scores.get("edrToolsUsed") == QUESTION_MAX_SCORES["edrToolsUsed"]
    edr_answer = security_questions.get("edrToolsUsed", security_questions.get("edrTools"))
    remote_answer = _normalize_answer(security_questions.get("remoteAccess"))
    no_remote_access = remote_answer in {"nobody", "no"}

    mfa_values = set(_selected_values(security_questions.get("useMFA", security_questions.get("mfaTypes"))))
    leadership_group = _normalize_answer(
        "Leadership and security teams (executives, cybersecurity, recovery teams)"
    )
    admin_group = _normalize_answer(
        "Admins or high-access users (IT admins, developers, finance/HR, vendors, etc.)"
    )
    all_employees_group = _normalize_answer("All Employees")
    mfa_all_employees = all_employees_group in mfa_values
    mfa_covers_key_groups = leadership_group in mfa_values and admin_group in mfa_values
    admin_lockout_missing = security_questions.get("lockAccount") == "No"
    admin_lockout_full = scores.get("lockAccount") == QUESTION_MAX_SCORES["lockAccount"]

    no_vendor_access = _has_selected_value(
        security_questions,
        ["thirdPartyControls"],
        {
            "we don t give vendors access to our systems or data",
            "i do not share data or systems access with other businesses",
        }
    )

    has_backups = scores.get("dataBacking", 0) > 0
    daily_backups = scores.get("dataBacking") == QUESTION_MAX_SCORES["dataBacking"]
    restore_testing_full = scores.get("restoreBackupTest") == QUESTION_MAX_SCORES["restoreBackupTest"]
    no_card_storage = _card_storage_is_best_available(
        security_questions.get("storeCustomersCardNumber")
    )
    email_security_full = scores.get("emailSecurityService") == QUESTION_MAX_SCORES["emailSecurityService"]
    admin_alert_full = (
        scores.get("receiveAlert", scores.get("ReceiveAlertApp"))
        == QUESTION_MAX_SCORES["receiveAlert"]
    )
    external_banner_full = scores.get("externalEmailsFlagged") == QUESTION_MAX_SCORES["externalEmailsFlagged"]
    network_segmented = scores.get("networkDivided") == QUESTION_MAX_SCORES["networkDivided"]
    app_approval_full = scores.get("thirdPartyAppApproval") == QUESTION_MAX_SCORES["thirdPartyAppApproval"]
    ir_plan_uploaded = (
        scores.get("noIncidentResponse", scores.get("incidentResponsePlan"))
        == QUESTION_MAX_SCORES["noIncidentResponse"]
    )
    patching_full = scores.get("fixProblems") == QUESTION_MAX_SCORES["fixProblems"]

    for tier in tiers:
        recommendations = tier.get("recommendations_improved_rating", {}).get("recommendations", [])
        for rec in recommendations:
            # Pentest evidence is allowed to override a best questionnaire answer.
            if rec.get("source_type") == "pentest":
                continue

            text = _normalize_answer(
                " ".join(str(rec.get(field, "")) for field in ("vulnerability", "recommendation", "example"))
            )

            if edr_is_implemented and (
                ("absence" in text or "lack" in text or "missing" in text or "deploy" in text)
                and ("edr" in text or "endpoint detection" in text or "threat detection" in text)
            ):
                rec["vulnerability"] = "Existing threat detection operations can be tuned"
                rec["recommendation"] = (
                    "Tune and operationalize the existing threat detection tools instead of deploying EDR as a missing control."
                )
                rec["example"] = (
                    f"The selected tools ({edr_answer}) should be configured with alert owners, severity rules, "
                    "automated isolation for confirmed malware, and monthly review of detections across endpoints."
                )

            if mfa_covers_key_groups and (
                "mfa" in text
                and (
                    "key employee" in text
                    or "it staff" in text
                    or "cybersecurity" in text
                    or "recovery team" in text
                    or "executive" in text
                    or "admin" in text
                    or "high access" in text
                )
            ):
                rec["vulnerability"] = "MFA is not yet extended to all employees"
                rec["recommendation"] = (
                    "Expand MFA from covered key and high-access users to all remaining employees."
                )
                rec["example"] = (
                    "Current MFA already covers executives, cybersecurity, recovery teams, IT admins, "
                    "developers, finance/HR, vendors, and other high-access users. Extend enrollment to "
                    "all remaining standard employee accounts."
                )

            if mfa_all_employees and (
                "mfa" in text
                or "multi factor" in text
                or "multi-factor" in text
                or "verification code" in text
            ):
                rec["vulnerability"] = "MFA is already required for all employees"
                rec["recommendation"] = (
                    "Do not implement or expand MFA. Keep the existing all-employee requirement, including for remote logins."
                )
                rec["example"] = (
                    "MFA is already required for all employees. Maintain identity-provider reports showing enrollment, exceptions, disabled legacy authentication, and periodic reviews of bypassed accounts."
                )

            if no_vendor_access and (
                "vendor" in text
                and ("access" in text or "portal" in text or "credential" in text or "review" in text)
            ):
                rec["vulnerability"] = "Vendor access policy should remain explicit and verifiable"
                rec["recommendation"] = (
                    "Document and periodically verify the existing no-vendor-access policy instead of creating vendor access workflows."
                )
                rec["example"] = (
                    "Because vendors are not given systems or data access, maintain evidence such as identity-provider "
                    "reviews, procurement checks, and exceptions logs confirming no active vendor accounts exist."
                )

            if no_remote_access and (
                "remote access" in text or "vpn" in text or "rdp" in text or "remote login" in text
            ):
                rec["vulnerability"] = "Remote access policy should remain explicit and verified"
                rec["recommendation"] = (
                    "Document and periodically verify that remote access remains disabled or unused instead of deploying remote-access controls."
                )
                rec["example"] = (
                    "Because users do not log in from outside the office, maintain identity-provider and VPN reviews "
                    "showing no active remote-access paths, with an exception process if remote work is later introduced."
                )

            if has_backups and (
                "backup" in text
                and ("implement" in text or "establish" in text or "create" in text or "start" in text)
                and "restore" not in text
            ):
                rec["vulnerability"] = "Backup restore testing needs verification"
                rec["recommendation"] = (
                    "Test and document backup restores instead of treating backups as absent."
                )
                rec["example"] = (
                    "The company already has a backup cadence. Schedule restore tests for critical systems, "
                    "record recovery time, and verify restored data integrity."
                )
            if daily_backups and "backup" in text and ("frequency" in text or "daily" in text or "increase" in text):
                rec["vulnerability"] = "Backup restore resilience should be validated"
                rec["recommendation"] = (
                    "Validate restore reliability for existing daily backups rather than increasing backup frequency."
                )
                rec["example"] = (
                    "Daily backups are already in place. Run restore drills for priority systems and document "
                    "successful recovery evidence."
                )
            if restore_testing_full and "restore" in text and "backup" in text:
                rec["vulnerability"] = "Backup program monitoring can be improved"
                rec["recommendation"] = (
                    "Monitor backup job success and failed backup alerts instead of recommending more restore testing."
                )
                rec["example"] = (
                    "Restore testing is already at the maximum questionnaire score. Add monitoring for failed jobs, "
                    "missed schedules, and backup storage capacity."
                )

            if no_card_storage and (
                "card" in text or "pci" in text or "payment card" in text
            ) and ("store" in text or "storage" in text or "remove" in text or "token" in text):
                rec["vulnerability"] = "Card handling policy should remain documented"
                rec["recommendation"] = (
                    "Document and verify the current no-card-storage approach instead of recommending removal of stored card data."
                )
                rec["example"] = (
                    "Because full card numbers are not stored on company systems, keep processor attestations, "
                    "data-flow diagrams, and periodic checks confirming card data remains outside the environment."
                )
            if email_security_full and "email" in text and (
                "deploy" in text or "implement" in text or "add" in text or "missing" in text
            ):
                rec["vulnerability"] = "Email security configuration can be tuned"
                rec["recommendation"] = (
                    "Tune the existing email security service instead of deploying email filtering as a missing control."
                )
                rec["example"] = (
                    "Dedicated email security is already present. Review phishing policies, link detonation, "
                    "attachment scanning, and impersonation protection settings."
                )
            if admin_alert_full and ("admin" in text and "alert" in text):
                rec["vulnerability"] = "Administrator alert triage can be formalized"
                rec["recommendation"] = (
                    "Define triage ownership and escalation for existing administrator activity alerts."
                )
                rec["example"] = (
                    "An administrator alerting application is already provided. Assign owners, severity thresholds, "
                    "and response timelines for unusual admin activity."
                )
            if external_banner_full and ("external" in text and ("email" in text or "banner" in text or "flag" in text)):
                rec["vulnerability"] = "External email warning effectiveness can be reviewed"
                rec["recommendation"] = (
                    "Review user response to existing external email banners instead of recommending adding banners."
                )
                rec["example"] = (
                    "External email flagging is already enabled. Test whether employees recognize and report "
                    "impersonation attempts using simulated phishing exercises."
                )
            if network_segmented and ("segment" in text or "flat network" in text or "network divided" in text):
                rec["vulnerability"] = "Network segmentation rules can be validated"
                rec["recommendation"] = (
                    "Validate and monitor existing network segmentation instead of recommending segmentation as absent."
                )
                rec["example"] = (
                    "The network is already divided. Review firewall rules, guest Wi-Fi isolation, and access paths "
                    "between payment, office, and operational systems."
                )
            if app_approval_full and ("third party app" in text or "third-party app" in text or "oauth" in text):
                rec["vulnerability"] = "Third-party app approval reviews can be strengthened"
                rec["recommendation"] = (
                    "Review approved third-party apps periodically instead of adding an approval process as missing."
                )
                rec["example"] = (
                    "Admin approval is already required. Run quarterly reviews of connected Microsoft/Google apps, "
                    "OAuth scopes, and stale approvals."
                )
            if ir_plan_uploaded and ("incident response plan" in text or "incident response document" in text):
                rec["vulnerability"] = "Incident response plan testing can be improved"
                rec["recommendation"] = (
                    "Test and update the uploaded incident response plan instead of recommending creating one."
                )
                rec["example"] = (
                    "An incident response plan is already uploaded. Run a tabletop exercise, update contact lists, "
                    "and record lessons learned."
                )
            if patching_full and ("patch" in text or "security update" in text or "updates within 60" in text):
                rec["vulnerability"] = "Patch process evidence can be strengthened"
                rec["recommendation"] = (
                    "Track patch compliance evidence instead of recommending starting 60-day patching."
                )
                rec["example"] = (
                    "The 60-day update requirement is already met or committed. Maintain patch reports, exception "
                    "tracking, and overdue remediation ownership."
                )
            if admin_lockout_missing and (
                "lockout" in text
                or "failed login" in text
                or "failed attempt" in text
                or "brute force" in text
            ):
                rec["vulnerability"] = "Admin accounts are not locked after three failed login attempts"
                rec["recommendation"] = (
                    "Configure admin accounts to lock automatically after three failed login attempts, with security-team alerts and a documented unlock process."
                )
                rec["example"] = (
                    "For administrator accounts in Active Directory or the identity provider, set a three-failed-attempt lockout threshold, alert the security team on lockouts, and require helpdesk or security review before re-enabling access."
                )
            if admin_lockout_full and (
                "lockout" in text
                or "failed login" in text
                or "failed attempt" in text
                or "brute force" in text
            ):
                rec["vulnerability"] = "Admin lockout monitoring can be strengthened"
                rec["recommendation"] = (
                    "Review lockout alerts and unlock procedures instead of recommending admin lockout as a missing control."
                )
                rec["example"] = (
                    "Admin lockout is already implemented. Test alert routing, review failed-login trends, and verify that unlocks require appropriate approval."
                )

    return parsed


def calculate_individual_reductions(recommendations, current_p_value, new_p_value):
    """
    Calculate individual likelihood_reduction for each recommendation based on impact_contribution
    
    Args:
        recommendations: List of recommendation dicts with 'impact_contribution' field
        current_p_value: Starting breach likelihood percentage
        new_p_value: Target breach likelihood percentage
    
    Returns:
        List of recommendations with added 'likelihood_reduction' field (whole number percentages)
    """
    if not recommendations:
        return recommendations

    total_reduction_needed = round(current_p_value) - round(new_p_value)
    total_assigned = 0.0
    
    # Sum all contributions
    total_contribution = sum(
        float(rec.get('impact_contribution', 0))
        for rec in recommendations
    )
    if total_contribution == 0:
        # If no contributions, distribute equally
        total_contribution = len(recommendations)
        for rec in recommendations:
            rec['impact_contribution'] = 1.0

    if total_reduction_needed <= 0:
        # The tier does not lower likelihood, so there is nothing to distribute.
        # Reporting a floor of 1% here would imply a reduction that never happens.
        for rec in recommendations:
            contribution = float(rec.get('impact_contribution', 0.0))
            rec['impact_contribution_factor'] = contribution / total_contribution
            rec['individual_reduction'] = 0
            rec['likelihood_reduction'] = "0%"
        return recommendations

    # Calculate individual reductions
    heap = list()
    for i, rec in  enumerate(recommendations):
        contribution = float(rec.get('impact_contribution', 0.0))
        rec['impact_contribution_factor'] = (contribution/total_contribution)
        individual_reduction = (contribution / total_contribution) * total_reduction_needed
        rec['individual_reduction'] = max(1, int(floor(individual_reduction)))
        if individual_reduction - rec['individual_reduction'] > 0:
            heapq.heappush(heap, (-(individual_reduction - rec['individual_reduction']), i))
        total_assigned +=rec['individual_reduction']
        rec['likelihood_reduction'] = f"{int(rec['individual_reduction'])}%"
    while total_reduction_needed - total_assigned >= 1 and len(heap) > 0:
        _, index = heapq.heappop(heap)
        rec = recommendations[index]
        rec['individual_reduction']+=1
        rec['likelihood_reduction'] =  f"{int(round(rec['individual_reduction']))}%"
        total_assigned+=1
    return recommendations


def validate_improvement_json(
    response_text,
    expected_tiers,
    tier_starting_likelihoods,
    max_scored_keys=None,
    questionnaire_keys=None,
    pentest_finding_names=None,
):
    """
    Validate improvement JSON with contribution-based approach
    
    Returns: (parsed_json, missing_tiers, is_valid, validation_errors, tier_new_likelihoods)
    """
    validation_errors = []
    tier_new_likelihoods = {}
    max_scored_keys = set(max_scored_keys or [])
    questionnaire_keys = set(questionnaire_keys or questions_noPentest)
    pentest_finding_names = {
        _normalize_answer(name)
        for name in (pentest_finding_names or [])
    }
    
    try:
        data = json.loads(response_text)
        
        if "grade_improvement_recommendations" not in data:
            return None, expected_tiers, False, ["Missing grade_improvement_recommendations key"], {}
        
        grade_recs = data["grade_improvement_recommendations"]
        present_ratings = set()
        
        for i, item in enumerate(grade_recs):
            if "improved_rating" not in item:
                validation_errors.append(f"Tier {i+1}: Missing improved_rating field")
                continue
            
            rating = normalize_rating(item["improved_rating"])
            present_ratings.add(rating)
            
            # Get expected starting likelihood for this tier
            if rating not in tier_starting_likelihoods:
                validation_errors.append(f"Rating {rating}: No starting likelihood found")
                continue
            
            expected_start = tier_starting_likelihoods[rating]
            
            if "recommendations_improved_rating" not in item:
                validation_errors.append(f"Rating {rating}: Missing recommendations_improved_rating")
                continue
            
            rating_data = item["recommendations_improved_rating"]
            
            # Validate new_overall_likelihood_of_breach exists
            if "new_overall_likelihood_of_breach" not in rating_data:
                validation_errors.append(f"Rating {rating}: Missing new_overall_likelihood_of_breach")
                continue
            
            new_likelihood = extract_numeric_value(rating_data["new_overall_likelihood_of_breach"])
            
            # Validate it falls within the target rating range
            target_min, target_max = RATING_PROBABILITY_MAP[rating]
            if not (target_min <= new_likelihood <= target_max):
                validation_errors.append(
                    f"Rating {rating}: new_overall_likelihood_of_breach ({new_likelihood}%) "
                    f"outside valid range ({target_min}% - {target_max}%)"
                )
                continue
            
            # Validate recommendations exist
            if "recommendations" not in rating_data:
                validation_errors.append(f"Rating {rating}: Missing recommendations array")
                continue
            
            recommendations = rating_data["recommendations"]
            
            if not 1 <= len(recommendations) <= 5:
                validation_errors.append(
                    f"Rating {rating}: Expected 1 to 5 recommendations, got {len(recommendations)}"
                )
                continue
            
            # Validate each recommendation has impact_contribution
            for j, rec in enumerate(recommendations):
                source_error = validate_recommendation_sources(
                    rec,
                    max_scored_keys,
                    questionnaire_keys,
                    pentest_finding_names,
                )
                if source_error:
                    validation_errors.append(
                        f"Rating {rating}, Recommendation {j+1}: {source_error}"
                    )

                if "impact_contribution" not in rec:
                    validation_errors.append(
                        f"Rating {rating}, Recommendation {j+1}: Missing impact_contribution"
                    )
                    continue
                
                try:
                    contribution = float(rec["impact_contribution"])
                    if contribution <= 0:
                        validation_errors.append(
                            f"Rating {rating}, Recommendation {j+1}: impact_contribution must be > 0"
                        )
                except (ValueError, TypeError):
                    validation_errors.append(
                        f"Rating {rating}, Recommendation {j+1}: impact_contribution must be a number"
                    )
            
            # Store the new likelihood for next tier
            tier_new_likelihoods[rating] = new_likelihood
        
        # Find missing tiers
        expected_set = set(expected_tiers)
        missing_tiers = expected_set - present_ratings
        
        if missing_tiers:
            validation_errors.append(f"Missing rating tiers: {', '.join(missing_tiers)}")
        
        is_valid = len(missing_tiers) == 0 and len(validation_errors) == 0
        
        return data, list(missing_tiers), is_valid, validation_errors, tier_new_likelihoods
        
    except json.JSONDecodeError as e:
        return None, expected_tiers, False, [f"JSON parsing error: {str(e)}"], {}


EVIDENCE_RULE = """
EVIDENCE RULE
Every "vulnerability" string must describe a gap shown directly by a questionnaire answer printed above or by a provided pentest finding.
A control that no question asks about has NOT been assessed. Never state or imply that the company lacks it.
Examples of unassessed controls: device posture or compliance checks, ZTNA, SIEM, privileged access management, DLP, application allowlisting, zero trust architecture.
"""

BEST_ANSWER_RULE = """
BEST-ANSWER RULE
A control whose points equal its maximum is the best answer the company could give. Make NO recommendation in that control's area: not a gap, not an upgrade, not an enhancement, not monitoring, testing, documentation, or maturity work.
Examples: if MFA covers all employees, do not suggest phishing-resistant MFA or hardware keys; if backups are daily, do not suggest immutable or offsite backups; if remote access uses VPN with a code, do not suggest device posture checks or ZTNA.
Per-control maximums printed above are authoritative. There is no separate overall questionnaire score to use.
Points below the maximum do not mean a control is partial or weak. Never mention points, scores, or maximums in the vulnerability, recommendation, or example text.
"""

GENERAL_ITEM_RULE = """
For source_type "general", the vulnerability field must not claim a current weakness. Describe the practice directly. Do not add a label or prefix such as "Opportunity:".
A general item must not fall in the area of any control listed as fully implemented, including upgrades or enhancements to it.
General items may only cover topics that no question asks about.
"""

PENTEST_FINDING_RULE = """
A recommendation with source_type "pentest" must remediate the named finding itself. Do not use a finding as the source for a recommendation about a different control.
A pentest finding may address a best-answer or max-score control only when that finding is itself about that control.
"""


def format_security_response_line(response, points_label="Points Earned"):
    key = response.get("key")
    if key in CONTEXT_ONLY_QUESTION_KEYS:
        return (
            f"- Key: {key} (context only, not a control; not a valid source key)\n"
            f"  Response: {response['answer']}"
        )
    if key == "storeCustomersCardNumber" and _card_storage_is_best_available(response.get("answer")):
        return (
            f"- Key: {key} | {response['question']}\n"
            f"  Response: {response['answer']} | Status: Best available answer. "
            "No recommendation allowed for this question."
        )
    if response.get("max_score") is not None:
        line = (
            f"- Key: {key} | {response['question']}\n"
            f"  Response: {response['answer']} | {points_label}: {response['score']}/{response['max_score']}"
        )
        if key == "thirdPartyControls" and _third_party_controls_are_best_available(response.get("answer")):
            line += " | Best available answer, no recommendation needed"
        return line
    return (
        f"- {response['question']}\n"
        f"  Response: {response['answer']} | {points_label}: {response['score']}"
    )


def format_tier_requirements(improvement_tiers, tier_starting_likelihoods):
    blocks = []
    for index, tier in enumerate(improvement_tiers):
        start_likelihood = tier_starting_likelihoods.get(tier)
        target_min, target_max = RATING_PROBABILITY_MAP[tier]
        if start_likelihood is None and index > 0:
            start_text = f"the value you choose for {improvement_tiers[index - 1]}"
        elif start_likelihood is None:
            start_text = "the current breach likelihood"
        else:
            start_text = f"{start_likelihood}%"
        blocks.append(
            f"""
**{tier}:**
- Starting likelihood: {start_text}
- Target range: {target_min}% - {target_max}%
- You must choose new_overall_likelihood_of_breach within {target_min}% - {target_max}%
- Provide 1 to 5 recommendations. impact_contribution is any positive number; only relative size matters.
"""
        )
    return "\n".join(blocks)


def get_noPentest_improvement_system_prompt(
    securityResponses,
    company_data,
    loss_data,
    current_rating,
    improvement_tiers,
    tier_starting_likelihoods,
    mandatory_priority_gaps=None,
    protected_control_facts=None,
    max_scored_controls=None,
):
    """
    System prompt for improvement recommendations (no pentest) with contribution-based approach
    """
    sec_resp_text = "\n".join(
        format_security_response_line(response, "Points Earned")
        for response in securityResponses
    )
    tier_requirements_text = format_tier_requirements(improvement_tiers, tier_starting_likelihoods)
    mandatory_priority_text = format_mandatory_priority_gaps(mandatory_priority_gaps or [])
    protected_control_facts = protected_control_facts or "No protected control facts were triggered."
    max_scored_controls = max_scored_controls or "No questionnaire controls earned maximum points."

    system_prompt = f"""
### CRITICAL INSTRUCTIONS

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH rating tier listed below.

---

### REQUIRED OUTPUT TIERS (MUST INCLUDE ALL {len(improvement_tiers)} TIERS)

""" + "\n".join([f"{i+1}. {tier}" for i, tier in enumerate(improvement_tiers)]) + f"""

**CURRENT RATING:** {current_rating}
**CURRENT BREACH LIKELIHOOD:** {loss_data.get('P_value')}%

---

### COMPANY INFORMATION

- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry_name')}
- Annual Revenue: ${company_data.get('annual_revenue')} USD

---

### SECURITY QUESTIONNAIRE RESULTS

{sec_resp_text}

{BEST_ANSWER_RULE}

### MAX-SCORE QUESTIONNAIRE CONTROLS — INELIGIBLE FOR RECOMMENDATIONS

{max_scored_controls}

These controls are excluded from questionnaire and general recommendations.

### PROTECTED CONTROL FACTS

{protected_control_facts}

These facts are authoritative. Recommendations must not contradict them. If a
control is implemented or a group is already covered, recommend only the actual
remaining gap, not the already-covered control.

{EVIDENCE_RULE}

---

### TIER-SPECIFIC REQUIREMENTS

{tier_requirements_text}

---

### HOW TO GENERATE RECOMMENDATIONS

For EACH tier:

1. **Choose target likelihood**: Pick a value within the target range for new_overall_likelihood_of_breach
2. **Provide 1 to 5 recommendations**: Each addressing a different real gap. Stop when no assessed gap remains. Do not invent items to reach 5.
3. **Assign impact_contribution**: Any positive number. Only relative size matters. Higher means more of the likelihood drop.

### MANDATORY TOP-PRIORITY QUESTIONNAIRE GAPS

{mandatory_priority_text}

If any mandatory priority gaps are listed above, the FIRST tier's recommendations
MUST start with those gaps in the listed order before any other recommendation.
This requirement overrides industry-specific controls and all other prioritization rules.
Only include these mandatory recommendations when the matching questionnaire answer triggered the gap.

**We will calculate likelihood_reduction for each recommendation using:**
`likelihood_reduction = (impact_contribution / sum_of_all_contributions) × total_reduction_needed`

---

### REQUIREMENTS FOR EACH RECOMMENDATION

Each recommendation MUST include:
1. **vulnerability**: Current gap/weakness (e.g., "Insufficient endpoint protection")
2. **recommendation**: Specific improvement action (e.g., "Deploy EDR across all endpoints")
3. **example**: Concrete industry-specific implementation example (≤125 words) . Ensure that the example belongs to the industry {company_data.get('primary_industry_name')}
4. **impact_contribution**: Any positive number. Only relative size matters (e.g., 0.1 or 30)
5. **source_type**: "questionnaire" or "general"
6. **source_question_key**: Exact questionnaire key when source_type is
   "questionnaire"; otherwise null. Never use remoteAccessTypes.
7. **source_pentest_finding**: Always null because no pentest findings were provided

Any recommendation related to a questionnaire topic MUST use source_type
"questionnaire" and its exact source_question_key. Do not label a questionnaire
recommendation as "general".

{GENERAL_ITEM_RULE}

**DO NOT include likelihood_reduction or reduction_in_total_loss - these will be calculated**

**DO NOT include the score score of the questionnaire or the CVSS score or the risk level in the vulnerability.**   
---

### CRITICAL: UNIQUE RECOMMENDATIONS

Each tier MUST have completely different recommendations:
- Tier 1: Start with all triggered mandatory top-priority questionnaire gaps, then use the remaining slots for the most critical assessed vulnerabilities
- Tier 2: Focus on security controls with demonstrated gaps (such as MFA, EDR, or encryption only when their current response is not already at maximum)
- Tier 3 and later: address remaining questionnaire gaps not yet covered in earlier tiers. Only when none remain, use general best practices relevant to the industry. Do not describe those as a current weakness, and do not prefix them with a label.

---

### OUTPUT FORMAT

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks)
2. Use EXACTLY these field names: vulnerability, recommendation, example,
   impact_contribution, source_type, source_question_key, source_pentest_finding
3. impact_contribution: Any positive number (e.g., 0.25, 1.5, 3.0). Only relative size matters.
4. new_overall_likelihood_of_breach: Whole number with % (e.g., "35%")
5. Must be within the target rating's range
6. Include ALL {len(improvement_tiers)} tiers
7. Each tier has 1 to 5 recommendations, not a fixed 5

**REQUIRED JSON STRUCTURE:**

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "FIRST_TIER",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{
                        "vulnerability": "Current gap",
                        "recommendation": "Improvement action",
                        "example": "Implementation example",
                        "impact_contribution": 30.5,
                        "source_type": "questionnaire",
                        "source_question_key": "a_non_max_question_key",
                        "source_pentest_finding": null
                    }},
                    {{
                        "vulnerability": "Second gap",
                        "recommendation": "Second action",
                        "example": "Second example",
                        "impact_contribution": 25.3,
                        "source_type": "general",
                        "source_question_key": null,
                        "source_pentest_finding": null
                    }},
                    {{
                        "vulnerability": "Third gap",
                        "recommendation": "Third action",
                        "example": "Third example",
                        "impact_contribution": 50,
                        "source_type": "general",
                        "source_question_key": null,
                        "source_pentest_finding": null
                    }},
                    {{
                        "vulnerability": "Fourth gap",
                        "recommendation": "Fourth action",
                        "example": "Fourth example",
                        "impact_contribution": 1.5,
                        "source_type": "general",
                        "source_question_key": null,
                        "source_pentest_finding": null
                    }},
                    {{
                        "vulnerability": "Fifth gap",
                        "recommendation": "Fifth action",
                        "example": "Fifth example",
                        "impact_contribution": 10,
                        "source_type": "general",
                        "source_question_key": null,
                        "source_pentest_finding": null
                    }}
                ],
                "new_overall_likelihood_of_breach": "40%"
            }}
        }}
    ]
}}

---

### EXAMPLE

Current: 52%, Target tier: B+ (33.33% - 44.43%)

You choose: new_overall_likelihood_of_breach = 40%
Your impact_contributions: [30, 25, 20, 15, 30], sum = 120
Total reduction needed = 12 percentage points
Reductions: 30/120 x 12 = 3.0, 25/120 x 12 = 2.5, 20/120 x 12 = 2.0, 15/120 x 12 = 1.5, 30/120 x 12 = 3.0
Python then rounds these to whole percentages. Do not output likelihood_reduction yourself.

---

### FINAL CHECKLIST

- [ ] Included ALL {len(improvement_tiers)} tiers
- [ ] Each tier has 1 to 5 recommendations
- [ ] Each recommendation ≤125 words
- [ ] Each new_overall_likelihood_of_breach is within its target range
- [ ] Each recommendation has impact_contribution (any positive number)
- [ ] Every recommendation has valid source metadata
- [ ] No recommendation concerns a max-score or best-answer questionnaire control
- [ ] All recommendations are unique across tiers
- [ ] NO likelihood_reduction or reduction_in_total_loss fields

Return ONLY the JSON output, nothing else.
"""
    
    return system_prompt


def format_pentest_finding(vuln):
    """One finding for the recommendation prompt. Extra detail is omitted when absent."""
    line = (
        f"- {vuln.get('Vulnerability name', '')} | "
        f"CVSS: {vuln.get('CVSS Score', '')} | "
        f"Risk: {vuln.get('Risk Score/Level', '')}"
    )
    details = []
    description = str(vuln.get("description") or "").strip()
    remediation = str(vuln.get("remediation") or "").strip()
    if description:
        details.append(f"  Description: {description}")
    if remediation:
        details.append(f"  Remediation: {remediation}")
    if not details:
        return line
    return "\n".join([line, *details])


def get_Pentest_improvement_system_prompt(
    vulnerabilities,
    securityResponses,
    company_data,
    loss_data,
    current_rating,
    improvement_tiers,
    tier_starting_likelihoods,
    mandatory_priority_gaps=None,
    protected_control_facts=None,
    max_scored_controls=None,
):
    """
    System prompt for improvement recommendations (with pentest) with contribution-based approach
    """
    vuln_text = "\n".join(format_pentest_finding(vuln) for vuln in vulnerabilities)

    sec_resp_text = "\n".join(
        format_security_response_line(response, "Points")
        for response in securityResponses
    )
    tier_requirements_text = format_tier_requirements(improvement_tiers, tier_starting_likelihoods)
    mandatory_priority_text = format_mandatory_priority_gaps(mandatory_priority_gaps or [])
    protected_control_facts = protected_control_facts or "No protected control facts were triggered."
    max_scored_controls = max_scored_controls or "No questionnaire controls earned maximum points."

    system_prompt = f"""
### CRITICAL INSTRUCTIONS

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH rating tier listed below.

---

### REQUIRED OUTPUT TIERS (MUST INCLUDE ALL {len(improvement_tiers)} TIERS)

""" + "\n".join([f"{i+1}. {tier}" for i, tier in enumerate(improvement_tiers)]) + f"""

**CURRENT RATING:** {current_rating}
**CURRENT BREACH LIKELIHOOD:** {loss_data.get('P_value')}%

---

### COMPANY INFORMATION

- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry_name')}
- Annual Revenue: ${company_data.get('annual_revenue')} USD

---

### PENETRATION TEST FINDINGS

{vuln_text}

---

### SECURITY QUESTIONNAIRE RESULTS

{sec_resp_text}

{BEST_ANSWER_RULE}

### MAX-SCORE QUESTIONNAIRE CONTROLS — INELIGIBLE FOR QUESTIONNAIRE RECOMMENDATIONS

{max_scored_controls}

Exception: a recommendation may address one of these controls only when it remediates a provided pentest finding that is itself about that control. Do not attach an unrelated finding to a best-answer control.

### PROTECTED CONTROL FACTS

{protected_control_facts}

These facts are authoritative. Recommendations must not contradict them. If a
control is implemented or a group is already covered, recommend only the actual
remaining gap, not the already-covered control.

{EVIDENCE_RULE}

---

### TIER-SPECIFIC REQUIREMENTS

{tier_requirements_text}

---

### HOW TO GENERATE RECOMMENDATIONS

For EACH tier:

1. **Choose target likelihood**: Pick a value within the target range for new_overall_likelihood_of_breach
2. **Provide 1 to 5 recommendations**: Address pentest findings and assessed questionnaire gaps. Stop when none remain. Do not invent items to reach 5.
3. **Assign impact_contribution**: Any positive number. Only relative size matters. Higher means more of the likelihood drop. Give larger weights to critical findings than to lower-severity ones.

### MANDATORY TOP-PRIORITY QUESTIONNAIRE GAPS

{mandatory_priority_text}

If any mandatory priority gaps are listed above, the FIRST tier's recommendations
MUST start with those gaps in the listed order before any pentest finding or any
other recommendation. This requirement overrides Critical/High CVSS severity,
pentest severity, industry-specific controls, and all other prioritization rules.
Only include these mandatory recommendations when the matching questionnaire
answer triggered the gap.

**We will calculate likelihood_reduction for each recommendation using:**
`likelihood_reduction = (impact_contribution / sum_of_all_contributions) × total_reduction_needed`

---

### REQUIREMENTS FOR EACH RECOMMENDATION

Address in priority order:
- Triggered mandatory top-priority questionnaire gaps
- Provided pentest findings, each recommendation remediating that named finding
- Security questionnaire weaknesses that are not best answers
- Only when those are exhausted: general best practices under the EVIDENCE RULE, not stated as a current weakness and not prefixed with a label

Each recommendation MUST include:
1. **vulnerability**: Current gap/weakness shown by an answer or a pentest finding (e.g., "Unpatched CVE-2023-XXXX")
2. **recommendation**: Specific improvement action (e.g., "Apply security patch within 48 hours")
3. **example**: Concrete industry-specific implementation (≤125 words) . Ensure that the example belongs to the industry {company_data.get('primary_industry_name')}
4. **impact_contribution**: Any positive number. Only relative size matters (e.g., 0.30 or 30)
5. **source_type**: "questionnaire", "pentest", or "general"
6. **source_question_key**: Exact questionnaire key for questionnaire sources;
   otherwise null. Never use remoteAccessTypes.
7. **source_pentest_finding**: Exact provided Vulnerability name for pentest
   sources; otherwise null

Any recommendation related to a questionnaire topic MUST use source_type
"questionnaire" unless it is directly backed by a provided pentest finding. Do
not label a questionnaire recommendation as "general".

{GENERAL_ITEM_RULE}

{PENTEST_FINDING_RULE}

**DO NOT include likelihood_reduction or reduction_in_total_loss - these will be calculated**
**DO NOT include the score score of the questionnaire or the CVSS score or the risk level in the vulnerability.**  

---

### CRITICAL: UNIQUE RECOMMENDATIONS

Each tier MUST have completely different recommendations:
- Tier 1: Start with all triggered mandatory top-priority questionnaire gaps, then use the remaining slots for the most critical pentest findings
- Tier 2: Remaining pentest findings and questionnaire controls that are not best answers
- Tier 3 and later: address remaining questionnaire gaps or pentest findings not yet covered in earlier tiers. Only when none remain, use general best practices relevant to the industry. Do not describe those as a current weakness, and do not prefix them with a label.

---

### OUTPUT FORMAT

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks)
2. Use EXACTLY these field names: vulnerability, recommendation, example,
   impact_contribution, source_type, source_question_key, source_pentest_finding
3. impact_contribution: Any positive number (e.g., 25, 1.5, 30). Only relative size matters.
4. new_overall_likelihood_of_breach: Whole number with % (e.g., "35%")
5. Must be within the target rating's range
6. Include ALL {len(improvement_tiers)} tiers
7. Each tier has 1 to 5 recommendations, not a fixed 5

**REQUIRED JSON STRUCTURE:**

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "FIRST_TIER",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{
                        "vulnerability": "Current gap",
                        "recommendation": "Improvement action",
                        "example": "Implementation example",
                        "impact_contribution": 30.2,
                        "source_type": "pentest",
                        "source_question_key": null,
                        "source_pentest_finding": "Exact provided Vulnerability name"
                    }}
                ],
                "new_overall_likelihood_of_breach": "40%"
            }}
        }}
    ]
}}

---

### FINAL CHECKLIST

- [ ] Included ALL {len(improvement_tiers)} tiers
- [ ] Each tier has 1 to 5 recommendations
- [ ] Each recommendation ≤125 words
- [ ] Each new_overall_likelihood_of_breach is within its target range
- [ ] Each recommendation has impact_contribution (any positive number)
- [ ] Every recommendation has valid source metadata
- [ ] Max-score or best-answer controls appear only when the named pentest finding is about that control
- [ ] Each pentest recommendation remediates its named finding
- [ ] All recommendations are unique across tiers
- [ ] NO likelihood_reduction or reduction_in_total_loss fields

Return ONLY the JSON output, nothing else.
"""
    
    return system_prompt


def get_improvement_recommendations(
    parsed_data,
    assessment,
    company_data,
    loss_data
):
    """
    Generate improvement recommendations with contribution-based approach
    """
    securityQuestions = parsed_data.get("securityQuestions", {})
    mandatory_priority_gaps = get_mandatory_priority_gaps(securityQuestions)

    scores = get_security_scores(
        securityQuestions,
        int(company_data.get("num_employees", 0))
    )
    scores = add_auxiliary_question_scores(securityQuestions, scores)
    protected_control_facts = get_protected_control_facts(securityQuestions, scores)

    combined_list = []

    for key, question_text in questions_noPentest.items():
        if key == "remoteAccessTypes":
            answer = {
                emp: int(securityQuestions.get(f"{emp}_count", securityQuestions.get(emp, 0)))
                for emp in securityQuestions.get("remoteAccessTypes", [])
            }
        elif key == "receiveAlert":
            answer = _get_admin_alert_answer(securityQuestions)
        elif key == "emailSecurityService":
            answer = _get_email_security_answer(securityQuestions)
        elif key == "noIncidentResponse":
            answer = _get_incident_response_answer(securityQuestions)
        elif key == "edrToolsUsed":
            answer = _first_non_blank(
                securityQuestions,
                ("edrToolsUsed", "edrTools"),
            )
        elif key == "useMFA":
            answer = _first_non_blank(
                securityQuestions,
                ("useMFA", "mfaTypes"),
            )
        else:
            answer = securityQuestions.get(key)

        if answer in (None, "", []):
            continue

        score_key = {
            "receiveAlert": "ReceiveAlertApp",
            "noIncidentResponse": "incidentResponsePlan",
        }.get(key, key)
        score = scores.get(key, scores.get(score_key, 0))

        if key == "pentestCriticalResolved":
            score = 10 if answer == "Yes" else 0
        elif key == "pentestCritical30Days":
            score = 4 if answer == "Yes" else 0

        combined_list.append({
            "key": key,
            "question": question_text,
            "answer": answer,
            "score": score,
            "max_score": 4 if assessment == "PenTest" and key == "remoteAccess" else QUESTION_MAX_SCORES.get(key),
        })

    max_scored_keys = get_max_scored_question_keys(combined_list)
    ineligible_keys = get_recommendation_blocked_keys(securityQuestions, max_scored_keys)
    max_scored_controls = format_max_scored_controls(combined_list, max_scored_keys)
    mandatory_priority_gaps = [
        gap for gap in mandatory_priority_gaps
        if gap["question_key"] not in max_scored_keys
    ]
    vulnerability_data = parsed_data.get("assessmentData", {}).get("vulnerabilityData") or []
    pentest_finding_names = [
        vulnerability.get("Vulnerability name")
        for vulnerability in vulnerability_data
        if isinstance(vulnerability, dict) and vulnerability.get("Vulnerability name")
    ]

    # Get current rating and improvement tiers
    current_rating_raw = loss_data.get('Rating', 'B')
    current_likelihood = extract_numeric_value(loss_data.get('P_value', 50.0))
    current_rating, improvement_tiers = get_improvement_tiers(
        current_rating_raw,
        current_likelihood
    )
    insurability_blockers = get_insurability_blockers(securityQuestions)

    if not improvement_tiers:
        return json.dumps({
            "message": (
                "No graded improvement is available; breach likelihood is already in the "
                "highest band."
            ),
            "current_rating": current_rating,
            "insurability_blockers": insurability_blockers
        })

    # Build tier starting likelihoods
    tier_starting_likelihoods = {improvement_tiers[0]: current_likelihood}
    
    # For subsequent tiers, we'll update after getting LLM results
    for i in range(1, len(improvement_tiers)):
        # Placeholder - will be updated with actual LLM output
        tier_starting_likelihoods[improvement_tiers[i]] = None

    # Select prompt
    if (
        assessment == "PenTest"
        and vulnerability_data
    ):
        system_prompt = get_Pentest_improvement_system_prompt(
            vulnerability_data,
            combined_list,
            company_data,
            loss_data,
            current_rating,
            improvement_tiers,
            tier_starting_likelihoods,
            mandatory_priority_gaps,
            protected_control_facts,
            max_scored_controls,
        )
    else:
        system_prompt = get_noPentest_improvement_system_prompt(
            combined_list,
            company_data,
            loss_data,
            current_rating,
            improvement_tiers,
            tier_starting_likelihoods,
            mandatory_priority_gaps,
            protected_control_facts,
            max_scored_controls,
        )

    # LLM call with validation and retry
    max_retries = 3
    # print(f"prompt {system_prompt}")
    for attempt in range(max_retries):
        try:
            response_text = ""
            with anthropic_client.messages.stream(
                    model="claude-sonnet-5",
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": "Generate recommendations to improve the rating of my company"}
                    ],
                    thinking={"type": "disabled"},
                    max_tokens=10000
            ) as stream:

                response_text = stream.get_final_text()

            # Clean JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Validate
            parsed, missing, is_valid, errors, tier_new_likelihoods = validate_improvement_json(
                response_text,
                improvement_tiers,
                tier_starting_likelihoods,
                ineligible_keys,
                questions_noPentest.keys(),
                pentest_finding_names,
            )
            
            if is_valid and parsed:
                parsed = enforce_mandatory_priority_recommendations(
                    parsed,
                    mandatory_priority_gaps,
                    company_data
                )
                parsed = dedupe_recommendations(parsed)

                # Calculate individual reductions for each tier
                prev_likelihood = current_likelihood
                
                for tier_item in parsed["grade_improvement_recommendations"]:
                    rating_data = tier_item["recommendations_improved_rating"]
                    new_likelihood = extract_numeric_value(rating_data["new_overall_likelihood_of_breach"])
                    
                    # Calculate individual reductions
                    rating_data["recommendations"] = calculate_individual_reductions(
                        rating_data["recommendations"],
                        prev_likelihood,
                        new_likelihood
                    )
                    
                    # Update for next tier
                    prev_likelihood = new_likelihood

                parsed["insurability_blockers"] = insurability_blockers
                parsed = attach_recommendation_basis(parsed)
                parsed = strip_recommendation_source_metadata(parsed)

                return json.dumps(parsed, indent=2)
            
            # Retry with feedback
            if errors and attempt < max_retries - 1:
                error_details = "\n".join(errors)
                
                # Update tier starting likelihoods based on actual results
                if tier_new_likelihoods:
                    for i, tier in enumerate(improvement_tiers):
                        if i == 0:
                            tier_starting_likelihoods[tier] = current_likelihood
                        else:
                            prev_tier = improvement_tiers[i-1]
                            if prev_tier in tier_new_likelihoods:
                                tier_starting_likelihoods[tier] = tier_new_likelihoods[prev_tier]
                
                # Regenerate prompt with updated starting likelihoods
                if (
                    assessment == "PenTest"
                    and vulnerability_data
                ):
                    system_prompt = get_Pentest_improvement_system_prompt(
                        vulnerability_data,
                        combined_list,
                        company_data,
                        loss_data,
                        current_rating,
                        improvement_tiers,
                        tier_starting_likelihoods,
                        mandatory_priority_gaps,
                        protected_control_facts,
                        max_scored_controls,
                    )
                else:
                    system_prompt = get_noPentest_improvement_system_prompt(
                        combined_list,
                        company_data,
                        loss_data,
                        current_rating,
                        improvement_tiers,
                        tier_starting_likelihoods,
                        mandatory_priority_gaps,
                        protected_control_facts,
                        max_scored_controls,
                    )
                
                correction = f"\n\n**ERRORS IN PREVIOUS ATTEMPT:**\n{error_details}\n\n"
                
                if missing:
                    correction += f"**MISSING TIERS**: {', '.join(missing)}\n\n"
                
                correction += "**CRITICAL REMINDERS:**\n"
                correction += "- new_overall_likelihood_of_breach MUST be within the target range\n"
                correction += "- Each tier needs 1 to 5 recommendations, not exactly 5\n"
                correction += "- impact_contribution is any positive number; only relative size matters\n"
                correction += "- Do not recommend a best-answer or max-score control, and do not describe it as missing\n"
                correction += "- Do not claim a gap the questionnaire did not ask about\n"
                correction += "- Include valid source metadata for every recommendation\n"
                correction += "- remoteAccessTypes is not a valid source key\n"
                correction += "- All recommendations must be unique across tiers\n"
                
                system_prompt += correction
                continue
            
        except Exception as e:
            if attempt == max_retries - 1:
                return json.dumps({
                    "error": f"Cannot generate recommendations: {str(e)}",
                    "current_rating": current_rating,
                    "expected_tiers": improvement_tiers,
                    "insurability_blockers": insurability_blockers
                })
            continue
    
    return json.dumps({
        "error": "Failed after retries",
        "current_rating": current_rating,
        "expected_tiers": improvement_tiers,
        "insurability_blockers": insurability_blockers
    })
