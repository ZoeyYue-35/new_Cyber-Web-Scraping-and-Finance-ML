import json
import numpy as np

REMOTE_ACCESS_DEDUCTION_TABLE = [
    # (upper_bound_percentage, deduction)
    (4.9, 0),
    (9.9, 0.15),
    (14.9, 0.26),
    (19.9, 0.35),
    (24.9, 0.44),
    (29.9, 0.51),
    (34.9, 0.57),
    (39.9, 0.62),
    (44.9, 0.66),
    (49.9, 0.70),
    (54.9, 0.73),
    (59.9, 0.76),
    (64.9, 0.78),
    (69.9, 0.80),
    (74.9, 0.82),
    (79.9, 0.84),
    (84.9, 0.86),
    (89.9, 0.88),
    (94.9, 0.89),
    (99.9, 0.90),
    (100.0, 1.0)
]

NO_THREAT_DETECTION_VALUES = {
    "we do not have this",
    "we do not use security software to detect threats",
    "none",
    "no",
}

NO_BACKUP_VALUES = {
    "we do not backup our files",
    "we do not back up our files",
    "we do not backup our data",
    "we do not back up our data",
    "we don't back up our data",
    "we don t back up our data",
    "we don’t back up our data",
}

NO_EMAIL_SECURITY_VALUES = {
    "we do not use a dedicated security service",
    "we do not use email security software",
    "none",
    "no",
}

NO_ADMIN_ALERT_VALUES = {
    "we do not use any application",
    "none",
    "no",
}


def _normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("'", " ").replace("’", " ").strip().lower().split())


def _as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _get_mfa_score(value):
    """Score both the current list response and the legacy Yes/No response."""
    selections = {_normalize_text(item) for item in _as_list(value)}

    if "yes" in selections or "all employees" in selections:
        return 10

    group_points = {
        "leadership and security teams (executives, cybersecurity, recovery teams)": 3,
        "admins or high-access users (it admins, developers, finance/hr, vendors, etc.)": 6,
    }
    return min(10, sum(points for group, points in group_points.items() if group in selections))


def _is_no_threat_detection(value):
    values = _as_list(value)
    if not values:
        return True

    normalized_values = {_normalize_text(item) for item in values}
    return any(value in NO_THREAT_DETECTION_VALUES for value in normalized_values)


def _first_non_empty(*values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _score_named_text_field(value, absent_values):
    if value in (None, "", []):
        return 0
    normalized_values = {_normalize_text(item) for item in _as_list(value)}
    return 0 if normalized_values & absent_values else 4


def _score_incident_response_plan(security_data):
    if security_data.get("noIncidentResponse") is True:
        return 0

    value = _first_non_empty(
        security_data.get("incidentFile"),
        security_data.get("incidentResponsePlan"),
        security_data.get("noIncidentResponse"),
    )
    if value in (None, "", []):
        return 0
    if _normalize_text(value) in {"true", "yes", "we do not have an incident response document"}:
        return 0
    return 8


def get_remote_access_deduction(percentage: float) -> float:
    """
    Given a percentage (0 to 100) of employees with remote access, return the
    deduction factor according to the table. Returns a value between 0 and 1.
    """
    for upper_bound, deduction in REMOTE_ACCESS_DEDUCTION_TABLE:
        if percentage <= upper_bound:
            return deduction
    return 1.0

def get_security_scores(security_data, total_employees):
    scores = {}

    # useMFA
    scores["useMFA"] = _get_mfa_score(
        _first_non_empty(security_data.get("useMFA"), security_data.get("mfaTypes"))
    )

    # remoteAccess
    remote_access_map = {
        "vpnWithCode": 3,
        "vpnWithPassword": 2,
        "vpnWithRemote": 1,
        "withoutVpn": 0,
        "nobody": 5,
        "No": 4,
        "Yes": 0,
    }
    remote_answer = security_data.get("remoteAccess")
    scores["remoteAccess"] = remote_access_map.get(remote_answer, 0)

    if remote_answer not in ("nobody", "No", None, ""):
        total_remote_employees = 0
        for category in security_data.get("remoteAccessTypes", []) or []:
            total_remote_employees += int(_first_non_empty(
                security_data.get(f"{category}_count"),
                security_data.get(category),
                0
            ) or 0)

        if total_employees:
            remote_ratio_pct = (total_remote_employees / total_employees) * 100
            scores["remoteAccess"] = max(
                0,
                scores["remoteAccess"] - get_remote_access_deduction(remote_ratio_pct)
            )
        scores["remoteAccessTypes"] = total_remote_employees

    # pentest critical/high-risk remediation
    if security_data.get("pentestCriticalResolved") == "Yes":
        scores["pentestCriticalResolved"] = 10
    elif security_data.get("pentestCritical30Days") == "Yes":
        scores["pentestCriticalResolved"] = 4
    elif "pentestCriticalResolved" in security_data or "pentestCritical30Days" in security_data:
        scores["pentestCriticalResolved"] = 0

    # lockAccount
    scores["lockAccount"] = 3 if security_data.get("lockAccount") == "Yes" else 0

    # Admin-account unusual activity alert app
    scores["ReceiveAlertApp"] = _score_named_text_field(
        security_data.get("ReceiveAlertApp"),
        NO_ADMIN_ALERT_VALUES
    )

    # Dedicated email security service
    if security_data.get("noEmailSecurityService") is True:
        scores["emailSecurityService"] = 0
    else:
        scores["emailSecurityService"] = _score_named_text_field(
            security_data.get("emailSecurityService"),
            NO_EMAIL_SECURITY_VALUES
        )

    # External email banner
    scores["externalEmailsFlagged"] = 4 if security_data.get("externalEmailsFlagged") == "Yes" else 0

    # Money-request verification
    verify_map = {"doNotVerify": 0, "verifyWithRequest": 1, "verifyWithOnFile": 4}
    scores["verifyMoneyRequests"] = verify_map.get(security_data.get("verifyMoneyRequests"), 0)

    # Dual authorization
    scores["dualAuthorization"] = 4 if security_data.get("dualAuthorization") == "Yes" else 0

    # securityTraining
    securityTraining_train_map = {
        "Once a year": 1,
        "Twice a year": 2,
        "Every 3 months": 3,
        "We don't provide cybersecurity training": 0,
        "We don’t provide cybersecurity training": 0,
        "We do not offer any training": 0,
    }
    scores["securityTraining"] = securityTraining_train_map.get(security_data.get("securityTraining"), 0)
    
    # fixProblems    
    if security_data.get("fixProblems") == "Yes":
        scores["fixProblems"] = 5
    else:
        scores["fixProblems"] = 5 if security_data.get("updatesIn30Days") == "Yes" else 0
    scores["updatesIn30Days"] = scores["fixProblems"]

    # dataEncryption
    if security_data.get("dataEncryption") == "Yes":
        scores["dataEncryption"] = 4
    else: scores["dataEncryption"] = 0

    # dataBacking
    backup_map = {
        "Daily": 10,
        "Weekly": 8,
        "Monthly": 6,
        "Once a month": 6,
        "Every 3 months": 4,
        "Every 6 months": 2,
        "Once a year": 1,
        "We don't back up our data": 0,
        "We don’t back up our data": 0,
        "We do not backup our files": 0,
    }
    scores["dataBacking"] = backup_map.get(security_data.get("dataBacking"), 0)

    # Backup restore testing
    restore_map = {
        "monthly": 6,
        "every3Months": 4,
        "every6Months": 2,
        "onceAYear": 1,
        "never": 0,
        "Monthly": 6,
        "Every 3 months": 4,
        "Every 6 months": 2,
        "Once a year": 1,
        "We have never tested a restore": 0,
    }
    if _normalize_text(security_data.get("dataBacking")) in NO_BACKUP_VALUES:
        scores["restoreBackupTest"] = 0
    else:
        scores["restoreBackupTest"] = restore_map.get(security_data.get("restoreBackupTest"), 0)

    # Card number storage — form options are Yes / No (payment processor).
    card_map = {
        "Yes": 1,
        "No - our payment processor handles and stores them": 3,
        "No": 3,
    }
    scores["storeCustomersCardNumber"] = card_map.get(security_data.get("storeCustomersCardNumber"), 0)

    # Network segmentation
    scores["networkDivided"] = 4 if security_data.get("networkDivided") == "Yes" else 0

    # Third-party app admin approval
    scores["thirdPartyAppApproval"] = 4 if security_data.get("thirdPartyAppApproval") == "Yes" else 0

    # thirdPartyControls
    tpm_map = {
        "Yes": 3,
        "No": 1,
        "We don't give vendors access to our systems or data": 5,
        "We don’t give vendors access to our systems or data": 5,
        "I do not share data or systems access with other businesses": 5,
    }
    scores["thirdPartyControls"] = tpm_map.get(security_data.get("thirdPartyControls"), 0)

    # edrToolsUsed
    td_val = security_data.get("edrToolsUsed", security_data.get("edrTools"))
    if _is_no_threat_detection(td_val):
        scores["edrToolsUsed"] = 0
    else:
        scores["edrToolsUsed"] = 10
    scores["edrTools"] = scores["edrToolsUsed"]

    # Incident response plan upload
    scores["incidentResponsePlan"] = _score_incident_response_plan(security_data)
    scores["noIncidentResponse"] = scores["incidentResponsePlan"]

    return scores
