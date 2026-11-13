import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

NO_THREAT_DETECTION_VALUES = {
    "we do not have this",
    "we do not use security software to detect threats",
    "none",
    "no",
}

NO_BACKUP_VALUES = {
    "we do not backup our files",
    "we do not back up our data",
    "we do not back up our files",
    "we do not backup our data",
    "we don't back up our data",
}

NO_EMAIL_SECURITY_VALUES = {
    "no",
    "none",
    "we do not use a dedicated security service",
    "we don't use a dedicated security service",
}

NO_ADMIN_ALERT_APP_VALUES = {
    "no",
    "none",
    "we do not use any application",
    "we don't use any application",
}

NO_VENDOR_ACCESS_VALUES = {
    "we don't give vendors access to our systems or data",
    "we do not give vendors access to our systems or data",
    "i do not share data or systems access with other businesses",
}

PROCESSOR_CARD_STORAGE_VALUES = {
    "no - our payment processor handles and stores them",
    "no – our payment processor handles and stores them",
    "no — our payment processor handles and stores them",
    "no",
}

def _score_new_security_questions(security_data):
    scores = {}

    # Email security service: named service = 4, none = 0.
    # PenTest flow asks this as Yes/No, so accept that too.
    service_name = _normalize_text(security_data.get("emailSecurityService"))
    no_service = security_data.get("noEmailSecurityService") is True or service_name in NO_EMAIL_SECURITY_VALUES
    if service_name and not no_service:
        scores["email_security_points"] = 4
    else:
        scores["email_security_points"] = 0

    # Admin-account alerting app: named app = 4, none = 0.
    alert_app = _normalize_text(_first_non_empty(
        security_data.get("receiveAlert"),
        security_data.get("ReceiveAlertApp"),
    ))
    no_alert_app = security_data.get("noReceiveAlertApp") is True or alert_app in NO_ADMIN_ALERT_APP_VALUES
    if alert_app and not no_alert_app:
        scores["admin_alert_app_points"] = 4
    else:
        scores["admin_alert_app_points"] = 0

    # Money-request verification
    verify_map = {"doNotVerify": 0, "verifyWithRequest": 1, "verifyWithOnFile": 4}
    scores["money_verification_points"] = verify_map.get(
        security_data.get("verifyMoneyRequests"), 0)

    # Dual authorization
    scores["dual_authorization_points"] = 4 if security_data.get("dualAuthorization") == "Yes" else 0

    # Backup restore testing — only meaningful when they actually back up
    restore_map = {"monthly": 6, "every3Months": 4, "every6Months": 2, "onceAYear": 1, "never": 0}
    if _normalize_text(security_data.get("dataBacking")) in NO_BACKUP_VALUES:
        scores["backup_restore_test_points"] = 0
    else:
        scores["backup_restore_test_points"] = restore_map.get(
            security_data.get("restoreBackupTest"), 0)

    # Card number storage — form options are Yes / No (payment processor).
    card_answer = _normalize_text(security_data.get("storeCustomersCardNumber"))
    if card_answer in PROCESSOR_CARD_STORAGE_VALUES:
        scores["card_storage_points"] = 3
    elif card_answer == "yes":
        scores["card_storage_points"] = 1
    else:
        scores["card_storage_points"] = 0

    # Third-party app admin approval
    scores["third_party_app_points"] = 4 if security_data.get("thirdPartyAppApproval") == "Yes" else 0
    scores["external_email_flag_points"] = 4 if security_data.get("externalEmailsFlagged") == "Yes" else 0
    scores["network_segmentation_points"] = 4 if security_data.get("networkDivided") == "Yes" else 0

    return scores


def validate_all_questions_answered(security_data, assessment_type="no_pentest"):
    """
    Validate that all 9 main required security questions are answered.
    
    Args:
        security_data: Dictionary containing security assessment responses
        assessment_type: "pentest" or "no_pentest" (default)
    
    Returns:
        dict: {
            'all_answered': bool,
            'total_questions': int,
            'answered_questions': int,
            'missing_questions': list,
            'question_status': dict
        }
    """
    
    # Current frontend questionnaire keys. Legacy backend keys are kept as
    # fallbacks so older saved payloads still validate and score.
    required_questions = {
        1: {
            'key': 'useMFA',
            'fallback_keys': ['mfaTypes'],
            'name': 'Multi-Factor Authentication'
        },
        2: {
            'key': 'remoteAccess', 
            'name': 'Remote Access'
        },
        3: {
            'key': 'lockAccount',
            'fallback_keys': ['specialAccess'],
            'name': 'Account Lockout / Special Access Controls'
        },
        4: {
            'key': 'securityTraining',
            'name': 'Security Training'
        },
        5: {
            'key': 'fixProblems',
            'fallback_keys': ['updatesIn30Days', 'securityPatches'],
            'name': 'Security Patches'
        },
        6: {
            'key': 'dataEncryption',
            'name': 'Data Encryption'
        },
        7: {
            'key': 'dataBacking',
            'name': 'Data Backup'
        },
        8: {
            'key': 'thirdPartyControls',
            'name': 'Third Party Controls'
        },
        9: {
            'key': 'edrToolsUsed',
            'fallback_keys': ['edrTools'],
            'name': 'Threat Detection/EDR Tools'
        },
        10: {
            'key': 'receiveAlert',
            'fallback_keys': ['noReceiveAlertApp', 'ReceiveAlertApp'],
            'name': 'Administrator Alerting Application'
        },
        11: {
            'key': 'emailSecurityService',
            'fallback_keys': ['noEmailSecurityService'],
            'name': 'Email Security Service'
        },
        12: {
            'key': 'externalEmailsFlagged',
            'name': 'External Email Warning'
        },
        13: {
            'key': 'verifyMoneyRequests',
            'name': 'Money Request Verification'
        },
        14: {
            'key': 'dualAuthorization',
            'name': 'Dual Authorization'
        },
        15: {
            'key': 'restoreBackupTest',
            'name': 'Backup Restore Testing'
        },
        16: {
            'key': 'storeCustomersCardNumber',
            'name': 'Card Number Storage'
        },
        17: {
            'key': 'networkDivided',
            'name': 'Network Segmentation'
        },
        18: {
            'key': 'thirdPartyAppApproval',
            'name': 'Third-Party App Approval'
        },
        19: {
            'key': 'noIncidentResponse',
            'fallback_keys': [
                'incidentFileUrl',
                'incidentFile',
                'IncidentResponse',
                'incidentResponsePlan',
                'incidentResponseDocument',
            ],
            'name': 'Incident Response Plan'
        }
    }
    
    if assessment_type == "PenTest":
        required_questions[0] = {
            'key': 'pentestCriticalResolved',
            'name': 'Pentest Critical Issues Resolved'
        }
        # These controls are only asked in the no-pentest frontend path.
        for question_number in (10, 12, 17):
            required_questions.pop(question_number, None)
    
    # Track validation results
    question_status = {}
    missing_questions = []
    answered_count = 0
    total_questions = len(required_questions)
    
    for q_num, question_info in required_questions.items():
        key = question_info['key']
        keys_to_check = [key] + question_info.get('fallback_keys', [])
        name = question_info['name']
        
        # Check if main question is answered (not None, not empty string, not empty list)
        # "None" string is considered a valid answer for some questions
        is_answered = False
        answered_value = 'Not provided'
        
        for candidate_key in keys_to_check:
            if candidate_key not in security_data:
                continue

            value = security_data[candidate_key]
            if value is not None and value != "" and value != []:
                if isinstance(value, list):
                    is_answered = len(value) > 0 and not all(item == '' for item in value)
                else:
                    is_answered = True

            if is_answered:
                key = candidate_key
                answered_value = value
                break
        
        question_status[q_num] = {
            'name': name,
            'key': key,
            'answered': is_answered,
            'value': answered_value
        }
        
        if is_answered:
            answered_count += 1
        else:
            missing_questions.append({
                'question_number': q_num,
                'question_name': name,
                'key': key
            })
    
    return {
        'all_answered': answered_count == total_questions,
        'answered_questions': answered_count,
    }


def mapping_values():
    mappings = {
    "securityPlanPractice": {
        "Once a year": 1,
        "Twice a year": 2,
        "Every 3 months": 3,
        "Every month": 4,
        "Never": 0
    },
    "documentedSecurityPlan": {
        "Yes": 2,
        "No": 0
    },
    "incidentResponseTraining": {
        "Once a year": 3,
        "Twice a year": 6,
        "Every 3 months": 9,
        "We do not offer any training": 0
    },
    "alternativeTraining": {
        "Yes": 1,
        "No": 0
    },
    "computerSafety": {
        "Every 3 months": 9,
        "Once a Year": 3,
        "Twice a Year": 6,
        "We do not offer any training": 0
    },
    "fixProblems": {
        "Yes": 5,
        "No": 0
    },
    "updatesIn30Days": {
        "Yes": 5,
        "No": 0
    },
    "dataEncryption": {
        "Yes": 4,
        "No": 0
    },
    "dataBacking": {
        "Daily": 10,
        "Weekly": 8,
        "Monthly": 6,
        "Once a month": 6,
        "Every 3 months": 4,
        "Every 6 months": 2,
        "Once a year": 1,
        "We do not backup our files": 0,
        "We don't back up our data": 0,
    },
    "thirdPartyControls": {
        "Yes": 3,
        "No": 1,
        "I do not share data or systems access with other businesses": 5,
        "We don't give vendors access to our systems or data": 5,
    },
    "securityAlerts": {
        "Yes": 1,
        "No": 0
    },
    # "handlingAlerts": {
    #     "Right away": 4,
    #     "Within an hour": 2,
    #     "By the end of the day": 0.67,
    #     "We do not check security alerts": 0
    # },
    "edrTools": {
        "Yes": 10,
        "No": 0
    },
    "edrToolsUsed": {
        "CrowdStrike Falcon": 10,
        "Microsoft Defender": 10,
        "SentinelOne": 10,
        "Other": 10,
        "We do not use security software to detect threats": 0,
    },
    "remoteAccess": {
        "vpnWithCode": 3, "vpnWithPassword": 2, "vpnWithRemote": 1,
        "withoutVpn": 0, "nobody": 5
    },
    "emailSecurityService": {"provided": 4, "none": 0},
    "receiveAlert": {"provided": 4, "none": 0},
    "externalEmailsFlagged": {"Yes": 4, "No": 0},
    "verifyMoneyRequests": {"doNotVerify": 0, "verifyWithRequest": 1, "verifyWithOnFile": 4},
    "dualAuthorization": {"Yes": 4, "No": 0},
    "restoreBackupTest": {"monthly": 6, "every3Months": 4, "every6Months": 2, "onceAYear": 1, "never": 0},
    "storeCustomersCardNumber": {"Yes": 1, "No": 3},
    "networkDivided": {"Yes": 4, "No": 0},
    "thirdPartyAppApproval": {"Yes": 4, "No": 0},
}
    return mappings

def check_automatic_denial(answers, remote_access_ratios=None, is_pentest=False, business_data=None):
    """
    Returns a list of triggered combination numbers using a data-driven approach.
    'answers' is a dict with question => numeric points that can be used to define 
    the booleans below (e.g. no_mfa, no_threat_detection, etc.)
    """
    triggered = []
    if _has_selection(answers, ["edrToolsUsed", "edrTools"], NO_THREAT_DETECTION_VALUES):
        triggered.append("no_edr")

    if _has_selection(answers, ["useMFA", "mfaTypes"], {"Nobody", "None"}):
        triggered.append("no_mfa")

    if _has_selection(answers, "dataBacking", {
        "We do not backup our files",
        "We do not back up our data",
        "We do not back up our files",
        "We do not backup our data",
        "We don't back up our data"
    }):
        triggered.append("no_backups")

    return triggered


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


def _parse_positive_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return 0


def _get_remote_access_count(security_data, category):
    return _parse_positive_int(
        _first_non_empty(
            security_data.get(f"{category}_count"),
            security_data.get(category),
            0,
        )
    )


def _score_remote_access(security_data, num_employees, pentest=False):
    remote_answer = security_data.get("remoteAccess")

    if remote_answer in ("nobody", "No"):
        return 4 if pentest else 5

    base_points = 0 if pentest else {
        "vpnWithCode": 3,
        "vpnWithPassword": 2,
        "vpnWithRemote": 1,
        "withoutVpn": 0,
    }.get(remote_answer, 0)

    total_remote_employees = sum(
        _get_remote_access_count(security_data, category)
        for category in security_data.get("remoteAccessTypes", [])
    )
    employee_count = max(_parse_positive_int(num_employees), 1)
    remote_percentage = (total_remote_employees / employee_count) * 100
    point_deduction = get_remote_access_deduction(remote_percentage)
    return max(0, base_points - point_deduction)


def _score_backup_frequency(value):
    backup_map = {
        "daily": 10,
        "weekly": 8,
        "monthly": 6,
        "once a month": 6,
        "every 3 months": 4,
        "every 6 months": 2,
        "once a year": 1,
    }
    normalized_value = _normalize_text(value)
    if normalized_value in NO_BACKUP_VALUES:
        return 0
    return backup_map.get(normalized_value, 0)


def _score_third_party_controls(value):
    normalized_value = _normalize_text(value)
    if normalized_value in NO_VENDOR_ACCESS_VALUES:
        return 5
    if normalized_value == "yes":
        return 3
    if normalized_value == "no":
        return 1
    return 0


def _score_incident_response(security_data, confidence_rating):
    has_plan = False

    if security_data.get("IncidentResponse") is True:
        has_plan = True
    if security_data.get("noIncidentResponse") is False:
        has_plan = True

    for key in (
        "incidentFileUrl",
        "incidentFile",
        "incidentResponsePlan",
        "incidentResponseDocument",
    ):
        if security_data.get(key) not in (None, "", [], False):
            has_plan = True
            break

    if has_plan:
        confidence_rating["Incident Response Document"] += 1
        return 8
    return 0


def _has_selection(security_data, keys, matches):
    if isinstance(keys, str):
        keys = [keys]

    normalized_matches = {_normalize_text(match) for match in matches}
    for key in keys:
        for value in _as_list(security_data.get(key)):
            if _normalize_text(value) in normalized_matches:
                return True
    return False


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


def _has_recent_ransomware_incident(security_data, business_data=None):
    business_data = business_data or {}

    recent_incident_flag = False
    for key in ("recentBreach", "cyberIncident"):
        if _normalize_text(_first_non_empty(
            business_data.get(key),
            security_data.get(key)
        )) in {"yes", "true", "1"}:
            recent_incident_flag = True
            break

    incident_count = _parse_positive_int(
        _first_non_empty(
            business_data.get("incidentCount"),
            business_data.get("numIncident"),
            security_data.get("incidentCount"),
            security_data.get("numIncident")
        )
    )

    primary_cause = _normalize_text(
        _first_non_empty(
            business_data.get("primaryCause"),
            security_data.get("primaryCause")
        )
    )

    return recent_incident_flag and incident_count > 0 and "ransomware" in primary_cause

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

def get_remote_access_deduction(percentage: float) -> float:
    """
    Given a percentage (0 to 100) of employees with remote access, return the
    deduction factor according to the table. Returns a value between 0 and 1.
    """
    for upper_bound, deduction in REMOTE_ACCESS_DEDUCTION_TABLE:
        if percentage <= upper_bound:
            return deduction
    return 1.0


def mapping_values_business():
    # Mapping values for business-related data
    location_scores = {
        "CA": 3, "NY": 3, "TX": 3, "MA": 3, "FL": 3, "IL": 3, 
        "PA": 3, "VA": 3, "GA": 3, "NJ": 3
    }
    budget_scores = {
        "1-3.5%": 1,
        "3.6-6.9%": 2,
        ">7%": 3
    }
    industry_risk_scores = {
        "high-risk": 3,
        "low-risk": 1
    }
    data_type_scores = {
        "PHI": 3,  # Protected Health Information
        "PII": 0.25,  # Personally Identifiable Information
        "PCI": 1,  # Payment Card Information
        "SSN": 0.25, # Social Security Number
        "Credit Card Details": 1,
        "Health Information": 2,
        "Financial": 0.75,
        "Personal": 0.25
    }
    insurance_coverage_scores = {
        "Business Interruption Loss": 1,
        "Incident Response Costs": 1,
        "Regulatory Fines and Penalties": 1
    }
    return location_scores, budget_scores, industry_risk_scores, data_type_scores, insurance_coverage_scores


def parse_business_data(business_data, company_size):
    scores = {}
    location_scores, budget_scores, industry_risk_scores, data_type_scores, insurance_coverage_scores = mapping_values_business()
    
    # Location scoring
    state = business_data.get("region")
    scores["location"] = location_scores.get(state, 0)
    
    # Cybersecurity budget scoring
    budget = business_data.get("cybersecurityBudget")
    scores["cybersecurityBudget"] = budget_scores.get(budget, 0)
    
    # Industry risk scoring (use NAICS codes)
    industry = business_data.get("primaryIndustry")
    scores["industryRisk"] = 3 if industry in ["51", "52", "54"] else 1  # Example: high-risk NAICS codes
    
    # Breach scoring
    breach = business_data.get("recentBreach", "No")
    scores["recentBreach"] = 1 if breach == "Yes" else 3
    
    # Data exposed scoring
    data_exposed = business_data.get("dataExposed", [])
    scores["dataExposed"] = sum([data_type_scores.get(data, 0) for data in data_exposed])
    
    # Records exposed scoring
    records_exposed = business_data.get("recordsExposed")
    if records_exposed == "51-100":
        scores["recordsExposed"] = 1
    elif records_exposed == "101-500":
        scores["recordsExposed"] = 2
    elif records_exposed == "501+":
        scores["recordsExposed"] = 3
    else:
        scores["recordsExposed"] = 0  # Default for unknown ranges
    
    #company size scoring
    if company_size == 'micro':
        scores["companySize"] = 1
    elif company_size == 'small':
        scores["companySize"] = 2
    elif company_size == 'medium':
        scores["companySize"] = 3
    
    return scores


def questionnaire_scoring_function(security_data, assessment_type, confidence_rating, num_employees, business_data=None):
    """
    Checks if pentest is uploaded, then provides scoring of security data
    """
    validation_result = validate_all_questions_answered(security_data, assessment_type)
    if validation_result['all_answered']:
        confidence_rating["Security Questionnaire"] = 1

    if assessment_type == "PenTest":
        confidence_rating["PenTest"] = 1
        return parse_security_data_with_pentest(security_data, num_employees, confidence_rating, business_data=business_data)
    else:
        return parse_security_data_no_pentest(security_data, num_employees, confidence_rating, business_data=business_data)

def parse_security_data_with_pentest(security_data, num_employees, confidence_rating, business_data=None):
    """
    If a Pen Test is uploaded, then provide scoring of security data
    """
    scores = {
        "pentest_points": 0,
        "mfa_points": 0,
        "remote_access_points": 0,
        "employee_training_points": 0,
        "incident_response_points": 0,
        "patch_management_points": 0,
        "encryption_points": 0,
        "backup_points": 0,
        "third_party_points": 0,
        "threat_detection_points": 0,
        "special_access_points": 0,
        "email_security_points": 0,
        "external_email_flag_points": 0,
        "money_verification_points": 0,
        "dual_authorization_points": 0,
        "backup_restore_test_points": 0,
        "card_storage_points": 0,
        "network_segmentation_points": 0,
        "third_party_app_points": 0,
        "admin_alert_app_points": 0,
    }

    # 0) Pentest Critical Issues Resolved
    if security_data.get("pentestCriticalResolved") == "Yes":
        scores["pentest_points"] = 10
    else:
        if security_data.get("pentestCritical30Days") == "Yes":
            scores["pentest_points"] = 4
        else:
            scores["pentest_points"] = 0

    # 1) MFA => Critical/Special Users=3, All Employees=10, None=0
    scores["mfa_points"] = _get_mfa_score(
        _first_non_empty(security_data.get("useMFA"), security_data.get("mfaTypes"))
    )

    scores["remote_access_points"] = _score_remote_access(security_data, num_employees, pentest=True)

    # 3) Special Access - lockAccount
    if security_data.get("lockAccount") == "Yes":
        scores["special_access_points"] = 3
    else:
        scores["special_access_points"] = 0

    # 4) Employee Training => once=1, twice=2, 3mon=3, none=0
    train_map_pt = {
        "Once a year": 1,
        "Twice a year": 2,
        "Every 3 months": 3,
        "We don't provide cybersecurity training": 0
    }
    scores["employee_training_points"] = train_map_pt.get(security_data.get("securityTraining"), 0)

    # 5) Patch mgmt => yes=5, no=0
    if security_data.get("fixProblems") == "Yes":
        scores["patch_management_points"] = 5
    else:
        scores["patch_management_points"] = 0
    
    # 6) Encryption => yes=4, no=0
    if security_data.get("dataEncryption") == "Yes":
        scores["encryption_points"] = 4
    else:
        scores["encryption_points"] = 0

    # 7) Backup => daily=10, weekly=8, monthly=6, quarterly=4, semiannual=2, annual=1, none=0
    scores["backup_points"] = _score_backup_frequency(security_data.get("dataBacking"))

    # 8) Third-Party => yes=3, no=1, i do not share=5
    scores["third_party_points"] = _score_third_party_controls(security_data.get("thirdPartyControls"))

    # 9) Threat detection => None=0, selected EDR/security tool=10
    td_val = _first_non_empty(security_data.get("edrToolsUsed"), security_data.get("edrTools"))
    if _is_no_threat_detection(td_val):
        scores["threat_detection_points"] = 0
    else:
        scores["threat_detection_points"] = 10

    # 10) Incident Response - check noIncidentResponse flag
    scores["incident_response_points"] = _score_incident_response(security_data, confidence_rating)

    scores.update(_score_new_security_questions(security_data))
    triggered = check_automatic_denial(security_data, is_pentest=True, business_data=business_data)
    fail = False
    if triggered:
        fail = True
    return scores, fail, confidence_rating

def parse_security_data_no_pentest(security_data, num_employees, confidence_rating, business_data=None):
    scores = {
            "pentest_points": 0,
            "mfa_points": 0,
            "remote_access_points": 0,
            "employee_training_points": 0,
            "incident_response_points": 0,
            "patch_management_points": 0,
            "encryption_points": 0,
            "backup_points": 0,
            "third_party_points": 0,
            "threat_detection_points": 0,
            "special_access_points": 0,
            "email_security_points": 0,
            "external_email_flag_points": 0,
            "money_verification_points": 0,
            "dual_authorization_points": 0,
            "backup_restore_test_points": 0,
            "card_storage_points": 0,
            "network_segmentation_points": 0,
            "third_party_app_points": 0,
            "admin_alert_app_points": 0,
        }

    # 1) MFA => Critical/Special Users=3, All Employees=10, None=0
    scores["mfa_points"] = _get_mfa_score(
        _first_non_empty(security_data.get("useMFA"), security_data.get("mfaTypes"))
    )

    # 2) Key employees remote access => nobody=5, otherwise method score minus percentage deduction
    scores["remote_access_points"] = _score_remote_access(security_data, num_employees, pentest=False)
    
    # 3) Special Access - lockAccount
    if security_data.get("lockAccount") == "Yes":
        scores["special_access_points"] = 3
    else:
        scores["special_access_points"] = 0
    

    # 4) Employee Training => once=1, twice=2, 3mon=3, none=0
    train_map = {
        "Once a year": 1,
        "Twice a year": 2,
        "Every 3 months": 3,
        "We don't provide cybersecurity training": 0
    }
    scores["employee_training_points"] = train_map.get(security_data.get("securityTraining"), 0)

    # 5) Patch mgmt => fixed=5, committed within 30 days=5, no=0
    if security_data.get("fixProblems") == "Yes":
        scores["patch_management_points"] = 5
    else:
        if security_data.get("updatesIn30Days") == "Yes":
            scores["patch_management_points"] = 5
        else:
            scores["patch_management_points"] = 0
    
    # 6) Encryption => yes=4, no=0
    if security_data.get("dataEncryption") == "Yes":
        scores["encryption_points"] = 4
    else:
        scores["encryption_points"] = 0

    # 7) Backup => daily=10, weekly=8, monthly=6, quarterly=4, semiannual=2, annual=1, none=0
    scores["backup_points"] = _score_backup_frequency(security_data.get("dataBacking"))

    # 8) Third-Party => yes=3, no=1, i do not share=5
    scores["third_party_points"] = _score_third_party_controls(security_data.get("thirdPartyControls"))

    # 9) Threat detection => None=0, selected EDR/security tool=10
    td_val = _first_non_empty(security_data.get("edrToolsUsed"), security_data.get("edrTools"))
    if _is_no_threat_detection(td_val):
        scores["threat_detection_points"] = 0
    else:
        scores["threat_detection_points"] = 10

    # 10) Incident Response - check noIncidentResponse flag
    scores["incident_response_points"] = _score_incident_response(security_data, confidence_rating)

    scores.update(_score_new_security_questions(security_data))
    triggered = check_automatic_denial(security_data, business_data=business_data)
    fail = False
    if triggered:
        fail = True

    return scores, fail, confidence_rating
