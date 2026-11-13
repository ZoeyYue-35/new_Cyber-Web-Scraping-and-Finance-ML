import json
import os

# Constants for R multipliers and score limits
class Constants:
    BUSINESS_MAX_SCORE = 20
    BUSINESS_MIN_SCORE = 6.25
    SECURITY_MAX_SCORE = 35
    SECURITY_MIN_SCORE = 4

class DataProcessor:

    def process_questionnaire(data):
        """Process questionnaire dictionary to calculate total capped score."""

        if data['type'] == 'business':
            max_score = Constants.BUSINESS_MAX_SCORE
            min_score = Constants.BUSINESS_MIN_SCORE
        elif data['type'] == 'security':
            max_score = Constants.SECURITY_MAX_SCORE
            min_score = Constants.SECURITY_MIN_SCORE
        total_score = 0

        for key,score in data.items():
            if key == 'type':
                continue
            total_score += score
        
        #clamped_score = max(min(total_score, max_score), min_score)
        
        return total_score
    
class ValueCalculator:
    """Class to calculate final value from processed scores."""
    MAX_SCORE_NO_PENTEST = 102
    MAX_SCORE_PENTEST = 99
    def calculate_value(Q, assessment=None, PT=None):
        if PT is not None:
            print('Pentest')
            # 0.35*PT contributes 0-14; questionnaire term contributes 0-6.095
            return 0.35 * PT + (128 / 21) * (ValueCalculator.MAX_SCORE_PENTEST - Q) / ValueCalculator.MAX_SCORE_PENTEST
        else:
            print('No Pentest')
            return (ValueCalculator.MAX_SCORE_NO_PENTEST - Q) * 20 / (ValueCalculator.MAX_SCORE_NO_PENTEST - 1)

        # if Q and assessment:
        #     return 0.75 * assessment + 0.78 * Q


def v_calc(company_size, security_data, assessment_data, business_data):
    # File paths
    #cv_file_name = 'V_calc/PT.json'
    assessment_name = assessment_data['assessmentType']

    S_Q = DataProcessor.process_questionnaire(security_data)
    B_Q = DataProcessor.process_questionnaire(business_data)

    # Default normalization for non-PenTest paths; the PenTest branch overrides
    # this below. Guarantees S_bar is always defined so v_calc never raises an
    # UnboundLocalError when no assessment type is supplied.
    S_bar = S_Q / ValueCalculator.MAX_SCORE_NO_PENTEST

    # Process data to obtain PT, Q, and assessment values

    if assessment_name == "PenTest":

        PT = assessment_data['assessmentScore']
        PT = 40 - PT
        assessment_value = None
        S_bar = S_Q / ValueCalculator.MAX_SCORE_PENTEST
        if assessment_data['vulnerabilityData'] == [] and PT == 40:
            PT = None
    elif not assessment_name:
        PT = None
        if 'cisAnswers' in list(assessment_data.keys()):
            questions = assessment_data['cisAnswers']
        
        elif 'sigLiteAnswers' in list(assessment_data.keys()):
            questions = assessment_data['sigLiteAnswers']
        
        else:
            assessment_value = 0
        
    else:
        assessment_value = assessment_data['assessmentScore']
        PT = None
        S_bar = S_Q / ValueCalculator.MAX_SCORE_NO_PENTEST
    
    
    Q = (S_Q + B_Q)
    # Calculate final V value
    V = ValueCalculator.calculate_value(S_Q, assessment = assessment_value, PT = PT)

    # if company_size == 'micro':
    #     return V / 5
    # elif company_size == 'small':
    #     return 20 + V / 1.25
    # else:
    #     return 50 + V * 9.5

    return V, S_bar
