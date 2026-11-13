import openai
import os
import json
from .parse_security_LLM import get_security_scores
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment")

openai.api_key = OPENAI_API_KEY

questions_noPentest = {
    "useMFA": "All key employees (i.e. IT Staff, Cybersecurity professionals, Recovery Teams, Executive Leadership) and root users (i.e. System/Database/Network Administrators) use multi-factor authentication (MFA) to access company systems.",
    "remoteAccess": "Key employees (i.e. IT Staff, Cybersecurity professionals, Recovery Teams, Executive Leadership) work from home and access your company's systems outside the office.",
    "remoteAccessTypes": "Please mark which key workers have remote network access:", 
    "lockAccount": "Are special [privileged] accounts locked after multiple failed login attempts?",
    "noReceiveAlertApp": "Provide the name of any application you use to receive alerts on unusual activity on administrator accounts:",
    "securityTraining": "How often does your company provide formal cybersecurity training to employees (including privileged and critical users)?",
    "alternativeTraining": "Your company offers other ways to teach employees, such as workshops or team meetings.",
    "incidentResponseTraining": "Your company regularly practices what to do in the event of a cyber incident.",
    "documentProcedure": "Your company has an internal document that highlights who does what in case of a cyber breach.",
    "fixProblems": "Your company installs security updates within 60 days for all employee-used systems, software, and devices (Windows, Apple, or Third-party)",
    "updatesIn30Days": "Will your company begin applying all security patches and updates within 60 days going forward?",
    "dataEncryption": "Sensitive information (customer or business data) is encrypted both at rest and in transit.",
    "dataBacking": "All data is backed up.",
    "thirdPartyControls": "You have rules or tools to make sure third parties cannot access your systems without permission.",
    "edrToolsUsed": "Your company uses one or more of the following tools to detect threats in your network."
}

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
    
    # Handle various formats
    if rating in ["UNINSURABLE", "U (UNINSURABLE)"]:
        return "U"
    
    if rating in RATING_ORDER:
        return rating
    
    # Default fallback
    return "B"


def get_improvement_tiers(current_rating):
    """
    Get all improvement tiers above current rating
    Returns: (normalized_current_rating, list_of_improvement_tiers)
    """
    normalized = normalize_rating(current_rating)
    
    try:
        current_idx = RATING_ORDER.index(normalized)
    except ValueError:
        # If somehow invalid, default to B
        normalized = "B"
        current_idx = RATING_ORDER.index(normalized)
    
    # If already at top, no improvements
    if normalized == "A+":
        return normalized, []
    
    # Return all higher tiers
    improvement_tiers = RATING_ORDER[current_idx + 1:]
    
    return normalized, improvement_tiers


def extract_recommendations_from_previous(recommendation_data):
    """Extract all previous recommendations to avoid duplication"""
    previous_recs = set()
    
    if not recommendation_data:
        return previous_recs
    
    # Extract from pentest recommendations
    if "pentest_recommendations" in recommendation_data:
        for item in recommendation_data["pentest_recommendations"]:
            if isinstance(item, dict) and "recommendation" in item:
                previous_recs.add(item["recommendation"].lower().strip())
    
    # Extract from security responses
    if "security_responses" in recommendation_data:
        for item in recommendation_data["security_responses"]:
            if isinstance(item, dict) and "recommendation" in item:
                previous_recs.add(item["recommendation"].lower().strip())
    
    return previous_recs


def extract_numeric_value(value_str):
    """Extract numeric value from string (handles $, %, commas, etc.)"""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return 0.0
    
    # Remove common symbols and parse
    cleaned = value_str.replace('$', '').replace(',', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def call_llm_with_retry(system_prompt, max_retries=3, expected_tiers=None, current_breach_likelihood=None):
    """
    Call LLM with retry logic for missing tiers, invalid JSON, or incorrect likelihood ranges
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.7  # Slight randomness to avoid identical retries
            )
            
            if not response or not response.get("choices"):
                continue
            
            response_text = response["choices"][0]["message"]["content"]
            
            # Try to clean common JSON formatting issues
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Validate structure
            if expected_tiers:
                parsed, missing, is_valid, errors = validate_and_fix_json_structure(
                    response_text, 
                    expected_tiers,
                    current_breach_likelihood
                )
                
                if is_valid and parsed:
                    return response_text, parsed
                
                # If there are validation errors, update prompt for retry
                if errors and attempt < max_retries - 1:
                    error_details = "\n".join(errors)
                    
                    # Build correction instructions
                    correction_instructions = f"\n\n**CRITICAL ERRORS IN PREVIOUS ATTEMPT:**\n{error_details}\n\n"
                    
                    if missing:
                        correction_instructions += f"**MISSING RATINGS**: You MUST include these ratings: {', '.join(missing)}\n\n"
                    
                    # Add specific likelihood range requirements
                    correction_instructions += "**LIKELIHOOD RANGE REQUIREMENTS:**\n"
                    for tier in expected_tiers:
                        min_p, max_p = RATING_PROBABILITY_MAP[tier]
                        correction_instructions += f"- Rating {tier}: new_overall_likelihood_of_breach MUST be between {min_p}% and {max_p}%\n"
                    
                    correction_instructions += "\nEnsure each rating's likelihood falls within its valid range."
                    
                    system_prompt += correction_instructions
                    continue
            
            # Try to parse anyway
            try:
                parsed = json.loads(response_text)
                return response_text, parsed
            except:
                if attempt < max_retries - 1:
                    system_prompt += "\n\n**JSON FORMATTING ERROR**: Your previous response had invalid JSON. Ensure you return ONLY valid JSON with no markdown formatting, no explanations, and proper escaping of special characters."
                    continue
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            continue
    
    raise Exception("Failed to get valid LLM response after retries")


def get_recommendations(parsed_data, assessment, company_data, loss_data):
    securityQuestions = parsed_data.get("securityQuestions", {})

    # Compute security scores
    scores = get_security_scores(
        securityQuestions,
        int(company_data.get("num_employees", 0))
    )

    combined_list = []

    for key, question_text in questions_noPentest.items():
        if key not in securityQuestions:
            continue

        if key == "remoteAccessTypes":
            answer = {
                emp: int(securityQuestions.get(emp, 0))
                for emp in securityQuestions.get("remoteAccessTypes", [])
            }
        else:
            answer = securityQuestions.get(key)

        combined_list.append({
            "question": question_text,
            "answer": answer,
            "score": scores.get(key, 0)
        })

    # Select system prompt
    if (
        assessment == "PenTest"
        and parsed_data.get("assessmentData", {}).get("vulnerabilityData")
    ):
        system_prompt = get_pentest_system_prompt(
            parsed_data["assessmentData"]["vulnerabilityData"],
            combined_list,
            company_data,
            loss_data
        )
    else:
        system_prompt = get_noPentest_system_prompt(
            combined_list,
            company_data,
            loss_data
        )

    # LLM call with retry - ADD current_breach_likelihood parameter
    try:
        current_likelihood = loss_data.get('P_value', 50.0)
        response_text, _ = call_llm_with_retry(
            system_prompt, 
            max_retries=2,
            expected_tiers=None,  # No tiers for initial recommendations
            current_breach_likelihood=current_likelihood
        )
        return response_text
    except Exception as e:
        return json.dumps({"error": f"Recommendations cannot be generated: {str(e)}"})

def get_improvement_recommendations(
    parsed_data,
    assessment,
    company_data,
    loss_data,
    recommendation_data
):
    # Parse existing recommendations
    try:
        recommendation_data = json.loads(recommendation_data) if isinstance(recommendation_data, str) else recommendation_data
    except:
        recommendation_data = {}

    securityQuestions = parsed_data.get("securityQuestions", {})

    # Compute scores
    scores = get_security_scores(
        securityQuestions,
        int(company_data.get("num_employees", 0))
    )

    combined_list = []

    for key, question_text in questions_noPentest.items():
        if key not in securityQuestions:
            continue

        if key == "remoteAccessTypes":
            answer = {
                emp: int(securityQuestions.get(emp, 0))
                for emp in securityQuestions.get("remoteAccessTypes", [])
            }
        else:
            answer = securityQuestions.get(key)

        combined_list.append({
            "question": question_text,
            "answer": answer,
            "score": scores.get(key, 0)
        })

    # Get current rating and improvement tiers
    current_rating_raw = loss_data.get('Rating', 'B')
    current_rating, improvement_tiers = get_improvement_tiers(current_rating_raw)

    # If already A+, return early
    if not improvement_tiers:
        return json.dumps({
            "message": "Company already at A+ rating. No further improvements available.",
            "current_rating": "A+"
        })

    # Extract previous recommendations for deduplication
    previous_recommendations = extract_recommendations_from_previous(recommendation_data)

    # Select correct prompt
    if (
        assessment == "PenTest"
        and parsed_data.get("assessmentData", {}).get("vulnerabilityData")
    ):
        system_prompt = get_Pentest_improvement_system_prompt(
            parsed_data["assessmentData"]["vulnerabilityData"],
            combined_list,
            company_data,
            loss_data,
            recommendation_data,
            current_rating,
            improvement_tiers,
            previous_recommendations
        )
    else:
        system_prompt = get_noPentest_improvement_system_prompt(
            combined_list,
            company_data,
            loss_data,
            recommendation_data,
            current_rating,
            improvement_tiers,
            previous_recommendations
        )

    # LLM call with validation
    try:
        print(loss_data)
        current_likelihood = loss_data.get('P_value', 50.0)
        response_text, parsed = call_llm_with_retry(
            system_prompt, 
            max_retries=3,
            expected_tiers=improvement_tiers,
            current_breach_likelihood=current_likelihood
        )
        return response_text
    except Exception as e:
        return json.dumps({
            "error": f"Improvement Recommendations cannot be generated: {str(e)}",
            "current_rating": current_rating,
            "expected_tiers": improvement_tiers
        })


def get_pentest_system_prompt(vulnerabilities, securityResponses, company_data, loss_data):
    system_prompt = (
        f"Given a {company_data['company_size']}-sized {company_data['primary_industry']} company with an annual revenue of {company_data['annual_revenue']} USD. The company has a likelihood of breach probability of {loss_data['P_value']}% at the moment. "
        f"In the event of a successful breach, the estimated result is that the company would experience a financial loss of {loss_data['Total']['total_loss']} USD before implementing any security improvements. "
        "The company performed a pentest and discovered the following vulnerabilities along with their risk score: \n\n"
        + "\n".join([f"{vuln['Vulnerability name']} - CVSS Score: {vuln['CVSS Score']} - Risk Level: {vuln['Risk Score/Level']}" for vuln in vulnerabilities]) +
        "\n\nAlong with the pentest the company answered to some security questions, the response and a score for each response is provided: \n\n"
        + "\n".join([f"{response['question']} - Response: {response['answer']} - Points Earned: {response['score']}" for response in securityResponses]) +
        "\n\nThe maximum possible score for this questionnaire is 48 points. All questions contribute a fixed number of points towards the overall score of 48 points. "
        "An overall questionnaire score closer to 48 means a lower likelihood of breach. A questionnaire score further from 48 suggests an increased likelihood of breach. "
        "\n\n### Task:\n"
        "Based on the vulnerabilities found in the pentest provide specific recommendations (in 125 words or less) for all the vulnerabilities the company can implement based on the company's industry. "
        "Also, based on the security question responses provide specific recommendations (in 125 words or less) in 5 different areas the company can implement based on the company's industry. "
        "Each recommendation must be very specific and directly address the corresponding problem or finding. "
        "For each finding, also provide a specific example of what a company can do to improve the situation—this example must also be tailored to the exact problem.\n"
        "For each recommendation provide a breach likelihood reduction (expressed as a percentage) and reduction in total loss if this recommendation is implemented independently. "
        "Each recommendation's reduction value should be calculated as if it were the only change made. "
        "Also, provide the new overall breach likelihood probability (as an overall percentage) and the revised total financial loss in USD if all recommendations are implemented together. "
        "Use a non-linear risk decay model to calculate the final overall breach likelihood probability.\n"
        "\n\nProvide the response in VALID JSON format only (no markdown, no explanations) with this exact structure:\n"
        "\n\n**CRITICAL: Use 'vulnerability' as the key name for BOTH pentest_recommendations AND security_responses. Do NOT use 'area'.**\n\n"
        '''{
            "pentest_recommendations": [
                {
                    "vulnerability": "vulnerability_name_here",
                    "recommendation": "specific_recommendation",
                    "example": "concrete_example",
                    "reduction_in_total_loss": "numeric_value_without_currency",
                    "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
                }
            ],
            "new_overall_likelihood_of_breach": "numeric_value_with_percent_sign",
            "new_estimated_total_loss": "numeric_value_without_currency",
            "security_responses": [
                {
                    "vulnerability": "security_area_name_here",
                    "recommendation": "specific_recommendation",
                    "example": "concrete_example",
                    "reduction_in_total_loss": "numeric_value_without_currency",
                    "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
                }
            ]
        }'''
    )
    return system_prompt

def get_noPentest_system_prompt(securityResponses, company_data, loss_data):
    system_prompt = (
        f"Given a {company_data['company_size']}-sized {company_data['primary_industry']} company with an annual revenue of {company_data['annual_revenue']} USD. The company has a likelihood of breach probability of {loss_data['P_value']}% at the moment. "
        f"In the event of a successful breach, the estimated result is that the company would experience a financial loss of {loss_data['Total']['total_loss']} USD before implementing any security improvements. "
        "The company answered to some security questions, the response and a score for each response is provided: \n\n"
        + "\n".join([f"{response['question']} - Response: {response['answer']} - Points Earned: {response['score']}" for response in securityResponses]) +
        "\n\nThe maximum possible score for this questionnaire is 48 points. "
        "\n\n### Task:\n"
        "Based on the responses provide specific recommendations (in 125 words or less) in 5 different areas the company can implement based on the company's industry. "
        "Each recommendation must be very specific and directly address the corresponding problem or finding. "
        "For each finding, also provide a specific example of what a company can do to improve the situation—this example must also be tailored to the exact problem.\n"
        "For each recommendation provide a breach likelihood reduction (expressed as a percentage) and reduction in total loss if this recommendation is implemented independently. "
        "Also, provide the new overall breach likelihood probability (as an overall percentage) and the revised total financial loss in USD if all recommendations are implemented together. "
        "Use a non-linear risk decay model to calculate the final overall breach likelihood probability.\n"
        "\n\nProvide the response in VALID JSON format only (no markdown, no explanations) with this exact structure:\n"
        "\n\n**CRITICAL: Use 'vulnerability' as the key name consistently. Do NOT use 'area'.**\n\n"
        '''{
            "security_responses": [
                {
                    "vulnerability": "security_area_name_here",
                    "recommendation": "specific_recommendation",
                    "example": "concrete_example",
                    "reduction_in_total_loss": "numeric_value_without_currency",
                    "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
                }
            ],
            "new_overall_likelihood_of_breach": "numeric_value_with_percent_sign",
            "new_estimated_total_loss": "numeric_value_without_currency"
        }'''
    )
    return system_prompt


def get_noPentest_improvement_system_prompt(
    securityResponses, 
    company_data, 
    loss_data, 
    recommendation_data,
    current_rating,
    improvement_tiers,
    previous_recommendations
):
    # Format security responses
    sec_resp_text = "\n".join(
        f"{r['question']} - Response: {r['answer']} - Points Earned: {r['score']}"
        for r in securityResponses
    )

    # Format previous recommendations for exclusion
    prev_recs_text = "NONE"
    if previous_recommendations:
        prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

    # Create explicit tier list
    tier_list = "\n".join([f"- {tier}" for tier in improvement_tiers])

    system_prompt = f"""
### CRITICAL INSTRUCTIONS - READ CAREFULLY

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH of the following rating tiers. MISSING ANY TIER IS A CRITICAL FAILURE.

**REQUIRED OUTPUT TIERS (YOU MUST INCLUDE ALL OF THESE):**
{tier_list}

**CURRENT RATING:** {current_rating}

### Company Information
- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry')}
- Annual Revenue: {company_data.get('annual_revenue')} USD
- Current Breach Likelihood: {loss_data.get('P_value')}%
- Estimated Total Loss (pre-improvement): {loss_data.get('Total', {}).get('total_loss')} USD

### Security Questionnaire Responses
{sec_resp_text}

### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT ANY OF THESE)
{prev_recs_text}

### Rating-to-Probability Mapping
| Rating | Probability Range (%) |
|--------|----------------------|
| A+     | 2.0 - 11.10         |
| A      | 11.11 - 22.21       |
| A-     | 22.22 - 33.32       |
| B+     | 33.33 - 44.43       |
| B      | 44.44 - 55.54       |
| B-     | 55.55 - 66.65       |
| U      | 66.66 - 98.0        |

### Requirements for EACH Rating Tier

For EACH tier listed above, you MUST provide:
1. **Exactly 5 unique recommendations** (each ≤125 words total)
2. Each recommendation must include:
   - "recommendation": specific action to take
   - "reduction_in_total_loss": numeric value (no $ symbol)
   - "likelihood_reduction": percentage value (with % symbol)
3. Calculate:
   - "new_overall_likelihood_of_breach": MUST fall within the target rating's probability range (see table above)
   - "new_estimated_total_loss": numeric value (no $ symbol)

**CRITICAL LIKELIHOOD REQUIREMENTS:**
{chr(10).join([f"- For rating {tier}: new_overall_likelihood_of_breach MUST be between {RATING_PROBABILITY_MAP[tier][0]}% and {RATING_PROBABILITY_MAP[tier][1]}%" for tier in improvement_tiers])}

Each rating tier should show progressive improvement. The likelihood should decrease as the rating improves.

### STRICT JSON FORMAT (NO MARKDOWN, NO EXPLANATIONS)

YOU MUST RETURN EXACTLY THIS STRUCTURE WITH ALL {len(improvement_tiers)} TIERS:

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "{improvement_tiers[0] if improvement_tiers else 'A+'}",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{
                        "recommendation": "Deploy EDR on all endpoints...",
                        "reduction_in_total_loss": "50000",
                        "likelihood_reduction": "8%"
                    }},
                    {{
                        "recommendation": "Implement MFA for all remote access...",
                        "reduction_in_total_loss": "40000",
                        "likelihood_reduction": "6%"
                    }},
                    {{
                        "recommendation": "Third recommendation...",
                        "reduction_in_total_loss": "30000",
                        "likelihood_reduction": "5%"
                    }},
                    {{
                        "recommendation": "Fourth recommendation...",
                        "reduction_in_total_loss": "25000",
                        "likelihood_reduction": "4%"
                    }},
                    {{
                        "recommendation": "Fifth recommendation...",
                        "reduction_in_total_loss": "20000",
                        "likelihood_reduction": "3%"
                    }}
                ],
                "new_overall_likelihood_of_breach": "15.5%",
                "new_estimated_total_loss": "800000"
            }}
        }}{"," if len(improvement_tiers) > 1 else ""}
        {f'''{{
            "improved_rating": "{improvement_tiers[1] if len(improvement_tiers) > 1 else 'A+'}",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}}
                ],
                "new_overall_likelihood_of_breach": "...",
                "new_estimated_total_loss": "..."
            }}
        }}''' if len(improvement_tiers) > 1 else ""}
    ]
}}

**CRITICAL**: Include a block for EVERY tier in the list above. Missing even one tier is unacceptable.
"""
    
    return system_prompt


def get_Pentest_improvement_system_prompt(
    vulnerabilities,
    securityResponses,
    company_data,
    loss_data,
    recommendation_data,
    current_rating,
    improvement_tiers,
    previous_recommendations
):
    # Format vulnerabilities
    vuln_text = "\n".join(
        f"{vuln['Vulnerability name']} - CVSS: {vuln['CVSS Score']} - Risk: {vuln['Risk Score/Level']}"
        for vuln in vulnerabilities
    )

    # Format security responses
    sec_resp_text = "\n".join(
        f"{r['question']} - Response: {r['answer']} - Points: {r['score']}"
        for r in securityResponses
    )

    # Format previous recommendations
    prev_recs_text = "NONE"
    if previous_recommendations:
        prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

    # Create tier list
    tier_list = "\n".join([f"- {tier}" for tier in improvement_tiers])

    system_prompt = f"""
### CRITICAL INSTRUCTIONS - READ CAREFULLY

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH of the following rating tiers. MISSING ANY TIER IS A CRITICAL FAILURE.

**REQUIRED OUTPUT TIERS (YOU MUST INCLUDE ALL OF THESE):**
{tier_list}

**CURRENT RATING:** {current_rating}

### Company Information
- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry')}
- Annual Revenue: {company_data.get('annual_revenue')} USD
- Current Breach Likelihood: {loss_data.get('P_value')}%
- Estimated Total Loss (pre-improvement): {loss_data.get('Total', {}).get('total_loss')} USD

### Penetration Test Vulnerabilities
{vuln_text}

### Security Questionnaire Responses
{sec_resp_text}

### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT ANY OF THESE)
{prev_recs_text}

### Rating-to-Probability Mapping
| Rating | Probability Range (%) |
|--------|----------------------|
| A+     | 2.0 - 11.10         |
| A      | 11.11 - 22.21       |
| A-     | 22.22 - 33.32       |
| B+     | 33.33 - 44.43       |
| B      | 44.44 - 55.54       |
| B-     | 55.55 - 66.65       |
| U      | 66.66 - 98.0        |

### Requirements for EACH Rating Tier

For EACH tier listed above, you MUST provide:
1. **Exactly 5 unique recommendations** (each ≤125 words total)
2. Each recommendation must:
   - Address vulnerabilities or questionnaire weaknesses
   - Be specific to the company's industry
   - Include concrete implementation examples
   - NOT repeat any previous recommendations
3. Each recommendation includes:
   - "recommendation": specific action
   - "reduction_in_total_loss": numeric (no $)
   - "likelihood_reduction": percentage (with %)
4. Calculate:
   - "new_overall_likelihood_of_breach": MUST be within target rating range (see table above)
   - "new_estimated_total_loss": numeric (no $)

**CRITICAL LIKELIHOOD REQUIREMENTS:**
{chr(10).join([f"- For rating {tier}: new_overall_likelihood_of_breach MUST be between {RATING_PROBABILITY_MAP[tier][0]}% and {RATING_PROBABILITY_MAP[tier][1]}%" for tier in improvement_tiers])}

Each rating tier should show progressive improvement. The likelihood should decrease as the rating improves.

### STRICT JSON FORMAT (NO MARKDOWN, NO EXPLANATIONS)

Return EXACTLY this structure with ALL {len(improvement_tiers)} tiers:

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "{improvement_tiers[0] if improvement_tiers else 'A+'}",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
                    {{"recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}}
                ],
                "new_overall_likelihood_of_breach": "...",
                "new_estimated_total_loss": "..."
            }}
        }}{", (continue for all remaining tiers)" if len(improvement_tiers) > 1 else ""}
    ]
}}

**CRITICAL**: You must include ALL {len(improvement_tiers)} rating tiers. Each tier needs exactly 5 recommendations.
"""
    
    return system_prompt


def validate_likelihood_in_range(likelihood, target_rating):
    """
    Check if likelihood falls within the valid range for the target rating
    Returns: (is_valid, corrected_likelihood)
    """
    if target_rating not in RATING_PROBABILITY_MAP:
        return False, likelihood
    
    min_prob, max_prob = RATING_PROBABILITY_MAP[target_rating]
    likelihood_val = extract_numeric_value(likelihood)
    
    # Check if within range (with small tolerance)
    if min_prob <= likelihood_val <= max_prob:
        return True, likelihood
    
    # Correct to midpoint of range
    corrected = round((min_prob + max_prob) / 2, 2)
    return False, f"{corrected}%"


def validate_and_fix_json_structure(response_text, expected_tiers, current_breach_likelihood):
    """
    Validate JSON structure, check for missing rating tiers, and validate breach likelihood ranges
    Returns: (parsed_json, missing_tiers, is_valid, validation_errors)
    """
    validation_errors = []
    
    try:
        data = json.loads(response_text)
        
        if "grade_improvement_recommendations" not in data:
            return None, expected_tiers, False, ["Missing grade_improvement_recommendations key"]
        
        # Get ratings present in response
        present_ratings = set()
        for item in data["grade_improvement_recommendations"]:
            if "improved_rating" in item:
                rating = normalize_rating(item["improved_rating"])
                present_ratings.add(rating)
                
                # Validate likelihood is in correct range
                if "recommendations_improved_rating" in item:
                    recs = item["recommendations_improved_rating"]
                    if "new_overall_likelihood_of_breach" in recs:
                        likelihood = recs["new_overall_likelihood_of_breach"]
                        is_valid, corrected = validate_likelihood_in_range(likelihood, rating)
                        
                        if not is_valid:
                            validation_errors.append(
                                f"Rating {rating}: likelihood {likelihood} is outside valid range "
                                f"{RATING_PROBABILITY_MAP[rating]}. Should be around {corrected}"
                            )
                            # Auto-correct the value
                            recs["new_overall_likelihood_of_breach"] = corrected
        
        # Find missing tiers
        expected_set = set(expected_tiers)
        missing_tiers = expected_set - present_ratings
        
        if missing_tiers:
            validation_errors.append(f"Missing rating tiers: {', '.join(missing_tiers)}")
        
        is_valid = len(missing_tiers) == 0 and len(validation_errors) == 0
        
        return data, list(missing_tiers), is_valid, validation_errors
        
    except json.JSONDecodeError as e:
        return None, expected_tiers, False, [f"JSON parsing error: {str(e)}"]


def call_llm_with_retry(system_prompt, max_retries=3, expected_tiers=None, current_breach_likelihood=None):
    """
    Call LLM with retry logic for missing tiers, invalid JSON, or incorrect likelihood ranges
    """
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.7
            )
            
            if not response or not response.get("choices"):
                continue
            
            response_text = response["choices"][0]["message"]["content"]
            
            # Clean JSON formatting issues
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Validate structure ONLY if expected_tiers is provided
            if expected_tiers:
                parsed, missing, is_valid, errors = validate_and_fix_json_structure(
                    response_text, 
                    expected_tiers,
                    current_breach_likelihood
                )
                
                if is_valid and parsed:
                    return response_text, parsed
                
                # If there are validation errors, update prompt for retry
                if errors and attempt < max_retries - 1:
                    error_details = "\n".join(errors)
                    
                    correction_instructions = f"\n\n**CRITICAL ERRORS IN PREVIOUS ATTEMPT:**\n{error_details}\n\n"
                    
                    if missing:
                        correction_instructions += f"**MISSING RATINGS**: You MUST include these ratings: {', '.join(missing)}\n\n"
                    
                    correction_instructions += "**LIKELIHOOD RANGE REQUIREMENTS:**\n"
                    for tier in expected_tiers:
                        min_p, max_p = RATING_PROBABILITY_MAP[tier]
                        correction_instructions += f"- Rating {tier}: new_overall_likelihood_of_breach MUST be between {min_p}% and {max_p}%\n"
                    
                    correction_instructions += "\nEnsure each rating's likelihood falls within its valid range."
                    
                    system_prompt += correction_instructions
                    continue
            else:
                # For initial recommendations (no tiers), just parse JSON
                try:
                    parsed = json.loads(response_text)
                    return response_text, parsed
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        system_prompt += "\n\n**JSON FORMATTING ERROR**: Your previous response had invalid JSON. Ensure you return ONLY valid JSON with no markdown formatting, no explanations, and proper escaping of special characters."
                        continue
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            continue
    
    raise Exception("Failed to get valid LLM response after retries")

def get_recommendations(parsed_data, assessment, company_data, loss_data):
    securityQuestions = parsed_data.get("securityQuestions", {})

    # Compute security scores
    scores = get_security_scores(
        securityQuestions,
        int(company_data.get("num_employees", 0))
    )

    combined_list = []

    for key, question_text in questions_noPentest.items():
        if key not in securityQuestions:
            continue

        if key == "remoteAccessTypes":
            answer = {
                emp: int(securityQuestions.get(emp, 0))
                for emp in securityQuestions.get("remoteAccessTypes", [])
            }
        else:
            answer = securityQuestions.get(key)

        combined_list.append({
            "question": question_text,
            "answer": answer,
            "score": scores.get(key, 0)
        })

    # Select system prompt
    if (
        assessment == "PenTest"
        and parsed_data.get("assessmentData", {}).get("vulnerabilityData")
    ):
        system_prompt = get_pentest_system_prompt(
            parsed_data["assessmentData"]["vulnerabilityData"],
            combined_list,
            company_data,
            loss_data
        )
    else:
        system_prompt = get_noPentest_system_prompt(
            combined_list,
            company_data,
            loss_data
        )

    # LLM call with retry
    try:
        response_text, _ = call_llm_with_retry(system_prompt, max_retries=2)
        return response_text
    except Exception as e:
        return json.dumps({"error": f"Recommendations cannot be generated: {str(e)}"})


def get_improvement_recommendations(
    parsed_data,
    assessment,
    company_data,
    loss_data,
    recommendation_data
):
    # Parse existing recommendations
    try:
        recommendation_data = json.loads(recommendation_data) if isinstance(recommendation_data, str) else recommendation_data
    except:
        recommendation_data = {}

    securityQuestions = parsed_data.get("securityQuestions", {})

    # Compute scores
    scores = get_security_scores(
        securityQuestions,
        int(company_data.get("num_employees", 0))
    )

    combined_list = []

    for key, question_text in questions_noPentest.items():
        if key not in securityQuestions:
            continue

        if key == "remoteAccessTypes":
            answer = {
                emp: int(securityQuestions.get(emp, 0))
                for emp in securityQuestions.get("remoteAccessTypes", [])
            }
        else:
            answer = securityQuestions.get(key)

        combined_list.append({
            "question": question_text,
            "answer": answer,
            "score": scores.get(key, 0)
        })

    # Get current rating and improvement tiers
    current_rating_raw = loss_data.get('Rating', 'B')
    current_rating, improvement_tiers = get_improvement_tiers(current_rating_raw)

    # If already A+, return early
    if not improvement_tiers:
        return json.dumps({
            "message": "Company already at A+ rating. No further improvements available.",
            "current_rating": "A+"
        })

    # Extract previous recommendations for deduplication
    previous_recommendations = extract_recommendations_from_previous(recommendation_data)

    # Select correct prompt
    if (
        assessment == "PenTest"
        and parsed_data.get("assessmentData", {}).get("vulnerabilityData")
    ):
        system_prompt = get_Pentest_improvement_system_prompt(
            parsed_data["assessmentData"]["vulnerabilityData"],
            combined_list,
            company_data,
            loss_data,
            recommendation_data,
            current_rating,
            improvement_tiers,
            previous_recommendations
        )
    else:
        system_prompt = get_noPentest_improvement_system_prompt(
            combined_list,
            company_data,
            loss_data,
            recommendation_data,
            current_rating,
            improvement_tiers,
            previous_recommendations
        )

    # LLM call with validation
    try:
        response_text, parsed = call_llm_with_retry(
            system_prompt, 
            max_retries=3,
            expected_tiers=improvement_tiers
        )
        return response_text
    except Exception as e:
        return json.dumps({
            "error": f"Improvement Recommendations cannot be generated: {str(e)}",
            "current_rating": current_rating,
            "expected_tiers": improvement_tiers
        })


def get_pentest_system_prompt(vulnerabilities, securityResponses, company_data, loss_data):
    """
    System prompt for initial recommendations when pentest data is available
    """
    system_prompt = f"""
### CONTEXT

You are a cybersecurity risk assessment engine providing recommendations for a company with the following profile:

**Company Profile:**
- Size: {company_data['company_size']}
- Industry: {company_data['primary_industry']}
- Annual Revenue: ${company_data['annual_revenue']} USD
- Current Breach Likelihood: {loss_data['P_value']}%
- Estimated Financial Loss (if breached): ${loss_data['Total']['total_loss']} USD

**Penetration Test Findings:**
The company conducted a penetration test and identified the following vulnerabilities:

""" + "\n".join([f"- {vuln['Vulnerability name']} | CVSS Score: {vuln['CVSS Score']} | Risk Level: {vuln['Risk Score/Level']}" for vuln in vulnerabilities]) + """

**Security Questionnaire Results:**
The company completed a security questionnaire (maximum score: 48 points). Higher scores indicate better security posture and lower breach likelihood.

""" + "\n".join([f"- {response['question']}\n  Response: {response['answer']} | Points Earned: {response['score']}/48" for response in securityResponses]) + """

---

### YOUR TASK

Generate two types of recommendations:

1. **Pentest Recommendations**: Address EACH vulnerability found in the penetration test
2. **Security Response Recommendations**: Address the 5 MOST CRITICAL security questionnaire weaknesses

**Requirements for EACH Recommendation:**
- Maximum 125 words per recommendation
- Phrase the "vulnerability" field as a current gap (e.g., "Lack of Multi-Factor Authentication")
- Phrase the "recommendation" field as an improvement action (e.g., "Implement MFA for all key employees")
- Include a concrete, industry-specific example
- Calculate independent impact (as if this is the ONLY change made)
- Use non-linear risk decay model for overall calculations

**Calculation Rules:**
- Each recommendation's impact is INDEPENDENT (don't assume other fixes are applied)
- "reduction_in_total_loss": Financial savings if this ONE recommendation is implemented
- "likelihood_reduction_in_percentage": Breach probability reduction if this ONE recommendation is implemented
- "new_overall_likelihood_of_breach": Final probability after implementing ALL recommendations (both pentest AND security responses)
- "new_estimated_total_loss": Final financial loss after implementing ALL recommendations (both pentest AND security responses)

---

### OUTPUT FORMAT REQUIREMENTS

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks, no explanations)
2. Use EXACTLY these field names - do not add, remove, or rename any fields
3. The "vulnerability" field MUST describe the current gap/weakness
4. Currency values: Include commas, NO dollar signs (e.g., "50,000" not "$50,000" or "50000")
5. Percentages: Whole numbers only, NO decimals, NO spaces (e.g., "8%" not "8.5%" or "8 %")
6. Escape all quotes in text fields using backslash before quotes
7. If no reduction is possible, use "0" for numeric fields

**REQUIRED JSON STRUCTURE:**

{{
    "pentest_recommendations": [
        {{
            "vulnerability": "Current gap/weakness being addressed",
            "recommendation": "Specific improvement action to take",
            "example": "Concrete industry-specific implementation example",
            "reduction_in_total_loss": "Numeric value with commas, no currency symbol",
            "likelihood_reduction_in_percentage": "Whole number percentage with % symbol, no spaces"
        }}
    ],
    "security_responses": [
        {{
            "vulnerability": "Current gap/weakness being addressed",
            "recommendation": "Specific improvement action to take",
            "example": "Concrete industry-specific implementation example",
            "reduction_in_total_loss": "Numeric value with commas, no currency symbol",
            "likelihood_reduction_in_percentage": "Whole number percentage with % symbol, no spaces"
        }}
    ],
    "new_overall_likelihood_of_breach": "Whole number percentage after ALL recommendations",
    "new_estimated_total_loss": "Numeric value with commas after ALL recommendations"
}}

**EXAMPLE (format reference only):**

{{
    "pentest_recommendations": [
        {{
            "vulnerability": "Unpatched Microsoft Message Queuing RCE vulnerability",
            "recommendation": "Apply security patch CVE-2023-21554 to all affected systems within 48 hours",
            "example": "Use Windows Update or WSUS to deploy the patch to all servers running MSMQ service, then verify installation with vulnerability scanner",
            "reduction_in_total_loss": "125,000",
            "likelihood_reduction_in_percentage": "12%"
        }}
    ],
    "security_responses": [
        {{
            "vulnerability": "Lack of multi-factor authentication for key personnel",
            "recommendation": "Deploy MFA across all administrative and executive accounts using authentication apps",
            "example": "Implement Microsoft Authenticator or Google Authenticator for all users with elevated privileges, including IT staff and C-suite executives",
            "reduction_in_total_loss": "85,000",
            "likelihood_reduction_in_percentage": "9%"
        }}
    ],
    "new_overall_likelihood_of_breach": "32%",
    "new_estimated_total_loss": "890,000"
}}

---

### FINAL CHECKLIST BEFORE RESPONDING

- [ ] Addressed ALL pentest vulnerabilities
- [ ] Provided exactly 5 security response recommendations
- [ ] Each recommendation ≤125 words
- [ ] All "vulnerability" fields describe current gaps
- [ ] All currency values have commas, no $ symbol
- [ ] All percentages are whole numbers with % symbol, no spaces
- [ ] JSON is valid and properly escaped
- [ ] Used ONLY the exact field names specified
- [ ] Calculated new_overall_likelihood_of_breach considering ALL recommendations together
- [ ] Calculated new_estimated_total_loss considering ALL recommendations together

Return ONLY the JSON output, nothing else.
"""
    return system_prompt


def get_noPentest_system_prompt(securityResponses, company_data, loss_data):
    """
    System prompt for initial recommendations when NO pentest data is available
    """
    system_prompt = f"""
### CONTEXT

You are a cybersecurity risk assessment engine providing recommendations for a company with the following profile:

**Company Profile:**
- Size: {company_data['company_size']}
- Industry: {company_data['primary_industry']}
- Annual Revenue: ${company_data['annual_revenue']} USD
- Current Breach Likelihood: {loss_data['P_value']}%
- Estimated Financial Loss (if breached): ${loss_data['Total']['total_loss']} USD

**Security Questionnaire Results:**
The company completed a security questionnaire (maximum score: 48 points). Higher scores indicate better security posture and lower breach likelihood.

""" + "\n".join([f"- **{tier}**: new_overall_likelihood_of_breach must be between {RATING_PROBABILITY_MAP[tier][0]}% and {RATING_PROBABILITY_MAP[tier][1]}%" for tier in improvement_tiers]) + """
---

### YOUR TASK

Generate **5 CRITICAL security recommendations** based on the questionnaire weaknesses that will have the greatest impact on reducing breach likelihood and financial loss.

**Requirements for EACH Recommendation:**
- Maximum 125 words per recommendation
- Phrase the "vulnerability" field as a current gap (e.g., "Lack of Multi-Factor Authentication")
- Phrase the "recommendation" field as an improvement action (e.g., "Implement MFA for all key employees")
- Include a concrete, industry-specific example
- Calculate independent impact (as if this is the ONLY change made)
- Use non-linear risk decay model for overall calculations

**Calculation Rules:**
- Each recommendation's impact is INDEPENDENT (don't assume other fixes are applied)
- "reduction_in_total_loss": Financial savings if this ONE recommendation is implemented
- "likelihood_reduction_in_percentage": Breach probability reduction if this ONE recommendation is implemented
- "new_overall_likelihood_of_breach": Final probability after implementing ALL 5 recommendations
- "new_estimated_total_loss": Final financial loss after implementing ALL 5 recommendations

---

### OUTPUT FORMAT REQUIREMENTS

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks, no explanations)
2. Use EXACTLY these field names - do not add, remove, or rename any fields
3. The "vulnerability" field MUST describe the current gap/weakness
4. Currency values: Include commas, NO dollar signs (e.g., "50,000" not "$50,000" or "50000")
5. Percentages: Whole numbers only, NO decimals, NO spaces (e.g., "8%" not "8.5%" or "8 %")
6. Escape all quotes in text fields using backslash before quotes
7. If no reduction is possible, use "0" for numeric fields

**REQUIRED JSON STRUCTURE:**

{{
    "security_responses": [
        {{
            "vulnerability": "Current gap/weakness being addressed",
            "recommendation": "Specific improvement action to take",
            "example": "Concrete industry-specific implementation example",
            "reduction_in_total_loss": "Numeric value with commas, no currency symbol",
            "likelihood_reduction_in_percentage": "Whole number percentage with % symbol, no spaces"
        }}
    ],
    "new_overall_likelihood_of_breach": "Whole number percentage after ALL recommendations",
    "new_estimated_total_loss": "Numeric value with commas after ALL recommendations"
}}

**EXAMPLE (format reference only):**

{{
    "security_responses": [
        {{
            "vulnerability": "Lack of multi-factor authentication for key personnel",
            "recommendation": "Deploy MFA across all administrative and executive accounts using authentication apps",
            "example": "Implement Microsoft Authenticator or Google Authenticator for all users with elevated privileges, including IT staff and C-suite executives",
            "reduction_in_total_loss": "85,000",
            "likelihood_reduction_in_percentage": "9%"
        }},
        {{
            "vulnerability": "Insufficient data backup frequency",
            "recommendation": "Increase backup frequency to daily automated backups with offsite storage",
            "example": "Configure Veeam or Acronis to perform daily incremental backups with weekly full backups stored in AWS S3 Glacier for disaster recovery",
            "reduction_in_total_loss": "65,000",
            "likelihood_reduction_in_percentage": "7%"
        }},
        {{
            "vulnerability": "Delayed security patch deployment",
            "recommendation": "Implement automated patch management to deploy critical updates within 15 days",
            "example": "Use Windows WSUS or third-party tools like ManageEngine to automate patch deployment across all endpoints and servers",
            "reduction_in_total_loss": "72,000",
            "likelihood_reduction_in_percentage": "8%"
        }},
        {{
            "vulnerability": "Absence of endpoint detection and response tools",
            "recommendation": "Deploy EDR solution across all company endpoints for threat detection",
            "example": "Install CrowdStrike Falcon or Microsoft Defender for Endpoint on all workstations and servers to monitor and respond to threats in real-time",
            "reduction_in_total_loss": "95,000",
            "likelihood_reduction_in_percentage": "10%"
        }},
        {{
            "vulnerability": "Inadequate employee security awareness training",
            "recommendation": "Implement quarterly security training with phishing simulations for all employees",
            "example": "Use KnowBe4 or Proofpoint to deliver interactive training modules and monthly phishing tests to reduce social engineering risks",
            "reduction_in_total_loss": "55,000",
            "likelihood_reduction_in_percentage": "6%"
        }}
    ],
    "new_overall_likelihood_of_breach": "28%",
    "new_estimated_total_loss": "780,000"
}}

---

### FINAL CHECKLIST BEFORE RESPONDING

- [ ] Provided exactly 5 security response recommendations
- [ ] Each recommendation ≤125 words
- [ ] All "vulnerability" fields describe current gaps
- [ ] All currency values have commas, no $ symbol
- [ ] All percentages are whole numbers with % symbol, no spaces
- [ ] JSON is valid and properly escaped
- [ ] Used ONLY the exact field names specified
- [ ] Calculated new_overall_likelihood_of_breach considering ALL 5 recommendations together
- [ ] Calculated new_estimated_total_loss considering ALL 5 recommendations together

Return ONLY the JSON output, nothing else.
"""
    return system_prompt


def get_noPentest_improvement_system_prompt(
    securityResponses, 
    company_data, 
    loss_data, 
    recommendation_data,
    current_rating,
    improvement_tiers,
    previous_recommendations
):
    """
    System prompt for improvement recommendations (no pentest) - generates recommendations to move up rating tiers
    """
    # Format security responses
    sec_resp_text = "\n".join(
        f"- {r['question']}\n  Response: {r['answer']} | Points Earned: {r['score']}/48"
        for r in securityResponses
    )

    # Format previous recommendations for exclusion
    prev_recs_text = "NONE"
    if previous_recommendations:
        prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

    # Create explicit tier list
    tier_list = "\n".join([f"{i+1}. {tier}" for i, tier in enumerate(improvement_tiers)])

    # Generate likelihood requirements for each tier
    likelihood_requirements = "\n".join([
        f"- **{tier}**: new_overall_likelihood_of_breach must be between {RATING_PROBABILITY_MAP[tier][0]}% and {RATING_PROBABILITY_MAP[tier][1]}%"
        for tier in improvement_tiers
    ])

    system_prompt = f"""
### CRITICAL INSTRUCTIONS

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH rating tier listed below. **MISSING ANY TIER IS A CRITICAL FAILURE.**

---

### REQUIRED OUTPUT TIERS (MUST INCLUDE ALL)

{tier_list}

**CURRENT RATING:** {current_rating}
**CURRENT BREACH LIKELIHOOD:** {loss_data.get('P_value')}%

You MUST generate recommendations for ALL {len(improvement_tiers)} tiers listed above.

---

### COMPANY INFORMATION

- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry')}
- Annual Revenue: ${company_data.get('annual_revenue')} USD
- Current Breach Likelihood: {loss_data.get('P_value')}%
- Estimated Total Loss (pre-improvement): ${loss_data.get('Total', {}).get('total_loss')} USD

---

### SECURITY QUESTIONNAIRE RESULTS

{sec_resp_text}

---

### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT)

The following recommendations have already been provided. DO NOT repeat, rephrase, or reuse ANY of these:

{prev_recs_text}

**CRITICAL:** All new recommendations MUST be completely unique and different from the above.

---

### RATING-TO-PROBABILITY MAPPING

| Rating | Breach Likelihood Range |
|--------|-------------------------|
| A+     | 2% - 11%               |
| A      | 11% - 22%              |
| A-     | 22% - 33%              |
| B+     | 33% - 44%              |
| B      | 44% - 55%              |
| B-     | 55% - 66%              |
| U      | 66% - 98%              |

---

### REQUIREMENTS FOR EACH RATING TIER

**For EACH tier, you MUST provide:**

1. **Exactly 5 unique recommendations** (each ≤125 words)
2. Each recommendation MUST include:
   - "vulnerability": Current gap/weakness being addressed
   - "recommendation": Specific improvement action to implement
   - "example": Concrete industry-specific implementation example
   - "reduction_in_total_loss": Financial impact if implemented (numeric with commas, no $)
   - "likelihood_reduction": Breach probability reduction (whole number %, no spaces)

3. **Cumulative Logic**: Each tier builds on the previous tier
   - Tier 1 recommendations assume ONLY the initial recommendations are implemented
   - Tier 2 recommendations assume initial + Tier 1 recommendations are implemented
   - Tier 3 recommendations assume initial + Tier 1 + Tier 2 recommendations are implemented
   - And so on...

4. **Progressive Improvement**: Each tier MUST have:
   - LOWER "new_overall_likelihood_of_breach" than the previous tier
   - LOWER "new_estimated_total_loss" than the previous tier
   - Likelihood MUST fall within the target rating's range

---

### CRITICAL LIKELIHOOD REQUIREMENTS

{likelihood_requirements}

Each tier's likelihood MUST be LOWER than the previous tier and MUST fall within its valid range.

**Example Progressive Improvement:**
- Current (B): 50%
- B+: 40% (within 33%-44%)
- A-: 28% (within 22%-33%)
- A: 18% (within 11%-22%)
- A+: 8% (within 2%-11%)

---

### OUTPUT FORMAT REQUIREMENTS

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks, no explanations)
2. Use EXACTLY these field names - do not add, remove, or rename any fields
3. The "vulnerability" field MUST describe the current gap/weakness
4. Currency values: Include commas, NO dollar signs (e.g., "50,000" not "$50,000")
5. Percentages: Whole numbers only, NO decimals, NO spaces (e.g., "8%" not "8.5%" or "8 %")
6. Escape all quotes in text fields
7. Include ALL {len(improvement_tiers)} rating tiers - missing even one is unacceptable

**REQUIRED JSON STRUCTURE:**

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "FIRST_TIER_FROM_LIST",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{
                        "vulnerability": "Current gap/weakness being addressed",
                        "recommendation": "Specific improvement action to implement",
                        "example": "Concrete industry-specific implementation example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Second gap/weakness",
                        "recommendation": "Second improvement action",
                        "example": "Second concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Third gap/weakness",
                        "recommendation": "Third improvement action",
                        "example": "Third concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Fourth gap/weakness",
                        "recommendation": "Fourth improvement action",
                        "example": "Fourth concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Fifth gap/weakness",
                        "recommendation": "Fifth improvement action",
                        "example": "Fifth concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }}
                ],
                "new_overall_likelihood_of_breach": "Must be within target range for this tier",
                "new_estimated_total_loss": "Numeric with commas, no $"
            }}
        }},
        {{
            "improved_rating": "SECOND_TIER_FROM_LIST",
            "recommendations_improved_rating": {{
                "recommendations": [
                    ... (5 MORE RECOMMENDATIONS) ...
                ],
                "new_overall_likelihood_of_breach": "LOWER than previous tier, within this tier's range",
                "new_estimated_total_loss": "LOWER than previous tier"
            }}
        }}
        ... CONTINUE FOR ALL {len(improvement_tiers)} TIERS ...
    ]
}}

---

### FINAL CHECKLIST BEFORE RESPONDING

- [ ] Included ALL {len(improvement_tiers)} rating tiers
- [ ] Each tier has exactly 5 recommendations
- [ ] Each recommendation ≤125 words
- [ ] All "vulnerability" fields describe current gaps
- [ ] All currency values have commas, no $ symbol
- [ ] All percentages are whole numbers with % symbol, no spaces
- [ ] Each tier's likelihood is LOWER than previous and within valid range
- [ ] No recommendations repeat previous recommendations
- [ ] JSON is valid and properly escaped
- [ ] Used ONLY the exact field names specified

**CRITICAL:** You MUST include ALL {len(improvement_tiers)} tiers. Missing even one tier is unacceptable.

Return ONLY the JSON output, nothing else.
"""
    
    return system_prompt


def get_Pentest_improvement_system_prompt(
    vulnerabilities,
    securityResponses,
    company_data,
    loss_data,
    recommendation_data,
    current_rating,
    improvement_tiers,
    previous_recommendations
):
    """
    System prompt for improvement recommendations (with pentest) - generates recommendations to move up rating tiers
    """
    # Format vulnerabilities
    vuln_text = "\n".join(
        f"- {vuln['Vulnerability name']} | CVSS: {vuln['CVSS Score']} | Risk: {vuln['Risk Score/Level']}"
        for vuln in vulnerabilities
    )

    # Format security responses
    sec_resp_text = "\n".join(
        f"- {r['question']}\n  Response: {r['answer']} | Points: {r['score']}/48"
        for r in securityResponses
    )

    # Format previous recommendations
    prev_recs_text = "NONE"
    if previous_recommendations:
        prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

    # Create tier list
    tier_list = "\n".join([f"{i+1}. {tier}" for i, tier in enumerate(improvement_tiers)])

    # Generate likelihood requirements
    likelihood_requirements = "\n".join([
        f"- **{tier}**: new_overall_likelihood_of_breach must be between {RATING_PROBABILITY_MAP[tier][0]}% and {RATING_PROBABILITY_MAP[tier][1]}%"
        for tier in improvement_tiers
    ])

    system_prompt = f"""
### CRITICAL INSTRUCTIONS

You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH rating tier listed below. **MISSING ANY TIER IS A CRITICAL FAILURE.**

---

### REQUIRED OUTPUT TIERS (MUST INCLUDE ALL)

{tier_list}

**CURRENT RATING:** {current_rating}
**CURRENT BREACH LIKELIHOOD:** {loss_data.get('P_value')}%

You MUST generate recommendations for ALL {len(improvement_tiers)} tiers listed above.

---

### COMPANY INFORMATION

- Size: {company_data.get('company_size')}
- Industry: {company_data.get('primary_industry')}
- Annual Revenue: ${company_data.get('annual_revenue')} USD
- Current Breach Likelihood: {loss_data.get('P_value')}%
- Estimated Total Loss (pre-improvement): ${loss_data.get('Total', {}).get('total_loss')} USD

---

### PENETRATION TEST FINDINGS

{vuln_text}

---

### SECURITY QUESTIONNAIRE RESULTS

{sec_resp_text}

---

### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT)

The following recommendations have already been provided. DO NOT repeat, rephrase, or reuse ANY of these:

{prev_recs_text}

**CRITICAL:** All new recommendations MUST be completely unique and different from the above.

---

### RATING-TO-PROBABILITY MAPPING

| Rating | Breach Likelihood Range |
|--------|-------------------------|
| A+     | 2% - 11%               |
| A      | 11% - 22%              |
| A-     | 22% - 33%              |
| B+     | 33% - 44%              |
| B      | 44% - 55%              |
| B-     | 55% - 66%              |
| U      | 66% - 98%              |

---

### REQUIREMENTS FOR EACH RATING TIER

**For EACH tier, you MUST provide:**

1. **Exactly 5 unique recommendations** (each ≤125 words)
2. Recommendations should address:
   - Remaining pentest vulnerabilities not yet fixed
   - Security questionnaire weaknesses
   - Industry-specific security controls
   - Compliance requirements (NIST, CIS, CISA, ISO 27001)

3. Each recommendation MUST include:
   - "vulnerability": Current gap/weakness being addressed
   - "recommendation": Specific improvement action to implement
   - "example": Concrete industry-specific implementation example
   - "reduction_in_total_loss": Financial impact if implemented (numeric with commas, no $)
   - "likelihood_reduction": Breach probability reduction (whole number %, no spaces)

4. **Cumulative Logic**: Each tier builds on the previous tier
   - Tier 1 recommendations assume ONLY the initial recommendations are implemented
   - Tier 2 recommendations assume initial + Tier 1 recommendations are implemented
   - Tier 3 recommendations assume initial + Tier 1 + Tier 2 recommendations are implemented
   - And so on...

5. **Progressive Improvement**: Each tier MUST have:
   - LOWER "new_overall_likelihood_of_breach" than the previous tier
   - LOWER "new_estimated_total_loss" than the previous tier
   - Likelihood MUST fall within the target rating's range

---

### CRITICAL LIKELIHOOD REQUIREMENTS

{likelihood_requirements}

Each tier's likelihood MUST be LOWER than the previous tier and MUST fall within its valid range.

**Example Progressive Improvement:**
- Current (B): 50%
- B+: 40% (within 33%-44%)
- A-: 28% (within 22%-33%)
- A: 18% (within 11%-22%)
- A+: 8% (within 2%-11%)

---

### OUTPUT FORMAT REQUIREMENTS

**CRITICAL FORMATTING RULES:**
1. Return ONLY valid JSON (no markdown, no code blocks, no explanations)
2. Use EXACTLY these field names - do not add, remove, or rename any fields
3. The "vulnerability" field MUST describe the current gap/weakness
4. Currency values: Include commas, NO dollar signs (e.g., "50,000" not "$50,000")
5. Percentages: Whole numbers only, NO decimals, NO spaces (e.g., "8%" not "8.5%" or "8 %")
6. Escape all quotes in text fields
7. Include ALL {len(improvement_tiers)} rating tiers - missing even one is unacceptable

**REQUIRED JSON STRUCTURE:**

{{
    "grade_improvement_recommendations": [
        {{
            "improved_rating": "FIRST_TIER_FROM_LIST",
            "recommendations_improved_rating": {{
                "recommendations": [
                    {{
                        "vulnerability": "Current gap/weakness being addressed",
                        "recommendation": "Specific improvement action to implement",
                        "example": "Concrete industry-specific implementation example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Second gap/weakness",
                        "recommendation": "Second improvement action",
                        "example": "Second concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Third gap/weakness",
                        "recommendation": "Third improvement action",
                        "example": "Third concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Fourth gap/weakness",
                        "recommendation": "Fourth improvement action",
                        "example": "Fourth concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }},
                    {{
                        "vulnerability": "Fifth gap/weakness",
                        "recommendation": "Fifth improvement action",
                        "example": "Fifth concrete example",
                        "reduction_in_total_loss": "Numeric with commas, no $",
                        "likelihood_reduction": "Whole number with %, no spaces"
                    }}
                ],
                "new_overall_likelihood_of_breach": "Must be within target range for this tier",
                "new_estimated_total_loss": "Numeric with commas, no $"
            }}
        }},
        {{
            "improved_rating": "SECOND_TIER_FROM_LIST",
            "recommendations_improved_rating": {{
                "recommendations": [
                    ... (5 MORE RECOMMENDATIONS) ...
                ],
                "new_overall_likelihood_of_breach": "LOWER than previous tier, within this tier's range",
                "new_estimated_total_loss": "LOWER than previous tier"
            }}
        }}
        ... CONTINUE FOR ALL {len(improvement_tiers)} TIERS ...
    ]
}}

---

### FINAL CHECKLIST BEFORE RESPONDING

- [ ] Included ALL {len(improvement_tiers)} rating tiers
- [ ] Each tier has exactly 5 recommendations
- [ ] Each recommendation ≤125 words
- [ ] All "vulnerability" fields describe current gaps
- [ ] All currency values have commas, no $ symbol
- [ ] All percentages are whole numbers with % symbol, no spaces
- [ ] Each tier's likelihood is LOWER than previous and within valid range
- [ ] No recommendations repeat previous recommendations
- [ ] Recommendations address both pentest findings and security weaknesses
- [ ] JSON is valid and properly escaped
- [ ] Used ONLY the exact field names specified

**CRITICAL:** You MUST include ALL {len(improvement_tiers)} tiers. Missing even one tier is unacceptable.

Return ONLY the JSON output, nothing else.
"""
    
    return system_prompt


# def get_pentest_system_prompt(vulnerabilities, securityResponses, company_data, loss_data):
#     system_prompt = (
#         f"Given a {company_data['company_size']}-sized {company_data['primary_industry']} company with an annual revenue of {company_data['annual_revenue']} USD. The company has a likelihood of breach probability of {loss_data['P_value']}% at the moment. "
#         f"In the event of a successful breach, the estimated result is that the company would experience a financial loss of {loss_data['Total']['total_loss']} USD before implementing any security improvements. "
#         "The company performed a pentest and discovered the following vulnerabilities along with their risk score: \n\n"
#         + "\n".join([f"{vuln['Vulnerability name']} - CVSS Score: {vuln['CVSS Score']} - Risk Level: {vuln['Risk Score/Level']}" for vuln in vulnerabilities]) +
#         "\n\nAlong with the pentest the company answered to some security questions, the response and a score for each response is provided: \n\n"
#         + "\n".join([f"{response['question']} - Response: {response['answer']} - Points Earned: {response['score']}" for response in securityResponses]) +
#         "\n\nThe maximum possible score for this questionnaire is 48 points. All questions contribute a fixed number of points towards the overall score of 48 points. "
#         "An overall questionnaire score closer to 48 means a lower likelihood of breach. A questionnaire score further from 48 suggests an increased likelihood of breach. "
#         "\n\n### Task:\n"
#         "Based on the vulnerabilities found in the pentest provide specific recommendations (in 125 words or less) for all the vulnerabilities the company can implement based on the company's industry. "
#         "Also, based on the security question responses provide specific recommendations (in 125 words or less) in 5 different areas the company can implement based on the company's industry. "
#         "Each recommendation must be very specific and directly address the corresponding problem or finding. "
#         "For each finding, also provide a specific example of what a company can do to improve the situation—this example must also be tailored to the exact problem.\n"
#         "For each recommendation provide a breach likelihood reduction (expressed as a percentage) and reduction in total loss if this recommendation is implemented independently. "
#         "Each recommendation's reduction value should be calculated as if it were the only change made. "
#         "Also, provide the new overall breach likelihood probability (as an overall percentage) and the revised total financial loss in USD if all recommendations are implemented together. "
#         "Use a non-linear risk decay model to calculate the final overall breach likelihood probability.\n"
#         "\n\nProvide the response in VALID JSON format only (no markdown, no explanations) with this exact structure:\n"
#         '''{
#             "pentest_recommendations": [
#                 {
#                     "vulnerability": "vulnerability_name",
#                     "recommendation": "specific_recommendation",
#                     "example": "concrete_example",
#                     "reduction_in_total_loss": "numeric_value_without_currency",
#                     "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
#                 }
#             ],
#             "new_overall_likelihood_of_breach": "numeric_value_with_percent_sign",
#             "new_estimated_total_loss": "numeric_value_without_currency",
#             "security_responses": [
#                 {
#                     "vulnerability": "vulnerability_name",
#                     "recommendation": "specific_recommendation",
#                     "example": "concrete_example",
#                     "reduction_in_total_loss": "numeric_value_without_currency",
#                     "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
#                 }
#             ]
#         }'''
#     )
#     return system_prompt


# def get_noPentest_system_prompt(securityResponses, company_data, loss_data):
#     system_prompt = (
#         f"Given a {company_data['company_size']}-sized {company_data['primary_industry']} company with an annual revenue of {company_data['annual_revenue']} USD. The company has a likelihood of breach probability of {loss_data['P_value']}% at the moment. "
#         f"In the event of a successful breach, the estimated result is that the company would experience a financial loss of {loss_data['Total']['total_loss']} USD before implementing any security improvements. "
#         "The company answered to some security questions, the response and a score for each response is provided: \n\n"
#         + "\n".join([f"{response['question']} - Response: {response['answer']} - Points Earned: {response['score']}" for response in securityResponses]) +
#         "\n\nThe maximum possible score for this questionnaire is 48 points. "
#         "\n\n### Task:\n"
#         "Based on the responses provide specific recommendations (in 125 words or less) in 5 different areas the company can implement based on the company's industry. "
#         "Each recommendation must be very specific and directly address the corresponding problem or finding. "
#         "For each finding, also provide a specific example of what a company can do to improve the situation—this example must also be tailored to the exact problem.\n"
#         "For each recommendation provide a breach likelihood reduction (expressed as a percentage) and reduction in total loss if this recommendation is implemented independently. "
#         "Also, provide the new overall breach likelihood probability (as an overall percentage) and the revised total financial loss in USD if all recommendations are implemented together. "
#         "Use a non-linear risk decay model to calculate the final overall breach likelihood probability.\n"
#         "\n\nProvide the response in VALID JSON format only (no markdown, no explanations) with this exact structure:\n"
#         '''{
#             "security_responses": [
#                 {
#                     "vulnerability": "vulnerability_name",
#                     "recommendation": "specific_recommendation",
#                     "example": "concrete_example",
#                     "reduction_in_total_loss": "numeric_value_without_currency",
#                     "likelihood_reduction_in_percentage": "numeric_value_with_percent_sign"
#                 }
#             ],
#             "new_overall_likelihood_of_breach": "numeric_value_with_percent_sign",
#             "new_estimated_total_loss": "numeric_value_without_currency"
#         }'''
#     )
#     return system_prompt


# def get_noPentest_improvement_system_prompt(
#     securityResponses, 
#     company_data, 
#     loss_data, 
#     recommendation_data,
#     current_rating,
#     improvement_tiers,
#     previous_recommendations
# ):
#     # Format security responses
#     sec_resp_text = "\n".join(
#         f"{r['question']} - Response: {r['answer']} - Points Earned: {r['score']}"
#         for r in securityResponses
#     )

#     # Format previous recommendations for exclusion
#     prev_recs_text = "NONE"
#     if previous_recommendations:
#         prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

#     # Create explicit tier list
#     tier_list = "\n".join([f"- {tier}" for tier in improvement_tiers])

#     system_prompt = f"""
# ### CRITICAL INSTRUCTIONS - READ CAREFULLY

# You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH of the following rating tiers. MISSING ANY TIER IS A CRITICAL FAILURE.

# **REQUIRED OUTPUT TIERS (YOU MUST INCLUDE ALL OF THESE):**
# {tier_list}

# **CURRENT RATING:** {current_rating}

# ### Company Information
# - Size: {company_data.get('company_size')}
# - Industry: {company_data.get('primary_industry')}
# - Annual Revenue: {company_data.get('annual_revenue')} USD
# - Current Breach Likelihood: {loss_data.get('P_value')}%
# - Estimated Total Loss (pre-improvement): {loss_data.get('Total', {}).get('total_loss')} USD

# ### Security Questionnaire Responses
# {sec_resp_text}

# ### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT ANY OF THESE)
# {prev_recs_text}

# ### Rating-to-Probability Mapping
# | Rating | Probability Range (%) |
# |--------|----------------------|
# | A+     | 2.0 - 11.10         |
# | A      | 11.11 - 22.21       |
# | A-     | 22.22 - 33.32       |
# | B+     | 33.33 - 44.43       |
# | B      | 44.44 - 55.54       |
# | B-     | 55.55 - 66.65       |
# | U      | 66.66 - 98.0        |

# ### Requirements for EACH Rating Tier

# For EACH tier listed above, you MUST provide:
# 1. **Exactly 5 unique recommendations** (each ≤125 words total)
# 2. Each recommendation must include:
#    - "recommendation": specific action to take
#    - "reduction_in_total_loss": numeric value (no $ symbol)
#    - "likelihood_reduction": percentage value (with % symbol)
# 3. Calculate:
#    - "new_overall_likelihood_of_breach": must fall within the target rating's probability range
#    - "new_estimated_total_loss": numeric value (no $ symbol)

# ### STRICT JSON FORMAT (NO MARKDOWN, NO EXPLANATIONS)

# YOU MUST RETURN EXACTLY THIS STRUCTURE WITH ALL {len(improvement_tiers)} TIERS:

# {{
#     "grade_improvement_recommendations": [
#         {{
#             "improved_rating": "{improvement_tiers[0] if improvement_tiers else 'A+'}",
#             "recommendations_improved_rating": {{
#                 "recommendations": [
#                     {{  
#                         "vulnerability": "vulnerability_name", 
#                         "recommendation": "Deploy EDR on all endpoints...",
#                         "reduction_in_total_loss": "50000",
#                         "likelihood_reduction": "8%"
#                     }},
#                     {{
#                         "vulnerability": "vulnerability_name", 
#                         "recommendation": "Implement MFA for all remote access...",
#                         "reduction_in_total_loss": "40000",
#                         "likelihood_reduction": "6%"
#                     }},
#                     {{
#                         "vulnerability": "vulnerability_name", 
#                         "recommendation": "Third recommendation...",
#                         "reduction_in_total_loss": "30000",
#                         "likelihood_reduction": "5%"
#                     }},
#                     {{
#                         "vulnerability": "vulnerability_name", 
#                         "recommendation": "Fourth recommendation...",
#                         "reduction_in_total_loss": "25000",
#                         "likelihood_reduction": "4%"
#                     }},
#                     {{
#                         "vulnerability": "vulnerability_name", 
#                         "recommendation": "Fifth recommendation...",
#                         "reduction_in_total_loss": "20000",
#                         "likelihood_reduction": "3%"
#                     }}
#                 ],
#                 "new_overall_likelihood_of_breach": "15.5%",
#                 "new_estimated_total_loss": "800000"
#             }}
#         }}{"," if len(improvement_tiers) > 1 else ""}
#         {f'''{{
#             "improved_rating": "{improvement_tiers[1] if len(improvement_tiers) > 1 else 'A+'}",
#             "recommendations_improved_rating": {{
#                 "recommendations": [
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}}
#                 ],
#                 "new_overall_likelihood_of_breach": "...",
#                 "new_estimated_total_loss": "..."
#             }}
#         }}''' if len(improvement_tiers) > 1 else ""}
#     ]
# }}

# **CRITICAL**: Include a block for EVERY tier in the list above. Missing even one tier is unacceptable.
# """
    
#     return system_prompt


# def get_Pentest_improvement_system_prompt(
#     vulnerabilities,
#     securityResponses,
#     company_data,
#     loss_data,
#     recommendation_data,
#     current_rating,
#     improvement_tiers,
#     previous_recommendations
# ):
#     # Format vulnerabilities
#     vuln_text = "\n".join(
#         f"{vuln['Vulnerability name']} - CVSS: {vuln['CVSS Score']} - Risk: {vuln['Risk Score/Level']}"
#         for vuln in vulnerabilities
#     )

#     # Format security responses
#     sec_resp_text = "\n".join(
#         f"{r['question']} - Response: {r['answer']} - Points: {r['score']}"
#         for r in securityResponses
#     )

#     # Format previous recommendations
#     prev_recs_text = "NONE"
#     if previous_recommendations:
#         prev_recs_text = "\n".join([f"- {rec}" for rec in previous_recommendations])

#     # Create tier list
#     tier_list = "\n".join([f"- {tier}" for tier in improvement_tiers])

#     system_prompt = f"""
# ### CRITICAL INSTRUCTIONS - READ CAREFULLY

# You are a cybersecurity improvement recommendation engine. Your task is to generate EXACTLY ONE recommendation block for EACH of the following rating tiers. MISSING ANY TIER IS A CRITICAL FAILURE.

# **REQUIRED OUTPUT TIERS (YOU MUST INCLUDE ALL OF THESE):**
# {tier_list}

# **CURRENT RATING:** {current_rating}

# ### Company Information
# - Size: {company_data.get('company_size')}
# - Industry: {company_data.get('primary_industry')}
# - Annual Revenue: {company_data.get('annual_revenue')} USD
# - Current Breach Likelihood: {loss_data.get('P_value')}%
# - Estimated Total Loss (pre-improvement): {loss_data.get('Total', {}).get('total_loss')} USD

# ### Penetration Test Vulnerabilities
# {vuln_text}

# ### Security Questionnaire Responses
# {sec_resp_text}

# ### PREVIOUS RECOMMENDATIONS (DO NOT REPEAT ANY OF THESE)
# {prev_recs_text}

# ### Rating-to-Probability Mapping
# | Rating | Probability Range (%) |
# |--------|----------------------|
# | A+     | 2.0 - 11.10         |
# | A      | 11.11 - 22.21       |
# | A-     | 22.22 - 33.32       |
# | B+     | 33.33 - 44.43       |
# | B      | 44.44 - 55.54       |
# | B-     | 55.55 - 66.65       |
# | U      | 66.66 - 98.0        |

# ### Requirements for EACH Rating Tier

# For EACH tier listed above, you MUST provide:
# 1. **Exactly 5 unique recommendations** (each ≤125 words total)
# 2. Each recommendation must:
#    - Address vulnerabilities or questionnaire weaknesses
#    - Be specific to the company's industry
#    - Include concrete implementation examples
#    - NOT repeat any previous recommendations
# 3. Each recommendation includes:
#    - "recommendation": specific action
#    - "reduction_in_total_loss": numeric (no $)
#    - "likelihood_reduction": percentage (with %)
# 4. Calculate:
#    - "new_overall_likelihood_of_breach": MUST be within target rating range
#    - "new_estimated_total_loss": numeric (no $)

# ### STRICT JSON FORMAT (NO MARKDOWN, NO EXPLANATIONS)

# Return EXACTLY this structure with ALL {len(improvement_tiers)} tiers:

# {{
#     "grade_improvement_recommendations": [
#         {{
#             "improved_rating": "{improvement_tiers[0] if improvement_tiers else 'A+'}",
#             "recommendations_improved_rating": {{
#                 "recommendations": [
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}},
#                     {{"vulnerability": "...", "recommendation": "...", "reduction_in_total_loss": "...", "likelihood_reduction": "..."}}
#                 ],
#                 "new_overall_likelihood_of_breach": "...",
#                 "new_estimated_total_loss": "..."
#             }}
#         }}{", (continue for all remaining tiers)" if len(improvement_tiers) > 1 else ""}
#     ]
# }}

# **CRITICAL**: You must include ALL {len(improvement_tiers)} rating tiers. Each tier needs exactly 5 recommendations.
# """
    
#     return system_prompt
