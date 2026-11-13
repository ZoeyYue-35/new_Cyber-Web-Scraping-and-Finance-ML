import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import scipy.stats as stats
# import pymc3 as pm
import warnings
import logging
import json
import os
from utils.helper_func import *
from utils.V_calc.alt_v import v_calc
from utils.V_calc.parse_security import questionnaire_scoring_function, parse_business_data
from utils.V_calc.get_llm_recommendations import get_improvement_recommendations, extract_numeric_value
# Suppress all warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
from utils.p_value import get_p_value
from utils.bricking_model import BrickingModel
from utils.system_failure_model import SystemFailureModel
from utils.ransomware_model import RansomwareModel
from interpolate_losses import FastLossLookup
import heapq

"""Add weight based off the NAICS code of the company, and historical information of that industry"""

def run_monte_carlo_simulations(
    lambda_value, mu_samples, sigma_samples, company_data, subcategory_proportions,
    num_sims=200000, num_years=1):

    primary_industry = company_data['primary_industry']
    primary_industry, primary_industry_code, country_label = get_industry_name(primary_industry, company_data['location'])
    converted_code = convert_code_to_naics(primary_industry_code, country_label)
    total_non_fine_losses = [] 
    events_per_sim = []
    subcategory_losses_list = {category: [] for category in subcategory_proportions.keys()}

    # Ensure that the number of simulations matches the number of mu and sigma samples
    num_samples = len(mu_samples)
    if num_sims > num_samples:
        raise ValueError("num_sims cannot be greater than the number of posterior samples.")
    
    for sim in range(num_sims):
        mu = mu_samples[sim]
        sigma = sigma_samples[sim]
        
        # Simulate number of events per year
        events_per_year = np.random.poisson(lam=lambda_value, size=num_years)
        events_per_sim.append(int(np.sum(events_per_year)))
        total_loss_simulation = 0
        simulation_fines = 0
        total_non_fine_losses_val = 0
        
        for num_events in events_per_year:
            if num_events > 0:
                # Simulate loss amounts
                loss_amounts = np.random.lognormal(mean=mu, sigma=sigma, size=num_events)
                yearly_loss_no_fines = np.sum(loss_amounts)

                
            else:
                 yearly_loss_no_fines = 0

            val = yearly_loss_no_fines/num_events if num_events > 0 else 0
            total_non_fine_losses_val += val
        

        industry_scaled_loss = get_industry_map(total_non_fine_losses_val, converted_code)

        total_non_fine_losses.append(industry_scaled_loss)
        
        # Allocate losses to subcategories
        

        for category, proportion in subcategory_proportions.items():
            sub_loss = industry_scaled_loss * proportion
            subcategory_losses_list[category].append(sub_loss)
        
    for category in subcategory_losses_list:
        subcategory_losses_list[category] = np.array(subcategory_losses_list[category])
    
    subcategory_losses_list['Privacy Liability Coverage'] = 0
    return total_non_fine_losses, subcategory_losses_list, np.array(events_per_sim)
    

  
def model(data, plot=False, use_ai=True, num_simulations = 200_000):
    # Load data from JSON
    # data = load_questionnaire_data('questionnaire_answers.json')

    business_data = data['businessQuestions']
    security_data = data['securityQuestions']
    assessment_data = data["assessmentData"]

    display_currency = detect_currency_from_country(business_data.get('country'))
    usd_to_target_rate = get_usd_to_target_rate(display_currency)
    business_data = normalize_business_currency_fields(business_data, usd_to_target_rate)

    is_valid_assessment = assessment_data.get('is_valid_assessment')

    if is_valid_assessment is True:
        assessment_data['assessmentType'] = "PenTest"
    else:
        assessment_data['assessmentType'] = None

    # effective assessment_type for functions that expect a simple value
    assessment_type = assessment_data.get('assessmentType')

    data_type_multipliers = {
        'Personally Identifiable Information (PII)': {'Notification': 1.0, 'Forensics': 1.0, 'Crisis Management': 1.0},
        'Payment Card Information (PCI)': {'Notification': 1.0, 'Forensics': 1.5, 'Crisis Management': 1.0},
        'Protected Health Information (PHI)': {'Notification': 1.0, 'Forensics': 1.7, 'Crisis Management': 1.0},
    }
    

    effective_regulations = get_effective_regulations(business_data)

    # Parse and extract company data
    company_data = {
        # Contact Information
        'name': business_data['name'],
        'phone': business_data['phone'],
        'email': business_data['email'],

        #General Information
        'company_name': business_data['companyName'],
        'location': business_data['country'],
        'primary_industry': business_data['primaryIndustry'],
        'num_employees': parse_number(business_data['employeeCount']),
        'annual_revenue': coerce_amount(business_data['annualRevenue']),
        'prev_year_revenue1': coerce_amount(business_data.get('revenue2022', '0')),
        'prev_year_revenue2': coerce_amount(business_data.get('revenue2023', '0')),
        'regulations': effective_regulations,
        'regulation_records': parse_regulations(business_data),
        'dataExposed': business_data.get('dataExposed', []),

        #Data Handling
        'numIncidents': 0,
        'personalRecords': business_data['personalRecords'],
        'recent_breach': business_data['recentBreach'],
        'insurance_coverage': business_data['insuranceCoverage'],
        'recent_breach': business_data['recentBreach'],
    }
    confidence_rating = {
        "Incident Response Document": 0,
        "Security Questionnaire": 1,
        "PenTest": 0,
    }

    security_scores, automatic_failure, confidence_rating = questionnaire_scoring_function(
        security_data,
        assessment_type,
        confidence_rating,
        num_employees=company_data['num_employees'],
        business_data=business_data,
    )
    
    # Determine company size
    num_employees = company_data['num_employees']
    company_size = determine_company_size(
        num_employees=num_employees,
        current_revenue=company_data['annual_revenue'],
        prev_year_revenue1=company_data['prev_year_revenue1'],
        prev_year_revenue2=company_data['prev_year_revenue2']
    )
    privacy_company_size = company_size

    business_scores = parse_business_data(business_data, company_size)

    #get coverages
    coverages = business_data['insuranceCoverage']
    coverage_values = {}
    if coverages:
        for i in coverages:
            coverage_values[i] = business_data[i]

    # print(f"coverages: {coverages} \n coverage_values: {coverage_values}")

    # Handle companies larger than 'medium'
    if company_size not in ['micro', 'small', 'medium']:
        company_size = 'medium' 
    company_data['company_size'] = company_size

    business_scores['type'] = 'business'
    security_scores['type'] = 'security'
    
    # Get industry code for p-value scaling
    primary_industry, primary_industry_code, country_label = get_industry_name(company_data['primary_industry'], company_data['location'])
    company_data['primary_industry_name'] = primary_industry
    converted_industry_code = convert_code_to_naics(primary_industry_code, country_label)
    industry_benchmark= get_industry_benchmark(converted_industry_code)
    # Calculate p_value using the imported function, unless an override is provided
    p_value_override = None
    
    try:
        p_value_override = data.get('assessmentData', {}).get('p_value_override')
    except Exception:
        p_value_override = None
    if p_value_override is not None:
        try:    
            p_value = float(p_value_override)
            _, S_bar = get_p_value(
                company_size=company_data['company_size'], 
                security_data=security_scores, 
                assessment_data=assessment_data, 
                business_data=business_scores,
                industry_code=converted_industry_code,
                industry_benchmark= industry_benchmark
            )
            # Clamp to [0, 1)
            p_value = max(0.02, min(0.98, p_value))

        except Exception:
            p_value, S_bar = get_p_value(
                company_size=company_data['company_size'], 
                security_data=security_scores, 
                assessment_data=assessment_data, 
                business_data=business_scores,
                industry_code=converted_industry_code,
                industry_benchmark= industry_benchmark
            )
    else:
        p_value, S_bar = get_p_value(
            company_size=company_data['company_size'], 
            security_data=security_scores, 
            assessment_data=assessment_data, 
            business_data=business_scores,
            industry_code=converted_industry_code,
            industry_benchmark= industry_benchmark
        )


    #parse number of record types
    total_records, notification_costs, data_driven_forensics_costs, pci_records, pii_records, phi_records = count_num_records(company_data, business_data, data_type_multipliers, p_value)

    #subquestions information from each of the regulation types, for use in calculating regulatory fines
    regulation_subquestions = company_data['regulation_records']
    primary_industry, primary_industry_code, country_label = get_industry_name(
            company_data['primary_industry'], 
            company_data['location']
        )
    recent_breach = company_data['recent_breach']
    privacy_liability_coverage_loss, regulatory_fines, pci_detail = calculate_privacy_liability_coverage_loss(
        regulations=company_data['regulations'],
        pci_records= pci_records,
        pii_records= pii_records,
        phi_records= phi_records,
        annual_revenue=company_data['annual_revenue'],
        regulation_subquestions = regulation_subquestions,
        s_bar = S_bar,
        naics_code = convert_code_to_naics(primary_industry_code, country_label),
        security_data = security_data,
        recent_breach = recent_breach,
        num_employees = num_employees,
        p_value = p_value,
        assessment_data = assessment_data,
        business_data = business_data,
        company_size = privacy_company_size,
    )

   #number of records breached impacts the notification costs/ incident response costs 

    # Calculate λ
    lambda_value = calculate_lambda(p_value)
    # Run simulations
    num_years = 1

    # Calculate μ and σ
    mu_samples, sigma_samples = calculate_lognormal_params(num_samples=num_simulations, user_size = company_data['company_size'], annual_revenue = company_data['annual_revenue'])


    # Prepare historical data
    (affected_counts_df, response_costs_df, litigated_cases_df,
     fines_penalties_df, economic_loss_df, company_size_df) = load_historical_data()

    subcategory_proportions = calculate_subcategory_proportions(
        response_costs_df, litigated_cases_df, fines_penalties_df, economic_loss_df
    )
    total_losses, subcategory_losses_list, events_per_sim = run_monte_carlo_simulations(
        lambda_value, mu_samples, sigma_samples, company_data,
        subcategory_proportions, num_sims=num_simulations, num_years=num_years
    )    

    bricking_model = BrickingModel(
        num_employees=num_employees,
        naics_code=converted_industry_code,
        security_data=security_data,
    )

    system_failure_model = SystemFailureModel(
        num_employees=num_employees,
        naics_code=converted_industry_code,
        security_data=security_data,
        annual_revenue=company_data['annual_revenue'],
        prev_year_revenue1=company_data['prev_year_revenue1'],
        prev_year_revenue2=company_data['prev_year_revenue2'],
        company_size=company_data['company_size'],
        p_value=p_value,
        recent_breach=recent_breach,
        # privacy_cost_per_incident intentionally omitted — see anti-double-counting note
    )

    ransomware_model = RansomwareModel(
        num_employees=num_employees,
        naics_code=converted_industry_code,
        security_data=security_data,
        annual_revenue=company_data['annual_revenue'],
        prev_year_revenue1=company_data['prev_year_revenue1'],
        prev_year_revenue2=company_data['prev_year_revenue2'],
        company_size=company_data['company_size'],
        p_value=p_value,
        recent_breach=recent_breach,
        # privacy_cost_per_incident omitted — anti-double-counting
    )

    # subcategory_losses_list['Privacy Liability Coverage'] = np.array(privacy_liability_coverage_loss)
    pci_dss_loss = float(regulatory_fines.get('PCI DSS', 0))
    privacy_fines = {k: v for k, v in regulatory_fines.items() if k != 'PCI DSS'}
    subcategory_losses_list['Privacy Liability Coverage'] = np.array(list(privacy_fines.values()) or [0])
    
    # Analyze results
    lower_bound_total_loss = np.percentile(total_losses, 2.5)
    upper_bound_total_loss = np.percentile(total_losses, 97.5)

    # print(f"\n--- Risk Assessment for {company_data['company_name']} ---")
    # print(f"Expected Total Loss: ${expected_total_loss:,.2f}")
    # print(f"95% Credible Interval: (${lower_bound_total_loss:,.2f}, ${upper_bound_total_loss:,.2f})")

    # print("\nExpected Subcategory Losses:")
    # for category in subcategory_proportions.keys():
    #     expected_sub_loss = np.mean(subcategory_losses_list[category])
    #     print(f"{category}: ${expected_sub_loss:,.2f}")

    

    results = {
        'company_name': company_data['company_name'],
        'expected_total_loss': 0,
        'expected_subcategory_losses': {},
    }

    expected_total_loss = 0
    response_costs = 0

    technical_response_proportion = 0.8
    crisis_proportion = 0.2

    for category in subcategory_losses_list.keys():
        
        subcategory_losses = subcategory_losses_list[category]
        if category == 'Privacy Liability Coverage':
            expected_sub_loss = np.sum(subcategory_losses)
        else:
            expected_sub_loss = np.mean(subcategory_losses)
        expected_sub_loss = expected_sub_loss.item()
        expected_total_loss += round(expected_sub_loss, 2)

        if category == 'Incident Response Costs':

            response_costs += expected_sub_loss
            
            if 'Incident Response' not in results['expected_subcategory_losses'].keys():
                results['expected_subcategory_losses']['Incident Response Costs'] = {}
            
            results['expected_subcategory_losses']['Incident Response Costs']['Total'] = round(expected_sub_loss, 2)
            
        elif category == 'Litigation Costs':
            
            response_costs += expected_sub_loss

            results['expected_subcategory_losses']['Incident Response Costs']['Legal'] = round(expected_sub_loss, 2)
            

        else:
            # Handle other categories
            results['expected_subcategory_losses'][category] = round(expected_sub_loss, 2)

    response_costs += notification_costs
    results['expected_subcategory_losses']['Incident Response Costs']['Total'] = round(response_costs, 2)

    expected_bricking = bricking_model.expected_annual_loss()
    

   # Reallocate incident response losses using breach exposure and control weakness.
    incident_response_breakdown = calculate_incident_response_costs_subcategory_losses(
        company_data=company_data,
        business_data=business_data,
        security_data=security_data,
        total_ir_loss=response_costs,
    )

    results['expected_subcategory_losses']['Incident Response Costs']['Notification'] = incident_response_breakdown['Notification']
    results['expected_subcategory_losses']['Incident Response Costs']['Forensics'] = incident_response_breakdown['Forensics']
    results['expected_subcategory_losses']['Incident Response Costs']['Legal'] = incident_response_breakdown['Legal']
    results['expected_subcategory_losses']['Incident Response Costs']['Technical Response'] = incident_response_breakdown['Technical Response']
    results['expected_subcategory_losses']['Incident Response Costs']['Crisis Management'] = incident_response_breakdown['Crisis Management']
    
    # Add Bricking as a completely independent category
    results['expected_subcategory_losses']['Bricking'] = round(expected_bricking, 2)
    expected_total_loss += expected_bricking

    expected_system_failure = system_failure_model.expected_annual_loss()
    results['expected_subcategory_losses']['Malicious System Failure'] = round(expected_system_failure, 2)
    expected_total_loss += expected_system_failure

    expected_ransomware = ransomware_model.expected_annual_loss()
    results['expected_subcategory_losses']['Ransomware'] = round(expected_ransomware, 2)
    expected_total_loss += expected_ransomware
    
    # Add PCI DSS as its own category (moved out of Privacy Liability Coverage)
    results['expected_subcategory_losses']['PCI DSS'] = round(pci_dss_loss, 2)
    expected_total_loss += pci_dss_loss


    #analyze results
    #expected_total_loss += np.array(list(regulatory_fines.values()))[0]

    expected_total_loss += notification_costs
    results['expected_total_loss'] = round(expected_total_loss, 2)

    coverage_gaps = get_coverage_gap(results['expected_subcategory_losses'], business_data)

    coverage_gaps['Rating'], coverage_gaps['P_value'] = get_insurability_rating(
        p_value,
        automatic_failure=automatic_failure,
    )

    coverage_gaps['Incident Response Costs Subcat'] = {}
    coverage_gaps['Incident Response Costs Subcat']['Legal'] = results['expected_subcategory_losses']['Incident Response Costs']['Legal']
    coverage_gaps['Incident Response Costs Subcat']['Forensics'] = results['expected_subcategory_losses']['Incident Response Costs']['Forensics']
    coverage_gaps['Incident Response Costs Subcat']['Technical Response'] = results['expected_subcategory_losses']['Incident Response Costs']['Technical Response']
    coverage_gaps['Incident Response Costs Subcat']['Crisis Management'] = results['expected_subcategory_losses']['Incident Response Costs']['Crisis Management']
    coverage_gaps['Incident Response Costs Subcat']['Notification'] = results['expected_subcategory_losses']['Incident Response Costs']['Notification']

    coverage_gaps = calculate_business_interruption_subcategory_losses(coverage_gaps, results)

    # Incident response subcategories are now allocated from scenario features instead of a fixed residual split.

    final_json_key_names = {
        'Bricking': 'Ruined Hardware',
        'Malicious System Failure': 'Intentional Crash',
        'Ransomware': 'Digital Extortion',
    }
    serializable_results = {}
    for key, value in coverage_gaps.items():
        output_key = final_json_key_names.get(key, key)
        if isinstance(value, dict):
            serializable_results[output_key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[output_key] = convert_to_serializable(value)
    
    confidence_level = ""
    if sum(confidence_rating.values()) == 3:
        confidence_level = "High"
    elif sum(confidence_rating.values()) == 2:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    serializable_results['Annual Revenue'] = company_data['annual_revenue']
    serializable_results['company_name'] = company_data['company_name']
    primary_industry, primary_industry_code, country_label = get_industry_name(company_data['primary_industry'], company_data['location'])
    converted_code = convert_code_to_naics(primary_industry_code, country_label)
    print(primary_industry_code)
    print(converted_code)
    
    serializable_results['industry_benchmark'] = get_industry_benchmark(converted_code)
    print(confidence_level)
    serializable_results['confidence_rating'] = confidence_level
    serializable_results['Total']['total_loss'] = results['expected_total_loss'] 

    serializable_results['Privacy Liability Coverage Subcat'] = privacy_fines
    # serializable_results['Bricking Diagnostics'] = bricking_model.diagnostics(
    #     expected_ir_loss=response_costs,
    #     lambda_value=lambda_value,
    # )


    # UPDATED MAIN FUNCTION SNIPPET
    if use_ai:
        # Resolve the industry code used by the loss lookup.
        primary_industry, primary_industry_code, country_label = get_industry_name(
            company_data['primary_industry'], 
            company_data['location']
        )
        converted_code = int(convert_code_to_naics(primary_industry_code, country_label))
        
        # Generate improvement recommendations through A+, independently of the
        # industry benchmark used by the risk model and assessment output.
        improvement_recommendations = get_improvement_recommendations(
            data, 
            assessment_type, 
            company_data, 
            serializable_results
        )
        
        serializable_results["improvement_recommendations"] = json.loads(improvement_recommendations)
        
        # Calculate reduction_in_total_loss using lookup table
        base_path = "data/upsell_table"
        try:
            lookup = FastLossLookup(base_path, [converted_code])
            
            if 'grade_improvement_recommendations' in serializable_results['improvement_recommendations']:
                current_p_value = extract_numeric_value(
                    serializable_results.get('P_value', 50.0)
                )
                current_loss = serializable_results['Total']['total_loss']
                current_lookup_loss = lookup.get_loss_for_p_value(
                    converted_code,
                    company_data['annual_revenue'],
                    pci_records,
                    pii_records,
                    phi_records,
                    current_p_value / 100.0
                )
                
                for grade_rec in serializable_results['improvement_recommendations']['grade_improvement_recommendations']:
                    # print(f"current p value {current_p_value}")
                    if 'recommendations_improved_rating' in grade_rec:
                        rating_data = grade_rec['recommendations_improved_rating']
                        
                        # Get new likelihood for this tier
                        new_likelihood_str = rating_data.get('new_overall_likelihood_of_breach', '0%')
                        new_likelihood_val = float(new_likelihood_str.strip('%'))
                        # An automatic questionnaire failure can force a lower rating
                        # even when the numeric likelihood is already below that
                        # rating's range. Remediation must never increase likelihood.
                        new_likelihood_val = min(new_likelihood_val, current_p_value)
                        rating_data['new_overall_likelihood_of_breach'] = (
                            f"{new_likelihood_val:g}%"
                        )
                        new_p_value = new_likelihood_val / 100.0
                        # print(f"current p value {current_p_value}, new p value {new_p_value}")

                        # Use the lookup table to calculate the relative change, then
                        # apply that change to the current modeled loss. Directly
                        # subtracting a lookup loss from the Monte Carlo loss mixes
                        # incompatible baselines and can produce negative reductions.
                        tier_lookup_loss = lookup.get_loss_for_p_value(
                            converted_code,
                            company_data['annual_revenue'],
                            pci_records,
                            pii_records,
                            phi_records,
                            new_p_value
                        )
                        if current_lookup_loss > 0:
                            tier_new_loss = current_loss * min(
                                1.0,
                                tier_lookup_loss / current_lookup_loss
                            )
                        else:
                            tier_new_loss = current_loss
                        rating_data['new_estimated_total_loss'] = round(tier_new_loss, 2)
                        improvement = max(0.0, current_loss - tier_new_loss)
                        
                        # Calculate reduction for each recommendation
                        # print(f"improvement = {improvement}, current_loss = {current_loss}, new tier loss {tier_new_loss}")
                        remaining_improvement = 0
                        heap = list()
                        if 'recommendations' in rating_data:
                            for i, rec in enumerate(rating_data['recommendations']):
                                # Get likelihood_reduction for this recommendation
                                likelihood_reduction_factor = extract_numeric_value(rec.get('impact_contribution_factor', '0'))
                                
                                # Calculate p_value after applying this recommendation
                                # Get loss at this reduced p_value
                                # print(f"Likelihood reduction factor {likelihood_reduction_factor}")
                                reduction_in_loss = improvement * likelihood_reduction_factor
                                rounded_off_loss = reduction_in_loss
                                if reduction_in_loss > 1_000_000:
                                    rounded_off_loss = (reduction_in_loss // 1_000_000) * 1_000_000
                                    remaining_improvement+= reduction_in_loss - rounded_off_loss
                                    heapq.heappush(heap, ((rounded_off_loss - reduction_in_loss), i))

                                rec['reduction_in_total_loss'] = rounded_off_loss
                            while remaining_improvement > 1_000_000 and len(heap) > 0:
                                _, index = heapq.heappop(heap)
                                rating_data['recommendations'][index]['reduction_in_total_loss']+=1_000_000
                                remaining_improvement-=1_000_000
                            if remaining_improvement > 0:
                                rating_data['new_estimated_total_loss'] =  round(
                                    rating_data['new_estimated_total_loss'] + remaining_improvement, 2
                                )
                            for rec in  rating_data['recommendations']:
                                rec['reduction_in_total_loss'] = f"{rec['reduction_in_total_loss']:,.2f}"
                        
                        # Update current_p_value for next tier (cumulative)
                        current_p_value = new_likelihood_val
                        current_lookup_loss = tier_lookup_loss
                        current_loss = rating_data['new_estimated_total_loss']
                        
        except Exception as e:
            print(f"Error calculating losses: {str(e)}")

    #print(serializable_results)
    
    if plot:
        plot_loss_exceedance_curve(total_losses)

    serializable_results['base_currency'] = 'USD'
    serializable_results['currency'] = display_currency
    serializable_results['exchange_rate_to_currency'] = usd_to_target_rate

    if display_currency != 'USD':
        serializable_results = convert_outputs_to_target_currency(serializable_results, usd_to_target_rate)
        serializable_results['base_currency'] = 'USD'
        serializable_results['currency'] = display_currency
        serializable_results['exchange_rate_to_currency'] = usd_to_target_rate
    
    
    return serializable_results


if __name__ == '__main__':

    with open('./input_Pentest.json') as f:
        data = json.load(f)
    
    output = model(data, use_ai = True)
    with open('./output.json', 'w') as f:
        json.dump(output, f, indent=4)
