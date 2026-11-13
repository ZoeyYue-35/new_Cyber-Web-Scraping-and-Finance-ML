import pandas as pd
import numpy as np
import json
from model.financial_services_finloss import model
from utils.helper_func import convert_to_serializable

"""
Model Ensemble for Financial Services FinLoss
Runs multiple simulations with different random seeds and aggregates results
to provide more robust predictions with confidence intervals.
"""

def run_ensemble_model(data, num_models=5, confidence_level=95, plot=False, random_seed=42):
    """
    Run multiple model simulations with different random seeds and aggregate results.
    
    Args:
        data (dict): Input data dictionary containing businessQuestions, securityQuestions, etc.
        num_models (int): Number of models to run in the ensemble (default: 5)
        confidence_level (float): Confidence level for intervals (default: 95)
        plot (bool): Whether to plot results (default: False)
        random_seed (int): Base random seed for reproducibility (default: 42)
    
    Returns:
        dict: Aggregated results with means and confidence intervals
    """
    print(f"Running ensemble with {num_models} models...")
    
    # Store results from each model run
    all_results = []
    
    # Generate fixed seeds based on the base random seed for reproducibility
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 10000, size=num_models)  # Generate fixed seeds
    print(f"Using fixed random seeds: {seeds[:5]}{'...' if num_models > 5 else ''}")
    
    # Run the model multiple times with different seeds
    for i, seed in enumerate(seeds):
        print(f"Running model {i+1}/{num_models} with seed {seed}...")
        
        # Set the seed in numpy before running the model
        np.random.seed(seed)
        
        # Run the model
        result = model(data, plot=False)
        all_results.append(result)
    
    print("Aggregating results...")
    
    # Aggregate results
    ensemble_results = aggregate_results(all_results, confidence_level)
    
    # Add metadata
    ensemble_results['ensemble_metadata'] = {
        'num_models': num_models,
        'confidence_level': confidence_level,
        'seeds_used': seeds.tolist(),
        'company_name': all_results[0]['company_name'],
        'industry_benchmark': all_results[0]['industry_benchmark'],
        'confidence_rating': all_results[0]['confidence_rating'],
        'Annual Revenue': all_results[0]['Annual Revenue']
    }
    
    # Use the recommendations from the first model (they should be similar across runs)
    ensemble_results['recommendations'] = all_results[0]['recommendations']
    ensemble_results['improvement_recommendations'] = all_results[0]['improvement_recommendations']
    
    if plot:
        plot_ensemble_results(ensemble_results)
    
    return ensemble_results


def aggregate_results(all_results, confidence_level=95):
    """
    Aggregate results from multiple model runs.
    
    Args:
        all_results (list): List of result dictionaries from each model run
        confidence_level (float): Confidence level for intervals
    
    Returns:
        dict: Aggregated results with means and confidence intervals
    """
    num_models = len(all_results)
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    # Initialize aggregated results structure
    aggregated = {
        'Total': {},
        'P_value': {},
        'Rating': {},
        'Incident Response Costs': {},
        'Business Interruption Loss': {},
        'Privacy Liability Coverage': {},
        'Incident Response Costs Subcat': {},
        'Business Interruption Loss SubCats': {},
        'Privacy Liability Coverage Subcat': {}
    }
    
    # Aggregate Total losses
    total_losses = [r['Total']['total_loss'] for r in all_results]
    total_losses_with_coverage = [r['Total']['total_loss_with_coverage'] for r in all_results]
    
    aggregated['Total'] = {
        'total_loss': {
            'mean': np.mean(total_losses),
            'std': np.std(total_losses),
            'lower_ci': np.percentile(total_losses, lower_percentile),
            'upper_ci': np.percentile(total_losses, upper_percentile),
            'min': np.min(total_losses),
            'max': np.max(total_losses)
        },
        'total_loss_with_coverage': {
            'mean': np.mean(total_losses_with_coverage),
            'std': np.std(total_losses_with_coverage),
            'lower_ci': np.percentile(total_losses_with_coverage, lower_percentile),
            'upper_ci': np.percentile(total_losses_with_coverage, upper_percentile),
            'min': np.min(total_losses_with_coverage),
            'max': np.max(total_losses_with_coverage)
        }
    }
    
    # Aggregate P_value and Rating
    p_values = [r['P_value'] for r in all_results]
    aggregated['P_value'] = {
        'mean': np.mean(p_values),
        'std': np.std(p_values),
        'lower_ci': np.percentile(p_values, lower_percentile),
        'upper_ci': np.percentile(p_values, upper_percentile)
    }
    
    # Get most common rating
    ratings = [r['Rating'] for r in all_results]
    aggregated['Rating'] = {
        'most_common': max(set(ratings), key=ratings.count),
        'all_ratings': ratings
    }
    
    # Aggregate main coverage categories
    for category in ['Incident Response Costs', 'Business Interruption Loss', 'Privacy Liability Coverage']:
        category_data = {}
        for key in ['loss', 'coverage', 'deductible', 'existing_coverage', 'coverage_gap']:
            values = [r[category][key] for r in all_results]
            category_data[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower_ci': np.percentile(values, lower_percentile),
                'upper_ci': np.percentile(values, upper_percentile),
                'min': np.min(values),
                'max': np.max(values)
            }
        aggregated[category] = category_data
    
    # Aggregate Incident Response Costs Subcategories
    for subcat in ['Legal', 'Technical Response', 'Crisis Management', 'Notification']:
        values = [r['Incident Response Costs Subcat'][subcat] for r in all_results]
        aggregated['Incident Response Costs Subcat'][subcat] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'lower_ci': np.percentile(values, lower_percentile),
            'upper_ci': np.percentile(values, upper_percentile),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Aggregate Business Interruption Loss Subcategories
    for subcat in ['Loss Net Income', 'Continuing Expenses', 'Contingent Business Interruption']:
        values = [r['Business Interruption Loss SubCats'][subcat] for r in all_results]
        aggregated['Business Interruption Loss SubCats'][subcat] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'lower_ci': np.percentile(values, lower_percentile),
            'upper_ci': np.percentile(values, upper_percentile),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Aggregate Privacy Liability Coverage Subcategories
    # This typically has regulatory fines which might be constant
    first_privacy_subcat = all_results[0]['Privacy Liability Coverage Subcat']
    for key in first_privacy_subcat.keys():
        values = [r['Privacy Liability Coverage Subcat'][key] for r in all_results]
        if np.std(values) == 0:  # If all values are the same (e.g., regulatory fines)
            aggregated['Privacy Liability Coverage Subcat'][key] = {
                'value': values[0],
                'is_constant': True
            }
        else:
            aggregated['Privacy Liability Coverage Subcat'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower_ci': np.percentile(values, lower_percentile),
                'upper_ci': np.percentile(values, upper_percentile),
                'is_constant': False
            }
    
    # Convert to serializable format
    serializable_results = {}
    for key, value in aggregated.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable_nested(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)
    
    return serializable_results


def convert_to_serializable_nested(obj):
    """
    Recursively convert nested structures to serializable format.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable_nested(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return convert_to_serializable(obj)


def plot_ensemble_results(ensemble_results):
    """
    Plot the ensemble results (placeholder for future implementation).
    """
    # TODO: Implement visualization of ensemble results
    pass


def print_ensemble_summary(ensemble_results):
    """
    Print a human-readable summary of the ensemble results.
    """
    print("\n" + "="*80)
    print("ENSEMBLE MODEL SUMMARY")
    print("="*80)
    
    metadata = ensemble_results['ensemble_metadata']
    print(f"\nCompany: {metadata['company_name']}")
    print(f"Models Run: {metadata['num_models']}")
    print(f"Confidence Level: {metadata['confidence_level']}%")
    print(f"Industry Benchmark: {metadata['industry_benchmark']}")
    print(f"Confidence Rating: {metadata['confidence_rating']}")
    
    print("\n" + "-"*80)
    print("TOTAL LOSS ESTIMATES")
    print("-"*80)
    
    total_loss = ensemble_results['Total']['total_loss']
    print(f"Mean Total Loss: ${total_loss['mean']:,.2f}")
    print(f"Standard Deviation: ${total_loss['std']:,.2f}")
    print(f"{metadata['confidence_level']}% Confidence Interval: (${total_loss['lower_ci']:,.2f}, ${total_loss['upper_ci']:,.2f})")
    print(f"Range: ${total_loss['min']:,.2f} - ${total_loss['max']:,.2f}")
    
    total_with_cov = ensemble_results['Total']['total_loss_with_coverage']
    print(f"\nMean Total Loss (with coverage): ${total_with_cov['mean']:,.2f}")
    print(f"{metadata['confidence_level']}% Confidence Interval: (${total_with_cov['lower_ci']:,.2f}, ${total_with_cov['upper_ci']:,.2f})")
    
    print("\n" + "-"*80)
    print("RISK RATING")
    print("-"*80)
    
    p_value = ensemble_results['P_value']
    rating = ensemble_results['Rating']
    print(f"Mean P-value: {p_value['mean']:.2f}%")
    print(f"{metadata['confidence_level']}% Confidence Interval: ({p_value['lower_ci']:.2f}%, {p_value['upper_ci']:.2f}%)")
    print(f"Most Common Rating: {rating['most_common']}")
    print(f"All Ratings: {', '.join(rating['all_ratings'])}")
    
    print("\n" + "-"*80)
    print("LOSS BREAKDOWN BY CATEGORY")
    print("-"*80)
    
    for category in ['Incident Response Costs', 'Business Interruption Loss', 'Privacy Liability Coverage']:
        loss = ensemble_results[category]['loss']
        gap = ensemble_results[category]['coverage_gap']
        print(f"\n{category}:")
        print(f"  Mean Loss: ${loss['mean']:,.2f} (±${loss['std']:,.2f})")
        print(f"  {metadata['confidence_level']}% CI: (${loss['lower_ci']:,.2f}, ${loss['upper_ci']:,.2f})")
        print(f"  Mean Coverage Gap: ${gap['mean']:,.2f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Example usage
    with open('./input.json') as f:
        data = json.load(f)
    
    # Run ensemble with 10 models using fixed random seed for reproducibility
    ensemble_output = run_ensemble_model(data, num_models=10, confidence_level=95, random_seed=42)
    
    # Print summary
    print_ensemble_summary(ensemble_output)
    
    # Save to file
    with open('./ensemble_output.json', 'w') as f:
        json.dump(ensemble_output, f, indent=4)
    
    print("\nEnsemble results saved to 'ensemble_output.json'")
