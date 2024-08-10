#MCMCで最適なものを決定していくよ

import numpy as np
import matplotlib.pyplot as plt

def mcmc_ndm(subject_data, c_type, c_path, thr, seed_candidates, ref_list, n_subgroups, n_iterations=1000000):
    log_likelihoods = []
    
    for i in range(n_iterations):
        # Run EM or similar optimization within MCMC step
        gammas, membership_probs, seed_regions = em_ndm_with_seed(
            subject_data, c_type, c_path, thr, seed_candidates, ref_list, n_subgroups
        )
        
        # Calculate log-likelihood
        log_likelihood = compute_total_log_likelihood(subject_data, gammas, membership_probs, seed_regions)
        log_likelihoods.append(log_likelihood)
    
    return log_likelihoods

# Calculate the total log-likelihood for the entire dataset
def compute_total_log_likelihood(subject_data, gammas, membership_probs, seed_regions):
    total_log_likelihood = 0
    for i, subject in enumerate(subject_data):
        subgroup_log_likelihoods = []
        for k, gamma in enumerate(gammas):
            x_t_norm = run_ndm(c_type, c_path, thr, seed_regions[k], ref_list, gamma=gamma)
            subgroup_log_likelihoods.append(compute_likelihood(subject, x_t_norm))
        total_log_likelihood += np.log(np.sum(np.exp(subgroup_log_likelihoods) * membership_probs[i]))
    return total_log_likelihood

def mcmc_ndm_with_seed_search(subject_data, c_type, c_path, thr, seed_candidates, ref_list, n_subgroups, n_iterations=1000000):
    best_log_likelihood = -np.inf
    best_seed_regions = None
    best_gammas = None
    log_likelihoods = []

    for i in range(n_iterations):
        # Run EM or similar optimization within MCMC step
        gammas, membership_probs, seed_regions = em_ndm_with_seed(
            subject_data, c_type, c_path, thr, seed_candidates, ref_list, n_subgroups
        )
        
        # Calculate log-likelihood
        log_likelihood = compute_total_log_likelihood(subject_data, gammas, membership_probs, seed_regions)
        log_likelihoods.append(log_likelihood)
        
        # If this is the best likelihood we've seen, update the best parameters
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_seed_regions = seed_regions
            best_gammas = gammas

    return best_seed_regions, best_gammas, log_likelihoods