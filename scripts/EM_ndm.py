#subject_dataにはnumpy配列が入る。pandasで読み込んで1回valuesで変換することが期待される。
#

def compute_likelihood(subject, x_t_norm):
    # 二乗誤差に基づく尤度の計算
    #######尤度指標はちゃんと考えないといけない。今回は流石に一つ一つの部位の萎縮の程度を考えたいから二乗和誤差を使うような気がするけどね...
    likelihood = -np.sum((subject - x_t_norm) ** 2)
    return likelihood

def update_gamma(subject_data, membership_probs, c_type, c_path, thr, seed_region, ref_list):
    # gamma の最適化
    def objective_function(gamma):
        x_t_norm = run_ndm(c_type, c_path, thr, seed_region, ref_list, gamma=gamma)
        likelihoods = np.array([compute_likelihood(subject, x_t_norm) for subject in subject_data])
        return -np.sum(membership_probs * likelihoods)  # 負の尤度を最小化する

    result = minimize(objective_function, x0=5e-5, bounds=[(1e-6, 1e-3)])  # 例として最適化
    return result.x[0]  # 最適化された gamma を返す

def update_seed_region(subject_data, membership_probs, seed_candidates, c_type, c_path, thr, ref_list):
    best_seed = None
    best_likelihood = -np.inf

    for seed in seed_candidates:
        x_t_norm = run_ndm(c_type, c_path, thr, seed, ref_list)
        likelihoods = np.array([compute_likelihood(subject, x_t_norm) for subject in subject_data])
        total_likelihood = np.sum(membership_probs * likelihoods)
        
        if total_likelihood > best_likelihood:
            best_likelihood = total_likelihood
            best_seed = seed
    
    return best_seed


def em_ndm_with(subject_data, c_type, c_path, thr, seed_candidates, ref_list, n_subgroups, tol=1e-6, max_iter=100):
    # Initialize parameters
    gammas = np.random.rand(n_subgroups) * 1e-4  # random initial guesses for gamma
    membership_probs = np.random.rand(len(subject_data), n_subgroups)
    membership_probs /= membership_probs.sum(axis=1, keepdims=True)  # normalize to sum to 1
    seed_regions = np.random.choice(seed_candidates, size=n_subgroups)  # Randomly initialize seed regions

    log_likelihood_old = -np.inf
    converged = False
    iter_count = 0

    while not converged and iter_count < max_iter:
        iter_count += 1
        
        # E-Step: Calculate expected log-likelihood and update membership probabilities
        log_likelihoods = np.zeros((len(subject_data), n_subgroups))
        for i, subject in enumerate(subject_data):
            for k in range(n_subgroups):
                # Run NDM with current parameters for subgroup k
                x_t_norm = run_ndm(c_type, c_path, thr, seed_regions[k], ref_list, gammas[k])
                log_likelihoods[i, k] = compute_likelihood(subject, x_t_norm)
        
        # Update membership probabilities
        membership_probs = np.exp(log_likelihoods)
        membership_probs /= membership_probs.sum(axis=1, keepdims=True)  # normalize to sum to 1

        # M-Step: Update gamma and seed regions based on new membership probabilities
        for k in range(n_subgroups):
            gammas[k] = update_gamma(subject_data, membership_probs[:, k], c_type, c_path, thr, seed_regions[k], ref_list)
            seed_regions[k] = update_seed_region(subject_data, membership_probs[:, k], seed_candidates, c_type, c_path, thr, ref_list)

        # Calculate the total log-likelihood
        log_likelihood_new = np.sum(np.log(np.sum(np.exp(log_likelihoods), axis=1)))

        # Check for convergence (e.g., change in log-likelihood or parameters)
        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            converged = True
        
        log_likelihood_old = log_likelihood_new

    return gammas, membership_probs, seed_regions