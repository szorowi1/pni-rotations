data {

    // Metadata
    int N;                    // number of subjects
    int T;                    // number of trials
    int subj_ix[T];           // subject index
    int block_ix[T];          // block index
    int mood_ix[T];           // Mood rating index
    
    // Data
    real                    delta[T];
    real<lower=-1, upper=1>  M[N, 9];
    matrix[N,2]                  h12;
    
}
parameters {

    // Group-level (hyper)parameters
    real          mu_pr;
    real<lower=0> sigma; 

    // Subject-level parameters (raw)
    vector[N]  eta_h_pr;
    vector[N]        h3;

}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=1>[N]  eta_h;
    
    for (i in 1:N) {
        eta_h[i]  = Phi_approx( mu_pr + sigma * eta_h_pr[i] );
    }

}
model {

    // Generated data
    matrix[N,3] h;
    h[:,1:2] = h12;
    h[:,3] = h3;
    
    // Priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    eta_h_pr ~ normal(0, 1);
    h3 ~ normal(0, 1);

    // Likelihood
    for (i in 1:T) {

        // Update recent prediction-error history. 
        h[subj_ix[i], block_ix[i]] += eta_h[subj_ix[i]] * (delta[i] - h[subj_ix[i], block_ix[i]]);
        
        // Predict mood.
        if (mood_ix[i] > 0) {
            M[subj_ix[i], mood_ix[i]] ~ normal( tanh( h[subj_ix[i], block_ix[i]] ), 1 );
        }
        
    }
    
}
generated quantities {

    // Transformed group-level parameters.
    real mu_eta_h;         // History rate
    
    // Posterior predictive check.
    real M_pred[N, 9];     // Predicted moods.    
    
    // Transform parameters.
    mu_eta_h = Phi_approx(mu_pr);
    
    { // Local section (to avoid saving h-values)
    
        // Initialize local values
        matrix[N,3] h;
        h[:,1:2] = h12;
        h[:,3] = h3;
        
        // Iterate over trials.
        for (i in 1:T) {
        
            // Update recent prediction-error history. 
            h[subj_ix[i], block_ix[i]] += eta_h[subj_ix[i]] * (delta[i] - h[subj_ix[i], block_ix[i]]);
            
            // Predict mood.
            if (mood_ix[i] > 0) {
                M_pred[subj_ix[i], mood_ix[i]] = tanh( h[subj_ix[i], block_ix[i]] );
            }
            
        }
    }

}