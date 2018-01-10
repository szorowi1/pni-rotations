data {

    // Metadata
    int  N;                            // number of subjects
    int  B;                            // number of blocks
    int  T;                            // number of trials
    
    // Data
    int  X[N, B, T, 2];                // presented machines, range [1-9]
    int  Y[N, B, T];                   // choice data, range [-1-2] where missing = -1
    real R[N, B, T];                   // outcome data, range {0.00, 0.25}
    
}
parameters {

    // Group-level (hyper)parameters
    vector[2] mu_pr;
    vector<lower=0>[2] sigma; 

    // Subject-level parameters (raw)
    vector[N] beta_pr;
    vector[N] eta_v_pr;

}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=50>[N]    beta;
    vector<lower=0,upper=1>[N]     eta_v;
    
    for (i in 1:N) {
        beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_pr[i] ) * 50;
        eta_v[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_v_pr[i] );
    }
    
}
model {
    
    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    
    // Subject-level priors
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N) {
    
        // Generated data
        vector[9] Q;
        Q = rep_vector(0, 9);
  
        for (j in 1:B) {
        
            for (k in 1:T) {
                        
                real delta;
                delta = 0;
                        
                // Execute only for non-missing responses.
                if ( Y[i,j,k] > 0 ) {
                
                    // Likelihood of observed choice.
                    Y[i,j,k] ~ categorical_logit( beta[i] * Q[X[i,j,k,:]] );
                    
                    // Compute reward prediction error.
                    delta = R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];
                    
                    // Update expectations.
                    Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;
                
                }
                
            }
        
        }
    
    }
    
}
generated quantities {

    // Transformed group-level parameters.
    real mu_beta;            // Inverse temperature
    real mu_eta_v;           // Learning rate
    
    // Posterior predictive check / log-likelihood values.
    int  Y_pred[N, B, T];    // Simulated choice data
    vector[N] log_lik;       // Model log-likelihood
    
    // Transform parameters.
    mu_beta = Phi_approx( mu_pr[1] ) * 50;
    mu_eta_v = Phi_approx( mu_pr[2] );
    
    // Initialize stored data.
    log_lik = rep_vector(0, N);

    { // Local section (to avoid saving Q-values)
    
        for (i in 1:N) {

            // Generated data
            vector[9] Q;
            Q = rep_vector(0, 9);

            for (j in 1:B) {

                for (k in 1:T) {

                    real delta;
                    delta = 0;

                    // Execute only for non-missing responses.
                    if ( Y[i,j,k] > 0 ) {

                        // Log-likelihood of observed choice.
                        log_lik[i] += categorical_logit_lpmf( Y[i,j,k] | beta[i] * Q[X[i,j,k,:]] );

                        // Predict choice given current model.
                        Y_pred[i,j,k] = categorical_logit_rng( beta[i] * Q[X[i,j,k,:]] );

                        // Compute reward prediction error.
                        delta = R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];

                        // Update expectations.
                        Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;

                    } 
                   
               }
                
            }
        
        }
    
    }
    
}