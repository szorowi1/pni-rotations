data {

    // Metadata
    int  N;                            // number of subjects
    int  T;                            // number of trials
    int<lower=1> ix[T];                // subject index
    
    // Data
    int<lower=1, upper=9>  X[T, 2];    // presented machines
    int<lower=1, upper=2>  Y[T];       // choice data
    real R[T];                         // outcome data
    real q;                            // Initial Q value
    
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
    vector<lower=0,upper=50>[N] beta;
    vector<lower=0,upper=1>[N]  eta_v;
    
    for (i in 1:N) {
        beta[i]   = Phi_approx( mu_pr[1] + sigma[1] * beta_pr[i] ) * 50;
        eta_v[i]  = Phi_approx( mu_pr[2] + sigma[2] * eta_v_pr[i] );
    }
    
}
model {

    // Generated data
    vector[9] Q[N];
    for (i in 1:N) { Q[i] = rep_vector(q, 9); }
    
    // Priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);

    // Likelihood
    for (i in 1:T) {

        // Likelihood of observed choice.
        Y[i] ~ categorical_logit( beta[ix[i]] * Q[ix[i], X[i]] );

        // Value updating (reward prediction error).
        Q[ix[i], X[i,Y[i]]] += eta_v[ix[i]] * ( R[i] - Q[ix[i], X[i,Y[i]]] );
                
    }

}
generated quantities {

    // Transformed group-level parameters.
    real  mu_beta;          // Inverse temperature
    real  mu_eta_v;         // Learning rate
    
    // Posterior predictive check / log-likelihood values.
    int  Y_pred[T];         // Simulated choice data
    vector[N] log_lik;      // Model log-likelihood
    
    // Transform parameters.
    mu_beta = Phi_approx( mu_pr[1] ) * 50;
    mu_eta_v = Phi_approx( mu_pr[2] );
    
    // Initialize log-likelihood.
    log_lik = rep_vector(0, N);
    
    { // Local section (to avoid saving Q-values)
    
        // Initialize local values
        vector[9] Q[N];
        for (i in 1:N) { Q[i] = rep_vector(q, 9); }
        
        // Iterate over trials.
        for (i in 1:T) {
        
            // Log-likelihood of observed choice.
            log_lik[ix[i]] += categorical_logit_lpmf( Y[i] | beta[ix[i]] * Q[ix[i], X[i]] );
            
            // Predict choice given current model.
            Y_pred[i] = categorical_logit_rng( beta[ix[i]] * Q[ix[i], X[i]] );
            
            // Value updating (reward prediction error).
            Q[ix[i], X[i,Y[i]]] += eta_v[ix[i]] * ( R[i] - Q[ix[i], X[i,Y[i]]] );
        
        }
    }
}