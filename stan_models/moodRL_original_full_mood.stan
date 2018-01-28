// Full mood RL model with partial pooling and fully-estimated h-values (original model).

data {

    // Metadata
    int  N;                            // number of subjects
    int  B;                            // number of blocks
    int  T;                            // number of trials
    
    // Data
    int  X[N, B, T, 2];                // presented machines, range [1-9]
    int  Y[N, B, T];                   // choice data, range [-1-2] where missing = -1
    int  R[N, B, T];                   // outcome data, range [0, 1]
    
    // Initial values
    real h12[N, 2];                    // Initial h-values, blocks 1-2, arctanh transformed

}
parameters {

    // Group-level (hyper)parameters
    vector[4] mu_pr;
    vector<lower=0>[4] sigma; 

    // Subject-level parameters (raw)
    vector[N] beta_pr;
    vector[N] eta_v_pr;
    vector[N] eta_h_pr;
    vector[N] f_pr;

}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=20>[N]    beta;
    vector<lower=0,upper=1>[N]     eta_v;
    vector<lower=0,upper=1>[N]     eta_h;
    vector[N]     f;
    
    for (i in 1:N) {
        beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_pr[i] ) * 20;
        eta_v[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_v_pr[i] );
        eta_h[i] = Phi_approx( mu_pr[3] + sigma[3] * eta_h_pr[i] );
        f[i] = exp( (mu_pr[4] + sigma[4] * f_pr[i]) / 2 );
    }
    
}
model {
    
    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    
    // Subject-level priors
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    eta_h_pr ~ normal(0, 1);
    f_pr ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N) {
    
        // Initialize values
        vector[9] Q;
        real delta;
        real h;
        real m;

        Q = rep_vector(0, 9);
        delta = 0;
        h = 0;
        m = tanh(h);
  
        for (j in 1:B) {
        
            // Initialize h-value from pre-block questionnaire.
            if ( j < 3 ) { 
                h = h12[i,j];
                m = tanh(h);
            }
        
            for (k in 1:T) {

                // Section for choice data.
                if ( Y[i,j,k] > 0 ) {
                
                    // Likelihood of observed choice.
                    Y[i,j,k] ~ categorical_logit( beta[i] * Q[X[i,j,k,:]] );
                    
                    // Compute reward prediction error.
                    delta = (f[i] ^ m) * R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];
                    
                    // Update expectations.
                    Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;
                
                    // Update history of rewards.
                    h += eta_h[i] * (delta - h);
                    
                    // Update mood.
                    m = tanh( h );
                
                }
                
            }
        
        }
    
    }
    
}
generated quantities {
    
    // Posterior predictive check / log-likelihood values.
    real Y_pred[N, B, T];       // Simulated choice data
    real Y_log_lik[N, B, T];    // Model log-likelihood

    // Transformed group-level parameters.
    real mu_beta;            // Inverse temperature
    real mu_eta_v;           // Learning rate
    real mu_eta_h;           // History rate
    real mu_f;               // Mood bias
    
    // Transform parameters.
    mu_beta = Phi_approx( mu_pr[1] ) * 20;
    mu_eta_v = Phi_approx( mu_pr[2] );
    mu_eta_h = Phi_approx( mu_pr[3] );
    mu_f = exp( mu_pr[4] / 2 );

    { // Local section (to avoid saving Q-values)
    
        for (i in 1:N) {

            // Initialize values
            vector[9] Q;
            real delta;
            real h;
            real m;

            Q = rep_vector(0, 9);
            delta = 0;
            h = 0;
            m = tanh(h);

            for (j in 1:B) {

                // Initialize h-value from pre-block questionnaire.
                if ( j < 3 ) { 
                    h = h12[i,j];
                    m = tanh(h);
                }

                for (k in 1:T) {

                    // Section for observed choice data.
                    if ( Y[i,j,k] > 0 ) {

                        // Log-likelihood of observed choice.
                        Y_log_lik[i,j,k] = categorical_logit_lpmf( Y[i,j,k] | beta[i] * Q[X[i,j,k,:]] );

                        // Predict choice given current model.
                        Y_pred[i,j,k] = categorical_logit_rng( beta[i] * Q[X[i,j,k,:]] );

                        // Compute reward prediction error.
                        delta = (f[i] ^ m) * R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];

                        // Update expectations.
                        Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;

                        // Update history of rewards.
                        h += eta_h[i] * (delta - h);

                        // Update mood.
                        m = tanh( h );

                    // Section for missing choice data.
                    } else {
                    
                        // Log-likelihood of observed choice.
                        Y_log_lik[i,j,k] = 0;
                        
                        // Predict choice given current model.
                        Y_pred[i,j,k] = -1;
                        
                    }
                   
               }
                
            }
        
        }
    
    }
    
}