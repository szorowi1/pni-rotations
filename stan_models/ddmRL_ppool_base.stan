data {

    // Metadata
    int     N;       // number of subjects
    int     T;       // number of trials
    int ix[T];       // index of post-query trials, binary {0, 1}
    
    // Data  
    int  X[N, T, 2]; // stimuli presented on a trial
    int  Y[N, T];    // trial choice, {less valuable = 1, more valuable = 2, missing = -1}
    real Z[N, T];    // response time, in range [0,3] or missing = -1
    int  R[N, T];    // reward, binary {0, 1}
    real minRT[N];   // Minimum reaction time

}
parameters {

    // Group-level parameters
    vector[3] mu_pr;
    vector<lower=0>[3] sigma; 

    // Subject-level parameters
    vector[N] alpha_pr;                 // DDM: decision boundary
    vector[N] beta_pr;                  // DDM: drift rate scaling
    vector<lower=0,upper=1>[N] tau_b_pr;    // DDM: non-decision time (baseline)
    vector[N] eta_v_pr;                 // RL: learning rate
    
}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0>[N]                       alpha;
    vector<lower=0>[N]                        beta;
    vector<lower=0>[N]                       tau_b;
    vector<lower=0,upper=1>[N]               eta_v;
    
    for (i in 1:N) {
        alpha[i]  = Phi_approx( mu_pr[1] + sigma[1] * alpha_pr[i] ) * 4;
        beta[i]   = exp( mu_pr[2] + sigma[2] * beta_pr[i] );
        tau_b[i]  = tau_b_pr[i] * minRT[i];
        eta_v[i]  = Phi_approx( mu_pr[3] + sigma[3] * eta_v_pr[i] );
    }
    
}
model {

    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    
    // Subject-level priors
    alpha_pr ~ normal(0, 1);
    beta_pr ~ normal(0, 1);
    tau_b_pr ~ beta(1, 1);
    eta_v_pr ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N) {
    
        // Initialize values
        vector[9] Q;
        real drift;
        real delta;
        real   tau;

        Q = rep_vector(0, 9);
        drift = 0;
        delta = 0;
        tau = 0;
    
        for (j in 1:T) {

            if (Z[i,j] > 0) {

                // Compute drift rate (delta).
                drift = beta[i] * (Q[X[i,j,2]] - Q[X[i,j,1]]);
                
                // Compute non-decision time (tau).
                tau = tau_b[i];

                // Compute log-likelihood of response.
                Z[i,j] ~ wiener( alpha[i], tau, 0.5, drift );

                // Compute reward prediction error.
                delta = R[i, j] - Q[X[i,j, Y[i,j]]];

                // Update expectations.
                Q[X[i,j, Y[i,j]]] += eta_v[i] * delta;

            }
        
        }
    
    }
    
}
generated quantities {
    
    // Posterior predictive check / log-likelihood values.
    matrix[N, T] Z_log_lik;    // Model log-likelihood
    matrix[N, T] drift;        // Predicted drift rate

    // Transformed group-level parameters.
    real mu_alpha;           // Inverse temperature
    real mu_beta;            // Drift rate 
    real mu_eta_v;           // Learning rate
    
    // Transform parameters.
    mu_alpha = Phi_approx( mu_pr[1] ) * 4;
    mu_beta = exp( mu_pr[2] );
    mu_eta_v = Phi_approx( mu_pr[3] );

    // Pre-populate matrices.
    Z_log_lik = rep_matrix(0, N, T);
    drift = rep_matrix(-1, N, T);

    { // Local section (to avoid saving Q-values)
    
        for (i in 1:N) {

            // Initialize values
            vector[9] Q;
            real delta;
            real   tau;

            Q = rep_vector(0, 9);
            delta = 0;
            tau = 0;

            for (j in 1:T) {

                // Section for observed choice data.
                if (Z[i,j] > 0) {

                    // Compute drift rate (delta).
                    drift[i,j] = beta[i] * (Q[X[i,j,2]] - Q[X[i,j,1]]);

                    // Compute non-decision time (tau).
                    tau = tau_b[i];

                    // Compute log-likelihood of response.
                    Z_log_lik[i,j] = wiener_lpdf( Z[i,j] | alpha[i], tau, 0.5, drift[i,j] );

                    // Compute reward prediction error.
                    delta = R[i, j] - Q[X[i,j, Y[i,j]]];

                    // Update expectations.
                    Q[X[i,j, Y[i,j]]] += eta_v[i] * delta;
                    
                }
                
            }
        
        }
    
    }
    
}