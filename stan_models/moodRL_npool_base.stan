// Baseline (no mood) RL model with no pooling

data {

    // Metadata
    int  N;                            // number of subjects
    int  B;                            // number of blocks
    int  T;                            // number of trials
    
    // Data
    int  X[N, B, T, 2];                // presented machines, range [1-9]
    int  Y[N, B, T];                   // choice data, range [-1-2] where missing = -1
    int  R[N, B, T];                   // outcome data, range [0, 1]
    real M[N, B, 3];                   // Mood data, range (-1, 1)
    
    // Initial values
    real h12[N, 2];                    // Initial h-values, blocks 1-2, arctanh transformed
    
}
parameters {

    // Subject-level parameters (raw)
    vector[N] beta_pr;
    vector[N] eta_v_pr;
    
    // Missing data
    vector[N] h3;

}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=20>[N]    beta;
    vector<lower=0,upper=1>[N]     eta_v;
    
    for (i in 1:N) {
        beta[i]  = Phi_approx( beta_pr[i] ) * 20;
        eta_v[i] = Phi_approx( eta_v_pr[i] );
    }
    
}
model {

    // Subject-level priors
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    h3 ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N) {
    
        // Initialize Q-values
        vector[9] Q;
        Q = rep_vector(0, 9);
  
        for (j in 1:B) {
        
            // Initialize history/mood values
            real h;
            real m;

            if ( j < 3 ) { 
                h = h12[i,j];
            } else { 
                h = h3[i];
            }
            m = tanh(h);
        
            for (k in 1:T) {
                        
                real delta;
                delta = 0;
                        
                // Section for choice data.
                if ( Y[i,j,k] > 0 ) {
                
                    // Likelihood of observed choice.
                    Y[i,j,k] ~ categorical_logit( beta[i] * Q[X[i,j,k,:]] );
                    
                    // Compute reward prediction error.
                    delta = R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];
                    
                    // Update expectations.
                    Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;
                
                }
                
                // Section for mood data.
                if ( k == 7 ){
                    M[i,j,1] ~ normal( m, 0.1 );
                } else if ( k == 21 ) {
                    M[i,j,2] ~ normal( m, 0.1 );
                } else if ( k == 35 ) {
                    M[i,j,3] ~ normal( m, 0.1 );
               }
                
            }
        
        }
    
    }
    
}
generated quantities {
    
    // Posterior predictive check / log-likelihood values.
    real Y_pred[N, B, T];       // Simulated choice data
    real Y_log_lik[N, B, T];    // Model log-likelihood
    real M_pred[N, B, 3];       // Simulated mood data
    real M_log_lik[N, B, 3];    // Model log-likelihood
    
    { // Local section (to avoid saving Q-values)
    
        for (i in 1:N) {

            // Initialize Q-values
            vector[9] Q;
            Q = rep_vector(0, 9);

            for (j in 1:B) {

                // Initialize history/mood values
                real h;
                real m;

                if ( j < 3 ) { 
                    h = h12[i,j];
                } else { 
                    h = h3[i];
                }
                m = tanh(h);

                for (k in 1:T) {

                    real delta;
                    delta = 0;

                    // Section for choice data.
                    if ( Y[i,j,k] > 0 ) {

                        // Log-likelihood of observed choice.
                        Y_log_lik[i,j,k] = categorical_logit_lpmf( Y[i,j,k] | beta[i] * Q[X[i,j,k,:]] );

                        // Predict choice given current model.
                        Y_pred[i,j,k] = categorical_logit_rng( beta[i] * Q[X[i,j,k,:]] );

                        // Compute reward prediction error.
                        delta = R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];

                        // Update expectations.
                        Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;

                    } else {
                        Y_log_lik[i,j,k] = 0;
                        Y_pred[i,j,k] = -1;
                    }
                    
                    // Section for mood data.
                    if ( k == 7 ){
                        M_log_lik[i,j,1] = normal_lpdf( M[i,j,1] | m, 0.1 );
                        M_pred[i,j,1] = m;
                    } else if ( k == 21 ) {
                        M_log_lik[i,j,2] = normal_lpdf( M[i,j,2] | m, 0.1 );
                        M_pred[i,j,2] = m;
                    } else if ( k == 35 ) {
                        M_log_lik[i,j,3] = normal_lpdf( M[i,j,3] | m, 0.1 );
                        M_pred[i,j,3] = m;
                   }
                   
               }
                
            }
        
        }
    
    }
    
}