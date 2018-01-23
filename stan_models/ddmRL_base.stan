data {

    // Metadata
    int  N;                            // number of subjects
    int  B;                            // number of blocks
    int  T;                            // number of trials
    
    // Data
    int  X[N, B, T, 2];                // presented machines, range [1-9]
    int  Y[N, B, T];                   // choice data, range [-1-2] where missing = -1
    int  Z[N, B, T];                   // reaction time data, where missing = -1
    int  R[N, B, T];                   // outcome data, range [0, 1]
    real M[N, B, 3];                   // Mood data, range (-1, 1)

}
parameters {

    // Group-level (hyper)parameters
    vector[3] mu_pr;
    vector<lower=0>[3] sigma; 
    real<lower=0> sigma_m;

    // Subject-level parameters (raw)
    vector[N] alpha_pr;
    vector[N] tau_pr;
    vector[N] eta_v_pr;
    vector[N] beta_h;

}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0>[N]              alpha;
    vector<lower=0.1>[N]              tau;
    vector<lower=0,upper=1>[N]      eta_v;
    vector<lower=-1,upper=1>[N]    beta_m;
    
    for (i in 1:N) {
        alpha[i]  = exp( mu_pr[1] + sigma[1] * alpha_pr[i] );
        tau[i]    = Phi_approx( mu_p[2] + sigma[2] * tau_pr[i] ) * 0.2;
        eta_v[i]  = Phi_approx( mu_pr[3] + sigma[3] * eta_v_pr[i] );
        beta_m[i] = tanh( beta_h[i] );
    }
    
}
model {

    // Group-level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    sigma_m ~ gamma(1, 0.5);
    
    // Subject-level priors
    alpha_pr ~ normal(0, 1);
    tau_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    beta_h ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N) {
    
        // Initialize values
        vector[9] Q;
        real drift;
        real delta;

        Q = rep_vector(0, 9);
        drift = 0;
        delta = 0;
    
        for (j in 1:B) {
        
            for (k in 1:T) {
            
                if (Z[i,j,k] > 0) {
            
                    // Compute drift rate (delta)
                    delta = Q[X[i,j,k,2] - Q[X[i,j,k,1];

                    // Compute log-likelihood of response.
                    Z[i,j,k] ~ wiener( alpha[i], tau[i], 0.5, delta );

                    // Compute reward prediction error.
                    delta = R[i, j, k] - Q[X[i,j,k, Y[i,j,k]]];

                    // Update expectations.
                    Q[X[i,j,k, Y[i,j,k]]] += eta_v[i] * delta;
                    
                }
                
                // Section for mood data.
                if ( k == 7 ){
                    M[i,j,1] ~ normal( beta_m[i], sigma_m );
                } else if ( k == 21 ) {
                    M[i,j,2] ~ normal( beta_m[i], sigma_m );
                } else if ( k == 35 ) {
                    M[i,j,3] ~ normal( beta_m[i], sigma_m );
                }

            }
        
        }
    
    }
    
}