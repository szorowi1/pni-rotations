data {

    // Metadata
    int  B;                         // number of blocks
    int  T;                         // number of trials
    
    // Data
    int  X[B, T, 2];                // presented machines, range [1-9]
    int  Y[B, T];                   // choice data, range [-1-2] where missing = -1
    real Z[B, T];                   // reaction time data, where missing = -1
    int  R[B, T];                   // outcome data, range [0, 1]
    real M[B, 3];                   // Mood data, range (-1, 1)
}
parameters {

    // Subject-level parameters (raw)
    real alpha_pr;
    real tau_pr;
    real eta_v_pr;
    real beta_h;
    real<lower=0> sigma_m;

}
transformed parameters {

    // Subject-level parameters (transformed)
    real<lower=0>             alpha;
    real<lower=0.1>           tau;
    real<lower=0,upper=1>     eta_v;
    real<lower=-1,upper=1>    beta_m;
    
    alpha  = exp( alpha_pr );
    tau    = Phi_approx( tau_pr );
    eta_v  = Phi_approx( eta_v_pr );
    beta_m = tanh( beta_h );
    
}
model {
   
    // Initialize values
    vector[9] Q;
    real drift;
    real delta;

    Q = rep_vector(0, 9);
    drift = 0;
    delta = 0;
   
    // Subject-level priors
    alpha_pr ~ normal(0, 1);
    tau_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    beta_h ~ normal(0, 1);
    sigma_m ~ gamma(1, 0.5);
    
    // Likelihood
    for (j in 1:B) {

        for (k in 1:T) {

            if ( Y[j,k] > 0 ) {

                // Compute drift rate
                drift = 10 * (Q[X[j,k,2]] - Q[X[j,k,1]]);

                // Compute log-likelihood of response.
                Z[j,k] ~ wiener( alpha, tau, 0.5, drift );

                // Compute reward prediction error.
                delta = R[j, k] - Q[X[j,k, Y[j,k]]];
                
                // Update expectations.
                Q[X[j,k, Y[j,k]]] += eta_v * delta;

            }

            // Section for mood data.
            if ( k == 7 ){
                M[j,1] ~ normal( beta_m, sigma_m );
            } else if ( k == 21 ) {
                M[j,2] ~ normal( beta_m, sigma_m );
            } else if ( k == 35 ) {
                M[j,3] ~ normal( beta_m, sigma_m );
            }

        }

    }

}
generated quantities {

    real drift[B,T];
    
    {
        vector[9] Q;
        real delta;

        Q = rep_vector(0, 9);
        delta = 0;
    
        for (j in 1:B) {

            for (k in 1:T) {

                if ( Y[j,k] > 0 ) {

                    // Compute drift rate
                    drift[j,k] = 10 * (Q[X[j,k,2]] - Q[X[j,k,1]]);

                    // Compute reward prediction error.
                    delta = R[j, k] - Q[X[j,k, Y[j,k]]];

                    // Update expectations.
                    Q[X[j,k, Y[j,k]]] += eta_v * delta;

                } else {
                
                    drift[j,k] = -1;
                
                }
                
            }
            
        }
        
    }
}