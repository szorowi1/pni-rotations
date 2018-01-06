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
    vector[2] mu_p;
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
        beta[i]   = Phi_approx( mu_p[1] + sigma[1] * beta_pr[i] ) * 50;
        eta_v[i]  = Phi_approx( mu_p[2] + sigma[2] * eta_v_pr[i] );
    }
    
}
model {

    // Generated data
    vector[9] Q[N];
    for (i in 1:N) { Q[i] = rep_vector(q, 9); }
    
    // Priors
    mu_p ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);

    // Likelihood
    for (i in 1:T) {

        // Choice phase
        Y[i] ~ categorical_logit( beta[ix[i]] * Q[ix[i], X[i]] );

        // Learning phase
        Q[ix[i], X[i,Y[i]]] += eta_v[ix[i]] * ( R[i] - Q[ix[i], X[i,Y[i]]] );
                
    }

}