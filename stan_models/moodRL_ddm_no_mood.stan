data {

    // Metadata
    int  N;                      // number of subjects
    int  T;                      // number of total trials
    int subj_ix[T];              // index of subject for trial i
    int mood_ix[T];              // index of mood value for trial i
    
    // Data
    int  X[T, 2];                // presented machines, range [1-9]
    int  Y[T];                   // choice data, range [1, 2]
    int  R[T];                   // outcome data, range [0, 1]
    real Z[T];                   // RT data, range [0, 3]
    real M[N*9];                 // mood data, range (-1, 1)
    
    real minRT[N];               // Minimum RT.
    
}
parameters {

    // Group-level (hyper)parameters
    vector[3] mu_pr;
    vector<lower=0>[3] sigma; 
    real<lower=0> sigma_m;

    // Subject-level parameters (raw)
    vector[N] dm_pr;             // DDM drift rate modifier
    vector[N] db_pr;             // DDM decision boundary
    vector[N] eta_v_pr;          // RL learning rate
    
    vector[N] beta_h;            // Mood intercept
    vector<lower=0.1,upper=min(minRT)>[N] ndt;      // DDM non-decision time
    
}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=10>[N]    dm; 
    vector<lower=0,upper=5>[N]     db;
    vector<lower=0,upper=1>[N]     eta_v;
    
    for (i in 1:N) {
        dm[i] = Phi_approx( mu_pr[1] + sigma[1] * dm_pr[i] ) * 10;
        db[i] = Phi_approx( mu_pr[2] + sigma[2] * db_pr[i] ) * 5;
        eta_v[i] = Phi_approx( mu_pr[3] + sigma[3] * eta_v_pr[i] );
    }
    
}
model {

    // Generated data
    vector[T]      drift;        // Drift rates
    vector[T]      alpha;        // Decision boundary
    vector[T]      tau;          // Non-decision time
    vector[N*9]    Mhat;         // Estimated mood
    
    vector[9]      Q[N];         // Q-values
    real           delta;        // Reward prediction error
    real           h;            // Reward history
    real           m;            // Mood
    
    for (i in 1:N){ Q[i] = rep_vector(0, 9); }
    delta = 0;

    // Group-level priors (RL)
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    sigma_m ~ gamma(1, 0.5);
    
    // Subject-level priors (RL)
    eta_v_pr ~ normal(0, 1);
    db_pr ~ normal(0, 1);
    dm_pr ~ normal(0, 1);
    
    beta_h ~ normal(0, 1);
    for (i in 1:N) { ndt[i] ~ uniform(0.1, minRT[i]); }    
    
    // Precompute values.
    for (i in 1:T) {
    
        // Update and store drift rate.
        drift[i] = dm[subj_ix[i]] * ( Q[subj_ix[i], X[i,2]] - Q[subj_ix[i], X[i,1]] );
        Z[i] ~ wiener( db[subj_ix[i]], ndt[subj_ix[i]], 0.5, drift[i] );
        
        // Compute reward prediction error.
        delta = R[i] - Q[subj_ix[i], X[i,Y[i]]];
            
        // Update expectations.
        Q[subj_ix[i], X[i,Y[i]]] += eta_v[subj_ix[i]] * delta;

        // Update mood.
        m = tanh( beta_h[subj_ix[i]] );       
        
        // Store values.
        alpha[i] = db[subj_ix[i]];
        tau[i] = ndt[subj_ix[i]];
        if ( mood_ix[i] > 0 ) { M[mood_ix[i]] ~ normal(m, sigma_m); }
        
    }
    // Likelihood.
    Z ~ wiener( alpha, tau, rep_vector(0.5, T), drift );
    M ~ normal( Mhat, sigma_m );
    
}
generated quantities {

    // Transformed group-level parameters.
    real mu_dm;               // Drift rate modifier
    real mu_db;               // Decision boundary
    real mu_eta_v;            // Learning rate
    
    // Transform parameters.
    mu_dm = Phi_approx( mu_pr[1] ) * 10;
    mu_db = Phi_approx( mu_pr[2] ) * 5;
    mu_eta_v = Phi_approx( mu_pr[3] );

}