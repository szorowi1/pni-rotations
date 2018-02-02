functions {
    
    // Element-wise power function
    vector fpow(vector a, real b){
        vector[rows(a)] x;
        
        for (i in 1:rows(a)) {
            x[i] = pow(a[i], b);
        }
        return x;
    }

    // Vectorized shifted Wald lpdf
    real shifted_wald_lpdf(vector X, vector gamma, vector alpha, vector theta){
        vector[rows(X)] tmp1;
        vector[rows(X)] tmp2;
        vector[rows(X)] sx;
        
        sx = X - theta;
        tmp1 = alpha ./ sqrt( 2 * pi() * fpow(sx, 3) );
        tmp2 = exp( -1 * fpow(alpha - gamma .* sx, 2) ./ (2 * sx) );
        return sum(log(tmp1 .* tmp2));
    }
    
}
data {

    // Metadata
    int  N;                      // number of subjects
    int  T;                      // number of total trials
    int subj_ix[T];              // index of subject for trial i
    int mood_ix[T];              // index of mood value for trial i
    
    // Data
    int  X[T, 2];                // presented machines, range [1-9]
    int  Y[T];                   // choice data, range [0, 1]
    int  R[T];                   // outcome data, range [0, 1]
    vector[T] Z;                 // RT data, range [0, 3]
    vector[N*9] M;               // mood data, range (-1, 1)
        
}
parameters {

    // Group-level (hyper)parameters
    vector[4] mu_pr;
    vector<lower=0>[4] sigma; 
    real<lower=0> sigma_m;

    // Subject-level parameters (raw)
    vector[N] gamma_pr;          // Drift rate
    vector[N] alpha_pr;          // Decision boundary
    vector[N] beta_pr;           // Inverse temperature
    vector[N] eta_v_pr;          // RL learning rate
    
    vector[N] beta_h;            // Mood intercept
    vector<lower=0>[N] theta;    // Non-decision time
    
}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=10>[N]    gamma;
    vector<lower=0,upper=3>[N]     alpha;
    vector<lower=0,upper=20>[N]    beta;
    vector<lower=0,upper=1>[N]     eta_v;
    
    for (i in 1:N) {
        gamma[i] = Phi_approx( mu_pr[1] + sigma[1] * gamma_pr[i] ) * 10;
        alpha[i] = Phi_approx( mu_pr[2] + sigma[2] * alpha_pr[i] ) * 3;
        beta[i]  = Phi_approx( mu_pr[3] + sigma[3] * beta_pr[i] ) * 20;
        eta_v[i] = Phi_approx( mu_pr[4] + sigma[4] * eta_v_pr[i] );
    }
    
}
model {

    // Generated data
    vector[T]      dEV;          // Difference in expected value
    vector[T]      dr;           // Drift rate
    vector[T]      db;           // Decision boundary
    vector[T]      ndt;          // Non-decision time
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
    gamma_pr ~ normal(0, 1);
    alpha_pr ~ normal(0, 1);
    beta_pr  ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    
    theta ~ normal(0.5, 0.5);
    beta_h ~ normal(0, 1);
    
    // Precompute values.
    for (i in 1:T) {
    
        // Update and store difference in expected values / drift.
        dEV[i] = beta[subj_ix[i]] * ( Q[subj_ix[i], X[i,2]] - Q[subj_ix[i], X[i,1]] );
        dr[i] = gamma[subj_ix[i]] * fabs( Q[subj_ix[i], X[i,2]] - Q[subj_ix[i], X[i,1]] );
        
        // Compute reward prediction error.
        delta = R[i] - Q[subj_ix[i], X[i,Y[i]+1]];
            
        // Update expectations.
        Q[subj_ix[i], X[i,Y[i]+1]] += eta_v[subj_ix[i]] * delta;

        // Update mood.
        m = tanh( beta_h[subj_ix[i]] );       
        
        // Store values.
        db[i] = alpha[subj_ix[i]];
        ndt[i] = theta[subj_ix[i]];
        if ( mood_ix[i] > 0 ) { Mhat[mood_ix[i]] = m; }
        
    }
    // Likelihood.
    target += bernoulli_logit_lpmf( Y | dEV );
    target += shifted_wald_lpdf( Z | dr, db, ndt  );
    target += normal_lpdf( M | Mhat, sigma_m );
    
}
generated quantities {

    // Transformed group-level parameters.
    real mu_gamma;              // Drift rate
    real mu_alpha;              // Decision boundary
    real mu_beta;               // Inverse temperature
    real mu_eta_v;              // Learning rate
    
    // Transform parameters.
    mu_gamma = Phi_approx( mu_pr[1] ) * 10;
    mu_alpha = Phi_approx( mu_pr[2] ) * 3;
    mu_beta = Phi_approx( mu_pr[3] ) * 20;
    mu_eta_v = Phi_approx( mu_pr[4] );

}