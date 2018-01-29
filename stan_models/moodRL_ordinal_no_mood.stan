functions {

    vector ord_pdf( real mu, vector cutpoints ) {
        int N = rows(cutpoints) + 1;
        vector[N] pr;

        pr[1] = cauchy_cdf(cutpoints[1], mu, 1);
        for (i in 2:N-1) {
            pr[i] = cauchy_cdf(cutpoints[i], mu, 1) - cauchy_cdf(cutpoints[i-1], mu, 1);
        }
        pr[N] = 1 - cauchy_cdf(cutpoints[N-1], mu, 1);
        return pr;
    }

}
data {

    // Metadata
    int  N;                            // number of subjects
    int  T;                            // number of trials
    int  C;
    int mood_ix[T];  
    int subj_ix[T];
    
    // Data
    int  X[T, 2];                // presented machines, range [1-9]
    int  Y[T];                   // choice data, range [-1-2] where missing = -1
    int  R[T];                   // outcome data, range [0, 1]
    int  M[N*9];                 // mood data, range [1-9]
    
}
transformed data{
    
    vector[C-1] cutpoints;
    
    for (i in 1:C-1) {
        cutpoints[i] = i + 0.5;
    }

}
parameters {

    // Group-level parameters (RL)
    vector[2] mu_pr;
    vector<lower=0>[2] sigma; 

    // Subject-level parameters (RL)
    vector[N] beta_pr;
    vector[N] eta_v_pr;
    
    // Subject-level parameters (Mood)
    vector<lower=1,upper=C>[N] beta_m;
    //vector<lower=0>[N] sigma_m;
    
}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=20>[N]    beta;
    vector<lower=0,upper=1>[N]     eta_v;
    
    for (i in 1:N) {
        beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_pr[i] ) * 20;
        eta_v[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_v_pr[i] );
    }
    
}
model {

    // Generated data
    vector[9] Q[N];
    vector[T]  dEV;
    real delta;
    
    for (i in 1:N){ Q[i] = rep_vector(0, 9); }
    delta = 0;

    // Group-level priors (RL)
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    
    // Subject-level priors (RL)
    beta_pr ~ normal(0, 1);
    eta_v_pr ~ normal(0, 1);
    
    // Subject-level priors (Mood)
    beta_m ~ uniform(1,C);
        
    // Precompute RL values.
    for (i in 1:T) {
    
        // Compute difference in expected value.
        dEV[i] = beta[subj_ix[i]] * ( Q[subj_ix[i], X[i,2]] - Q[subj_ix[i], X[i,1]] );
        
        // Compute reward prediction error.
        delta = R[i] - Q[subj_ix[i], X[i,Y[i]+1]];
            
        // Update expectations.
        Q[subj_ix[i], X[i,Y[i]+1]] += eta_v[subj_ix[i]] * delta;
        
        // Compute likelihood of mood values.
        if ( mood_ix[i] > 0 ){
            M[mood_ix[i]] ~ categorical( ord_pdf( beta_m[subj_ix[i]], cutpoints ) );
        }
    
    }
    
    // Likelihood.
    Y ~ bernoulli_logit( dEV );
        
}
generated quantities {

    // Posterior predictive check / log-likelihood values.
    vector[T] Y_pred;         // Simulated choice data
    vector[N*9] M_pred;       // Simulated choice data
    vector[T] Y_log_lik;      // Model log-likelihood
    vector[N*9] M_log_lik;    // Model log-likelihood

    // Transformed group-level parameters.
    real mu_beta;             // Inverse temperature
    real mu_eta_v;            // Learning rate
    
    // Transform parameters.
    mu_beta = Phi_approx( mu_pr[1] ) * 20;
    mu_eta_v = Phi_approx( mu_pr[2] );
    
    { // Local section 
    
        // Generated data
        vector[9] Q[N];
        vector[C] theta;
        real dEV;
        real delta;

        for (i in 1:N){ Q[i] = rep_vector(0, 9); }
        delta = 0;

        for (i in 1:T) {

            // Compute difference in expected value.
            dEV = beta[subj_ix[i]] * ( Q[subj_ix[i], X[i,2]] - Q[subj_ix[i], X[i,1]] );

            // Compute log-likelihood of choice.
            Y_log_lik[i] = bernoulli_logit_lpmf( Y | dEV );
            
            // Predict choice given current model.
            Y_pred[i] = bernoulli_logit_rng( dEV );

            // Compute reward prediction error.
            delta = R[i] - Q[subj_ix[i], X[i,Y[i]+1]];

            // Update expectations.
            Q[subj_ix[i], X[i,Y[i]+1]] += eta_v[subj_ix[i]] * delta;

            // Compute likelihood of mood values.
            if ( mood_ix[i] > 0 ){
            
                // Compute probability of observing mood.
                theta = ord_pdf( beta_m[subj_ix[i]], cutpoints );
                
                // Compute log-likelihood of mood.
                M_log_lik[mood_ix[i]] = categorical_lpmf( M[mood_ix[i]] | theta );
                
                // Predict mood given model.
                M_pred[mood_ix[i]] = categorical_rng( theta );
            }

        }
               
    }

}