data {

    // Metadata
    int  T;                            // number of trials
    
    // Data
    int<lower=1, upper=9>  X[T, 2];    // presented machines
    int<lower=1, upper=2>  Y[T];       // choice data
    real R[T];                         // outcome data
    real q;                            // Initial Q value
    
}
parameters {

    real beta_pr;
    real eta_v_pr;

}
transformed parameters {

    real<lower=0,upper=50> beta;
    real<lower=0,upper=1>  eta_v;
    
    beta   = Phi_approx( beta_pr ) * 50;
    eta_v  = Phi_approx( eta_v_pr );
    
}
model {

    // Generated data
    vector[9] Q;
    Q = rep_vector(q, 9);
    
    // Priors
    beta_pr ~ normal(0,1);
    eta_v_pr ~ normal(0,1);
    
    // Likelihood
    for (i in 1:T) {

        // Choice phase
        Y[i] ~ categorical_logit( beta * Q[X[i]] );

        // Learning phase
        Q[X[i,Y[i]]] += eta_v * ( R[i] - Q[X[i,Y[i]]] );
                
    }

}