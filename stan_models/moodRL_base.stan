data {

    // Metadata
    int  T;                             // number of trials
    real q;                             // Initial Q value
    
    // Data
    int<lower= 1, upper=9>  X[T, 2];    // presented machines
    int<lower=-1, upper=2>  Y[T];       // choice data
    real R[T];                          // outcome data
    
}
parameters {

    real<lower=0, upper=20> Beta;
    real<lower=0, upper=1> Eta_v;

}
model {

    // Generated data
    vector[9] Q; 
    Q = rep_vector(q, 9);
    
    // Priors
    Beta ~ gamma(4.83, 1.37);
    Eta_v ~ beta(0.007, 0.018);
    
    // Likelihood
    for (i in 1:T) {

        // Choice phase
        Y[i] ~ categorical_logit( Beta * Q[X[i]] );

        // Learning phase
        Q[X[i,Y[i]]] += Eta_v * ( R[i] - Q[X[i,Y[i]]] );
                
    }

}
