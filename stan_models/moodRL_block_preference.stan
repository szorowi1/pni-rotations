data {

    // Metadata
    int  N;   
    int  M;
    
    // Data
    matrix[N, M]  X;       // Design matrix
    int           Y[N];    // Number of Block 2 > Block 1 trials
    int           T[N];    // Total number of trials
    
}
parameters {

    vector[M] beta;

}
model {

    // Priors
    beta ~ normal(0, 1);

    // Likelihood
    Y ~ binomial( T, inv_logit( X * beta ) );

}