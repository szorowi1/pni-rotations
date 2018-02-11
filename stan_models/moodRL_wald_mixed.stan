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
    int  B;                      // number of blocks
    int  C;                      // number of variables
    int subj_ix[T];              // index of subject for trial i
    int block_ix[T];             // index of block for trial i
    
    // Data
    vector[T] Z;                 // RT data, range [0, 3]
    real X[N,B,C];
        
}
parameters {

    // Group-level (hyper)parameters
    vector[C] mu_pr[2];
    vector<lower=0>[2] sigma; 

    // Subject-level parameters (raw)
    vector[N] gamma_pr[B];          // Drift rate
    vector[N] alpha_pr[B];          // Decision boundary
    vector<lower=0>[N] theta[B];    // Non-decision time
    
}
transformed parameters {

    // Subject-level parameters (transformed)
    vector<lower=0,upper=10>[N]    gamma[B];
    vector<lower=0,upper=5>[N]     alpha[B];
    
    for (j in 1:B) {
        gamma[j] = Phi_approx( to_matrix(X[:,j,:]) * mu_pr[1] + sigma[1] * gamma_pr[j] ) * 10;
        alpha[j] = Phi_approx( to_matrix(X[:,j,:]) * mu_pr[2] + sigma[2] * alpha_pr[j] ) * 5;
    }
    
}
model {

    // Generated data
    vector[T]  dr;           // Drift rate
    vector[T]  db;           // Decision boundary
    vector[T]  ndt;          // Non-decision time    

    // Hyperpriors
    mu_pr[1] ~ normal(0, 1);
    mu_pr[2] ~ normal(0, 1);
    sigma ~ gamma(1, 0.5);
    
    // Priors
    for (j in 1:B) {
        gamma_pr[j] ~ normal(0, 1);
        alpha_pr[j] ~ normal(0, 1);
        theta[j] ~ normal(0.5, 0.5);
    }
    
    // Precompute values.
    for (i in 1:T) {  
        dr[i] = gamma[block_ix[i], subj_ix[i]];
        db[i] = alpha[block_ix[i], subj_ix[i]];
        ndt[i] = theta[block_ix[i], subj_ix[i]];
    }
    
    // Likelihood.
    target += shifted_wald_lpdf( Z | dr, db, ndt );
    
}