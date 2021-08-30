data {
  int<lower=0> T;  // number of trials
  int<lower=1> N;  // number of neurons
  int<lower=1> K;  // number of color 
  vector[N] y[T];  // neural activity
  int<lower=0,upper=1> cue[T]; // upper or lower indicator
  vector[K] C_u[T]; // upper color, 2-hot simplex vector
  vector[K] C_l[T]; // lower color, 2-hot simplex vector
}
  
transformed data {
  real<upper=0> neg_log_K;
  neg_log_K = -log(K);
}

parameters {
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean
  matrix[N,K] mu_d_u; // upper distractor
  matrix[N,K] mu_d_l; // lower distractor
}

model {
  real trg[T];
  // prior
  for (k in 1:K){
    mu_u[:,k] ~ std_normal();
    mu_l[:,k] ~ std_normal();
    mu_d_u[:,k] ~ std_normal();
    mu_d_l[:,k] ~ std_normal();
  }
  // likelihood
  for (n in 1:T) {
    trg[n] = cue[n]*(normal_lpdf(y[n] | mu_u*C_u[n] + mu_d_l*C_l[n] ,1)) + (1-cue[n])*(normal_lpdf(y[n] | mu_l*C_l[n] + mu_d_u*C_u[n], 1));
  }
  target += sum(trg);
}
