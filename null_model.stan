data {
  int<lower=0> T;  // number of trials
  int<lower=1> N;  // number of neurons
  int<lower=1> K;  // number of color 
  vector[N] y[T];  // neural activity
  int<lower=0,upper=1> cue[T]; // upper or lower indicator
  int<lower=1, upper=K> C_u[T]; // upper color
  int<lower=1, upper=K> C_l[T]; // lower color
}

transformed data {
  real<upper=0> neg_log_K;
  neg_log_K = -log(K);
}

parameters {
  vector[N] mu_u[K]; // upper color mean
  vector[N] mu_l[K]; // lower color mean
  vector[N] mu_d_u[K]; // upper distractor
  vector[N] mu_d_l[K]; // lower distractor
}


model {
  // prior
  for (k in 1:K){
    mu_u[k] ~ std_normal();
    mu_l[k] ~ std_normal();
    mu_d_u[k] ~ std_normal();
    mu_d_l[k] ~ std_normal();
  }
  // likelihood
  for (n in 1:T) {
    target += cue[n]*(normal_lpdf(y[n] | mu_u[C_u[n]]+mu_d_l[C_l[n]],1)) + (1-cue[n])*(normal_lpdf(y[n] | mu_l[C_l[n]]+mu_d_u[C_u[n]], 1));
  }
}
