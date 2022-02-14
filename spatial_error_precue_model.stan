data {
  int<lower=0> T;  // number of trials
  int<lower=1> N;  // number of neurons
  int<lower=1> K;  // number of color 
  vector[N] y[T];  // neural activity
  int<lower=0,upper=1> cue[T]; // upper or lower indicator
  vector[K] C_u[T]; // upper color, 2-hot simplex vector
  vector[K] C_l[T]; // lower color, 2-hot simplex vector
  vector[3] p[T]; // probabilities
}

transformed data {
  vector[3] log_p[T];
  log_p = log(p);
}

parameters {
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean
  vector<lower=0>[N] vars;
}

model {
  real trg[T];
  real lp[3];
  // prior
  for (k in 1:K){
    mu_u[:,k] ~ std_normal();
    mu_l[:,k] ~ std_normal();
  }
  
  vars ~ inv_gamma(2,1);

  // hierarchical mus
  // simultaneous behavior

  // likelihood
  for (n in 1:T) {
    lp[1] = log_p[n][1] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    lp[2] = log_p[n][2] + normal_lpdf(y[n] | mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars));
    lp[3] = log_p[n][3] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    trg[n] = log_sum_exp(lp);
  }
  target += sum(trg);
}

generated quantities{
  real log_lik[T];
  vector[N] err_hat[T];
  for (n in 1:T) {
    // loglihood
    real lp[3];
    int trl_type;

    lp[1] = log_p[n][1] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    lp[2] = log_p[n][2] + normal_lpdf(y[n] | mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars));
    lp[3] = log_p[n][3] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    log_lik[n] = log_sum_exp(lp);

    // generative model

    trl_type = categorical_rng(p[n]);

    if (trl_type==1)
      err_hat[n] = to_vector(normal_rng( mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars)));
    else if (trl_type==2)
      err_hat[n] = to_vector(normal_rng( mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars)));
    else if (trl_type==3)
      err_hat[n] = to_vector(normal_rng( mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars)));

  }
}