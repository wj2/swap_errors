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
  // vector[N] logit_vars; // variances
  vector<lower=0>[N] vars;
  real logits; // log-odds of being a spatial error trial
}

transformed parameters {
  real log_p_spa;
  real log_p_cue;
  real<lower=0,upper=1> p_spa;

  // log-odds -> log-probabilities
  log_p_spa = logits - log(1+exp(logits)); 
  log_p_cue = -logits - log(1+exp(-logits)); // log(1-p)

  // log-odds -> probabilities
  p_spa = exp(logits) / (1+exp(logits));

}

model {
  real trg[T];
  real lp[3];
  real lp_swp[2];

  // prior
  for (k in 1:K){
    mu_u[:,k] ~ std_normal();
    mu_l[:,k] ~ std_normal();
  }
  
  vars ~ inv_gamma(2,1);
  logits ~ normal(0,1);

  // likelihood
  for (n in 1:T) {
    lp[1] = log_p[n][1] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    // spatial errors
    lp_swp[1] = log_p_spa + normal_lpdf(y[n] | mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars));
    // cue errors
    lp_swp[2] = log_p_cue + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    // swap errors (spatial and cue)
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);
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
    real lp_swp[2];
    int trl_type;
    int swp_type;

    lp[1] = log_p[n][1] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    // spatial errors
    lp_swp[1] = log_p_spa + normal_lpdf(y[n] | mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars));
    // cue errors
    lp_swp[2] = log_p_cue + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    // swap errors (spatial and cue)
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);
    lp[3] = log_p[n][3] + normal_lpdf(y[n] | mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars));
    log_lik[n] = log_sum_exp(lp);

    // generative model

    trl_type = categorical_rng(p[n]);
    swp_type = bernoulli_rng(p_spa);

    if (trl_type==1)
      err_hat[n] = to_vector(normal_rng( mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars)));
    else if (trl_type==2)
      if (swp_type==1)
        err_hat[n] = to_vector(normal_rng( mu_u*C_l[n] + mu_l*C_u[n], sqrt(vars)));
      else
        err_hat[n] = to_vector(normal_rng( mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars)));
    else if (trl_type==3)
      err_hat[n] = to_vector(normal_rng( mu_u*C_u[n] + mu_l*C_l[n], sqrt(vars)));

  }
}