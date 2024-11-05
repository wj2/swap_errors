functions {
  real color_likelihood(vector resp, vector cu, vector cl,
			matrix mu_u, matrix mu_l, vector vars, vector nu) {
    real out;
    out = student_t_lpdf(resp | nu, mu_u*cu + mu_l*cl, sqrt(vars));
    return out;
  }
 
  vector color_rng(vector cu, vector cl, matrix mu_u,
		   matrix mu_l, vector vars, vector nu) {
    vector[dims(vars)[1]] out;
    out = to_vector(student_t_rng(nu, mu_u*cu + mu_l*cl, sqrt(vars)));
    return out;
  }
}

data {
  int<lower=0> T;  // number of trials
  int<lower=1> N;  // number of neurons
  int<lower=1> K;  // number of color 
  vector[N] y[T];  // neural activity
  vector[K] C_u[T]; // upper color, 2-hot simplex vector
  vector[K] C_l[T]; // lower color, 2-hot simplex vector
  vector[3] p[T]; // probabilities

  int<lower=0> T_ex;  // number of trials
  vector[N] y_ex[T_ex];  // neural activity
  vector[K] C_u_ex[T_ex]; // upper color, 2-hot simplex vector
  vector[K] C_l_ex[T_ex]; // lower color, 2-hot simplex vector
  vector[3] p_ex[T_ex]; // probabilities
}

transformed data {
  vector[3] log_p[T];
  vector[3] log_p_ex[T_ex];
  log_p = log(p);
  log_p_ex = log(p_ex);
}

parameters {
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean

  vector<lower=2>[N] nu; // DOF for student T
  vector<lower=0>[N] vars_raw;
}

transformed parameters {
  vector<lower=0>[N] vars;
  vars = vars_raw .* (nu - 2) ./ nu;
}


model {
  real trg[T];
  real lp[3];
  real lp_swp[2];
  real nom;

  // prior
  for (k in 1:K){
    mu_u[:,k] ~ std_normal();
    mu_l[:,k] ~ std_normal();
  }
  
  vars_raw ~ inv_gamma(2, 1);
  nu ~ gamma(2, .1);

  // likelihood
  for (n in 1:T) {
    nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l, vars, nu);
    trg[n] = nom;
  }
  target += sum(trg);
}

generated quantities{
  real log_lik[T];
  real log_lik_ex[T_ex];
  vector[N] err_hat[T];

  for (n in 1:T) {
    // loglihood
    real lp[3];
    real lp_swp[2];
    int trl_type;
    int swp_type;

    log_lik[n] = color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l,vars, nu);

    // GENERATION
    err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, vars, nu);
  }
  
  // EXCLUDED
  for (n in 1:T_ex) {
    // loglihood
    log_lik_ex[n] = color_likelihood(y_ex[n], C_u_ex[n], C_l_ex[n],
				     mu_u, mu_l, vars, nu);
  }    
}
