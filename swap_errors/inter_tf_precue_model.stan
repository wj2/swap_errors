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
  vector[T] cue;
}

transformed data {
  vector[3] log_p[T];

  log_p = log(p);
}

parameters {
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean

  vector<lower=2>[N] nu; // DOF for student T
  vector<lower=0>[N] vars_raw;
  simplex[3] p_err; // probability of a cue, spatial, or no error
}

transformed parameters {
  vector[3] log_p_err;
  vector<lower=0>[N] vars;
  vars = vars_raw .* (nu - 2) ./ nu;
  
  log_p_err = log(p_err);
}


model {
  real trg[T];
  real lp[3];
  real lp_swp[3];
  real nom;

  // prior
  for (k in 1:K){
    mu_u[:,k] ~ std_normal();
    mu_l[:,k] ~ std_normal();
  }
  
  // vars_raw ~ inv_gamma(2, 1);
  vars_raw ~ normal(0, 10);
  // nu ~ gamma(2, .1);
  nu ~ normal(2, 20);

  p_err ~ dirichlet(rep_vector(1.5, 3));

  // likelihood
  for (n in 1:T) {
    nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l, vars, nu);

    lp[1] = log_p[n][1] + nom;
    // spatial errors
    lp_swp[1] = (log_p_err[1]
		 + color_likelihood(y[n], C_l[n], C_u[n], mu_u, mu_l, vars, nu));
    // spatial errors
    lp_swp[2] = (log_p_err[2]
		 + color_likelihood(y[n], (1 - cue[n])*C_u[n], cue[n]*C_l[n],
				    mu_u, mu_l, vars, nu));
    // no errors
    lp_swp[3] = log_p_err[3] + nom;
		 
    // swap errors (spatial and cue)                    
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);   
    lp[3] = log_p[n][3] + nom;
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
    real lp_swp[3];
    int trl_type;
    int swp_type;
    
    lp[1] = log_p[n][1] + color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l, vars, nu);
    // spatial errors
    lp_swp[1] = (log_p_err[1] + color_likelihood(y[n], C_l[n], C_u[n],
						 mu_u, mu_l, vars, nu));

    // forget error
    lp_swp[2] = (log_p_err[2] + color_likelihood(y[n], (1 - cue[n])*C_u[n], cue[n]*C_l[n],
						 mu_u, mu_l, vars, nu));
    // no errors
    lp_swp[3] = (log_p_err[3] + color_likelihood(y[n], C_u[n], C_l[n],
						 mu_u, mu_l, vars, nu));
		 
    // swap errors (spatial and cue)                    
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);
    lp[3] = log_p[n][3] + color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l,vars, nu);
    log_lik[n] = log_sum_exp(lp);

    // GENERATION
    trl_type = categorical_rng(p[n]);
    swp_type = categorical_rng(p_err);

    if (trl_type==1)
    {
      err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, vars, nu);
    }
    else if (trl_type==2)
    {
      if (swp_type==1)
      {
        err_hat[n] = color_rng(C_l[n], C_u[n], mu_u, mu_l, vars, nu);
      }

      else if (swp_type==2)
	err_hat[n] = color_rng((1 - cue[n])*C_u[n], cue[n]*C_l[n],
			       mu_u, mu_l, vars, nu);
      else if (swp_type==3)
      {
        err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, vars, nu);
      }
    }
    else if (trl_type==3)
    {
      err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, vars, nu);
    }
  }
}
