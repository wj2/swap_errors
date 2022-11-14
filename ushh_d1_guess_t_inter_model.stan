functions {
  real color_likelihood(vector resp, vector cu, vector cl,
			matrix mu_u, matrix mu_l, vector inter,
			vector vars, vector nu) {
    real out;
    out = student_t_lpdf(resp | nu, mu_u*cu + mu_l*cl + inter, sqrt(vars));
    return out;
  }
 
  vector color_rng(vector cu, vector cl, matrix mu_u,
		   matrix mu_l, vector inter, vector vars, vector nu) {
    vector[dims(vars)[1]] out;
    out = to_vector(student_t_rng(nu, mu_u*cu + mu_l*cl + inter, sqrt(vars)));
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
  vector[K] C_resp[T]; // 
  int<lower=0,upper=1> cue[T]; // upper or lower indicator
  vector[3] p[T]; // probabilities
  
}

transformed data {
  vector[3] log_p[T];
  log_p = log(p);
}

parameters {
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean
  vector[N] inter;

  vector<lower=2>[N] nu; // DOF for student T
  vector<lower=0>[N] vars_raw;
  simplex[2] p_err; // probability of a cue, spatial, or no error
  simplex[2] p_guess_err; // probability of a cue, spatial, or no error
}

transformed parameters {
  vector[2] log_p_err;
  vector[2] log_p_guess_err;
  vector<lower=0>[N] vars;
  vars = vars_raw .* (nu - 2) ./ nu;
  
  log_p_err = log(p_err);
  log_p_guess_err = log(p_guess_err);
}


model {
  real trg[T];
  real lp[3];
  real lp_swp[2];
  real lp_guess[2];
  real nom;
  real alpha; 

  // prior
  for (k in 1:K){
    mu_u[:,k] ~ normal(0, 5);
    mu_l[:,k] ~ normal(0, 5);
  }
  
  vars_raw ~ inv_gamma(2, 1);
  nu ~ gamma(2, .1);

  alpha = 1;
  p_err ~ dirichlet(rep_vector(alpha, 2));
  p_guess_err ~ dirichlet(rep_vector(alpha, 2));

  // likelihood
  for (n in 1:T) {
    nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l, inter, vars, nu);

    lp[1] = log_p[n][1] + nom;
    // spatial errors
    lp_swp[1] = (log_p_err[1]
		 + color_likelihood(y[n], C_l[n], C_u[n], mu_u, mu_l, inter, vars, nu));
    // no errors
    lp_swp[2] = log_p_err[2] + nom;
    // swap errors (spatial and cue)                    
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);

    if (cue[n] == 1) {
      lp_guess[1] = (log_p_guess_err[1]
		     + color_likelihood(y[n], C_resp[n], C_l[n], mu_u, mu_l,
					inter, vars, nu));
    } else {
      lp_guess[1] = (log_p_guess_err[1]
		     + color_likelihood(y[n], C_u[n], C_resp[n], mu_u, mu_l,
					inter, vars, nu));
    }
    lp_guess[2] = log_p_guess_err[2] + nom;
    lp[3] = log_p[n][3] + log_sum_exp(lp_guess);
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
    real lp_guess[2];
    int trl_type;
    int swp_type;
    int guess_type;
    real nom;

    nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u, mu_l,
			   inter, vars, nu);
    
    lp[1] = log_p[n][1] + nom;
    // spatial errors
    lp_swp[1] = (log_p_err[1] + color_likelihood(y[n], C_l[n], C_u[n], mu_u, mu_l,
						 inter, vars, nu));
    // no errors
    lp_swp[2] = log_p_err[2] + nom;
		 
    // swap errors (spatial and cue)                    
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);

    if (cue[n] == 1) {
      lp_guess[1] = (log_p_guess_err[1]
		     + color_likelihood(y[n], C_resp[n], C_l[n], mu_u, mu_l,
					inter, vars, nu));
    } else {
      lp_guess[1] = (log_p_guess_err[1]
		     + color_likelihood(y[n], C_u[n], C_resp[n], mu_u, mu_l,
					inter, vars, nu));
    }
    lp_guess[2] = log_p_guess_err[2] + nom;
    lp[3] = log_p[n][3] + log_sum_exp(lp_guess);
    log_lik[n] = log_sum_exp(lp);

    // generative model

    trl_type = categorical_rng(p[n]);
    swp_type = categorical_rng(p_err);
    guess_type = categorical_rng(p_guess_err);

    if (trl_type==1)
    {
      err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, inter, vars, nu);
    } else if (trl_type==2) {
      if (swp_type==1) {
        err_hat[n] = color_rng(C_l[n], C_u[n], mu_u, mu_l, inter, vars, nu);
      } else if (swp_type==2) {
        err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, inter, vars, nu);
      }
    } else if (trl_type==3) {
      if (guess_type == 1) {
	if (cue[n] == 1) {
	  err_hat[n] = color_rng(C_resp[n], C_l[n], mu_u, mu_l, inter, vars, nu);
	} else {
	  err_hat[n] = color_rng(C_u[n], C_resp[n], mu_u, mu_l, inter, vars, nu);
	}
      } else {
	err_hat[n] = color_rng(C_u[n], C_l[n], mu_u, mu_l, inter, vars, nu);
      }
    }
  }
}
