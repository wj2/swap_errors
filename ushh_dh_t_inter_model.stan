functions {
  real color_likelihood(vector resp, vector cu, vector cl,
			matrix mu_u, matrix mu_d_u,
			matrix mu_l, matrix mu_d_l,
			int cue, vector vars,
			vector nu, vector intercept_up, vector intercept_down) {
    real out;
    out = (cue*(student_t_lpdf(resp | nu, mu_u*cu + mu_d_l*cl + intercept_up,
			       sqrt(vars))) 
	   + (1-cue)*(student_t_lpdf(resp | nu, mu_l*cl + mu_d_u*cu + intercept_down,
				     sqrt(vars))));
    return out;
  }

  vector color_rng(vector cu, vector cl, matrix mu_u, matrix mu_d_u,
		   matrix mu_l, matrix mu_d_l, int cue, vector vars,
		   vector nu, vector intercept_up, vector intercept_down) {
    vector[dims(vars)[1]] out;
    out = to_vector(student_t_rng(nu, cue*(mu_u*cu + mu_d_l*cl + intercept_up)
				  + (1-cue)*(mu_l*cl + mu_d_u*cu + intercept_down),
				  sqrt(vars)));
    return out;
  }
}

data {
  int<lower=0> T;  // number of trials
  int<lower=1> N;  // number of neurons
  int<lower=1> K;  // number of color 
  vector[N] y[T];  // neural activity
  int<lower=0,upper=1> cue[T]; // upper or lower indicator
  vector[K] C_u[T]; // upper color, 2-hot simplex vector
  vector[K] C_l[T]; // lower color, 2-hot simplex vector
  vector[3] p[T]; // probabilities
  int<lower=1,upper=2> type[T]; // pro or retro tria ... must be all 0 if is_joint = False
  int<lower=0,upper=1> is_joint; // whether to use two separate simplices
}

transformed data {
  vector[3] log_p[T];
  log_p = log(p);
}

parameters {
  matrix[N,K] mu_c_pr; // cued prior
  matrix[N,K] mu_d_pr; // distractor prior
  matrix[N,K] mu_u; // upper color mean
  matrix[N,K] mu_l; // lower color mean
  matrix[N,K] mu_d_u; // upper distractor
  matrix[N,K] mu_d_l; // lower distractor

  real mu_u_type[2, N, K];
  real mu_l_type[2, N, K];
  real mu_d_u_type[2, N, K];
  real mu_d_l_type[2, N, K];
  
  vector[N] intercept_up;
  vector[N] intercept_down;
  vector<lower=2>[N] nu; // DOF for student T
  vector<lower=0>[N] vars_raw;
  vector<lower=0>[N] pr_var_c; // prior variances
  vector<lower=0>[N] pr_var_d; // prior variances
  vector<lower=0>[N] pr_var_c_ul; // prior variances
  vector<lower=0>[N] pr_var_d_ul; // prior variances
  simplex[3] p_err[is_joint ? 2:1]; // probability of a cue, spatial, or no error
}

transformed parameters {
  vector[3] log_p_err[is_joint ? 2:1];
  vector<lower=0>[N] vars;
  vars = vars_raw .* (nu - 2) ./ nu;
  
  log_p_err = log(p_err);
}

model {
  real trg[T];
  real lp[3];
  real lp_swp[3];
  real nom;

  matrix[N, K] mu_u_use;
  matrix[N, K] mu_l_use;
  matrix[N, K] mu_d_u_use;
  matrix[N, K] mu_d_l_use;

  // prior
  for (k in 1:K){
    mu_c_pr[:,k] ~ normal(0, 5);
    mu_d_pr[:,k] ~ normal(0, 5);
  }

  pr_var_c ~ inv_gamma(2,1);
  pr_var_d ~ inv_gamma(2,1);
  pr_var_c_ul ~ inv_gamma(2,1);
  pr_var_d_ul ~ inv_gamma(2,1);

  for (k in 1:K){
    mu_u[:,k] ~ normal(mu_c_pr[:,k], pr_var_c);
    mu_l[:,k] ~ normal(mu_c_pr[:,k], pr_var_c);
    mu_d_u[:,k] ~ normal(mu_d_pr[:,k], pr_var_d);
    mu_d_l[:,k] ~ normal(mu_d_pr[:,k], pr_var_d);

    for (i in 1:2) {
      mu_u_type[i, :, k] ~ normal(mu_u[:, k], pr_var_c_ul);
      mu_l_type[i, :, k] ~ normal(mu_l[:, k], pr_var_c_ul);
      mu_d_u_type[i, :, k] ~ normal(mu_d_u[:, k], pr_var_d_ul);
      mu_d_l_type[i, :, k] ~ normal(mu_d_l[:, k], pr_var_d_ul);
    }
  }
  intercept_up ~ normal(0, 5);
  intercept_down ~ normal(0, 5);
  
  vars_raw ~ inv_gamma(2, 1);
  nu ~ gamma(2, .1);

  for (t in 1:size(p_err)){
    p_err[t] ~ dirichlet(rep_vector(1.5,3));
  }

  // likelihood
  for (n in 1:T) {
    mu_u_use = to_matrix(mu_u_type[type[n]]);
    mu_l_use = to_matrix(mu_l_type[type[n]]);
    mu_d_u_use = to_matrix(mu_d_u_type[type[n]]);
    mu_d_l_use = to_matrix(mu_d_l_type[type[n]]);
    
    nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_d_u_use,
			   mu_l_use, mu_d_l_use, cue[n], vars, nu,
			   intercept_up, intercept_down);
    lp[1] = log_p[n][1] + nom;
    // spatial errors
    lp_swp[1] = (log_p_err[type[n]][1]
		 + color_likelihood(y[n], C_l[n], C_u[n], mu_u_use,
				    mu_d_u_use,
				    mu_l_use, mu_d_l_use, cue[n], vars, nu,
				    intercept_up, intercept_down));
    // cue errors
    lp_swp[2] = (log_p_err[type[n]][2]
		 + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_d_u_use,
				    mu_l_use, mu_d_l_use, 1 - cue[n], vars, nu,
				    intercept_up, intercept_down));
    // no errors
    lp_swp[3] = log_p_err[type[n]][3] + nom;
		 
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

    matrix[N, K] mu_u_use;
    matrix[N, K] mu_l_use;
    matrix[N, K] mu_d_u_use;
    matrix[N, K] mu_d_l_use;

    mu_u_use = to_matrix(mu_u_type[type[n]]);
    mu_l_use = to_matrix(mu_l_type[type[n]]);
    mu_d_u_use = to_matrix(mu_d_u_type[type[n]]);
    mu_d_l_use = to_matrix(mu_d_l_type[type[n]]);

    
    lp[1] = log_p[n][1] + color_likelihood(y[n], C_u[n], C_l[n],
					   mu_u_use, mu_d_u_use,
					   mu_l_use, mu_d_l_use, cue[n],
					   vars, nu,
					   intercept_up, intercept_down);
    // spatial errors
    lp_swp[1] = (log_p_err[type[n]][1]
		 + color_likelihood(y[n], C_l[n], C_u[n], mu_u_use,
				    mu_d_u_use,
				    mu_l_use, mu_d_l_use, cue[n], vars, nu,
				    intercept_up, intercept_down));
    // cue errors
    lp_swp[2] = (log_p_err[type[n]][2]
		 + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use,
				    mu_d_u_use,
				    mu_l_use, mu_d_l_use, 1 - cue[n],
				    vars, nu,
				    intercept_up, intercept_down));
    // no errors
    lp_swp[3] = (log_p_err[type[n]][3]
		 + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_d_u_use,
				    mu_l_use,
				    mu_d_l_use, cue[n], vars, nu,
				    intercept_up, intercept_down));
		 
    // swap errors (spatial and cue)                    
    lp[2] = log_p[n][2] + log_sum_exp(lp_swp);   
    lp[3] = log_p[n][3] + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use,
					   mu_d_u_use, mu_l_use,
					   mu_d_l_use, cue[n], vars, nu,
					   intercept_up, intercept_down);
    log_lik[n] = log_sum_exp(lp);

    // generative model

    trl_type = categorical_rng(p[n]);
    swp_type = categorical_rng(p_err[type[n]]);

    if (trl_type==1)
    {
      err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
			     mu_d_l_use,
			     cue[n], vars, nu, intercept_up, intercept_down);
    }
    else if (trl_type==2)
    {
      if (swp_type==1)
      {
        err_hat[n] = color_rng(C_l[n], C_u[n], mu_u_use, mu_d_u_use, mu_l_use,
			       mu_d_l_use,
			       cue[n], vars, nu, intercept_up, intercept_down);
      }
      else if (swp_type==2)
      {
        err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
			       mu_d_l_use,
			       1 - cue[n], vars, nu, intercept_up, intercept_down);
      }
      else if (swp_type==3)
      {
        err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
			       mu_d_l_use,
			       cue[n], vars, nu, intercept_up, intercept_down);
      }
    }
    else if (trl_type==3)
    {
      err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
			     mu_d_l_use,
			     cue[n], vars, nu, intercept_up, intercept_down);
    }

  }
}
