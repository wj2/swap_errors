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
  vector[K] C_resp[T]; // color reported
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

  real mu_u_type[3, N, K];
  real mu_l_type[3, N, K];
  real mu_d_u_type[2, N, K];
  real mu_d_l_type[2, N, K];
  
  vector[N] intercept_up;
  vector[N] intercept_down;
  matrix[3, N] i_up_type;
  matrix[3, N] i_down_type;

  vector<lower=2>[N] nu; // DOF for student T
  vector<lower=0>[N] vars_raw;
  vector<lower=0>[N] pr_var_c; // prior variances
  vector<lower=0>[N] pr_var_d; // prior variances
  vector<lower=0>[N] pr_var_c_ul; // prior variances
  vector<lower=0>[N] pr_var_d_ul; // prior variances
  vector<lower=0>[N] pr_var_i_up;
  vector<lower=0>[N] pr_var_i_down;

  simplex[3] p_err[is_joint ? 2:1]; // probability of a cue, spatial, or no error
  simplex[2] p_guess_err[is_joint ? 2:1]; // probability of a cue, spatial, or no error
}

transformed parameters {
  vector[3] log_p_err[is_joint ? 2:1];
  vector[2] log_p_guess_err[is_joint ? 2:1];
  vector<lower=0>[N] vars;
  vars = vars_raw .* (nu - 2) ./ nu;
  
  log_p_err = log(p_err);
  log_p_guess_err = log(p_guess_err);
}

model {
  real trg[T];
  real lp[3];
  real lp_swp[3];
  real lp_guess[2];
  real nom;

  matrix[N, K] mu_u_use;
  matrix[N, K] mu_l_use;
  matrix[N, K] mu_d_u_use;
  matrix[N, K] mu_d_l_use;
  vector[N] i_up_use;
  vector[N] i_down_use;

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

    for (i in 1:3) {
      mu_u_type[i, :, k] ~ normal(mu_u[:, k], pr_var_c_ul);
      mu_l_type[i, :, k] ~ normal(mu_l[:, k], pr_var_c_ul);
      if (i < 3) {
	mu_d_u_type[i, :, k] ~ normal(mu_d_u[:, k], pr_var_d_ul);
	mu_d_l_type[i, :, k] ~ normal(mu_d_l[:, k], pr_var_d_ul);
      }
    }
  }
  intercept_up ~ normal(0, 5);
  intercept_down ~ normal(0, 5);
  pr_var_i_up ~ inv_gamma(2,1);
  pr_var_i_down ~ inv_gamma(2,1);
  for (i in 1:3) {
    i_up_type[i] ~ normal(intercept_up, pr_var_i_up);
    i_down_type[i] ~ normal(intercept_down, pr_var_i_down);
  }
  
  vars_raw ~ inv_gamma(2, 1);
  nu ~ gamma(2, .1);

  for (t in 1:size(p_err)){
    p_err[t] ~ dirichlet(rep_vector(1.5,3));
    p_guess_err[t] ~ dirichlet(rep_vector(1.5, 2));
  }

  // likelihood
  for (n in 1:T) {
    mu_u_use = to_matrix(mu_u_type[type[n]]);
    mu_l_use = to_matrix(mu_l_type[type[n]]);
    i_up_use = i_up_type[type[n]]';
    i_down_use = i_down_type[type[n]]';
    if (type[n] < 3) { 
      mu_d_u_use = to_matrix(mu_d_u_type[type[n]]);
      mu_d_l_use = to_matrix(mu_d_l_type[type[n]]);
    
      nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_d_u_use,
			     mu_l_use, mu_d_l_use, cue[n], vars, nu,
			     i_up_use, i_down_use);
      lp[1] = log_p[n][1] + nom;
      // spatial errors
      lp_swp[1] = (log_p_err[type[n]][1]
		   + color_likelihood(y[n], C_l[n], C_u[n], mu_u_use,
				      mu_d_u_use,
				      mu_l_use, mu_d_l_use, cue[n], vars, nu,
				      i_up_use, i_down_use));
      // cue errors
      lp_swp[2] = (log_p_err[type[n]][2]
		   + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_d_u_use,
				      mu_l_use, mu_d_l_use, 1 - cue[n], vars, nu,
				      i_up_use, i_down_use));
      // no errors
      lp_swp[3] = log_p_err[type[n]][3] + nom;
		 
      // swap errors (spatial and cue)                    
      lp[2] = log_p[n][2] + log_sum_exp(lp_swp);

      // guess
      if (cue[n] == 1) {
	lp_guess[1] = (log_p_guess_err[type[n]][1]
		       + color_likelihood(y[n], C_resp[n], C_l[n],
					  mu_u_use, mu_d_u_use,
					  mu_l_use, mu_d_l_use, cue[n],
					  vars, nu,
					  i_up_use, i_down_use));
      } else {
	lp_guess[1] = (log_p_guess_err[type[n]][1]
		       + color_likelihood(y[n], C_u[n], C_resp[n],
					  mu_u_use, mu_d_u_use,
					  mu_l_use, mu_d_l_use, cue[n],
					  vars, nu,
					  i_up_use, i_down_use));
      }
      lp_guess[2] = log_p_guess_err[type[n]][2] + nom;
		    
      lp[3] = log_p[n][3] + log_sum_exp(lp_guess);
      trg[n] = log_sum_exp(lp);
    } else {
    
      nom = color_likelihood(y[n], C_u[n], C_l[n], mu_u_use, mu_u_use*0,
			     mu_l_use, mu_l_use*0, cue[n], vars, nu,
			     i_up_use, i_down_use);
      trg[n] = nom;
    }
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
    real lp_guess[2];
    int trl_type;
    int swp_type;
    int guess_type;
    real nom;

    matrix[N, K] mu_u_use;
    matrix[N, K] mu_l_use;
    matrix[N, K] mu_d_u_use;
    matrix[N, K] mu_d_l_use;

    vector[N] i_up_use;
    vector[N] i_down_use;

    mu_u_use = to_matrix(mu_u_type[type[n]]);
    mu_l_use = to_matrix(mu_l_type[type[n]]);
    i_up_use = i_up_type[type[n]]';
    i_down_use = i_down_type[type[n]]';

    if (type[n] < 3) {
      mu_d_u_use = to_matrix(mu_d_u_type[type[n]]);
      mu_d_l_use = to_matrix(mu_d_l_type[type[n]]);

      nom = color_likelihood(y[n], C_u[n], C_l[n],
			     mu_u_use, mu_d_u_use,
			     mu_l_use, mu_d_l_use, cue[n],
			     vars, nu,
			     i_up_use, i_down_use);
      lp[1] = log_p[n][1] + nom;
      // spatial errors
      lp_swp[1] = (log_p_err[type[n]][1]
		   + color_likelihood(y[n], C_l[n], C_u[n], mu_u_use,
				      mu_d_u_use,
				      mu_l_use, mu_d_l_use, cue[n], vars, nu,
				      i_up_use, i_down_use));
      // cue errors
      lp_swp[2] = (log_p_err[type[n]][2]
		   + color_likelihood(y[n], C_u[n], C_l[n], mu_u_use,
				      mu_d_u_use,
				      mu_l_use, mu_d_l_use, 1 - cue[n],
				      vars, nu,
				      i_up_use, i_down_use));
      // no errors
      lp_swp[3] = log_p_err[type[n]][3] + nom;
		 
      // swap errors (spatial and cue)                    
      lp[2] = log_p[n][2] + log_sum_exp(lp_swp);   
      // guess
      if (cue[n] == 1) {
	lp_guess[1] = (log_p_guess_err[type[n]][1]
		       + color_likelihood(y[n], C_resp[n], C_l[n],
					  mu_u_use, mu_d_u_use,
					  mu_l_use, mu_d_l_use, cue[n],
					  vars, nu,
					  i_up_use, i_down_use));
      } else {
	lp_guess[1] = (log_p_guess_err[type[n]][1]
		       + color_likelihood(y[n], C_u[n], C_resp[n],
					  mu_u_use, mu_d_u_use,
					  mu_l_use, mu_d_l_use, cue[n],
					  vars, nu,
					  i_up_use, i_down_use));
      }
      lp_guess[2] = log_p_guess_err[type[n]][2] + nom;
		    
      lp[3] = log_p[n][3] + log_sum_exp(lp_guess);
      log_lik[n] = log_sum_exp(lp);
      // generative model

      trl_type = categorical_rng(p[n]);
      swp_type = categorical_rng(p_err[type[n]]);
      guess_type = categorical_rng(p_guess_err[type[n]]);
      
      if (trl_type==1) {
	err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
			       mu_d_l_use,
			       cue[n], vars, nu, i_up_use, i_down_use);
      } else if (trl_type==2) {
	
	if (swp_type==1) {
	  err_hat[n] = color_rng(C_l[n], C_u[n], mu_u_use, mu_d_u_use, mu_l_use,
				 mu_d_l_use,
				 cue[n], vars, nu, i_up_use, i_down_use);
	} else if (swp_type==2) {
	  err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
				 mu_d_l_use,
				 1 - cue[n], vars, nu, i_up_use, i_down_use);
	} else if (swp_type==3) {
	  err_hat[n] = color_rng(C_u[n], C_l[n], mu_u_use, mu_d_u_use, mu_l_use,
				 mu_d_l_use,
				 cue[n], vars, nu, i_up_use, i_down_use);
	}
      } else if (trl_type==3) {
	if (guess_type == 1) {
	  if (cue[n] == 1) {
	    err_hat[n] = color_rng(C_resp[n], C_l[n],
				   mu_u_use, mu_d_u_use,
				   mu_l_use, mu_d_l_use,
				   cue[n], vars, nu,
				   i_up_use, i_down_use);
	  } else {
	    err_hat[n] = color_rng(C_u[n], C_resp[n],
				   mu_u_use, mu_d_u_use,
				   mu_l_use, mu_d_l_use,
				   cue[n], vars, nu,
				   i_up_use, i_down_use);
	  }
	} else {
	  err_hat[n] = color_rng(C_u[n], C_l[n],
				 mu_u_use, mu_d_u_use,
				 mu_l_use, mu_d_l_use,
				 cue[n], vars, nu,
				 i_up_use, i_down_use);
	}
      }
    } else {
      log_lik[n] = color_likelihood(y[n], C_u[n], C_l[n],
				    mu_u_use, mu_u_use*0,
				    mu_l_use, mu_l_use*0, cue[n],
				    vars, nu,
				    i_up_use, i_down_use);
      err_hat[n] = color_rng(C_u[n], C_l[n],
			     mu_u_use, mu_d_u_use*0,
			     mu_l_use, mu_d_l_use*0,
			     cue[n], vars, nu, i_up_use, i_down_use);
    }
  }
}
