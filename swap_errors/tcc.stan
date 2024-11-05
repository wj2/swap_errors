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
  int<lower=0> K; // number of bins
  
  vector<lower=0>[K] abs_errors;
  real<lower=0> sigma_dist;
  vector<lower=0>[K] bin_cents;
}

parameters {
  real<lower=0> dprime;
}

model {
  dprime ~ normal(0, prior_dprime);
  
  similarity_function = exp(-.5*(bin_cents/sigma_dist)**2);
  scaled_sf = dprime*similarity_function;

  normal(scaled_sf, 1);
}

generated quantities{
  real log_lik[T];
  vector[N] err_hat[T];
}
