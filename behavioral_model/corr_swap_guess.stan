data {
//sizes 
  int<lower=0> T; // number of trials
  int<lower=1> S; // number of runs

  real<lower=0> report_var_var_mean;
  real<lower=0> report_var_var_var;
  real<lower=0> report_var_mean_mean;
  real<lower=0> report_var_mean_var;

  real<lower=0> swap_weight_var_mean;
  real<lower=0> swap_weight_var_var;
  real<lower=0> swap_weight_mean_mean;
  real<lower=0> swap_weight_mean_var;

  real<lower=0> guess_weight_var_mean;
  real<lower=0> guess_weight_var_var;
  real<lower=0> guess_weight_mean_mean;
  real<lower=0> guess_weight_mean_var;

  // actual data
  int<lower=1, upper=S> run_ind[T];
  vector[T] err;
  vector[T] dist_err;
  vector[T] dist_loc;
}

parameters {
  // prior-related
  real<lower=0> report_var_mean;
  real<lower=0> report_var_var;

  real swap_weight_mean;
  real<lower=0> swap_weight_var;

  real guess_weight_mean;
  real<lower=0> guess_weight_var;

  // data-related
  vector[S] report_var_raw;
  vector[S] swap_weight_raw;
  vector[S] guess_weight_raw;
}

transformed parameters {  
  vector<lower=0>[S] report_var;
  vector<lower=0>[S] outcome_sum;
  vector[S] swap_weight;
  vector[S] guess_weight;
  vector<lower=0, upper=1>[S] swap_prob;
  vector<lower=0, upper=1>[S] guess_prob;

  report_var = report_var_mean + report_var_raw*report_var_var;
  swap_weight = swap_weight_mean + swap_weight_raw*swap_weight_var;
  guess_weight = guess_weight_mean + guess_weight_raw*guess_weight_var;
  
  outcome_sum = exp(0) + exp(swap_weight) + exp(guess_weight);
  swap_prob = exp(swap_weight) ./ outcome_sum;
  guess_prob = exp(guess_weight) ./ outcome_sum;
}

model {
  // var declarations
  int run;
  vector[3] outcome_lps;
  
  // priors
  report_var_var ~ normal(report_var_var_mean, report_var_var_var);
  report_var_mean ~ normal(report_var_mean_mean, report_var_mean_var);

  swap_weight_var ~ normal(swap_weight_var_var, swap_weight_var_var);
  swap_weight_mean ~ normal(swap_weight_mean_mean, swap_weight_mean_var);

  guess_weight_var ~ normal(guess_weight_var_mean, guess_weight_var_var);
  guess_weight_mean ~ normal(guess_weight_mean_mean, guess_weight_mean_var);
  
  report_var_raw ~ normal(0, 1);
  swap_weight_raw ~ normal(0, 1);
  guess_weight_raw ~ normal(0, 1);

  // model
  for (t in 1:T) {
    run = run_ind[t];
    outcome_lps[1] = log(guess_prob[run]) + uniform_lpdf(err[t] | -pi(), pi());
    outcome_lps[2] = log(swap_prob[run])
      + normal_lpdf(dist_err[t] | 0, report_var[run]);
    outcome_lps[3] = log(1 - guess_prob[run] - swap_prob[run])
      + normal_lpdf(err[t] | 0, report_var[run]);
    
    target += log_sum_exp(outcome_lps);
  }
}

generated quantities {
  matrix[T, 3] outcome_lps;
  vector[T] log_lik;
  vector[T] err_hat;
  for (t in 1:T) {
    int run = run_ind[t];
    int outcome;
    real eh;
    outcome_lps[t, 1] = log(guess_prob[run])
      + uniform_lpdf(err[t] | -pi(), pi());
    outcome_lps[t, 2] = log(swap_prob[run])
      + normal_lpdf(dist_err[t] | 0, report_var[run]);
    outcome_lps[t, 3] = log(1 - guess_prob[run] - swap_prob[run])
      + normal_lpdf(err[t] | 0, report_var[run]);
    log_lik[t] = log_sum_exp(outcome_lps[t]);
    outcome = categorical_rng([guess_prob[run], swap_prob[run],
			       1 - guess_prob[run] - swap_prob[run]]');
    if (outcome == 1) {
      eh = uniform_rng(-pi(), pi());
    } else if (outcome == 2) {
      eh = normal_rng(dist_loc[t], report_var[run]);
    } else {
      eh = normal_rng(0, report_var[run]);
    }
    if (eh > pi()) {
      err_hat[t] = eh - 2*pi();
    } else if (eh < -pi()) {
      err_hat[t] = eh + 2*pi();
    } else {
      err_hat[t] = eh;
    }
  }
}
