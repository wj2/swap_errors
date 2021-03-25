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
}

parameters {
  // prior-related
  real<lower=0> report_var_mean;
  real<lower=0> report_var_var;

  real<lower=0> swap_weight_mean;
  real<lower=0> swap_weight_var;

  real<lower=0> guess_weight_mean;
  real<lower=0> guess_weight_var;

  real<lower=0, upper=1> off_err_mean;
  real<lower=0> off_err_var;

  real<lower=0, upper=1> swap_err_mean;
  real<lower=0> swap_err_var;

  // data-related
  vector[S] report_var_raw;
  vector[S] swap_weight_raw;
  vector[S] guess_weight_raw;
}

transformed parameters {  
  vector<lower=0>[S] report_var;
  vector<lower=0>[S] outcome_sum;
  vector<lower=0>[S] swap_weight;
  vector<lower=0>[S] guess_weight;
  vector<lower=0, upper=1>[S] swap_prob;
  vector<lower=0, upper=1>[S] guess_prob;

  report_var = report_var_mean + report_var_raw*report_var_var;
  swap_weight = swap_weight_mean + swap_weight_raw*swap_weight_var;
  guess_weight = guess_weight_mean + guess_weight_raw*guess_weight_var;
  
  outcome_sum = 1 + swap_weight + guess_weight;
  swap_prob = swap_weight ./ outcome_sum;
  guess_prob = guess_weight ./ outcome_sum;
}

model {
  // var declarations
  int run;
  vector[3] outcome_lps;
  
  // priors
  report_var_var ~ normal(report_var_var_mean, report_var_var_var);
  report_var_mean ~ normal(report_var_mean_mean, report_var_mean_var);

  swap_weight_var ~ normal(swap_weight_mean, swap_weight_var);
  swap_weight_mean ~ normal(swap_weight_mean, swap_weight_var);

  guess_weight_var ~ normal(guess_weight_mean, guess_weight_var);
  guess_weight_mean ~ normal(guess_weight_mean, guess_weight_var);
  
  report_var_raw ~ normal(0, 1);
  swap_weight_raw ~ normal(0, 1);
  guess_weight_raw ~ normal(0, 1);

  // model
  for (t in 1:T) {
    run = run_ind[t];

    outcome_lps[1] = log(guess_prob[run]) + uniform_lpdf(err | -pi(), pi());
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
    outcome_lps[t, 1] = log(guess_prob[run])
      + uniform_lpdf(err[t] | -pi(), pi());
    outcome_lps[t, 2] = log(swap_prob[run])
      + normal_lpdf(dist_err[t] | 0, report_var[run]);
    outcome_lps[t, 3] = log(1 - guess_prob[run] - swap_prob[run])
      + normal_lpdf(err[t] | 0, report_var[run]);
    log_lik[t] = log_sum_exp(outcome_lps[t]);
  }
}
