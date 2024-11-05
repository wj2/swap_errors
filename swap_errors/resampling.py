
import numpy as np
import arviz as az

model_manifest = {'observed_data':'y',
                  'log_likelihood':['log_lik', 'log_lik_ex'],
                  'posterior_predictive':'err_hat'}

class SwapModelSamplingWrapper(az.PyStan2SamplingWrapper):

    def sel_observations(self, idx):
        odict = self.idata_orig
        n_trls = odict['T'] 

        mask = np.ones(n_trls, dtype=bool)
        mask[idx] = False

        new_dict = {}
        new_dict.update(odict)
        new_dict['T'] = np.sum(mask)
        new_dict['y'] = odict['y'][mask]
        new_dict['C_u'] = odict['C_u'][mask]
        new_dict['C_l'] = odict['C_l'][mask]
        new_dict['cue'] = odict['cue'][mask]
        new_dict['p'] = odict['p'][mask]
        new_dict['type'] = odict['type'][mask]

        new_dict['T_ex'] = np.sum(~mask)
        new_dict['y_ex'] = odict['y'][~mask]
        new_dict['C_u_ex'] = odict['C_u'][~mask]
        new_dict['C_l_ex'] = odict['C_l'][~mask]
        new_dict['cue_ex'] = odict['cue'][~mask]
        new_dict['p_ex'] = odict['p'][~mask]
        new_dict['type_ex'] = odict['type'][~mask]
        
        return new_dict, 'log_lik_ex'
        
