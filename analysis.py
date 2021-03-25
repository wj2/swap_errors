
import numpy as np
import general.neural_analysis as na

def decode_color(data, tbeg, tend, twindow, tstep, time_key='SAMPLES_ON_diode',
                 color_key='TargetTheta', regions=None, n_folds=10, **kwargs):
    pops, xs = data.get_populations(twindow, tbeg, tend, tstep,
                                    time_zero_field=time_key, skl_axes=True)
    regs = data[color_key]
    outs = []
    for i, pop in enumerate(pops):
        tcs = na.pop_regression_skl(pop, regs[i], n_folds, mean=False,
                                    **kwargs)
        outs.append(tcs)
    return outs, xs
    
