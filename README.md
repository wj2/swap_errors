
## Dependencies 
This code relies on the usual python scientific computing environment and on
one of my other repositories: [general](https://github.com/wj2/general-neural).

Once you have these two repositories on your path and all the additional python
packages downloaded, the code here should work. 

## Generating figures from Alleman et al. (2023)
This code underlies all of the figures in Alleman et al. (2023) https://doi.org/10.1101/2023.10.09.561584 . We are in the process of making some of the underlying data available on figshare. The full raw data can be requested from the authors of the first paper arising from this dataset: https://doi.org/10.1038/s41586-021-03390-w . There are brief instructions for navigating the code and reproducing the figures. Please feel free to contact [me](wjeffreyjohnston@gmail.com) if you have any questions. 

### Reproducing the analyses
The main two files for running the analyses are ```figures.py``` and 
```figures.conf``` -- the former contains all the code and the latter contains 
all the relevant parameters in a hierarchical configuration file. So, first 
change any relevant paths or parameters in ```figures.conf```. The general structure is that the figure file contains subclasses of a ```SwapFigure``` object. These subclasses have methods prefixed with the word ```panel``` that generate (or in cases where the simulations are relatively expensive, only plot) the panels of all the figures in the paper. For example, to plot some of the panels from Figure 2,
```
import swap_errors.figures as swf

fig_key = retro_lm'
fig = swf.RetroLMFigure(data=fig_data.get(fig_key))
fig.panel_color_cue()
fig_data[fig_key] = fig.get_data()
``` 

