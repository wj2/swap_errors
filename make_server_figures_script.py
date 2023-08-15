
import os
import swap_errors.figures as swf

if __name__ == "__main__":
    fig_data = {}
    save_fig_templates = ("fig-{}.svg", "fig-{}.pdf")
    os.chdir("/burg/home/wjj2109/analysis")

    fig_key = "pro_lm"
    fig = swf.ProLMFigure(data=fig_data.get(fig_key))
    fig.panel_color_cue()
    fig_data[fig_key] = fig.get_data()
    list(fig.save(sft.format(fig_key)) for sft in save_fig_templates)

    fig_key = "retro_lm"
    fig = swf.RetroLMFigure(data=fig_data.get(fig_key))
    fig.panel_color_cue()
    fig_data[fig_key] = fig.get_data()
    list(fig.save(sft.format(fig_key)) for sft in save_fig_templates)

    fig_key = "rsf"
    fig = swf.RetroSwapFigure(data=fig_data.get(fig_key))
    fig.panel_d1()
    fig.panel_d2()
    fig.panel_decoding()
    fig.panel_corr()
    fig.panel_rate_differences()
    fig_data[fig_key] = fig.get_data()
    list(fig.save(sft.format(fig_key)) for sft in save_fig_templates)

    fig_key = "psf"
    fig = swf.ProSwapFigure(data=fig_data.get(fig_key))
    fig.panel_d1()
    fig.panel_d2()
    fig.panel_d2_decoding()
    fig.panel_decoding_comparison()
    fig_data[fig_key] = fig.get_data()
    list(fig.save(sft.format(fig_key)) for sft in save_fig_templates)
