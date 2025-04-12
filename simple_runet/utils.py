import os
import json
import numpy as np
import psutil
import matplotlib.pyplot as plt



def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss/(1e3)**3
    print('Memory Usage in Gb: {:.2f}'.format(mem))  # in GB 
    return mem


def plot0(y_true, y_pred, index, layer, 
          tstep0=1, fs=12, figsize=(10, 4.5), vmin=0, vmax=1, error_vmin=-0.2, error_vmax=0.2, 
          aspect=15, shrink=1, dpi=300, cmap='jet', error_cmap='coolwarm', 
          space_adjust={'wspace': None, 'hspace': None}, fname=None):
    time_steps = y_true.shape[1]
    nrows = 3
    ncols = time_steps
    kwargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    err_kwargs = {'vmin': error_vmin, 'vmax': error_vmax, 'cmap': error_cmap}
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(**space_adjust)
    for row in range(3):
        for i_tstep in range(time_steps):
            if row == 0:
                im = axs[row,i_tstep].imshow(y_true[index, i_tstep, :, :, layer], **kwargs)
                axs[row,i_tstep].set_title('Year {} '.format(i_tstep+tstep0), fontsize=fs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            elif row == 1:
                im = axs[row,i_tstep].imshow(y_pred[index, i_tstep, :, :, layer], **kwargs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            else: 
                error = y_true[index, i_tstep, :, :, layer] - y_pred[index, i_tstep, :, :, layer]
                im = axs[row,i_tstep].imshow(error, **err_kwargs)
                ticks = [error_vmin, 0.0, error_vmax]
            axs[row,i_tstep].set_xticks([])
            axs[row,i_tstep].set_yticks([])
        fig.colorbar(im, ax=axs[row, :], ticks=ticks, pad=.009, aspect=aspect, shrink=shrink)
    axs[0,0].set_ylabel('True', fontsize=fs)
    axs[1,0].set_ylabel('Pred', fontsize=fs)
    axs[2,0].set_ylabel('Error', fontsize=fs)

    if fname == None:
        pass
    else:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
     
    plt.show()
