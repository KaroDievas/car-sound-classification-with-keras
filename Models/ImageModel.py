import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.tf_misfit import cwt


class ImageModel:
    """
    Currently supports only mono signal
    """
    def generate_morlet_scalogram(self, signal, image_path):
        axis = signal.ndim - 1
        signal_length = signal.shape[axis]
        t = np.linspace(0, signal_length, signal_length)

        f_min = 1
        f_max = signal_length
        scalogram = cwt(signal, 0.001, 5, f_min, f_max)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

        ax.pcolormesh(x, y, np.abs(scalogram), cmap='hot')
        ax.set_yscale('log')
        ax.set_ylim(f_min, f_max)
        ax.axis('off')
        ax.set_axis_off()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.savefig(image_path, bbox_inches='tight', format='png', pad_inches=0, transparent=True, dpi=100)
