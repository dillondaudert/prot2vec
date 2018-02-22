# visualization functions
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_arrays(sample, output):
    """
    Create a figure with two plots: the original sample, and a corresponding
    prediction.
    """

    assert len(sample.shape) == 2 and len(output.shape) == 2

    cmap = mpl.colors.ListedColormap(['purple', 'white', 'black', 'orange'])
    bounds = [-2.5, -.5, .5, 1.5, 2.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # calculate the difference. Since the output may be longer, align the
    # difference to beginning of the arrays
    diff = sample - output[:sample.shape[0], :sample.shape[1]]
    diff *= -2.
    # find the areas where the prediction doesn't match the sample
    is_diff = diff != 0.
    # change those locations in output so they plot to the correct color
    output[:sample.shape[0],:sample.shape[1]][is_diff] = diff[is_diff]
    # plot images using the appropriate color map

    fig = plt.figure(1)
    plt.subplot(121)
    plt.imshow(sample, cmap=cmap, norm=norm)

    plt.subplot(122)
    img2 = plt.imshow(output, cmap=cmap, norm=norm)

    bar = plt.colorbar(img2, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-2, 0, 1, 2])
    bar.ax.set_yticklabels(["False 0", "True 0", "True 1", "False 1"])


    plt.show()
