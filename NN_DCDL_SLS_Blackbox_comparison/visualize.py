import matplotlib.pyplot as plt
import numpy as np

def visualize_pic(pic_array, label_array, class_names, title):
    """ show 10 first  pictures """
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if pic_array.shape[3] == 1:
            plt.imshow(pic_array[i,:,:,0], cmap=plt.cm.Greys)
        elif pic_array.shape[3] == 3:
            plt.imshow(pic_array[i], cmap=plt.cm.Greys)
        else:
            raise ValueError('Picture should have 1 or 3 channels not'.format(pic_array.shape[3]))
        plt.xlabel(class_names[np.argmax(label_array[i])])

    st = plt.suptitle(title, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()


def graph_with_error_bar(x_values, y_values, y_stdr, title ,x_axis_title,
                         y_axis_tile, fix_y_axis, ax_out,
                         save_path, plot_line, xticks_rotation  ):
    if not ax_out:
        fig, ax = plt.subplots()
    else:
        ax = ax_out
    ax.errorbar(x_values, y_values,
                yerr=y_stdr,
                fmt='o')
    if plot_line:
        line = 0 * np.array(y_values) + y_values[0]
        ax.plot(x_values, line, '--r')
    ax.set_xlabel(x_axis_title)
    plt.xticks(rotation=xticks_rotation)
    ax.set_ylabel(y_axis_tile)
    ax.set_title(title)
    if fix_y_axis:
        ax.set_ylim((0.5), (1))

    if save_path:
        plt.savefig(save_path,
                dpi=300)
    if not ax_out:
        plt.show()

def mark_small_values(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if val == 1:
        color = 'black'
    elif val < 0.05:
        color = 'red'
    else:
        color = 'grey'

    return 'background-color: %s' % color