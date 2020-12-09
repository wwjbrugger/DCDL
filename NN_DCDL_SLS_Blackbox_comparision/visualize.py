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
