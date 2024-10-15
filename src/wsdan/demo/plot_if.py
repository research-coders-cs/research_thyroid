try:
    import google.colab
    IS_COLAB = True
except:
    IS_COLAB = False

def is_colab():
    return IS_COLAB

#

import matplotlib.pyplot as plt

def get_plt():
    return plt

def plt_show(plt):
    if not is_colab():
        print('@@ plt_show(): \'q\' to close interactively')
    plt.show()

def plt_imshow_im(plt, im):
    plt.figure()
    plt.imshow(im)
    plt_show(plt)

def plt_imshow(plt, x):
    if isinstance(x, str):
        plt_imshow_im(plt, plt.imread(x))
    else:
        plt_imshow_im(plt, x)

def plt_imshow_tensor(plt, ten, cmap='gray'):
    img = ten.permute(1, 2, 0)  # <c, h, w> -> <h, w, c>
    plt.imshow(img, cmap=cmap)
    plt_show(plt)
