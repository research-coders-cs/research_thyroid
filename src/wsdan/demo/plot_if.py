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

def plt_show(_plt):
    if not is_colab():
        print('@@ plt_show(): \'q\' to close interactively')
    _plt.show()

def plt_imshow_im(_plt, im):
    _plt.figure()
    _plt.imshow(im)
    plt_show(_plt)

def plt_imshow(_plt, x):
    if isinstance(x, str):
        plt_imshow_im(_plt, _plt.imread(x))
    else:
        plt_imshow_im(_plt, x)

def plt_imshow_tensor(_plt, ten, cmap='gray'):
    img = ten.permute(1, 2, 0)  # <c, h, w> -> <h, w, c>
    _plt.imshow(img, cmap=cmap)
    plt_show(_plt)
