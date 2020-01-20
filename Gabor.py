import numpy as np
import cv2
import sys
'''
def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
                                #(ksize, ksize), sigma, theta, lambda, gamma, psi, ktype
        kern = cv2.getGaborKernel((ksize, ksize), 4.0,  theta, 10.0,    0.5,  0,    ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    return filters


def process(img, filters):
    for kern in filters:
        #fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        fimg = cv2.filter2D(img, -1, kern)
        cv2.imshow('filtered image', fimg)
        cv2.waitKey(0)
'''


if __name__ == '__main__':

    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'test.jpg'

    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 8, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    cv2.imshow('image', img)
    cv2.imshow('filtered image', filtered_img)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    #filters = build_filters()
    #process(img, filters)
    #cv2.destroyAllWindows()
