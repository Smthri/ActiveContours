import sys
import utils
import numpy as np
import skimage.io as skio
import skimage
from skimage.transform import resize
from numpy.linalg import inv, eig
from tqdm import tqdm
from skimage.filters import sobel, gaussian
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve1d

def Eext(img, w_line, w_edge):
    G = gaussian(img, 3)
    Pline = G
    #grad = np.gradient(img)
    Pedge = sobel(G)

    return w_line*Pline + w_edge*Pedge

def calc_iou(result, gt):
    img_gt = np.array(skimage.img_as_float32(skio.imread(gt, as_gray = True))).astype(np.uint8)
    img_result = np.array(skimage.img_as_float32(skio.imread(result, as_gray = True))).astype(np.uint8)

    intersection = img_result * img_gt
    union = img_result + img_gt
    union[union > 1] = 1

    print(f'IoU: {np.sum(intersection) / np.sum(union)}')

if __name__ == '__main__':
    if (len(sys.argv) < 10):
        print('Usage: python main.py (input_image) (initial_snake) (output_image) (alpha) (beta) (tau) (w_line) (w_edge) (kappa) \n\
            input_image - path to image \n\
            initial_snake - path to initial contour \n\
            output_image - path to save result image \n\
            alpha - continuity term \n\
            beta - smoothness term \n\
            tau - step for gradient descent \n\
            w_line - intensity weight for external energy \n\
            w_edge - edge weight for external energy \n\
            kappa - balloon force weight')
        sys.exit(0)

    input_image, initial_snake, \
    output_image, alpha, beta, \
    tau, w_line, w_edge, kappa = tuple(sys.argv[1:10])

    image = skimage.img_as_float(skio.imread(input_image))

    X = np.array([np.array(x.replace('\n', '').split(' ')) for x in open(initial_snake)]).astype(np.float64)

    P = Eext(image, float(w_line), float(w_edge))
    grad = np.gradient(P)
    #norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    k = 1
    #Fextx = -k * grad[1]
    #Fexty = -k * grad[0] # 0 is y axis, 1 is x axis

    Fextx = -k*convolve1d(P, [1, -1], axis=1)
    Fexty = -k*convolve1d(P, [1, -1], axis=0)

    print(f'Max and min P: {np.max(P)}, {np.min(P)}')
    print(f'Max and min gradient: x: ({np.max(Fextx), np.min(Fextx)}), y: {np.max(Fexty), np.min(Fexty)}')

    A = np.zeros(shape = (X.shape[0], X.shape[0]), dtype = np.float64)
    a = float(alpha)
    b = float(beta)
    t = float(tau)
    
    line = [b, -a - 4*b, 2*a + 6*b, -a - 4*b, b]
    for i, l in enumerate(A):
        np.put(l, [i - 2, i - 1, i, i + 1, i + 2], line, mode = 'wrap')
    print(f'Max and min of matrix A: {np.max(A)}, {np.min(A)}')
    '''
    a = np.roll(np.eye(len(X)), -1, axis = 0) + np.roll(np.eye(len(X)), -1, axis = 1) - 2*np.eye(len(X))
    b = np.roll(np.eye(len(X)), -2, axis = 0) + np.roll(np.eye(len(X)), -2, axis = 1) - 4*np.roll(np.eye(len(X)), -1, axis = 0) -\
        np.roll(np.eye(len(X)), -1, axis = 1) + 6*np.eye(len(X))
    A = -alpha*a + beta*b
    '''
    A = inv(A + t*np.eye(len(X)))

    for iteration in tqdm(range(500)):
        x = np.around(X[:, 0]).astype(np.int32)
        y = np.around(X[:, 1]).astype(np.int32)

        fx = Fextx[x, y]
        fy = Fexty[x, y]

        #fx = intp(X[:, 0], X[:, 1], dx=1, grid=False)
        #fy = intp(X[:, 0], X[:, 1], dy=1, grid=False)

        f = np.array([[fx[i], fy[i]] for i in range(0, len(fx))], dtype = np.float64)
        X = np.matmul(A, t*X + f)

        #dx = 3 * np.tanh(dX[:, 0])
        #dy = 3 * np.tanh(dX[:, 1])
        
        '''
        output = open(output_image, "w")
        for line in X:
            output.writelines(str(line[0]) + ' ' + str(line[1]) + '\n')
        output.close()
        '''

        #utils.save_mask(f'result/test_{iteration}.png', X, np.uint8(image * 255))

    X = X.astype(np.int32)
    utils.save_mask_withimg(output_image, X, np.uint8(image * 255))
    utils.save_mask('genmask.png', X, np.uint8(image * 255))
    calc_iou('genmask.png', 'images/astranaut_mask.png')

