import sys
import utils
import numpy as np
import skimage.io as skio
import skimage
from skvideo.io import vwrite
from scipy.ndimage import gaussian_filter
from numpy.linalg import inv, eig
from tqdm import tqdm
import skvideo

def Eext(img, w_line, w_edge):
    G = gaussian_filter(img, 1)
    Pline = img
    grad = np.gradient(img)
    Pedge = (grad[0] ** 2 + grad[1] ** 2)

    return w_line*Pline + w_edge*Pedge

def calc_iou(result, gt):
    img_result = np.array(skimage.img_as_float32(skio.imread(result, as_gray = True))).astype(np.uint8)
    img_gt = np.array(skimage.img_as_float32(skio.imread(gt, as_gray = True))).astype(np.uint8)

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

    image = np.array(skimage.img_as_float64(skio.imread(input_image)))

    X = np.clip(np.array([np.array(x.replace('\n', '').split(' ')) for x in open(initial_snake)]).astype(np.float32).astype(np.int32), 0, max(image.shape) - 1)

    P = Eext(image, float(w_line), float(w_edge))
    grad = np.gradient(P)
    #norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    k = 1
    Fextx = -k * grad[1]
    Fexty = -k * grad[0] # 0 is y axis, 1 is x axis
    
    print(f'Max and min P: {np.max(P)}, {np.min(P)}')
    print(f'Max and min gradient: x: ({np.max(Fextx), np.min(Fextx)}), y: {np.max(Fexty), np.min(Fexty)}')

    A = np.zeros(shape = (X.shape[0], X.shape[0]), dtype = np.float64)
    a = float(alpha)
    b = float(beta)
    t = float(tau)
    line = [b, -a - 4*b, 2*a + 6*b, -a - 4*b, b]
    for i, l in enumerate(A):
        np.put(l, [i - 2, i - 1, i, i + 1, i + 2], line, mode = 'wrap')
        A[i, :] *= t
        A[i, i] += 1
    print(f'Max and min of matrix A: {np.max(A)}, {np.min(A)}')
    A = inv(A)

    for iteration in tqdm(range(40)):
        fx = Fextx[X[:, 0], X[:, 1]]
        fy = Fexty[X[:, 0], X[:, 1]]

        f = np.array([[fx[i], fy[i]] for i in range(0, len(fx))])
        X = np.int32(np.matmul(A, X + t*f))
        
        '''
        output = open(output_image, "w")
        for line in X:
            output.writelines(str(line[0]) + ' ' + str(line[1]) + '\n')
        output.close()
        '''

        utils.save_mask(f'result/test_{iteration}.png', X, np.uint8(image * 255))

    utils.save_mask(output_image, X, np.uint8(image * 255))
    calc_iou(output_image, 'images/astranaut_mask.png')

