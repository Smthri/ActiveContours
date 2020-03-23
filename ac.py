from numpy.linalg import inv, norm
import numpy as np
from skimage.filters import sobel, gaussian
from scipy.ndimage import convolve1d
from tqdm import tqdm
from skimage import img_as_float
import utils
from scipy.interpolate import interp2d, RectBivariateSpline
from skimage.io import imread, imsave

def Eext(img, w_line, w_edge):
    G = gaussian(img, 3)
    Pline = G
    #grad = np.gradient(img)
    Pedge = gaussian(sobel(img), 3)
    #Pedge = sobel(G)
    #Pedge = np.abs(convolve1d(G, [-1, 0, 1], axis = 0)) + np.abs(convolve1d(G, [-1, 0, 1], axis = 1))

    return w_line*Pline + w_edge*Pedge

def active_contours(image, X, alpha, beta, tau, w_line, w_edge, bf):
    
    P = Eext(img_as_float(image), float(w_line), float(w_edge))

    intp = RectBivariateSpline(np.arange(P.shape[1]), np.arange(P.shape[0]), P.T, kx=2, ky=2, s=0)

    A = np.zeros(shape = (X.shape[0], X.shape[0]), dtype = np.float64)
    a = float(alpha)
    b = float(beta)
    t = float(tau)
    
    line = [b, -a - 4*b, 2*a + 6*b, -a - 4*b, b]
    for i, l in enumerate(A):
        np.put(l, [i - 2, i - 1, i, i + 1, i + 2], line, mode = 'wrap')
        if i < 2:
            A[i, -3:] = list(A[i, -2:]) + [0]
        elif i >= len(A) - 2:
            A[i, :3] = [0] + list(A[i, :2])

    '''
    a = np.roll(np.eye(len(X)), -1, axis = 0) + np.roll(np.eye(len(X)), -1, axis = 1) - 2*np.eye(len(X))
    b = np.roll(np.eye(len(X)), -2, axis = 0) + np.roll(np.eye(len(X)), -2, axis = 1) - 4*np.roll(np.eye(len(X)), -1, axis = 0) -\
        np.roll(np.eye(len(X)), -1, axis = 1) + 6*np.eye(len(X))
    A = -alpha*a + beta*b
    '''
    A = inv(A + t*np.eye(len(X)))

    convergence_order = 10
    Xsave = np.empty((convergence_order, len(X)))
    Ysave = np.empty((convergence_order, len(X)))

    x = X[:, 0]
    y = X[:, 1]


    for iteration in tqdm(range(2500)):
        #fx = Fextx[xn, yn]
        #fy = Fexty[xn, yn]

        #fx = np.array([intpx(x, y)[i, len(X) - i - 1] for i in range(len(X))])
        #fy = np.array([intpy(x, y)[i, len(X) - i - 1] for i in range(len(X))])

        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)

        #print(fx)

        #f = np.array([[fx[i], fy[i]] for i in range(0, len(fx))], dtype = np.float64)
        #X = np.matmul(A, t*X + f)

        xn = np.matmul(A, t*x + fx)
        yn = np.matmul(A, t*y + fy)

        #dx = x - xn
        #dx /= norm(dx)
        x = xn

        #dy = y - yn
        #dy /= norm(dy)
        y = yn

        j = iteration % (convergence_order + 1)
        if j < convergence_order:
            Xsave[j, :] = x
            Ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(Xsave - x[None, :]) + np.abs(Ysave - y[None, :]), 1))
            if dist < .1:
                break

        #dx = 3 * np.tanh(dX[:, 0])
        #dy = 3 * np.tanh(dX[:, 1])
        
        '''
        output = open(output_image, "w")
        for line in X:
            output.writelines(str(line[0]) + ' ' + str(line[1]) + '\n')
        output.close()
        '''
        
        X[:, 0] = x
        X[:, 1] = y
        #utils.save_mask_withimg(f'result/test_{iteration}.png', X, np.uint8(image * 255))

    X[:, 0] = x
    X[:, 1] = y
    return X.astype(np.int32)
