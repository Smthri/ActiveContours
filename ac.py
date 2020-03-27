from numpy.linalg import inv, norm
import numpy as np
from skimage.filters import sobel, gaussian
from tqdm import tqdm
from skimage import img_as_float
import utils
from scipy.interpolate import RectBivariateSpline
from skimage.io import imread, imsave

def Eext(img, w_line, w_edge):
    G = gaussian(img, 3)
    Pline = G
    Pedge = gaussian(sobel(img), 3)

    return w_line*Pline + w_edge*Pedge

def calc_normals(X, bf):
    kappa = float(bf)
    N = np.empty((0, 2), dtype=np.float64)

    for it in range(len(X)):
        A = X[it-1]
        B = X[0]
        if it < len(X) - 1:
            B = X[it+1]
        dx = A[0] - B[0]
        dy = A[1] - B[1]
        N = np.append(N, [[-dx, dy]], axis=0)

    return kappa * N

def active_contours(image, X, alpha, beta, tau, w_line, w_edge, bf):
    
    P = Eext(img_as_float(image), float(w_line), float(w_edge))

    interpolator = RectBivariateSpline(np.arange(P.shape[1]), np.arange(P.shape[0]), P.T, kx=2, ky=2, s=0)

    A = np.zeros(shape = (X.shape[0], X.shape[0]), dtype = np.float64)
    a = float(alpha)
    b = float(beta)
    t = float(tau)
    
    line = [b, -a - 4*b, 2*a + 6*b, -a - 4*b, b]
    for i, l in enumerate(A):
        np.put(l, [i - 2, i - 1, i, i + 1, i + 2], line, mode = 'wrap')
    A = inv(t*A + np.eye(len(X)))

    save_amount = 10
    x_save = np.empty((save_amount, len(X)))
    y_save = np.empty((save_amount, len(X)))

    x = X[:, 0]
    y = X[:, 1]


    for iteration in tqdm(range(2500)):
        Fx = interpolator(x, y, dx=1, grid=False)
        Fy = interpolator(x, y, dy=1, grid=False)

        '''
        nx = norm(Fx)
        ny = norm(Fy)

        if not nx == 0:
            Fx /= nx
            Fx /= t
        if not ny == 0:
            Fy /= ny
            Fy /= t
            '''
 
        #N = calc_normals(np.stack((x, y), axis=1), bf)
        #Fx = -N[:, 0] + Fx
        #Fy = -N[:, 1] + Fy

        x = np.matmul(A, x + t*Fx)
        y = np.matmul(A, y + t*Fy)

        j = iteration % (save_amount + 1)
        if j < save_amount:
            x_save[j, :] = x
            y_save[j, :] = y
        else:
            dist = np.min(np.max(np.abs(x_save - x) + np.abs(y_save - y), 1))
            if dist < .1:
                break

    X[:, 0] = x
    X[:, 1] = y
    return X.astype(np.int32)
