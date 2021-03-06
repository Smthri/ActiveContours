import sys
import utils
import numpy as np
import skimage.io as skio
import skimage
from ac import *

def calc_iou(result, gt):
    img_gt = np.array(skimage.img_as_float32(skio.imread(gt, as_gray = True))).astype(np.uint8)
    img_result = np.array(skimage.img_as_float32(skio.imread(result, as_gray = True))).astype(np.uint8)

    intersection = img_result * img_gt
    union = img_result + img_gt
    union[union > 1] = 1

    return np.sum(intersection) / np.sum(union)
    #print(f'IoU: {np.sum(intersection) / np.sum(union)}')

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

    C = active_contours(image, X[:-1], alpha, beta, tau, w_line, w_edge, kappa)
    C = np.append(C, [C[0]], axis = 0)

    #utils.save_mask_withimg('result.png', C, np.uint8(image * 255))
    utils.save_mask(output_image, C, np.uint8(image * 255))
    iou = calc_iou(output_image, input_image.replace('.png', '_mask.png'))
    print(iou)



    #X = active_contour(image, X, float(alpha), float(beta), float(w_line), float(w_edge), float(tau))
    
    #utils.save_mask_withimg(output_image, X, np.uint8(image * 255))
    #utils.save_mask('genmask.png', X, np.uint8(image * 255))
    #calc_iou('genmask.png', 'images/astranaut_mask.png')

