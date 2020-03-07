import sys
import utils

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

    print(input_image)

