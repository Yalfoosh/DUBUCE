import numpy as np
import skimage.io as skio


def torch_draw_filters(layer, file_path: str, border: int = 1, n_columns: int = 8):
    layer_w = layer.weight.clone().detach().numpy()

    N, C, W, H = layer_w.shape[:4]

    layer_w_T = layer_w.transpose(2, 3, 1, 0)
    layer_w_T -= layer_w_T.min()
    layer_w_T /= layer_w_T.max()

    n_rows = int(np.ceil(N / n_columns))

    width, height = [x * y + (x - 1) * border for x, y in zip([n_columns, n_rows],
                                                              [W, H])]
    image = np.zeros([height, width, C])

    for i in range(N):
        c = int(i % n_columns) * (W + border)
        r = int(i / n_columns) * (H + border)

        image[r: r + H, c: c + W, :] = layer_w_T[:, :, :, i]

    skio.imsave(file_path, np.array(image * 255., dtype=np.uint8))
