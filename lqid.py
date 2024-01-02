from cv2 import imread, resize, COLOR_BGR2GRAY, COLOR_BGR2RGB, cvtColor
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy import (
    max, min, mean, std, log, abs, array, rot90, stack, expand_dims, absolute, 
    sqrt, linspace, hstack, repeat, newaxis, clip, bincount, prod, inf
)
from numpy.random import randint, choice, standard_normal
from numpy.typing import NDArray
from typing import Dict
from scipy.signal import convolve2d
from copy import deepcopy
from skimage.measure import (
    blur_effect, centroid, euler_number, label, regionprops, perimeter, 
    shannon_entropy, find_contours
)
from skimage.graph import pixel_graph
from skimage.feature import CENSURE
from skimage.exposure import is_low_contrast, rescale_intensity
from skimage.filters import gaussian
from matplotlib.pyplot import imshow, show, axis
from pandas import DataFrame
from multiprocessing import Pool, cpu_count


KERNEL_LAPLACIAN_1 = array([[+0, -1, +0], [-1, +4, -1], [+0, -1, +0]])
KERNEL_LAPLACIAN_2 = array([[-1, -1, -1], [-1, +8, -1], [-1, -1, -1]])
KERNEL_SOBEL_1 = array([[-1, +0, +1], [-2, +0, +2], [-1, +0, +1]])
KERNEL_SOBEL_2 = array([[+1, +2, +1], [+0, +0, +0], [-1, -2, -1]])


def img2matrix(path:str, l:int=400, h:int=400) -> Dict:
    """Read an image file into a dictionary of RGB and gray matrices. 

    Args:
        path (str): Path to the images file.
        l (int, optional): Dimension 1 of the output matrix. Defaults to 400.
        h (int, optional): Dimension 2 of the output matrix. Defaults to 400.

    Returns:
        dict: Dictionary of RGB and gray matrices
    """
    img = imread(path)
    img = resize(img, dsize=(l,h))
    img_rgb =cvtColor(img, COLOR_BGR2RGB)
    img_rgb = img_rgb[int(0.10*h):int(0.90*h), int(0.10*l):int(0.90*l), :]
    img_gray = cvtColor(img, COLOR_BGR2GRAY)
    img_gray = img_gray[int(0.10*h):int(0.90*h), int(0.10*l):int(0.90*l)]
    return {"rgb": img_rgb, "gray": img_gray}


def quickshow(matrix:NDArray):
    """Plot a image matrix quickly.

    Args:
        matrix (NDArray): Image matrix. 
    """
    imshow(matrix, cmap="gray", vmin=0, vmax=255)
    axis('off')
    show()


def random_band(matrix:NDArray) -> NDArray:
    """Generate a new image with random horizontal or vertical bands.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random bands.
    """
    matrix = deepcopy(matrix)
    is_gray = len(matrix.shape) == 2
    if is_gray:
        matrix = expand_dims(matrix, axis=2)
    h, l, c = matrix.shape
    rot = choice(a=(True, False))
    dim_exp = h
    if rot:
        dim_exp = l
        matrix = rot90(matrix, k=1, axes=(0,1))
    r = randint(low=min((int(0.2*dim_exp/6.0), 1)), high=int(dim_exp/6.0))
    v = hstack((linspace(0, 255, r), linspace(255, 0, r)))
    m = repeat([v], (1-rot)*l+rot*h, axis=0).T
    m = repeat(m[:, :, newaxis], c, axis=2)
    i = randint(low=0, high=(1-rot)*h+rot*l-2*r)
    matrix[i:(i+2*r), :, :] = clip(
        matrix[i:(i+2*r), :, :] + choice(a=(-1,1)) * m, 
        a_min=0, a_max=255
    )
    if rot:
        matrix = rot90(matrix, k=3, axes=(0,1))
    if is_gray:
        matrix = matrix[:,:,0]
    return matrix.astype(int)


def random_noise(matrix:NDArray) -> NDArray:
    """Generate a new image with random noise.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random noise.
    """
    matrix = deepcopy(matrix)
    matrix = matrix \
        + randint(low=50, high=250) * standard_normal(size=matrix.shape)
    matrix = clip(matrix, a_min=0, a_max=255)
    return matrix.astype(int)


def random_blur(matrix:NDArray) -> NDArray:
    """Generate a new image with random blur.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random blur.
    """
    matrix = deepcopy(matrix)
    matrix = gaussian(matrix, sigma=randint(low=2, high=5), preserve_range=True)
    matrix = clip(matrix, a_min=0, a_max=255)
    return matrix.astype(int)


def random_exposure(matrix:NDArray) -> NDArray:
    """Generate a new image with random exposure.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random exposure.
    """
    matrix = deepcopy(matrix)
    matrix = rescale_intensity(
        matrix, out_range=(randint(low=150, high=200), 255)
    )
    return matrix.astype(int)


def random_dark(matrix:NDArray) -> NDArray:
    """Generate a new image with random darkness.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random darkness.
    """
    matrix = deepcopy(matrix)
    matrix = rescale_intensity(
        matrix, out_range=(0, randint(low=20, high=50))
    )
    return matrix.astype(int)


def random_constant(matrix:NDArray) -> NDArray:
    """Generate a new image with random constant color.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random constant color.
    """
    matrix = deepcopy(matrix)
    is_gray = len(matrix.shape) == 2
    if is_gray:
        matrix = expand_dims(matrix, axis=2)
    flip = choice(a=(0,1), p=(0.9, 0.1))
    extreme_value = choice(a=(0,255))
    for j in range(matrix.shape[2]):
        matrix[:,:,j] = (1-flip) * randint(low=0, high=256) \
            + flip * extreme_value
    if is_gray:
        matrix = matrix[:,:,0]
    return matrix


def random_glare(matrix:NDArray) -> NDArray:
    """Generate a new image with random glare.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        NDArray: Transformed image with random glare.
    """
    matrix = deepcopy(matrix)
    is_gray = len(matrix.shape) == 2
    ths = randint(low=200, high=230)
    if is_gray:
        mask = matrix > ths
    else:
        mask = (matrix[:,:,0] > ths) \
            + (matrix[:,:,1] > ths) + (matrix[:,:,2] > ths)
        mask = mask > 0
    masked_values = matrix[mask]
    matrix = rescale_intensity(matrix, out_range=(0, randint(low=40, high=80)))
    matrix[mask] = masked_values
    matrix = gaussian(matrix, sigma=1, preserve_range=True)
    matrix = clip(matrix, a_min=0, a_max=255)
    matrix = matrix.astype(int)
    return matrix


def conv(matrix:NDArray, kernel:NDArray) -> NDArray:
    """Matrix convolution operation with desired kernel.

    Args:
        matrix (NDArray): Image matrix
        kernel (NDArray): Filter

    Returns:
        NDArray: Convolved image.
    """
    matrix = array(matrix)
    kernel = array(kernel)
    assert len(kernel.shape) == 2, "Wrong kernel dimensions."
    if len(matrix.shape) == 2: # one channel 
        return convolve2d(matrix, rot90(rot90(kernel)))
    elif len(matrix.shape) == 3: # multi channels
        return stack(
            [
                convolve2d(matrix[:, :, j], rot90(rot90(kernel))) \
                    for j in range(matrix.shape[2])
            ], 
            axis=-1
        )
    return array([])


def get_conv_features_per_kernel(matrix:NDArray, kernel:NDArray) -> Dict:
    """Create convolution features with a specific kernel.

    Args:
        matrix (NDArray): Input image matrix.
        kernel (NDArray): Specific filter.

    Returns:
        Dict: Dictionary of convolution features. 
    """
    if len(matrix.shape) == 2:
        matrix = expand_dims(matrix, axis=2)
    dict_features = {}
    convolved_matrix = conv(matrix, kernel)
    dict_features["tot_var"] = convolved_matrix.var()
    '''
    dim_x = convolved_matrix.shape[0]
    dim_y = convolved_matrix.shape[1]
    split_x = [0, int(0.333 * dim_x), int(0.666 * dim_x), dim_x]
    split_y = [0, int(0.333 * dim_y), int(0.666 * dim_y), dim_y]
    for c in range(matrix.shape[2]):
        dict_features["c%i_mean" % (c)] = convolved_matrix[:, :, c].mean()
        dict_features["c%i_var" % c] = convolved_matrix[:, :, c].var()
        array_vars = []
        for i in range(len(split_x)-1):
            for j in range(len(split_y)-1):
                convolved_matrix_split = convolved_matrix[split_x[i]:split_x[i+1], split_y[j]:split_y[j+1], c]
                dict_features["c%ii%ij%i_mean" % (c, i, j)] = convolved_matrix_split.mean()
                dict_features["c%ii%ij%i_var" % (c, i, j)] = convolved_matrix_split.var()
                if dict_features["tot_var"] > 0:
                    dict_features["c%ii%ij%i_var_delta" % (c, i, j)] = (dict_features["c%ii%ij%i_var" % (c, i, j)] - dict_features["tot_var"]) / dict_features["tot_var"] * 100
                else: 
                    dict_features["c%ii%ij%i_var_delta" % (c, i, j)] = (dict_features["c%ii%ij%i_var" % (c, i, j)] - dict_features["tot_var"]) * 1000
                array_vars.append(dict_features["c%ii%ij%i_var" % (c, i, j)])
        dict_features["c%i_var_max" % c] = max(array_vars)
        dict_features["c%i_var_min" % c] = min(array_vars)
        dict_features["c%i_var_mean" % c] = mean(array_vars)
    '''
    return dict_features


def get_conv_features(dict_img:Dict) -> Dict:
    """Create convolution features with different kernels.

    Args:
        dict_img (Dict): Dictionary containing RGB matrix and gray matrix.

    Returns:
        Dict: Convolution features.
    """
    dict_features = {}
    for key_matrix, matrix in dict_img.items():
        for key_kernel, kernel in {
            "laplacian1": KERNEL_LAPLACIAN_1, 
            "laplacian2": KERNEL_LAPLACIAN_2, 
            "sobel1": KERNEL_SOBEL_1, 
            "sobel2": KERNEL_SOBEL_2
        }.items():
            dict_tmp = get_conv_features_per_kernel(matrix, kernel)
            for k, v in dict_tmp.items():
                dict_features["%s_%s_%s" % (key_matrix, key_kernel, k)] = v
    return dict_features


def fftifft(img:NDArray, r:float=4.0) -> NDArray: 
    """Fast Fourier transform.

    Args:
        img (NDArray): Image matrix. 
        r (float, optional): Radius of frequency regions to remove. 
        Defaults to 4.0.

    Returns:
        NDArray: Transformed image matrix.
    """
    img_fft = fftshift(fft2(img))
    if len(img.shape) == 3:
        h, l, _ = img.shape
        img_fft[
            int(h/2.0-h/r):int(h/2.0+h/r), 
            int(l/2.0-l/r):int(l/2.0+l/r), 
        :] = 0
    else:
        h, l = img.shape
        img_fft[
            int(h/2.0-h/r):int(h/2.0+h/r), 
            int(l/2.0-l/r):int(l/2.0+l/r)
        ] = 0
    img_ifft = ifft2(ifftshift(img_fft))
    img_ifft = 20 * log(abs(img_ifft)+1.0)
    den = img_ifft.max() - img_ifft.min()
    if den > 0:
        img_ifft = (img_ifft - img_ifft.min()) / den * 255
    else:
        img_ifft = img_ifft - img_ifft
    return img_ifft


def get_fft_features_per_r(matrix:NDArray, r:float=4.0) -> Dict:
    """Fast Fourier transform features for a specific r

    Args:
        matrix (NDArray): Image matrix.
        r (float, optional): Radius of frequency regions to remove. 
        Defaults to 4.0.

    Returns:
        Dict: FFT features.
    """
    if len(matrix.shape) == 2:
        matrix = expand_dims(matrix, axis=2)
    dict_features = {}
    fft_matrix = fftifft(matrix, r=r)
    dict_features["tot_mean"] = fft_matrix.mean()
    dict_features["tot_var"] = fft_matrix.var()
    '''
    dim_x = fft_matrix.shape[0]
    dim_y = fft_matrix.shape[1]
    split_x = [0, int(0.333 * dim_x), int(0.666 * dim_x), dim_x]
    split_y = [0, int(0.333 * dim_y), int(0.666 * dim_y), dim_y]
    for c in range(matrix.shape[2]):
        dict_features["c%i_mean" % (c)] = fft_matrix[:, :, c].mean()
        dict_features["c%i_var" % c] = fft_matrix[:, :, c].var()
        array_vars = []
        for i in range(len(split_x)-1):
            for j in range(len(split_y)-1):
                fft_matrix_split = fft_matrix[split_x[i]:split_x[i+1], split_y[j]:split_y[j+1], c]
                dict_features["c%ii%ij%i_mean" % (c, i, j)] = fft_matrix_split.mean()
                dict_features["c%ii%ij%i_var" % (c, i, j)] = fft_matrix_split.var()
                if dict_features["tot_var"] > 0:
                    dict_features["c%ii%ij%i_var_delta" % (c, i, j)] = (dict_features["c%ii%ij%i_var" % (c, i, j)] - dict_features["tot_var"]) / dict_features["tot_var"] * 100
                else:
                    dict_features["c%ii%ij%i_var_delta" % (c, i, j)] = (dict_features["c%ii%ij%i_var" % (c, i, j)] - dict_features["tot_var"]) * 1000
                array_vars.append(dict_features["c%ii%ij%i_var" % (c, i, j)])
        dict_features["c%i_var_max" % c] = max(array_vars)
        dict_features["c%i_var_min" % c] = min(array_vars)
        dict_features["c%i_var_mean" % c] = mean(array_vars)
    '''
    return dict_features


def get_fft_features(dict_img:Dict) -> Dict:
    """Fast Fourier transform features with different r.

    Args:
        dict_img (Dict): Dictionary containing RGB matrix and gray matrix.

    Returns:
        Dict: FFT features.
    """
    dict_features = {}
    for key_matrix, matrix in dict_img.items():
        for key_r, r in {
            "fft_r4": 4, 
            "fft_r8": 8, 
        }.items():
            dict_tmp = get_fft_features_per_r(matrix, r)
            for k, v in dict_tmp.items():
                dict_features["%s_%s_%s" % (key_matrix, key_r, k)] = v
    return dict_features


def colorfulness(matrix:NDArray)->float:
    """Compute the colorfulness.

    Args:
        matrix (NDArray): Input image matrix.

    Returns:
        float: Colorfulness.
    """
    # https://pyimagesearch.com/2017/06/05/
    # computing-image-colorfulness-with-opencv-and-python/
    R = matrix[:,:,0]
    G = matrix[:,:,1]
    B = matrix[:,:,2]
    rg = absolute(R - G)
    yb = absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (mean(rg), std(rg))
    (ybMean, ybStd) = (mean(yb), std(yb))
    stdRoot = sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)


def straight_lines(matrix:NDArray) -> int:
    """Detect horizontal and vertical straight lines.

    Args:
        matrix (NDArray): Image matrix.

    Returns:
        int: Number of straight lines.
    """
    matrix_high_contrast = (matrix > matrix.mean()) * 255
    axis0_var = matrix_high_contrast.var(axis=0)
    axis1_var = matrix_high_contrast.var(axis=1)
    return (axis0_var.min() == axis0_var).sum() \
            + (axis1_var.min() == axis1_var).sum()


def get_gp_features(dict_img:Dict) -> Dict:
    """Compute various image general features

    Args:
        dict_img (Dict): Image matrices dictionary. 

    Returns:
        Dict: Generale purpose features. 
    """
    dict_features = {}
    for key_matrix, matrix in dict_img.items():
        dict_features["%s_darkness" % (key_matrix)] = matrix.mean()
        dict_features["%s_var" % (key_matrix)] = matrix.var()
        dict_features["%s_straight_lines" % (key_matrix)] \
            = straight_lines(matrix)
        dict_features["%s_high_pixels" % (key_matrix)] = (matrix > 225).mean()
        dict_features["%s_low_pixels" % (key_matrix)] = (matrix < 30).mean()
        dict_features["%s_high_pixels_regions" % (key_matrix)] \
            = sum([1 for region in regionprops(label(matrix > 225)) \
            if region.area >= 10])
        dict_features["%s_low_pixels_regions" % (key_matrix)] \
            = sum([1 for region in regionprops(label(matrix < 30)) \
            if region.area >= 10])
        for j, c in enumerate(centroid(matrix)):
            dict_features["%s_centroid_%i" % (key_matrix, j)] = c
        dict_features["%s_euler_number" % (key_matrix)] \
            = euler_number(matrix > 100)
        dict_features["%s_graph" % (key_matrix)] = pixel_graph(matrix)[0].mean()
        dict_features["%s_low_contrast" % (key_matrix)] \
            = is_low_contrast(matrix) + 0
        if len(matrix.shape) == 3: # RGB
            for j in (0,1,2):
                dict_features["%s_c%i_mean" % (key_matrix, j)] \
                    = matrix[:,:,0].mean()
                dict_features["%s_c%i_var" % (key_matrix, j)] \
                    = matrix[:,:,0].var()
            dict_features["%s_colorfulness" % (key_matrix)] \
                = colorfulness(matrix)
        elif len(matrix.shape) == 2: # Gray 
            dict_features["%s_blur_effect" % (key_matrix)] \
                = blur_effect(matrix)
            dict_features["%s_regions" % (key_matrix)] \
                = sum([1 for region in regionprops(label(matrix)) \
                if region.area >= 50])
            dict_features["%s_perimeter" % (key_matrix)] \
                = perimeter(matrix > 200)
            dict_features["%s_shannon" % (key_matrix)] \
                = shannon_entropy(matrix)
            censure = CENSURE()
            censure.detect(matrix)
            dict_features["%s_censure" % (key_matrix)] \
                = censure.keypoints.shape[0]
            dict_features["%s_frequency" % (key_matrix)] \
                = max(bincount(matrix.reshape(-1))) / (prod(matrix.shape))
            dict_features["%s_contours" % (key_matrix)] \
                = len(find_contours(matrix))
    return dict_features


def get_features(img_path:str) -> Dict:
    """Wrapper function to compute all features. 

    Args:
        img_path (str): Path to image.

    Returns:
        Dict: Features.
    """
    dict_img = img2matrix(img_path)
    dict_features = {}
    dict_features.update(get_conv_features(dict_img))
    dict_features.update(get_fft_features(dict_img))
    dict_features.update(get_gp_features(dict_img))
    return dict_features


def get_features_from_list(list_files:list) -> DataFrame:
    """Wrapper function to compute features from a list of images

    Args:
        list_files (list): List of images paths.

    Returns:
        DataFrame: Features table.
    """
    with Pool(processes=cpu_count()-1) as pool:
        list_features = pool.map(get_features, list_files)
    df = DataFrame(list_features)
    for col in df.columns:
        df[col+"_missing"] = df[col].isnull() + 0
    df = df.fillna(0.0)
    df= df.replace([inf], 999)
    df= df.replace([-inf], -999)
    df["path"] = list_files
    return df