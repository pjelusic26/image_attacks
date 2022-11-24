import numpy as np
import cv2
import skimage.transform as trs
import scipy.ndimage as nd
from scipy.signal import wiener
from scipy.signal import convolve2d as conv2
from skimage import img_as_float, restoration
from PIL import Image, ImageFilter
import os

def img_read(filepath, format):
    if format == 'numpy':
        return np.array(Image.open(filepath))
    elif format == 'pil':
        return Image.open(filepath)
    else:
        raise AttributeError("Please provide a valid image format.")

def scaling(img, scale):
    ### IMG IS ARRAY ###
    dimension = int(img.shape[0] * scale)
    img_scale = trs.resize(img, (dimension, dimension), preserve_range = True)
    return np.array(img_scale).astype(np.uint8)

def rotation(img, angle):
    ### IMG IS PIL ###
    img = Image.fromarray(img, mode = 'L')
    img_rotate = img.rotate(angle)
    return np.array(img_rotate)

def jpeg_compression(img, compression_value):
    ### IMG IS PIL ###
    img = Image.fromarray(img, mode = 'L')
    img.save('img_jpeg.jpg', optimize = True, quality = compression_value)
    img_compressed = img_read('img_jpeg.jpg', format = 'numpy')
    os.remove('img_jpeg.jpg')
    return np.array(img_compressed)

def blur(img, blur_radius):
    ### IMG IS PIL ###
    img = Image.fromarray(img, mode = 'L')
    img_blur = img.filter(ImageFilter.GaussianBlur(radius = blur_radius))
    return np.array(img_blur)

def noise(img, noise_factor):
    ### IMG IS ARRAY ###
    gauss = np.random.normal(0, noise_factor, img.size)
    gauss = gauss.reshape(
        img.shape[0], img.shape[1]).astype('uint8')
    img_blur = img + gauss
    return img_blur

def print_scan(img, noise_value, blur_value):

    noise = np.random.normal(0, noise_value, img.size)
    noise = noise.reshape(
        img.shape[0], img.shape[1]).astype('uint8')
    img_noise = img + noise
    ps = nd.gaussian_filter(img_noise, sigma = blur_value)

    return ps

def enhance_laplacian(img):
    img_laplace = nd.filters.laplace(img)
    return img_laplace

def enhance_unblur(img, sigma = 3, alpha = 30):

    img_blur = nd.gaussian_filter(img, sigma)
    filter_blur = nd.gaussian_filter(img_blur, sigma)
    image_sharp = img_blur + alpha * (img_blur - filter_blur)

    return image_sharp

def enhance_deconvolution(img):

    img = img_as_float(img)
    rng = np.random.default_rng()

    psf = np.ones((5, 5)) / 25
    astro = conv2(img, psf, 'same')
    # Add Noise to Image
    astro_noisy = astro.copy()
    astro_noisy += (rng.poisson(lam = 25, size=astro.shape) - 10) / 255.

    # Restore Image using Richardson-Lucy algorithm
    img_deconv = restoration.richardson_lucy(astro_noisy, psf, 15)
    img_deconv = 255 * img_deconv
    img_deconv = img_deconv.astype(np.uint8)
    # print(f"Image Min: {np.amin(img_deconv)}")
    # print(f"Image Max: {np.amax(img_deconv)}")
    return img_deconv

def enhance(img, kernel = 'HEAVY'):

    if kernel == 'SHARP':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    elif kernel == 'HEAVY':
        kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])

    elif kernel == 'EDGE':
        kernel = np.array([[-1,-1,-1,-1,-1],
                            [-1,2,2,2,-1],
                            [-1,2,8,2,-1],
                            [-2,2,2,2,-1],
                            [-1,-1,-1,-1,-1]])/8.0

    img_sharp = cv2.filter2D(img, -1, kernel)

    return img_sharp