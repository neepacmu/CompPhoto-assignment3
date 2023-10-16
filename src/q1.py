
import cv2
import skimage
import scipy
import numpy as np


def gaussian_kernel_r(I, i_val, sigma_r):
    distance_squared = (I - i_val)**2
    temp = np.exp(-distance_squared / (2 * sigma_r))
    out = 1 / (np.sqrt(2 * np.pi) * sigma_r) * temp
    return out


def gamma_correct(img):

    thresh = 0.0404482

    img[img <= thresh] = img[img <= thresh]/12.92
    img[img > thresh] = ((img[img > thresh]+0.055)/1.055)**(2.4)

    return img


def bilateral_filtering(img, flash_light = None, lda = 0.01, sigma_r = 0.1, sigma_s=0.1, eps = 1e-6):

    out_image = np.zeros_like(img)

    for c in range(3):
        I = img[:,:,c]

        h,w = I.shape

        if flash_light is None:
            maxI = np.amax(I) + lda
            minI = np.amin(I) - lda
        else:
            maxI = np.amax(flash_light) + lda
            minI = np.amin(flash_light) - lda

        nb_segments = int((maxI - minI) // sigma_r + 1)
        segments = []
        ij_s = []

        for j in range(nb_segments + 1):
            ij = minI + j * ((maxI - minI) / nb_segments)
            ij_s.append(ij)

            if flash_light is not None:
                Gj = gaussian_kernel_r(flash_light[:,:,c], ij, sigma_r=sigma_r)
            else:
                Gj = gaussian_kernel_r(I, ij, sigma_r=sigma_r)

            Kj = cv2.GaussianBlur(Gj, (0,0), 5)
            Hj = Gj * I
            Hj_star = cv2.GaussianBlur(Hj, (0,0), sigmaX = sigma_s)

            Jj = Hj_star/(Kj + eps)
            segments.append(Jj)

        segments = np.array(segments)

        grid_rows = np.linspace(0, h-1, h)
        grid_cols = np.linspace(0, w-1, w)
        grid_ij = ij_s
        grid_points = (grid_ij, grid_rows, grid_cols)
    
        rows_query = np.linspace(0, h-1, h)
        cols_query = np.linspace(8, w-1, w)
        rows_query, cols_query = np.meshgrid (rows_query, cols_query, indexing="ij")

        rows_query = np.expand_dims (rows_query.flatten(), axis=1)
        cols_query = np.expand_dims(cols_query.flatten(), axis=1)
        
        if flash_light is None:
            intensities = I.flatten()
        else:
            intensities = flash_light[:,:,c].flatten()

        intensities = np.expand_dims (intensities.flatten(), axis=1)
        query_points = np.hstack([intensities, rows_query, cols_query])

        interpolated_mg = scipy.interpolate.interpn(grid_points, segments, query_points)
        interpolated_img = np.reshape(interpolated_mg, (h, w))

        out_image[:,:,c] = interpolated_img

    out_image = np.clip(out_image, 0, 1) 
    
    return out_image


def diff_f(image1, image2):
    difference = cv2.absdiff(image1, image2)
    
    # Convert the difference image to grayscale
    print(difference.shape)
    gray_difference = cv2.cvtColor(difference.astype('float32'), cv2.COLOR_BGR2GRAY)

    return gray_difference


# detail 0.35 0.5
# final 0.3 0.5
# joint 0.2 0.5
# bf 0.3 0.5

data = skimage.io.imread('data/lamp/lamp_ambient.tif')/255.0
F = skimage.io.imread('data/lamp/lamp_flash.tif')/255.0


data = skimage.io.imread('custom/img_bf_2.jpg')/255.0
F = skimage.io.imread('custom/flash_bf_2.jpg')/255.0

print(data.max())


n = 1
eps = 1e-2
F = F[::n,::n]
img = data[::n,::n]

h,w,_ = img.shape

print(h,w)

s = [0.5]
r = [0.30]

F_ISO = 200
A_ISO = 1600

F_ISO = 220
A_ISO = 18000

shadow_thresh = 0.1

out_dir = 'out'
out_dir = 'out_q3_1'

fr = F[:,:,0]
fg = F[:,:,1]
fb = F[:,:,2]

for sigma_s in s:
    for sigma_r in r:
        print(sigma_r, sigma_s)
        A_base = bilateral_filtering(img, sigma_r=sigma_r, sigma_s=sigma_s)
        A_nr = bilateral_filtering(img, flash_light=F, sigma_r=sigma_r, sigma_s=sigma_s)

        F_base = bilateral_filtering(F, sigma_r=sigma_r, sigma_s=sigma_s)

        weight = np.clip(F/F_base, 0, 1)

        A_detail = weight*A_nr

        A_l = gamma_correct(img.copy())
        F_l = gamma_correct(F.copy())
        
        A_l = A_l*F_ISO/A_ISO

        mask = (F_l-A_l) < shadow_thresh        
        
        A_final = (1-mask)*A_detail + mask*A_base
        A_final = np.clip(A_final, 0, 1)


        if False:
            diff = diff_f(A_nr, img)
            print(diff.shape)
            skimage.io.imsave(f'out/diff2_{str(sigma_r)}_{str(sigma_s)}.png', (diff*255.0).astype('uint8'))

            diff = diff_f(A_detail, A_nr)
            skimage.io.imsave(f'out/diff3_{str(sigma_r)}_{str(sigma_s)}.png', (diff*255.0).astype('uint8'))

            diff = diff_f(A_final, A_detail)
            skimage.io.imsave(f'out/diff4_{str(sigma_r)}_{str(sigma_s)}.png', (diff*255.0).astype('uint8'))

            diff = diff_f(A_base, img)
            skimage.io.imsave(f'out/diff1_{str(sigma_r)}_{str(sigma_s)}.png', (diff*255.0).astype('uint8'))
    

        skimage.io.imsave(f'{out_dir}/bf_{str(sigma_r)}_{str(sigma_s)}.png', (A_base*255.0).astype('uint8'))
        skimage.io.imsave(f'{out_dir}/joint_{str(sigma_r)}_{str(sigma_s)}.png', (A_nr*255.0).astype('uint8'))
        skimage.io.imsave(f'{out_dir}/detail_{str(sigma_r)}_{str(sigma_s)}.png', (A_detail*255.0).astype('uint8'))
        skimage.io.imsave(f'{out_dir}/final_{str(sigma_r)}_{str(sigma_s)}.png', (A_final*255.0).astype('uint8'))

