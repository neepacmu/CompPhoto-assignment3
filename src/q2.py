
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import signal


def gradient(img):

    Iy = np.diff(img, n = 1, axis = 0, append=0)
    Ix = np.diff(img, n = 1, axis = 1, append=0)

    return np.dstack([Ix, Iy])

def divergence(grad):
    Ixx = np.diff(grad[:,:,0], n = 1, axis = 1, append=0)
    Iyy = np.diff(grad[:,:,1], n = 1, axis = 0, append=0)

    return Ixx + Iyy

def lapl(image):

    filter = [[0, 1, 0],[1, -4, 1],[0,1,0]]
    out = signal.convolve2d(image, filter, mode = 'same', boundary= 'fill', fillvalue=0)

    return out

def cgd(D, I_init, B, Ib, eps = 0, N = 1000):

    Is = I_init*B + (1 - B)*Ib
    r = B * (D - lapl(Is))
    d = r
    delta_new = np.sum(r**2)

    n = 0
    while delta_new > eps and n < N:
        q = lapl(d)
        new = delta_new/(np.sum(q*d))

        Is = Is + B*(new*d)
        r = B*(r - new*q)
        delta_prev = delta_new
        delta_new = np.sum(r**2)
        beta = delta_new/delta_prev

        d = r + beta*d 

        n += 1

    return Is


def get_border(A):
    B = np.zeros_like(A)

    B[0,:] = 1
    B[-1,:] = 1
    B[:,0] = 1
    B[:,-1] = 1

    return B


def test_implementation(A):

    grad = gradient(A)
    lap2 = lapl(A)

    Iz = np.zeros_like(A)

    B = get_border(A)
    
    I_b = B*A

    I_init = cgd(lap2, Iz, 1 - B, I_b)

    print((I_init - A).sum())
    plt.imshow(I_init)
    plt.show()

    plt.imshow(A)
    plt.show()





if __name__ == "__main__":

    A_in = skimage.io.imread('data/museum/museum_ambient.png')
    F_in = skimage.io.imread('data/museum/museum_flash.png')
    n = 4
    sigma = 10
    t = 0.6

    F_in = F_in[::n,::n]/255.0
    A_in = A_in[::n,::n]/255.0

    h,w,_ = A_in.shape

    out_image = []

    for i in range(3):

        A = A_in[:,:,i]
        F = F_in[:,:,i]
        B = get_border(A)

        A_g = gradient(A.copy())
        F_g = gradient(F.copy())

        mask = np.abs(F_g[:,:,0]*A_g[:,:,0] + F_g[:,:,1]*A_g[:,:,1])

        temp_num = (np.sqrt(F_g[:,:,0]**2 + F_g[:,:,1]**2)) * (np.sqrt(A_g[:,:,0]**2 + A_g[:,:,1]**2))

        mask = mask/(temp_num + 1e-7)


        ws = np.tanh(sigma * (F - t))

        ws = (ws - np.min(ws))
        ws = ws/np.max(ws)
        
        
        phi_star_x = ws*A_g[:,:,0] + (1 - ws)*(mask*F_g[:,:,0] + (1 - mask)*A_g[:,:,0])
        phi_star_y = ws*A_g[:,:,1] + (1 - ws)*(mask*F_g[:,:,1] + (1 - mask)*A_g[:,:,1])
        plt.imsave(f'out_q2/phi_star_x.png', phi_star_x)
        plt.imsave(f'out_q2/phi_star_y.png', phi_star_y)

        plt.imsave(f'out_q2/A_x.png', (A_g[:,:,0]))
        plt.imsave(f'out_q2/A_y.png', (A_g[:,:,1]))

        plt.imsave(f'out_q2/F_x.png', (F_g[:,:,0]))
        plt.imsave(f'out_q2/F_y.png', (F_g[:,:,1]))

        phi_star = np.dstack([phi_star_x, phi_star_y])

        Iz = np.zeros_like(A)
        Ib = B*F

        D = divergence(phi_star)

        I_out = cgd(D, Iz, 1 - B, Ib)

        out_image.append(I_out)

    out_image = np.dstack(out_image)
    out_image = np.clip(out_image, 0, 1)

    skimage.io.imsave(f'out_q2/out.png', (out_image*255.0).astype('uint8'))
        








