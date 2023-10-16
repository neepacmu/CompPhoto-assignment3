
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

def cgd(D, I_init, B, Ib, eps = 1e-3, N = 3000):

    Is = I_init*B + (1 - B)*Ib
    r = B * (D - lapl(Is))
    d = r
    delta_new = np.sum(r**2)

    n = 0
    while delta_new > eps:
        #print(n)
        q = lapl(d)
        new = delta_new/(np.sum(q*d))

        Is = Is + B*(new*d)
        r = B*(r - new*q)
        delta_prev = delta_new
        delta_new = np.sum(r**2)
        beta = delta_new/delta_prev

        d = r + beta*d 

        n += 1

        print(n , delta_new, end = '\r', sep = ' ', flush=True)

    print("")

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
    lap1 = divergence(grad)
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

    A_in = skimage.io.imread('custom/img6.jpg')
    F_in = skimage.io.imread('custom/flash6.jpg')
    n = 1
    sigma = 60
    t = 0.6

    F_in = F_in[::n,::n]/255.0
    A_in = A_in[::n,::n]/255.0

    h,w,_ = A_in.shape

    #test_implementation(A_in[:,:,0])

    out_image = []
    print(h, w)

    for i in range(3):
        print(i)
        A = A_in[:,:,i]
        F = F_in[:,:,i]
        B = get_border(A)

        A_g = gradient(A.copy())
        F_g = gradient(F.copy())

        mask = np.abs(F_g[:,:,0]*A_g[:,:,0] + F_g[:,:,1]*A_g[:,:,1])

        temp_d = (np.sqrt(F_g[:,:,0]**2 + F_g[:,:,1]**2)) * (np.sqrt(A_g[:,:,0]**2 + A_g[:,:,1]**2))

        mask = mask/(temp_d + 1e-7)

        # plt.imshow(mask)
        # plt.show()

        ws = np.tanh(sigma * (F - t))

        ws = (ws - np.min(ws))
        ws = ws/np.max(ws)

        #mask = np.expand_dims(mask, axis = -1)
        skimage.io.imsave(f'out_q3/mask_{str(i)}.png', (mask*255.0).astype('uint8'))
        
        
        phi_star_x = ws*A_g[:,:,0] + (1 - ws)*(mask*F_g[:,:,0] + (1 - mask)*A_g[:,:,0])
        phi_star_y = ws*A_g[:,:,1] + (1 - ws)*(mask*F_g[:,:,1] + (1 - mask)*A_g[:,:,1])

        phi_star = np.dstack([phi_star_x, phi_star_y])

        Iz = np.zeros_like(A)
        Ib = B*F

        D = divergence(phi_star)

        I_out = cgd(D, A, 1 - B, Ib)

        out_image.append(I_out)

    out_image = np.dstack(out_image)
    out_image = np.clip(out_image, 0, 1)

    skimage.io.imsave(f'out_q3/out_{str(sigma)}_{str(t)}_flash.png', (out_image*255.0).astype('uint8'))
        








