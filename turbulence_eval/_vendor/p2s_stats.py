'''
This file contains the codes for generating correlation matrices for tilt
and higher-order abberations. 

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

import math
import numpy as np
from scipy.special import jv
import scipy.integrate as integrate

def tilt_mat(N, D, r0, L, save_path = './data', thre = 0.002, adj = 1, use_temp = False): 
    '''
    Return tilt correlation data without writing files.

    N is the square simulation size (128, 256, 512 or 1024); D is the aperture
    diameter, r0 the Fried parameter, and L the propagation distance, in metres.
    adj scales the spatial sampling interval. thre suppresses Fourier amplitudes
    below that fraction of the maximum amplitude.

    The result contains s_half (real and imaginary Fourier components) and const
    (the displacement scaling factor). save_path and use_temp are unused;
    the simulation adapter manages caching of the returned data.
    '''
    # N: image size
    # D: Aperture diameter
    # r0: Fried parameter
    # L: Propagation distance
    D_r0 = D/r0
    wavelength = 0.500e-6
    k = 2*np.pi/wavelength
    delta0  = L*wavelength/(2*D)
    delta0 *= adj# Adjusting factor
    c1 = 2*((24/5)*math.gamma(6/5))**(5/6);
    c2 = 4*c1/np.pi*(math.gamma(11/6))**2;
    smax = delta0/D*N
    spacing = delta0/D
    I0_arr, I2_arr = _calculate_integral(smax, spacing)

    i, j = np.int32(N/2), np.int32(N/2)
    [x,y] = np.meshgrid(np.arange(1,N+0.01,1),np.arange(1,N+0.01,1))
    s = np.sqrt((x-i)**2 + (y-j)**2)
    s *= spacing

    C0 = (_In_m(s, spacing, I0_arr) + _In_m(s, spacing, I2_arr))/_I0(0)
    C0[i,j] = 1
    C0_scaled = C0*_I0(0)*c2*((D_r0)**(5/3))/(2**(5/3))*((2*wavelength/(np.pi*D))**2)*2*np.pi
    Cfft =  np.fft.fft2(C0_scaled)
    S_half =  np.sqrt(Cfft)
    S_half_max = np.max(np.max(np.abs(S_half)))
    S_half[np.abs(S_half) < thre*S_half_max] = 0
    S_half_new = np.zeros((2,N,N))
    S_half_new[0] = np.real(S_half)
    S_half_new[1] = np.imag(S_half)
    data = {}
    data['s_half'] = S_half_new
    data['const'] = np.sqrt(2)*N*(L/delta0)
    
    return data


def _I0(s):
    I0_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2), 0, 1e3, limit = 100000)

    return I0_s

def _I2(s):
    I2_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2), 0, 1e3, limit = 100000)
    
    return I2_s

def _calculate_integral(s_max, spacing):
    s_arr = np.arange(0,s_max,spacing)
    I0_arr = np.float32(s_arr*0)
    I2_arr = np.float32(s_arr*0)
    for i in range(len(s_arr)):
        I0_arr[i] = _I0(s_arr[i])
        I2_arr[i] = _I2(s_arr[i])
        
    return I0_arr, I2_arr

def _In_m(s, spacing, In_arr):
    idx = np.int32(np.floor(s.flatten()/spacing))
    M,N = np.shape(s)[0], np.shape(s)[1]
    In = np.reshape(np.take(In_arr, idx), [M,N])

    return In
