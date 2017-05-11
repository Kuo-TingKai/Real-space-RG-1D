"""
real_space_RG_1D.py
Real-Space RG 1D
version 1.0

This code is a simple implementation of real-space Renormalization Group (RG) for 
one-dimensional (chain-lattice) quantum Ising model in transverse field.
Copyright (C) 2017  Jozef Genzor <jozef.genzor@gmail.com> 

This file is part of "Real-Space RG 1D".

"Real-Space RG 1D" is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

"Real-Space RG 1D" is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with "Real-Space RG 1D".  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy import linalg

id = np.eye(2)
s_x = np.matrix([[0, 1], [1, 0]])
s_z = np.matrix([[1, 0], [0, -1]])

D = 16

def construct_two_copies(H):
    dim  = H.shape[0]
    id_matrix = np.eye(dim)
    result = np.kron(id_matrix, H) + np.kron(H, id_matrix) 
    return result

def initialize_hamiltonian(h, h_help):
    H = - np.kron(s_z, s_z)
    H = H - h * construct_two_copies(s_x) - h_help * construct_two_copies(s_z)
    return H

def construct_new_hamiltonian(H, s1, s2):
    result = - np.kron(s1, s2)
    result += construct_two_copies(H)
    return result

def construct_new_mag_operator(M):
    return construct_two_copies(M) / 2

EPS = 1.E-14
h_help = 1.E-10
start = 0
end = 2
step = 1.E-2

f = open('data.txt', 'a')
f.write('# Ising model on one-dim chain\n')
f.write('# D=%d, h_help=%E\n' % (D, h_help))
f.write('# h\tenergy\t\t\ts_z\t\t\ts_x\t\t\titer\n')
f.close()

h = start 
while h <= end:    
    H = initialize_hamiltonian(h, h_help)
    M_z = construct_new_mag_operator(s_z)
    M_x = construct_new_mag_operator(s_x)
    
    s1, s2, s3 = s_z, s_z, s_z
    size = 2

    ground_energy = 1 
    ground_energy_new = 0
    
    mag_z = 1 
    mag_z_new = 0 
    
    mag_x = 1 
    mag_x_new = 0 
    
    print('# h\tenergy\t\t\ts_z\t\t\ts_x\t\t\titer')
    
    iter = 0
    while ( abs(ground_energy - ground_energy_new) > EPS 
           or abs(mag_z - mag_z_new) > EPS 
           or abs(mag_x - mag_x_new) > EPS ) and (iter < 100):
                
        #w, v = np.linalg.eigh(H)
        w, v = linalg.eigh(H)
            
        idx = w.argsort()[::1]

        w = w[idx]
        v = v[:,idx]
        if w.shape[0] > D:
            w = w[:D]
            v = v[:,:D]
        ground_energy = ground_energy_new
        ground_energy_new = w[0] / size
                   
        mag_z = mag_z_new
        mag_z_new = np.transpose(v[:, 0, None]) * M_z * v[:, 0, None]

        mag_x = mag_x_new
        mag_x_new = np.transpose(v[:, 0, None]) * M_x * v[:, 0, None]
            
        M_z = np.transpose(v) * M_z * v
        M_x = np.transpose(v) * M_x * v
        
        print('%.3f\t%.15f\t%.15f\t%.15f\t%d' % (h,  ground_energy_new, mag_z_new, mag_x_new, iter))
                
        H = np.diag(w)
        
        dim = s1.shape[0]
        id_matrix = np.eye(dim)
        s1 = np.transpose(v) * np.kron(id_matrix, s1) * v
        s2 = np.transpose(v) * np.kron(s2, id_matrix) * v
        H = construct_new_hamiltonian(H, s1, s2)
        M_z = construct_new_mag_operator(M_z)
        M_x = construct_new_mag_operator(M_x)
        size *= 2
        iter += 1
        
    f = open('data.txt', 'a')
    f.write('%.3f\t%.15f\t%.15f\t%.15f\t%d\n' % (h,  ground_energy_new, mag_z_new, mag_x_new, iter))
    f.close()
    h += step
    
