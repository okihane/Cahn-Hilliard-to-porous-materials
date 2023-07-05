import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

# Simulation cell parameters:
Nx=50
Ny=50
Nz=50
dx=1.0
dy=1.0
dz=1.0
# Time integration parameters:
nstep=6000
nprint=500
dtime=0.015
# Material specific Parameters:
c0=0.50
mobility=1.0
grad_coef=0.5
# prepare microstructure:
conr = np.random.rand(Nx, Ny, Nz)
con1 = np.ones((Nx, Ny, Nz))
con = c0*con1 + 0.1*conr
lap_con = np.zeros((Nx, Ny, Nz))
dummy = np.zeros((Nx, Ny, Nz))
lap_dummy = np.random.rand(Nx, Ny, Nz)
# Evolve
def lap_(con):
	return (np.roll(con, 1, axis=0) + np.roll(con, -1, axis=0) + np.roll(con, 1, axis=1) + np.roll(con, -1, axis=1) + np.roll(con, 1, axis=2) + np.roll(con, -1, axis=2) - 6 * con) / (dx*dx)
for istep in range(nstep):
    lap_con = lap_(con)
    # derivative of free energy:
    dfdcon=1.0*(2.0*con*(1-con)**2 -2.0*con**2*(1.0-con))
    #dfdcon=con**3 - con
    dummy = dfdcon-grad_coef*lap_con

    lap_dummy = lap_(dummy)
    # time integration:
    con += dtime*mobility*lap_dummy
    if((istep == 0) or ((istep+1) % nprint)==0):
        print(f'done step: {istep+1}')
        print('Maximum concentration = ', np.max(con[5]))
        print('Minimum concentration = ', np.min(con[5]))
        plt.imshow(con[Nx/2], cmap='bwr')
        plt.colorbar()
        plt.show()
# generate lammps data file
A = 3.3 # lattice
num1 = -1
num2 = -1
id1 = 0
id2 = 0
def writehead(fln,atoms):
	with open(fln,'w') as fw:
		fw.write('#ch eq to lammps data\n')
		fw.write(f'\t{atoms*2}  atoms\n')
		fw.write('\t  1\tatom  types\n\n')
		fw.write('\t0.00\t{:.2f}  xlo xhi\n'.format(A*Nx))
		fw.write('\t0.00\t{:.2f}  ylo yhi\n'.format(A*Ny))
		fw.write('\t0.00\t{:.2f}  zlo zhi\n\n'.format(A*Nz))
		fw.write('Masses\n\n')
		fw.write('\t  1\t92.90638\t# Nb\n\n')
		fw.write('Atoms\n\n')

atoms = 0
for i in range(Nx):
	for j in range(Ny):
		for k in range(Nz):
			if con[i,j,k] > 0.5:
				atoms+=1
writehead(f'ta{Nx}.lmp',atoms)
writehead(f'tb{Nx}.lmp',Nx*Ny*Nz-atoms)

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if con[i,j,k] > 0.5:
                num1 += 1
                id1 = num1 * 2
                pos = []
                pos.append([i*A,j*A,k*A])
                pos.append([(i+0.5)*A,(j+0.5)*A,(k+0.5)*A])
                with open(f'ta{Nx}.lmp','a') as fw:
                    for m in range(2):
                        fw.write('\t{}\t1\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(id1+m+1,pos[m][0],pos[m][1],pos[m][2]))
            else:
                num2 += 1
                id2 = num2 * 2
                pos = []
                pos.append([i*A,j*A,k*A])
                pos.append([(i+0.5)*A,(j+0.5)*A,(k+0.5)*A])
                with open(f'tb{Nx}.lmp','a') as fw:
                    for m in range(2):
                        fw.write('\t{}\t1\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(id2+m+1,pos[m][0],pos[m][1],pos[m][2]))
print(f'countA = {id1+2}')
print(f'countB = {id2+2}')
