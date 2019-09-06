# Euler.py

import time
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


# Solution Parameters
R = 1.0
GAMMA = 1.4
CV = R/(GAMMA-1.0)
CP = CV + R
L = 1.0    # Length of region (x direction), m
H = 1.0    # Height of region (y direction), m
NX = 50	   # Number of cells in X direction
NY = 50    # Number of cells in Y direction
DX = (L/NX) # Grid size in X direction
DY = (H/NY) # "      "     Y direction
DT = 1e-3 # Time step, in seconds (s). Static for now.
NO_STEPS = 10000 # Number of time steps to take
D = 0.9		# Anti-dissipation coefficient
G = 0.001        # Normalized gravity
USE_FAST = True		# Decide which routines to use - fast (True) or not (False)
K = 0.001

# ========== Function declarations ============


def Compute_State(P, U, NX, NY):
        # Compute Primitives P from U using vectors
        P[:,:,0] = U[:,:,0]   		# Water Height
        P[:,:,1] = U[:,:,1]/U[:,:,0]	# X vel
        P[:,:,2] = U[:,:,2]/U[:,:,0]	# Y vel
        P[:,:,3] = ((U[:,:,3]/U[:,:,0]) - 0.5*(P[:,:,1]*P[:,:,1]+P[:,:,2]*P[:,:,2]))/CV # Temp	
        CFL = (P[:,:,1] + 2.0*np.sqrt(GAMMA*R*P[:,:,3]))*DT/DX
        print(np.max(np.max(CFL)))
        print(np.max(np.max(P[:,:,2])))
        print(np.max(np.min(P[:,:,2])))
        return P

def Compute_Change_of_State(dU, FP, FM, P, DX, DY, NX, NY, PHI, direction):
        # Compute the change of state based on gradients of F
        FL = np.ndarray([NX,NY,4])
        FR = np.ndarray([NX,NY,4])

        if (direction == 0):
                # This is the X direction computation
                for x in range(NX):
                        if (x == 0):
                                # Left boundary - use reflective
                                FL[x,:,:] = -FM[x,:,:]
                                FL[x,:,1] = -FL[x,:,1]  # This is reversed
                                FR[x,:,:] = FM[x+1,:,:]
                        elif (x == (NX-1)):
                                # Right boundary - use reflective
                                FL[x,:,:] = FP[x-1,:,:]
                                FR[x,:,:] = -FP[x,:,:]
                                FR[x,:,1] = -FR[x,:,1]  # Reversed
                                
                                # Add an energy flux
                                FR[x,:,3] = FR[x,:,3] - (K/DX)*(2.0-P[x,:,3])
                                
                        else:
                                # Internal cell. All is fine and dandy.
                                FL[x,:,:] = FP[x-1,:,:]
                                FR[x,:,:] = FM[x+1,:,:]


                # Now to apply them to update the state
                dU = dU - PHI*(FP - FM + FR - FL)

        elif (direction == 1):
                # This is for the Y direction
                for y in range(NY):
                        if (y == 0):
                                # Left boundary - use reflective
                                FL[:,y,:] = -FM[:,y,:]
                                FL[:,y,2] = -FL[:,y,2] # Reverse this one
                                FR[:,y,:] = FM[:,y+1,:]
                        elif (y == (NY-1)):
                                # Right boundary - use reflective
                                FL[:,y,:] = FP[:,y-1,:]
                                FR[:,y,:] = -FP[:,y,:]
                                FR[:,y,2] = -FR[:,y,2]
                        else:
                             	# Internal cell.
                                FL[:,y,:] = FP[:,y-1,:]
                                FR[:,y,:] = FM[:,y+1,:]

                # Now to apply them to update the state
                dU = dU - PHI*(FP - FM + FR - FL)

        return dU



def Compute_Fluxes(P, Y, NX, NY, direction):
	# Compute split SHLL fluxes in each direction
        FP = np.ndarray([NX, NY, 4])
        FM = np.ndarray([NX, NY, 4])
        U = np.ndarray([NX, NY, 4])


        # Compute fluxes in particular direction as required
        F = np.ndarray([NX,NY,4])


        # Recompute a local copy of U based on P
        U[:,:,0] = P[:,:,0]
        U[:,:,1] = P[:,:,0]*P[:,:,1]
        U[:,:,2] = P[:,:,0]*P[:,:,2]
        U[:,:,3] = P[:,:,0]*(CV*P[:,:,3] + 0.5*(P[:,:,1]*P[:,:,1] + P[:,:,2]*P[:,:,2]))

        # Use the direction to access the correct velocity information
        a = np.sqrt(P[:,:,3]*GAMMA*R)
        Pressure = R*P[:,:,0]*P[:,:,3]

        if (direction == 0):
                # X direction.
                # Attempt to vectorize the computation
                vel = P[:,:,1]  # X velocity
                Fr = vel/a  # Mach number
                # Compute some important constants
                Z1 = 0.5*(D*Fr+1.0)
                Z2 = 0.5*D*a*(1.0-Fr*Fr)
                Z3 = 0.5*(D*Fr-1.0)

                # Compute X direction fluxes
                Pressure = R*P[:,:,0]*P[:,:,3]
                F[:,:,0] = U[:,:,1]   
                F[:,:,1] = U[:,:,1]*P[:,:,1] + Pressure 
                F[:,:,2] = U[:,:,1]*P[:,:,2] 
                F[:,:,3] = P[:,:,1]*(U[:,:,3] + Pressure)

                # Compute the SHLL forward and backward fluxes
                FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
                FP[:,:,1] = F[:,:,1]*Z1 + U[:,:,1]*Z2
                FP[:,:,2] = F[:,:,2]*Z1 + U[:,:,2]*Z2
                FP[:,:,3] = F[:,:,3]*Z1 + U[:,:,3]*Z2

                FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
                FM[:,:,1] = -F[:,:,1]*Z3 - U[:,:,1]*Z2
                FM[:,:,2] = -F[:,:,2]*Z3 - U[:,:,2]*Z2
                FM[:,:,3] = -F[:,:,3]*Z3 - U[:,:,3]*Z2

        elif (direction == 1):
                # Y direction
                vel = P[:,:,2]  # Y Velocity
                Fr = vel/a
                # Compute some important constants
                Z1 = 0.5*(D*Fr+1.0)
                Z2 = 0.5*D*a*(1.0-Fr*Fr)
                Z3 = 0.5*(D*Fr-1.0)

                # Compute Y direction fluxes
                F[:,:,0] = U[:,:,2]   
                F[:,:,1] = U[:,:,2]*P[:,:,1] 
                F[:,:,2] = U[:,:,2]*P[:,:,2] + Pressure + P[:,:,0]*G*Y
                F[:,:,3] = P[:,:,2]*(U[:,:,3] + Pressure + P[:,:,0]*G*Y)

                # Compute the SHLL forward and backward fluxes
                FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
                FP[:,:,1] = F[:,:,1]*Z1 + U[:,:,1]*Z2
                FP[:,:,2] = F[:,:,2]*Z1 + U[:,:,2]*Z2
                FP[:,:,3] = F[:,:,3]*Z1 + U[:,:,3]*Z2

                FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
                FM[:,:,1] = -F[:,:,1]*Z3 - U[:,:,1]*Z2
                FM[:,:,2] = -F[:,:,2]*Z3 - U[:,:,2]*Z2
                FM[:,:,3] = -F[:,:,3]*Z3 - U[:,:,3]*Z2

        return FP, FM

def Plot_Surface(Data,DX,DY,NX,NY):
        # Create a surface plot of the 2D data Data
        L = NX*DX
        H = NY*DY
        X,Y = np.mgrid[0:L:DX, 0:H:DY]
        Z = Data
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax.set_title('Spatial variation of Data')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

def Plot_Quiver(P,DX,DY,NX,NY):
        # Create a surface plot of the 2D data Data
        L = NX*DX
        H = NY*DY
        X,Y = np.mgrid[0:L:DX, 0:H:DY]
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        Q = ax1.quiver(X, Y, P[:,:,1], P[:,:,2],  units='width')
        #qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
        #plt.show()
        plt.savefig('Quiver.png')

def Plot_All_Surfaces(P, DX, DY, NX, NY):
        # Plot all results within a single figure
        L = NX*DX
        H = NY*DY
        X,Y = np.mgrid[0:L:DX, 0:H:DY]
        Z = P[:,:,0] 
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        fig,ax = plt.subplots(2,2,figsize=(15, 5),subplot_kw=dict(projection='3d'))
        ax[0,0].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[0,0].set_title('Rho')
        ax[0,0].set_xlabel('X')
        ax[0,0].set_ylabel('Y')
        # X Velocity
        Z = P[:,:,1] # X Vel
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax[0,1].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[0,1].set_title('X Velocity')
        ax[0,1].set_xlabel('X')
        ax[0,1].set_ylabel('Y')
        # Y Velocity
        Z = P[:,:,2] # Y Vel
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax[1,0].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[1,0].set_title('Y Velocity')
        ax[1,0].set_xlabel('X')
        ax[1,0].set_ylabel('Y')
        # Temp
        Z = P[:,:,3] 
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax[1,1].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[1,1].set_title('Temp')
        ax[1,1].set_xlabel('X')
        ax[1,1].set_ylabel('Y')
        #plt.show()
        plt.savefig('Results.png')

        return 0


def Init(U, P, X, Y, NX, NY):
        # Compute initial values of U and P
        for x in range(NX):
                for y in range(NY):
                        # Compute initial conditions based on location
                        X[x,y] = x*(DX+0.5)
                        Y[x,y] = y*(DY+0.5)
                        if ((x < 0.6*NX) and (x > 0.4*NX) and (y < 0.6*NY) and (y > 0.4*NY)):
                                #P[x,y,0] = 2.0   # Density
                                #P[x,y,3] = 0.5   # Density
                                P[x,y,0] = 1.0  # Temp
                                P[x,y,3] = 1.0  # Temp
                        else:
                                P[x,y,0] = 1.0
                                P[x,y,3] = 1.0  # Temp
                        # Flow stationary
                        P[x,y,1] = 0.0  # X speed
                        P[x,y,2] = 0.0  # Y speed

        # Now to compute U from P
        U[:,:,0] = P[:,:,0]     # Density is conservative
        U[:,:,1] = P[:,:,0]*P[:,:,1] # X Mom
        U[:,:,2] = P[:,:,0]*P[:,:,2] # Y Mom
        U[:,:,3] = P[:,:,0]*CV*P[:,:,3] # Energy

        return U, P, X, Y



# Create our nd arrays
X = np.ndarray([NX, NY])
Y = np.ndarray([NX, NY])
P = np.ndarray([NX, NY, 4])   # Primitives - height, water speed in x,y directions
U = np.ndarray([NX, NY, 4])   # Conservatives - height and momentum in x,y directions
FP = np.ndarray([NX, NY, 4])  # Forward (positive, P) fluxes of conserved quantities
FM = np.ndarray([NX, NY, 4])  # Backward (minus, M) fluxes
dU = np.ndarray([NX, NY, 4])  # Changes to conserved quantities

# Initialize the flow field
U,P,X,Y = Init(U,P,X,Y,NX,NY)

# Run the solver
tic = time.time()
for step in range(NO_STEPS):
        print("Time step %d of %d" % (step, NO_STEPS))

        # Reset dU
        dU = 0*dU

        # Calculate fluxes in X direction (direction = 0)
        FP, FM = Compute_Fluxes(P, Y, NX, NY, 0)
        # Update dU
        dU = Compute_Change_of_State(dU, FP, FM, P, DX, DY, NX, NY, (DT/DX), 0)

        # Update fluxes in Y direction (direction = 1)
        FP, FM = Compute_Fluxes(P, Y, NX, NY, 1)
        # Update dU
        dU = Compute_Change_of_State(dU, FP, FM, P, DX, DY, NX, NY, (DT/DY), 1)

        # Update dU
        U = U + dU

        P = Compute_State(P, U, NX, NY)

        # End of main transient time loop segment

# Simulation is completed now. Examine the results.
toc = time.time()
Elapsed_Time = toc - tic
print("Elapsed time = %g seconds" % Elapsed_Time)
# Plot FP
Plot_Quiver(P,DX,DY,NX,NY)
Plot_All_Surfaces(P,DX,DY,NX,NY)

