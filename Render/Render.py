# Hackathon Code - process geometry for meshing
# Author(s): Matt Smith
# Which one? Spidy matt or venom matt?
# Answer: Yes.

import json
import numpy as np
import sys
import progressbar
import time
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import struct

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

OBJECT_FILE = "geo.json"
MESH_FILE = "test.msh"
MAT_K = {'air': 30.0, 'epoxy': 2.6, 'steel': 50.2, 'aluminium': 205.0, 'copper': 385.0, 'hdpe': 0.48}
MAT_RHO = {'air': 1.25, 'epoxy': 1250.0, 'steel': 7700.0, 'aluminium': 2700.0, 'copper': 8960.0, 'hdpe': 900.0}
MAT_CM = {'air': 717.0, 'epoxy': 710.0, 'steel': 468.0, 'aluminium': 887.0, 'copper': 385.0, 'hdpe': 710.0}
UNIT_CONVERSION = {'MM': 1000.0, 'CM': 100.0, 'M': 1.0}

def create_mesh(D):

    # Define some materials
    # Real values, from all over the internet
    # SI Units
 
    NX = D["NX"]
    NY = D["NY"]
    NZ = D["NZ"]
    L = D["L"]/UNIT_CONVERSION[D["UNITS"]]
    H = D["H"]/UNIT_CONVERSION[D["UNITS"]]
    W = D["W"]/UNIT_CONVERSION[D["UNITS"]]
    print("L = %f" % L)
    # Create a mesh
    X = np.zeros([NX,NY,NZ])
    Y = np.zeros([NX,NY,NZ])
    Z = np.zeros([NX,NY,NZ])
    BODY = np.zeros([NX,NY,NZ])
    # Primitive (P) quantities
    P = np.zeros([NX,NY,NZ, 5])
    # Source energy term
    Q = np.zeros([NX,NY,NZ])
    # Heat conductivity terms - assume air by default
    K = np.ones([NX,NY,NZ])*MAT_K['air']
    # Specific heat capacity
    CM = np.ones([NX,NY,NZ])*MAT_CM['air']

    DX = float(L)/float(NX-1)
    DY = float(H)/float(NY-1)
    DZ = float(W)/float(NZ-1)
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                X[i,j,k] = (i+0.5)*DX
                Y[i,j,k] = (j+0.5)*DY
                Z[i,j,k] = (k+0.5)*DZ
                # Set the density by default to that of air
                P[i,j,k,0] = MAT_RHO['air']
                # Velocities are already 0
                # Set the temperature to room temp
                P[i,j,k,4] = 300.0                   


    # Catagorise shit
    count = 0
    max_count = len(D["objects"])

    print("Computing 3D Bodies and Assigning Material Properties")
    for z in progressbar.progressbar(range(max_count)):
        obj = D["objects"][z]
        i_min = int(obj['x_min']/(DX*UNIT_CONVERSION[D["UNITS"]]))
        i_max = int(obj['x_max']/(DX*UNIT_CONVERSION[D["UNITS"]]))
        j_min = int(obj['y_min']/(DY*UNIT_CONVERSION[D["UNITS"]]))
        j_max = int(obj['y_max']/(DY*UNIT_CONVERSION[D["UNITS"]]))
        k_min = int(obj['z_min']/(DZ*UNIT_CONVERSION[D["UNITS"]]))
        k_max = int(obj['z_max']/(DZ*UNIT_CONVERSION[D["UNITS"]]))

        # Correct minimum thickness
        if (i_min == i_max):
            i_max = i_min + 1
        if (j_min == j_max):
            j_max = j_min + 1
        if (k_min == k_max):
            k_max = k_min + 1

        # These need further verification

        # Set these bodies
        BODY[i_min:i_max, j_min:j_max, k_min:k_max] = obj['body']
        # Assign the material
        try:
            K[i_min:i_max, j_min:j_max, k_min:k_max] = MAT_K[obj['material']]
            CM[i_min:i_max, j_min:j_max, k_min:k_max] = MAT_CM[obj['material']]
            P[i_min:i_max, j_min:j_max, k_min:k_max] = MAT_RHO[obj['material']]
            Q[i_min:i_max, j_min:j_max, k_min:k_max] = obj['Q']

        except:
            print("Unexpected Error: %s" % sys.exc_info()[0])
            print("I bet you are missing a material definition.")
            sys.exit(1)

        # Flux the progress bar stream
        time.sleep(0.1)
        progressbar.streams.flush()



    # Compute some sums to verify
    VOL = DX*DY*DZ
    TOTAL_MASS = np.sum(np.sum(np.sum(P*VOL)))
    TOTAL_BODIES = np.sum(np.sum(np.sum(BODY > 0)))
    print("==== Gridding Report ====")
    print("Total mass = %f kg" % TOTAL_MASS)
    print("Total solid body cells = %d" % TOTAL_BODIES)

    return X, Y, Z, BODY, P, Q, K, CM


def write_input_file(P, BODY, Q, K, CM, D):


    NX = D['NX']
    NY = D['NY'] 
    NZ = D['NZ']
    L = D['L']
    H = D['H'] 
    W = D['W']
    SIM_TIME = D['SIM_TIME']

    # Write the file out
    filename = "geom.bin"
    with open(filename, "wb") as f:
        # Time 
        f.write(struct.pack('f', SIM_TIME))
        # Save number of cells (x,y,z)
        f.write(struct.pack('i', NX))
        f.write(struct.pack('i', NY))
        f.write(struct.pack('i', NZ))
        # Domain size
        f.write(struct.pack('f', L))
        f.write(struct.pack('f', H))
        f.write(struct.pack('f', W))

        # Write it all
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    f.write(struct.pack('f', P[i,j,k,0]))
                    f.write(struct.pack('f', P[i,j,k,1]))
                    f.write(struct.pack('f', P[i,j,k,2]))
                    f.write(struct.pack('f', P[i,j,k,3]))
                    f.write(struct.pack('f', P[i,j,k,4]))
                    f.write(struct.pack('i', int(BODY[i,j,k])))
                    f.write(struct.pack('f', Q[i,j,k]))
                    f.write(struct.pack('f', K[i,j,k]))
                    f.write(struct.pack('f', CM[i,j,k]))

    # And that's it


def export_mesh(BODY, D, filename):
    # Export anything with a body which is not 0

    with open(filename, "w") as f:
        f.write("[\n")
        for i in range(D['NX']):
            for j in range(D['NY']):
                for k in range(D['NZ']):
                    # For now give a normalized fake temp
                    fake_temp = int(15*float(i)/float(D['NX']))
                    if (BODY[i,j,k] != 0):
                        # Write it to file
                        f.write("[{}, {}, {}, {}, {}],\n".format(i,j,k,int(BODY[i,j,k]),fake_temp))

        f.write("]\n")


def Process_Error(info):
	if (str(info[0]) == "<type 'exceptions.KeyError'>"):
		print("Missing data detected in JSON file: Check value for %s" % info[1]);

	if (str(info[0]) == "<class 'json.decoder.JSONDecodeError'>"):
		print("JSON file corrupted or invalid")


def load_json_from_file(filename):

    try:
        with open(filename, "r") as f:
            print("Reading file %s" % filename)
            json_data = json.load(f)
    except:
        print("Error Detected during load")
        Process_Error(sys.exc_info())
        print("=== JSON Compile Log ===")
        print(str(sys.exc_info()[1]))
        return 

    # Load in simulation values from JSON, verify
    # TODO: Expand on this verification, its far too short.
    try:
        NX = json_data["NX"]
        NY = json_data["NY"]
        NZ = json_data["NZ"]

    except:
        print("Error Detected during parse")
        Process_Error(sys.exc_info())
        print("=== Parse Log ===")
        print(sys.exc_info())
        return

    return json_data


# Draw stuff, sanity check
def render(BODY, D):
    mb1 = BODY == 1
    mb2 = BODY == 2
    bus = BODY == 3
    voxels = mb1 | mb2 | bus
    colors =  np.empty(voxels.shape, dtype=object)
    colors[mb1] = 'red'
    colors[mb2] = 'blue'
    colors[bus] = 'green'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # This works
    #ax.voxels(BODY[::4, ::4, ::4], facecolors='red', edgecolor='k')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    plt.show()


# Start the ball rolling
def main():
    print("Hello there")
    json_dict = load_json_from_file(OBJECT_FILE)

    X,Y,Z,BODY,P,Q,K,CM = create_mesh(json_dict)

    # Export a fake world for Matt
    export_mesh(BODY, json_dict, MESH_FILE)

    # Write the input file for the solver
    write_input_file(P, BODY, Q, K, CM, json_dict)

    # Render it
    #render(BODY, json_dict)


if __name__ == "__main__":
    main()
