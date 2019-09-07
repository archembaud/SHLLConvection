/*
	GPU-accelerated 3D compressible flow solver for blastwave
	Matthew Smith
	Last Updated: 10th Oct, 2018
	Usage:
	If executable is solver.run, from the command prompt:
		solver.run aaaabbbbccccdddd.bin <enter>
	where aaaabbbbccccdddd.bin is the solver input file, and
	aaaabbbbccccdddd is the UID for the simulation.

	Change Log
	----------
	9th Oct: PC_1 files have been copied into PC_1 from ~/Blastwave/PC_3.
		 These are then modified to (i) parse input file, and (ii) use
		 geometry and simulation constants contained within it.
		 The work associated with this act is extensive.

*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "gpu_main.h"
#include <cstdlib>

// Important Solution Variables
float *d_Fp, *d_Fm, *d_Hp, *d_Hm, *d_Gp, *d_Gm;
float *d_U, *d_P, *d_dU;
float *h_P;
float *h_X, *d_X;
float *h_CM, *h_Q, *h_K;
float *d_CM, *d_Q, *d_K;
float *d_Pmax, *h_Pmax; // Maximum variables

// Body Information
char *h_Type, *d_Type;

// Temperature and pressure ratios
float T_RATIO[128];
float P_RATIO[128];
char NO_BLASTS;

// Key simulation parameters are global
int NX, NY, NZ, N;		// Number of cells in each coordinate direction, total number of cells
float L, W, H;			// Simulation domain size
float DX, DY, DZ;
int NO_STEPS;			// Nuumber of steps - this is for recording purposes only
float h_DT;			// Timestep held on host (CPU) - the GPU version is declared in gpumain.cu
float SIM_TIME;			// Total desired dimensionless simulation time
float TIME;			// Current simulation time
float SAVE_FREQ;		// The frequency to save at
int RED_MULT;			// Reduction multiplier

// Remember the UID
char UID[16];

// TraceID
int TRACE_ID[128];		// Store for now, use later on.
char NO_TRACES;

double Get_Wall_Time() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return (double)time.tv_sec + (double)time.tv_usec*0.000001;

}

int Parse_Input(char *Filename) {

	// Do nothing at this stage.
	FILE *pFile;
	float buffer_f[3];
	float buffer_p[5];
	int buffer_i[3];
	char buffer_c[3];
	char buffer_id[16];
	int body_count =0;
	int index;
	int count;

	// Reading the file
	pFile = fopen(Filename, "rb");
	if (pFile == NULL) {
		printf("Parse_File() error: Cannot open file\n");
		return 1;
	} 
	// Simulation time and frequency
    fread(buffer_f, sizeof(float), 1, pFile);
	// Check the simulation time
	if ((buffer_f[0] > 0.0) && (buffer_f[0] < 10.0)) {
		// All clear
		SIM_TIME = buffer_f[0];
	} else {
		printf("Parse_File() Error: Simulation Time invalid\n");
		fclose(pFile);
		return 1;
	}
	printf("	Time = %g\n", buffer_f[0]);

	// Load the number of cells in each coordinate direction
	fread(buffer_i, 3*sizeof(int), 1, pFile);
	if ((buffer_i[0] > 0) && (buffer_i[0] < 10000)) {
		printf("        NX: %d\n", buffer_i[0]);
		NX = buffer_i[0];
	} else {
		printf("ParseFile() Error: NX invalid (%d)\n", NX);
		fclose(pFile);
		return 1;
	}

	if ((buffer_i[1] > 0) && (buffer_i[1] < 10000)) {
		printf("        NY: %d\n", buffer_i[1]);
		NY = buffer_i[1];
	} else {
		printf("ParseFile() Error: NY invalid (%d)\n", NY);
		fclose(pFile);
		return 1;
	}

	if ((buffer_i[2] > 0) && (buffer_i[2] < 10000)) {
		printf("        NZ: %d\n", buffer_i[2]);
		NZ = buffer_i[2];
	} else {
		printf("ParseFile()  Error: NZ invalid (%d)\n",NZ);
		fclose(pFile);
		return 1;
	}

	// Set the total number of cells
	N = NX*NY*NZ;
	// Can allocate memory now
	Allocate_Memory();

	// Grab the domain size
	fread(buffer_f, 3*sizeof(float), 1, pFile);
	if ((buffer_f[0] > 0) && (buffer_f[0] < 100.0)) {
		printf("        L = %g\n", buffer_f[0]);
		L = buffer_f[0];
	} else {
		printf("ParseFile() Error: L invalid (%g)\n", L);
		fclose(pFile);
		return 1;
	}

	if ((buffer_f[1] > 0) && (buffer_f[1] < 100.0)) {
		printf("        W = %g\n", buffer_f[1]);
		W = buffer_f[1];
	} else {
		printf("ParseFile() Error: W invalid (%g)\n", W);
		fclose(pFile);
		return 1;
	}

    if ((buffer_f[2] > 0) && (buffer_f[2] < 100.0)) {
		printf("        H = %g\n", buffer_f[2]);
		H = buffer_f[2];
	} else {
		printf("ParseFile() Error: H invalid (%g)\n", H);
		fclose(pFile);
		return 1;
	}

	// Compute cell widths
	DX = (L/NX); DY = (W/NY); DZ = (H/NZ);

	index = 0; 
    // Now lets load the data for each cell
	for (int i = 0; i < NX*NY*NZ; i++) {
		index = 5*i;	
		// Read 5 values for P into the next buffer
		fread(buffer_p, 5*sizeof(float), 1, pFile);
		if ((buffer_p[0] > 0.0) && (buffer_p[0] < 10000.0)) {
			h_P[index] = buffer_p[0];
		} else {
			printf("Parsefile() Error: Invalid Density(%g)\n", buffer_p[0]);
			fclose(pFile);
			return 1;
		}
		// Don't bother parsing speeds, they can take on many valid values
		h_P[index+1] = buffer_p[1];
		h_P[index+2] = buffer_p[2];
		h_P[index+3] = buffer_p[3];
		// Check temperatures though
		if ((buffer_p[4] > 0.0) && (buffer_p[4] < 10000.0)) {
			h_P[index+4] = buffer_p[4];
		} else {
			printf("Parsefile() Error: Invalid Density(%g)\n", buffer_p[4]);
			fclose(pFile);
			return 1;
		}

		// Read the body in
		fread(buffer_i, sizeof(int), 1, pFile);
		h_Type[i] = buffer_i[0];

		// Read in Q, K and CM together
		fread(buffer_f, 3*sizeof(float), 1, pFile);
		// Don't parse Q
		h_Q[i] = buffer_f[0];
		// Check K, must be positive
		if (buffer_f[1] > 0.0) {
			h_K[i] = buffer_f[1];
		} else {
			printf("Parsefile() Error: Invalid Heat Transfer Coefficient\n");
			fclose(pFile);
			return 1;
		}
		// CM
		if (buffer_f[2] > 0.0) {
			h_CM[i] = buffer_f[2];
		} else {
			printf("Parsefile() Error: Invalid Specific Heat Constant\n");
			fclose(pFile);
			return 1;
		}
	}
	fclose(pFile);
}

void Init() {

	int count = 0, index;
	float cx,cy,cz, radius;
	float Den, Tmp, Press;
	// Set the simulation time
	TIME = 0.0;
	h_DT = 0.05*DX;

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			for (int k = 0; k < NZ; k++) {

				// Compute cell location
				index = 3*count;
				h_X[index] = DX*(i+0.5);
				h_X[index+1] = DY*(j+0.5);
				h_X[index+2] = DZ*(k+0.5);

				count++;
			}
		}
	}

	if (DEBUG) printf("\tInitialization complete\n");
}

void Save_Results() {
	FILE *pFile, *pFile2;
	int index;
	int i,j,k, cell;
	float pressure, cx, cy;
	float maxpressure = 0.0;
	float current_pressure;
	pFile = fopen("Results.txt","w");
	if (pFile == NULL) {
		printf("Error: Failed to open results file for saving data.\n");
	} else {
		for (i = 0; i < N; i++) {
			index = 5*i;
			fprintf(pFile,"%d\t%g\t%g\t%g\t%g\t%g\n", i,
                        h_P[index], h_P[index+1], h_P[index+2], h_P[index+3], h_P[index+4]);
		}
		if (DEBUG) printf("\tSave Complete\n");
		fclose(pFile);
	}
	// Save maximum values on minimum Z
	pFile = fopen("MaxP.txt", "w");
	if (pFile == NULL) {
		printf("Error: Failed to open MaxP.txt for saving data\n");
	} else {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				// Set the k value
				k = 0;
				// Later on we will search through valid k values to find the lowest
				// non-solid cell in the column.
				cell = k + j*NZ + i*NZ*NY; // Is this correct?
				index = cell*5;
				pressure = h_Pmax[index];
				// Record this for our reporting
				if (pressure > maxpressure) {
					maxpressure = pressure;
				}
				fprintf(pFile, "%g", pressure);
				if (j == (NY-1)) {
					fprintf(pFile, "\n");
				} else {
					fprintf(pFile, "\t");
				}
			}
		}
		fclose(pFile);
		if (DEBUG) printf("Max. Pressure in Simulation = %g\n", maxpressure);

	}
	// Save the x velocity
        pFile = fopen("Ux.txt", "w");
        if (pFile == NULL) {
                printf("Error: Failed to open Ux.txt for saving data\n");
        } else {
                for (i = 0; i < NX; i++) {
                        for (j = 0; j < NY; j++) {
                                // Set the k value
                                k = 0;
                                // Later on we will search through valid k values to find the lowest
                                // non-solid cell in the column.
                                cell = k + j*NZ + i*NZ*NY; // Is this correct?
                                index = cell*5;
                                fprintf(pFile, "%g", h_P[index+1]);
                                if (j == (NY-1)) {
                                        fprintf(pFile, "\n");
                                } else {
                                        fprintf(pFile, "\t");
                                }
                        }
                }
                fclose(pFile);
        }
	// Save the X and Y then
        pFile = fopen("CX.txt", "w");
	pFile2 = fopen("CY.txt", "w");
        if (pFile == NULL) {
                printf("Error: Failed to open CX.txt for saving data\n");
        } else {
                for (i = 0; i < NX; i++) {
                        for (j = 0; j < NY; j++) {
                                // Set the k value
                                k = 0;
				cx = (i+0.5)*DX; cy = (j+0.5)*DY;
                                // Later on we will search through valid k values to find the lowest
                                // non-solid cell in the column.
                                cell = k + j*NZ + i*NZ*NY; // Is this correct?
                                index = cell*5;
                                fprintf(pFile, "%g", cx);
				fprintf(pFile2, "%g", cy);
                                if (j == (NY-1)) {
                                        fprintf(pFile, "\n"); fprintf(pFile2, "\n");
                                } else {
                                        fprintf(pFile, "\t"); fprintf(pFile2, "\t");
                                }
                        }
                }
                fclose(pFile); fclose(pFile2);

        }

	// Plot Maximum Pressure and velocity (This is slow)
        system("python plotContour3.py");


}


int main(int argc, char *argv[]) {

	double start_time, end_time;

	char filename[100]; // Need to pass this on

	if (argc == 2) {
		strcpy(filename, argv[1]);
		printf("== Preparing to parse %s ==\n", filename);

	} else {
		printf("solver/main.cpp main() error: Incorrect number of arguments\n");
		printf("	Program usage: solver.run aaaabbbbccccdddd.bin\n");
		printf("where aaaabbbbccccdddd.bin is the solver input file.\n");
	}

	if (DEBUG) printf("==Parsing input file==\n");
	Parse_Input(filename);

	printf("==Commencing Simulation==\n");
	printf("\tNumber of cells = %d\n", N);
	printf("\tNumber of timesteps = %d\n", NO_STEPS);


	if (DEBUG) printf("==Initializing Problem==\n");
	Init();

	// Set the GPU dt
	if (DEBUG) printf("==Setting Init DT on GPU==\n");
	Set_GPU_DT();

	if (DEBUG) printf("==Checking DT on GPU==\n");
	// Perform debug
	Perform_Debug();

	if (DEBUG) printf("==Sending data to GPU==\n");
	Send_To_Device();	// gpu_main.cu

	if (DEBUG) printf("==Updating U on the device==\n");
	Update_GPU_U_From_P();

	// Start the clock
	start_time = Get_Wall_Time();

	// Start counting the number of steps
	NO_STEPS = 0;
	/*
	for (int step = 0; step < 500; step++) {

		if (DEBUG) printf("==Timestep %d of %d==\n", step, 500);

		if (DEBUG) printf("\tComputing Split Fluxes on device\n");
		Compute_GPU_Split_Fluxes();

		if (DEBUG) printf("\tCompute GPU State\n");
		Compute_GPU_State();

		NO_STEPS++; // Increment the number of time steps

		TIME+=h_DT; // Increment the simulation time
	}
	*/
	// Stop the Clock
	end_time = Get_Wall_Time();

	if (DEBUG) printf("==Compute GPU P Values==\n");
	Compute_GPU_Primitives();

	if (DEBUG) printf("==Getting data from GPU==\n");
	Get_From_Device();

	if (DEBUG) printf("==Saving data to file==\n");
	Save_Results();

	if (DEBUG) printf("==Freeing Memory==\n");
	Free_Memory();

	printf("==Simulation Complete==\n");
	printf("\tSimulated time = %g (dimensionless)\n", TIME);
	printf("\tGPU Time = %lf seconds\n", end_time-start_time);
	return 0;
}
