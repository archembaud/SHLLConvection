#include <stdio.h>
#include <math.h>
#include "gpu_main.h"

extern float *d_Fp, *d_Fm, *d_Gp, *d_Gm, *d_Hp, *d_Hm;
extern float *d_dU, *d_U, *d_P, *d_Pmax;
extern float *h_P, *h_Pmax;
extern float *h_X, *d_X;
extern char *h_Type, *d_Type;
extern float *h_CM, *h_Q, *h_K;
extern float *d_CM, *d_Q, *d_K;


__device__ float d_DT;  // DT stored on device
extern float h_DT;  // DT held on host

extern int NX, NY, NZ, N;
extern float DX, DY, DZ;

// Debugging variables
float *d_debug, *h_debug;

// Message Passing
float *h_message, *d_message;

__global__ void GPU_Set_DT(float *input) {
	// Set the device DT
	d_DT = input[0];
}

__global__ void GPU_Calc_Primitives(float *U, float *P, int N) {
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int index = 5*i;
	if (i < N) {
		// Compute P from U
		P[index] = U[index];
		P[index+1] = U[index+1]/U[index];
                P[index+2] = U[index+2]/U[index];
                P[index+3] = U[index+3]/U[index];
                P[index+4] = ((U[index+4]/U[index])-0.5*(P[index+1]*P[index+1]+P[index+2]*P[index+2]+P[index+3]*P[index+3]))/CV;
	}
}




void Compute_GPU_Primitives() {
	int TPB = 64;
	int BPG = (N+TPB-1)/TPB;
	GPU_Calc_Primitives<<<BPG,TPB>>>(d_U, d_P, N);
}

__global__ void GPU_Debug(float *debug) {

	debug[0] = d_DT;

}

void Perform_Debug() {
	// Do some debugging
	size_t size = 5*sizeof(float);
	cudaError_t error;
	h_debug = (float*)malloc(size);
	error = cudaMalloc((void**)&d_debug, size);
	if (DEBUG) printf("Allocation (d_debug) cuda error = %s\n", cudaGetErrorString(error));

	GPU_Debug<<<1,1>>>(d_debug);

	error = cudaMemcpy(h_debug, d_debug, sizeof(float),cudaMemcpyDeviceToHost);
        if (DEBUG) printf("Mem cpy (d_debug) cuda error = %s\n", cudaGetErrorString(error));


	printf("Device DT = %g\n", h_debug[0]);

	// Free stuff
	free(h_debug);
	cudaFree(d_debug);

}


void Set_GPU_DT() {
	cudaError_t error;
	h_message[0] = h_DT; // Set value on host
	printf("--Value of DT on host = %g\n", h_message[0]);
	error = cudaMemcpy(d_message, h_message, sizeof(float), cudaMemcpyHostToDevice);
	printf("CUDA Error (DT update) = %s\n", cudaGetErrorString(error));
	// Launch a single threaded block for this
	GPU_Set_DT<<<1,1>>>(d_message);

}

__global__ void GPU_Calc_State(float *U, float *P, float *Fp, float *Fm, float *Hp, float *Hm, float *Gp, float *Gm,
                               int NX, int NY, int NZ, int N, float DX, float DY, float DZ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
        int index = 5*i;
	int xcell, ycell, zcell;
	float FL[5], FR[5], dU[5]; // Left and Right Fluxes, Change to 0
	if (i < N) {
		xcell = (int)(i/(NY*NZ));
		ycell = (int)((i-xcell*NY*NZ)/NZ);
		zcell = i - xcell*NY*NZ - ycell*NZ;
		// Reset dU
		dU[0] = 0.0; dU[1] = 0.0; dU[2] = 0.0; dU[3] = 0.0; dU[4] = 0.0;
		// === X Direction Contribution ===
		if (xcell == 0) {
			// Neumann
			//FL[0] = Fp[index];
			//FL[1] = Fp[index+1];
			//FL[2] = Fp[index+2];
			//FL[3] = Fp[index+3];
			//FL[4] = Fp[index+4];
			// Reflectve
                        FL[0] = -Fm[index];
                        FL[1] = Fm[index+1];
                        FL[2] = -Fm[index+2];
                        FL[3] = -Fm[index+3];
                        FL[4] = -Fm[index+4];

			// TODO: Implement convective (or conductive) heat loss across bounds

			// FR is normal
			FR[0] = Fm[index + 5*NY*NZ];
			FR[1] = Fm[index + 5*NY*NZ + 1];
                        FR[2] = Fm[index + 5*NY*NZ + 2];
                        FR[3] = Fm[index + 5*NY*NZ + 3];
                        FR[4] = Fm[index + 5*NY*NZ + 4];
		} else if (xcell == (NX-1)) {
			// FL is normal
			FL[0] = Fp[index - 5*NY*NZ];
			FL[1] = Fp[index - 5*NY*NZ + 1];
			FL[2] = Fp[index - 5*NY*NZ + 2];
			FL[3] = Fp[index - 5*NY*NZ + 3];
			FL[4] = Fp[index - 5*NY*NZ + 4];
			// FR
			// Neumann
			//FR[0] = Fm[index];
			//FR[1] = Fm[index+1];
			//FR[2] = Fm[index+2];
			//FR[3] = Fm[index+3];
			//FR[4] = Fm[index+4];
			// Reflective
			FR[0] = -Fp[index];
			FR[1] = Fp[index+1];
			FR[2] = -Fp[index+2];
			FR[3] = -Fp[index+3];
			FR[4] = -Fp[index+4];

			// TODO: Heat loss here

		} else {
			// We would normally need to check for bodies now
			// If the cell to the left is not a body, do this.
			FL[0] = Fp[index-5*NY*NZ];
			FL[1] = Fp[index-5*NY*NZ+1];
                        FL[2] = Fp[index-5*NY*NZ+2];
                        FL[3] = Fp[index-5*NY*NZ+3];
                        FL[4] = Fp[index-5*NY*NZ+4];
			// If the cell to the right is not a body, do this
			FR[0] = Fm[index+5*NY*NZ];
			FR[1] = Fm[index+5*NY*NZ+1];
                        FR[2] = Fm[index+5*NY*NZ+2];
                        FR[3] = Fm[index+5*NY*NZ+3];
                        FR[4] = Fm[index+5*NY*NZ+4];

		}
		// Update U based on X contributions
		dU[0] = dU[0] - (d_DT/DX)*(Fp[index] - Fm[index] + FR[0] - FL[0]);
		dU[1] = dU[1] - (d_DT/DX)*(Fp[index+1] - Fm[index+1] + FR[1] - FL[1]);
                dU[2] = dU[2] - (d_DT/DX)*(Fp[index+2] - Fm[index+2] + FR[2] - FL[2]);
                dU[3] = dU[3] - (d_DT/DX)*(Fp[index+3] - Fm[index+3] + FR[3] - FL[3]);
                dU[4] = dU[4] - (d_DT/DX)*(Fp[index+4] - Fm[index+4] + FR[4] - FL[4]);

		// === Y Direction Contribution ===
		if (ycell == 0) {
			// Neumann
			//FL[0] = Hp[index];
			//FL[1] = Hp[index+1];
			//FL[2] = Hp[index+2];
			//FL[3] = Hp[index+3];
			//FL[4] = Hp[index+4];
			// Reflectve
			FL[0] = -Hm[index];
			FL[1] = -Hm[index+1];
			FL[2] = Hm[index+2];
			FL[3] = -Hm[index+3];
			FL[4] = -Hm[index+4];

			// TODO: Heat loss

			// FR is normal
			FR[0] = Hm[index + 5*NZ];
			FR[1] = Hm[index + 5*NZ + 1];
			FR[2] = Hm[index + 5*NZ + 2];
			FR[3] = Hm[index + 5*NZ + 3];
			FR[4] = Hm[index + 5*NZ + 4];
		} else if (ycell == (NY-1)) {
			// FL is normal
			FL[0] = Hp[index - 5*NZ];
			FL[1] = Hp[index - 5*NZ + 1];
			FL[2] = Hp[index - 5*NZ + 2];
			FL[3] = Hp[index - 5*NZ + 3];
			FL[4] = Hp[index - 5*NZ + 4];
			// FR
			// Neumann
			//FR[0] = Hm[index];
			//FR[1] = Hm[index+1];
			//FR[2] = Hm[index+2];
			//FR[3] = Hm[index+3];
			//FR[4] = Hm[index+4];
			// Reflective
			FR[0] = -Hp[index];
			FR[1] = -Hp[index+1];
			FR[2] = Hp[index+2];
			FR[3] = -Hp[index+3];
			FR[4] = -Hp[index+4];

			// TODO: HEAT LOSS

		} else {
			// We would normally need to check for bodies now
			// If the cell to the left is not a body, do this.
			FL[0] = Hp[index-5*NZ];
			FL[1] = Hp[index-5*NZ+1];
		        FL[2] = Hp[index-5*NZ+2];
			FL[3] = Hp[index-5*NZ+3];
			FL[4] = Hp[index-5*NZ+4];
			// If the cell to the right is not a body, do this
			FR[0] = Hm[index+5*NZ];
			FR[1] = Hm[index+5*NZ+1];
			FR[2] = Hm[index+5*NZ+2];
			FR[3] = Hm[index+5*NZ+3];
			FR[4] = Hm[index+5*NZ+4];
		}
		// Update U based on Y contributions
		dU[0] = dU[0] - (d_DT/DY)*(Hp[index] - Hm[index] + FR[0] - FL[0]);
		dU[1] = dU[1] - (d_DT/DY)*(Hp[index+1] - Hm[index+1] + FR[1] - FL[1]);
	        dU[2] = dU[2] - (d_DT/DY)*(Hp[index+2] - Hm[index+2] + FR[2] - FL[2]);
        	dU[3] = dU[3] - (d_DT/DY)*(Hp[index+3] - Hm[index+3] + FR[3] - FL[3]);
	        dU[4] = dU[4] - (d_DT/DY)*(Hp[index+4] - Hm[index+4] + FR[4] - FL[4]);

		// === Z Direction Contribution ===
		if (zcell == 0) {
			// Neumann
			//FL[0] = Gp[index];
			//FL[1] = Gp[index+1];
			//FL[2] = Gp[index+2];
			//FL[3] = Gp[index+3];
			//FL[4] = Gp[index+4];
			// Reflectve
        		FL[0] = -Gm[index];
			FL[1] = -Gm[index+1];
			FL[2] = -Gm[index+2];
			FL[3] = Gm[index+3];
			FL[4] = -Gm[index+4];

			// TODO: HEAT LOSS

			// FR is normal
			FR[0] = Gm[index + 5];
			FR[1] = Gm[index + 5 + 1];
			FR[2] = Gm[index + 5 + 2];
			FR[3] = Gm[index + 5 + 3];
			FR[4] = Gm[index + 5 + 4];
		} else if (zcell == (NZ-1)) {
			// FL is normal
			FL[0] = Gp[index - 5];
			FL[1] = Gp[index - 5 + 1];
			FL[2] = Gp[index - 5 + 2];
			FL[3] = Gp[index - 5 + 3];
			FL[4] = Gp[index - 5 + 4];
			// FR
			// Neumann
			//FR[0] = Gm[index];
			//FR[1] = Gm[index+1];
			//FR[2] = Gm[index+2];
			//FR[3] = Gm[index+3];
			//FR[4] = Gm[index+4];
			// Reflective
			FR[0] = -Gp[index];
			FR[1] = -Gp[index+1];
			FR[2] = -Gp[index+2];
			FR[3] = Gp[index+3];
			FR[4] = -Gp[index+4];

			// TODO: HEAT LOSS

		} else {
			// We would normally need to check for bodies now
			// If the cell to the left is not a body, do this.
			FL[0] = Gp[index-5];
			FL[1] = Gp[index-5+1];
		        FL[2] = Gp[index-5+2];
		        FL[3] = Gp[index-5+3];
		        FL[4] = Gp[index-5+4];
			// If the cell to the right is not a body, do this
			FR[0] = Gm[index+5];
			FR[1] = Gm[index+5+1];
			FR[2] = Gm[index+5+2];
			FR[3] = Gm[index+5+3];
			FR[4] = Gm[index+5+4];

		}
		// Update U based on Z contributions
		dU[0] = dU[0] - (d_DT/DZ)*(Gp[index] - Gm[index] + FR[0] - FL[0]);
		dU[1] = dU[1] - (d_DT/DZ)*(Gp[index+1] - Gm[index+1] + FR[1] - FL[1]);
		dU[2] = dU[2] - (d_DT/DZ)*(Gp[index+2] - Gm[index+2] + FR[2] - FL[2]);
		dU[3] = dU[3] - (d_DT/DZ)*(Gp[index+3] - Gm[index+3] + FR[3] - FL[3]);
		dU[4] = dU[4] - (d_DT/DZ)*(Gp[index+4] - Gm[index+4] + FR[4] - FL[4]);

		// Update the value of U
		U[index] = U[index] + dU[0];
		U[index+1] = U[index+1] + dU[1];
		U[index+2] = U[index+2] + dU[2];
		U[index+3] = U[index+3] + dU[3];
		U[index+4] = U[index+4] + dU[4];
	}
}



void Compute_GPU_State() {
	int TPB = 64;
	int BPG = (N+TPB-1)/TPB;
	GPU_Calc_State<<<BPG,TPB>>>(d_U, d_P, d_Fp, d_Fm, d_Hp, d_Hm, d_Gp, d_Gm, NX, NY, NZ, N, DX, DY, DZ);
	cudaDeviceSynchronize();
}







__global__ void GPU_Calc_Fluxes(float *p, float *pmax, float *u, float *x, float *Fp, float *Fm, float *Hp, float *Hm, 
                                float *Gp, float *Gm, int N) {
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int index = 5*i;
	int location_index = 3*i;
	float U[5], P[5], F[5];  // U, P and F held in registers (locally)
	float X[3]; // Hold position in registers
	float a, M, pressure;
	float MP1 , MM1, M2NEG;
	if (i < N) {
		// Use registers (U) to hold global values of u
		U[0] = u[index]; U[1] = u[index+1]; U[2] = u[index+2]; U[3] = u[index+3]; U[4] = u[index+4];
		// Use registers to hold global X values
		X[0] = x[location_index]; X[1] = x[location_index+1]; X[2] = x[location_index+2];
		// Update P using register varaibles
		P[0] = U[0];		// Density
		P[1] = U[1]/U[0];	// X Velocity
		P[2] = U[2]/U[0];	// Y Velocity
		P[3] = U[3]/U[0];	// Z velocity
		P[4] = ((U[4]/U[0])-0.5*(P[1]*P[1]+P[2]*P[2]+P[3]*P[3]))/CV; // Ideal Gas Temperature
		/// Update global memory values
		p[index] = P[0]; p[index+1] = P[1]; p[index+2] = P[2]; p[index+3] = P[3]; p[index+4] = P[4];
		pressure = P[0]*R*P[4];
		// Update maximum values
		if (pressure > pmax[index]) {
			pmax[index] = pressure;
		}
		if (fabs(P[1]) > fabs(pmax[index+1])) { pmax[index+1] = P[1]; }
                if (fabs(P[2]) > fabs(pmax[index+2])) { pmax[index+2] = P[2]; }
                if (fabs(P[3]) > fabs(pmax[index+3])) { pmax[index+3] = P[3]; }
                if (fabs(P[4]) > fabs(pmax[index+4])) { pmax[index+4] = P[4]; }

		a = sqrtf(GAMMA*R*P[4]); // Speed of sound
		// === Compute X fluxes ===
		M = P[1]/a;
		MP1 = 0.5*(M+1.0); MM1 = 0.5*(M-1.0); M2NEG = 0.5*a*(1.0-M*M);

		// Fluxes F (with gravity in X direction terms)
		F[0] = P[0]*P[1];
		F[1] = P[0]*(P[1]*P[1] + R*P[4] + GX*X[0]);
		F[2] = P[0]*P[1]*P[2];
		F[3] = P[0]*P[1]*P[3];
		F[4] = P[1]*(U[4] + P[0]*R*P[4] + P[0]*GX*X[0]);

		// Split Fluxes (Fp = Forward, Fm = Backward)
		Fp[index] =   F[0]*MP1 + U[0]*M2NEG;
		Fp[index+1] = F[1]*MP1 + U[1]*M2NEG;
		Fp[index+2] = F[2]*MP1 + U[2]*M2NEG;
		Fp[index+3] = F[3]*MP1 + U[3]*M2NEG;
		Fp[index+4] = F[4]*MP1 + U[4]*M2NEG;
		Fm[index] =   -F[0]*MM1 - U[0]*M2NEG;
		Fm[index+1] = -F[1]*MM1 - U[1]*M2NEG;
                Fm[index+2] = -F[2]*MM1 - U[2]*M2NEG;
                Fm[index+3] = -F[3]*MM1 - U[3]*M2NEG;
		Fm[index+4] = -F[4]*MM1 - U[4]*M2NEG;
                // === Compute Y fluxes ===
                M = P[2]/a;
                MP1 = 0.5*(M+1.0); MM1 = 0.5*(M-1.0); M2NEG = 0.5*a*(1.0-M*M);
                // Fluxes F
                F[0] = P[0]*P[2];
                F[1] = P[0]*P[2]*P[1];
                F[2] = P[0]*(P[2]*P[2] + R*P[4] + GY*X[1]);
                F[3] = P[0]*P[2]*P[3];
                F[4] = P[2]*(U[4] + P[0]*R*P[4] + P[0]*GY*X[1]);

                // Split Fluxes (Hp = Forward, Hm = Backward)
                Hp[index] =   F[0]*MP1 + U[0]*M2NEG;
                Hp[index+1] = F[1]*MP1 + U[1]*M2NEG;
                Hp[index+2] = F[2]*MP1 + U[2]*M2NEG;
                Hp[index+3] = F[3]*MP1 + U[3]*M2NEG;
                Hp[index+4] = F[4]*MP1 + U[4]*M2NEG;
                Hm[index] =   -F[0]*MM1 - U[0]*M2NEG;
                Hm[index+1] = -F[1]*MM1 - U[1]*M2NEG;
                Hm[index+2] = -F[2]*MM1 - U[2]*M2NEG;
                Hm[index+3] = -F[3]*MM1 - U[3]*M2NEG;
                Hm[index+4] = -F[4]*MM1 - U[4]*M2NEG;
                // === Compute Z fluxes ===
                M = P[3]/a;
                MP1 = 0.5*(M+1.0); MM1 = 0.5*(M-1.0); M2NEG = 0.5*a*(1.0-M*M);
                // Fluxes F
                F[0] = P[0]*P[3];
                F[1] = P[0]*P[3]*P[1];
                F[2] = P[0]*P[3]*P[2];
                F[3] = P[0]*(P[3]*P[3] + R*P[4] + GZ*X[2]);
                F[4] = P[3]*(U[4] + P[0]*R*P[4] + P[0]*GZ*X[2]);

                // Split Fluxes (Gp = Forward, Gm = Backward)
                Gp[index] =   F[0]*MP1 + U[0]*M2NEG;
                Gp[index+1] = F[1]*MP1 + U[1]*M2NEG;
                Gp[index+2] = F[2]*MP1 + U[2]*M2NEG;
                Gp[index+3] = F[3]*MP1 + U[3]*M2NEG;
                Gp[index+4] = F[4]*MP1 + U[4]*M2NEG;
                Gm[index] =   -F[0]*MM1 - U[0]*M2NEG;
                Gm[index+1] = -F[1]*MM1 - U[1]*M2NEG;
                Gm[index+2] = -F[2]*MM1 - U[2]*M2NEG;
                Gm[index+3] = -F[3]*MM1 - U[3]*M2NEG;
                Gm[index+4] = -F[4]*MM1 - U[4]*M2NEG;

	}
}


void Compute_GPU_Split_Fluxes() {
	int TPB = 64;
	int BPG = (N+TPB-1)/TPB;
	GPU_Calc_Fluxes<<<BPG,TPB>>>(d_P, d_Pmax, d_U, d_X, d_Fp, d_Fm, d_Hp, d_Hm, d_Gp, d_Gm, N);
	cudaDeviceSynchronize();
}




__global__ void GPU_U_From_P(float *P, float *U, int N) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int index = 5*i;
	if (i < N) {
		// Update values of U based on P
		U[index] = P[index];			// Mass per unit volume
		U[index+1] = P[index]*P[index+1];	// X Mom per unit volume
		U[index+2] = P[index]*P[index+2];
		U[index+3] = P[index]*P[index+3];
		U[index+4] = P[index]*(P[index+4]*CV+0.5*(P[index+1]*P[index+1]+P[index+2]*P[index+2]+P[index+3]*P[index+3]));
	}
}

void Update_GPU_U_From_P() {
	int TPB = 64;
	int BPG = (N+TPB-1)/TPB;
	GPU_U_From_P<<<BPG,TPB>>>(d_P, d_U, N);
}

void Allocate_Memory() {
	// Allocate memory
	cudaError_t Error;
	size_t size = 5*N*sizeof(float);
	// Conserved Quantity Variables
	Error = cudaMalloc((void**)&d_U, size);
	if (DEBUG) printf("\tError (d_U Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_dU, size);
        if (DEBUG) printf("\tError (d_dU Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_P, size);
        if (DEBUG) printf("\tError (d_P Allocation) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_Pmax, size);
        if (DEBUG) printf("\tError (d_Pmax Allocation) = %s\n",cudaGetErrorString(Error));

	// Flux Variables
        Error = cudaMalloc((void**)&d_Fp, size);
        if (DEBUG) printf("\tError (d_Fp Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_Fm, size);
        if (DEBUG) printf("\tError (d_Fm Allocation) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_Hp, size);
        if (DEBUG) printf("\tError (d_Hp Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_Hm, size);
        if (DEBUG) printf("\tError (d_Hm Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_Gp, size);
        if (DEBUG) printf("\tError (d_Gp Allocation) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_Gm, size);
        if (DEBUG) printf("\tError (d_Gm Allocation) = %s\n",cudaGetErrorString(Error));
	size = N*sizeof(char);
	Error = cudaMalloc((void**)&d_Type, size);
	if (DEBUG) printf("\tError (d_type Allocation) = %s\n", cudaGetErrorString(Error));

	// Location Variables
	size = 3*N*sizeof(float);
        Error = cudaMalloc((void**)&d_X, size);
        if (DEBUG) printf("\tError (d_X Allocation) = %s\n",cudaGetErrorString(Error));

	// CM, K and Q
	size = N*sizeof(float);
	Error = cudaMalloc((void**)&d_CM, size);
	if (DEBUG) printf("\tError (d_CM Allocation) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_K, size);
	if (DEBUG) printf("\tError (d_K Allocation) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_Q, size);
	if (DEBUG) printf("\tError (d_Q Allocation) = %s\n",cudaGetErrorString(Error));


	// Message
	cudaMalloc((void**)&d_message, sizeof(float));

	// Allocate memory on host for P
	size = 5*N*sizeof(float);
	h_P = (float*)malloc(size);
	h_Pmax = (float*)malloc(size);
	// Allocate memeory for type on host
	size = N*sizeof(char);
	h_Type = (char*)malloc(size);
	// Allocate location variable
	size = 3*N*sizeof(float);
	h_X = (float*)malloc(size);
	// Allocate CM, Q and d_K
	size = N*sizeof(float);
	h_CM = (float*)malloc(size);
	h_K = (float*)malloc(size);
	h_Q = (float*)malloc(size);

	// Message
	h_message = (float*)malloc(sizeof(float));
}

void Send_To_Device() {
	size_t size = 5*N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_P, h_P,size,cudaMemcpyHostToDevice);
	if (DEBUG) printf("\tError (Copy h_P -> d_P) = %s\n", cudaGetErrorString(Error));
	size = 3*N*sizeof(float);
    Error = cudaMemcpy(d_X, h_X,size,cudaMemcpyHostToDevice);
    if (DEBUG) printf("\tError (Copy h_X -> d_X) = %s\n", cudaGetErrorString(Error));

	// TODO: Need to move bodies across
	// ALSO need to move K, CM, Q across

	// May as well also get Pmax to zero
	cudaMemset(d_Pmax, 0, size);
}

void Get_From_Device() {
        size_t size = 5*N*sizeof(float);
        cudaError_t Error;
	// Copy Primitives
        Error = cudaMemcpy(h_P, d_P,size,cudaMemcpyDeviceToHost);
        if (DEBUG) printf("\tError (Copy d_P -> h_P) = %s\n", cudaGetErrorString(Error));
	// Copy Maximum values
        Error = cudaMemcpy(h_Pmax, d_Pmax,size,cudaMemcpyDeviceToHost);
        if (DEBUG) printf("\tError (Copy d_Pmax -> h_Pmax) = %s\n", cudaGetErrorString(Error));

}

void Free_Memory() {
	cudaError_t Error;
	// Free solution variables
	Error = cudaFree(d_U);
	if (DEBUG) printf("\tError (Free d_U) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_dU);
        if (DEBUG) printf("\tError (Free d_dU) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_P);
        if (DEBUG) printf("\tError (Free d_P) = %s\n", cudaGetErrorString(Error));
	Error = cudaFree(d_Pmax);
	if (DEBUG) printf("\tError (Free d_Pmax) = %s\n", cudaGetErrorString(Error));
	// Free fluxes
	Error = cudaFree(d_Fp);
        if (DEBUG) printf("\tError (Free d_Fp) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_Fm);
        if (DEBUG) printf("\tError (Free d_Fm) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_Hp);
        if (DEBUG) printf("\tError (Free d_Hp) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_Hm);
        if (DEBUG) printf("\tError (Free d_Hm) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_Gp);
        if (DEBUG) printf("\tError (Free d_Gp) = %s\n", cudaGetErrorString(Error));
        Error = cudaFree(d_Gm);
        if (DEBUG) printf("\tError (Free d_Gm) = %s\n", cudaGetErrorString(Error));
	Error = cudaFree(d_Type);
	if (DEBUG) printf("\tError (Free d_Type) = %s\n", cudaGetErrorString(Error));
	// Free location
	Error = cudaFree(d_X);
    if (DEBUG) printf("\tError (Free d_X) = %s\n", cudaGetErrorString(Error));

	// Free Q, K and CM
	Error = cudaFree(d_Q);
    if (DEBUG) printf("\tError (Free d_Q) = %s\n", cudaGetErrorString(Error));
	Error = cudaFree(d_K);
    if (DEBUG) printf("\tError (Free d_K) = %s\n", cudaGetErrorString(Error));
	Error = cudaFree(d_CM);
    if (DEBUG) printf("\tError (Free d_CM) = %s\n", cudaGetErrorString(Error));

	// Free message
	cudaFree(d_message);

	// Free host variables
	free(h_P); free(h_Type); free(h_message); free(h_Pmax);
	// Free location
	free(h_X);
	// Free CM, Q, K
	free(h_K); free(h_CM); free(h_Q);

}

