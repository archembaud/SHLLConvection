#define R 1.0
#define GAMMA 1.4
#define CV (R/(GAMMA-1.0))
#define DEBUG 1
#define GX 0.0
#define GY 0.001
#define GZ 0.0

void Allocate_Memory();
void Free_Memory();
void Send_To_Device();
void Update_GPU_U_From_P();
void Compute_GPU_Split_Fluxes();
void Compute_GPU_State();
void Compute_GPU_Primitives();
void Get_From_Device();
void Set_GPU_DT();
void Perform_Debug();
