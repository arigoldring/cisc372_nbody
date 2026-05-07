#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"
#include "config.h"

// ============================================================
//Error handling, wraps each cuda fuction in a error handing that prints
//where the program crashed then exits
// ============================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================
// BLOCK_SIZE = 16 cause assignment said so
// ============================================================
#define BLOCK_SIZE 16

// ============================================================
// d_hPos  : positions of all N bodies on the GPU
// d_hVel  : velocities of all N bodies on the GPU
// d_mass  : masses of all N bodies on the GPU
// d_accels: flat N×N matrix of pairwise accelerations on the GPU
// ============================================================
vector3 *d_hPos = NULL;
vector3 *d_hVel = NULL;
double  *d_mass = NULL;

static vector3 *d_accels    = NULL;  // N×N pairwise acceleration matrix
static int      gpu_initialized = 0; // 0 until initGPU() has run

// ============================================================
//initialize GPU variables
// ============================================================
static void initGPU() {
    size_t vec_size   = sizeof(vector3) * NUMENTITIES;          // bytes for N vectors
    size_t mass_size  = sizeof(double)  * NUMENTITIES;          // bytes for N doubles
    size_t accel_size = sizeof(vector3) * NUMENTITIES * NUMENTITIES; // bytes for N×N vectors

    // Allocate GPU memory for each array
    CUDA_CHECK(cudaMalloc((void**)&d_hPos,   vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_hVel,   vec_size));
    CUDA_CHECK(cudaMalloc((void**)&d_mass,   mass_size));
    CUDA_CHECK(cudaMalloc((void**)&d_accels, accel_size));

    // Upload initial CPU state to the GPU
    CUDA_CHECK(cudaMemcpy(d_hPos, hPos, vec_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hVel, hVel, vec_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, mass, mass_size,  cudaMemcpyHostToDevice));

    gpu_initialized = 1;
}

// ============================================================

// ============================================================
__global__ void computeAccels(vector3 *pos, double *mass,
                               vector3 *accels, int n) {
    // Shared memory arrays for the j-tile.
    // Each array holds BLOCK_SIZE entries — one per j-column in this block.
    // All BLOCK_SIZE rows (threadIdx.y values) in the block share these.
    __shared__ double s_jx[BLOCK_SIZE];    // x-component of pos[j] for this tile
    __shared__ double s_jy[BLOCK_SIZE];    // y-component of pos[j] for this tile
    __shared__ double s_jz[BLOCK_SIZE];    // z-component of pos[j] for this tile
    __shared__ double s_jmass[BLOCK_SIZE]; // mass[j] for this tile

    // Global (i, j) indices for this thread.
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // source body (column)
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // affected body (row)

    // Load this block's j-tile into shared memory.
    // Only the first row of threads (threadIdx.y == 0) does the load so
    // we don't have BLOCK_SIZE redundant global reads for the same data.
    // The guard (j < n) prevents out-of-bounds access for the last tile
    // when N is not a multiple of BLOCK_SIZE.
    if (threadIdx.y == 0 && j < n) {
        s_jx[threadIdx.x]    = pos[j][0]; // load x-position of source body j
        s_jy[threadIdx.x]    = pos[j][1]; // load y-position of source body j
        s_jz[threadIdx.x]    = pos[j][2]; // load z-position of source body j
        s_jmass[threadIdx.x] = mass[j];   // load mass of source body j
    }
    // Wait until all threads in the block have finished loading the tile
    // before anyone reads from shared memory below.
    __syncthreads();

    // Threads that fall outside the N×N matrix (padding from ceil division)
    // have nothing to compute — exit after the barrier.
    if (i >= n || j >= n) return;

    // A body exerts no gravitational pull on itself.
    if (i == j) {
        accels[i * n + j][0] = 0.0;
        accels[i * n + j][1] = 0.0;
        accels[i * n + j][2] = 0.0;
        return;
    }

    // Displacement vector from j to i: d = pos[i] - pos[j]
    double dx = pos[i][0] - s_jx[threadIdx.x]; // x-displacement, i relative to j
    double dy = pos[i][1] - s_jy[threadIdx.x]; // y-displacement
    double dz = pos[i][2] - s_jz[threadIdx.x]; // z-displacement

    double mag_sq   = dx*dx + dy*dy + dz*dz; // squared distance |d|²
    double mag      = sqrt(mag_sq);           // distance |d|

    // Scalar acceleration magnitude: a = -G * m_j / r²
    // Negative sign because gravity pulls i toward j (opposite to d).
    double accelmag = -GRAV_CONSTANT * s_jmass[threadIdx.x] / mag_sq;

    // Write the 3D acceleration vector into the N×N matrix.
    accels[i * n + j][0] = accelmag * dx / mag; // x-component
    accels[i * n + j][1] = accelmag * dy / mag; // y-component
    accels[i * n + j][2] = accelmag * dz / mag; // z-component
}

// ============================================================

// ============================================================
__global__ void sumAndUpdate(vector3 *pos, vector3 *vel,
                             vector3 *accels, int n, double interval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // which body this thread owns
    if (i >= n) return;                             // guard for padding threads

    // Accumulate the net acceleration on body i by summing its row in accels.
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = 0; j < n; j++) {          // iterate over all source bodies j
        ax += accels[i * n + j][0];        // add j's x-contribution to i
        ay += accels[i * n + j][1];        // add j's y-contribution to i
        az += accels[i * n + j][2];        // add j's z-contribution to i
    }

    // Euler integration — update velocity: v_new = v_old + a * dt
    vel[i][0] += ax * interval;
    vel[i][1] += ay * interval;
    vel[i][2] += az * interval;

    // Update position: p_new = p_old + v_new * dt
    // (uses the just-updated velocity, matching the original serial code)
    pos[i][0] += vel[i][0] * interval;
    pos[i][1] += vel[i][1] * interval;
    pos[i][2] += vel[i][2] * interval;
}

// ============================================================

// ============================================================
void compute() {
    if (!gpu_initialized) {
        initGPU(); // one-time GPU allocation + upload
    }

    int n = NUMENTITIES;


    dim3 blockDim2D(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim2D(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,  // blocks along j-axis (x)
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE   // blocks along i-axis (y)
    );

    computeAccels<<<gridDim2D, blockDim2D>>>(d_hPos, d_mass, d_accels, n);
    CUDA_CHECK(cudaGetLastError()); // check for launch configuration errors


    // --- Kernel 2 launch (1D) ---
    // One thread per body; 256 threads per block.
    int blockDim1D = 256;
    int gridDim1D  = (n + blockDim1D - 1) / blockDim1D; // enough blocks for N bodies

    sumAndUpdate<<<gridDim1D, blockDim1D>>>(d_hPos, d_hVel, d_accels, n, (double)INTERVAL);
    CUDA_CHECK(cudaGetLastError()); // check for launch configuration errors

   
}

// ============================================================
// syncToHost: Copy final GPU results back to CPU arrays.

// ============================================================
void syncToHost() {
    int n = NUMENTITIES;

    CUDA_CHECK(cudaDeviceSynchronize()); // wait for all GPU kernels to finish

    // Copy final positions and velocities from GPU back to CPU host arrays.
    CUDA_CHECK(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * n, cudaMemcpyDeviceToHost));
}
