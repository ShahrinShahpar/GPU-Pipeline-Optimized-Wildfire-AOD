// newgtry_gpu.cu — GPU-only G matrix (transition matrix) computation.
// Stripped from newgtry.cu: CPU functions removed, GPU kernels preserved.
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DG_VERIFY_MAIN \
//        newgtry_gpu.cu csv_io.cu -o g_matrix_verify
//
// Usage:
//   ./g_matrix_verify z1.csv z2.csv omega1.csv omega2.csv ref_G.csv out_G.csv

#include "newgtry_gpu.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// GPU kernels — unchanged from newgtry.cu
// ---------------------------------------------------------------------------

__global__ void f1_kernel(const double* in1, double* a1, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in1 + tid * 4;
        a1[tid] = 2*M_PI*(s[2]*k1+s[3]*k2)
                * sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                * cos(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f2_kernel(const double* in1, double* a2, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in1 + tid * 4;
        a2[tid] = (-2)*M_PI*(s[2]*k1+s[3]*k2)
                * cos(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                * cos(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f3_kernel(const double* in1, double* a3, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in1 + tid * 4;
        a3[tid] = 2*M_PI*(s[2]*k1+s[3]*k2)
                * sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                * sin(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f5_kernel(const double* in5, double* a5, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in5 + tid * 5;
        a5[tid] = (-4*M_PI*M_PI*(k1*k1+k2*k2)*s[2]*cos(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                  - 2*M_PI*(s[3]*k1+s[4]*k2)*sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2))
                * cos(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f6_kernel(const double* in5, double* a6, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in5 + tid * 5;
        a6[tid] = (-4*M_PI*M_PI*(k1*k1+k2*k2)*s[2]*sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                  - 2*M_PI*(s[3]*k1+s[4]*k2)*cos(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2))
                * cos(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f7_kernel(const double* in5, double* a7, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in5 + tid * 5;
        a7[tid] = (-4*M_PI*M_PI*(k1*k1+k2*k2)*s[2]*cos(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                  - 2*M_PI*(s[3]*k1+s[4]*k2)*sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2))
                * sin(2*M_PI*s[0]*m1+2*M_PI*s[1]*m2);
    }
}

__global__ void f8_kernel(const double* in5, double* a8, int N,
                           double k1, double k2, double m1, double m2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        const double* s = in5 + tid * 5;
        a8[tid] = (-4*M_PI*M_PI*(k1*k1+k2*k2)*s[2]*sin(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2)
                  - 2*M_PI*(s[3]*k1+s[4]*k2)*cos(2*M_PI*s[0]*k1+2*M_PI*s[1]*k2))
                * sin(2*M_PI*m1*s[0]+2*M_PI*m2*s[1]);
    }
}

// ---------------------------------------------------------------------------
// Host-side GPU reduction
// ---------------------------------------------------------------------------
static double gpu_sum(const double* d_ptr, int N)
{
    return thrust::reduce(
        thrust::device_ptr<const double>(d_ptr),
        thrust::device_ptr<const double>(d_ptr + N),
        0.0, thrust::plus<double>());
}

// ---------------------------------------------------------------------------
// g_matrix_gpu — compute G on GPU, return in host array h_G
// ---------------------------------------------------------------------------
void g_matrix_gpu(const double* d_z1, const double* d_z2,
                  const double* h_omega1, const double* h_omega2,
                  int N, int k1, int k2, double delta, double* h_G)
{
    const int Ngrid  = k1 + 2 * k2;   // 400
    const double dd  = delta * delta;
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    // Allocate device scratch arrays (one per phi function, reused)
    double *d_a1, *d_a2, *d_a3, *d_a5, *d_a6, *d_a7, *d_a8;
    cudaMalloc(&d_a1, N * sizeof(double));
    cudaMalloc(&d_a2, N * sizeof(double));
    cudaMalloc(&d_a3, N * sizeof(double));
    cudaMalloc(&d_a5, N * sizeof(double));
    cudaMalloc(&d_a6, N * sizeof(double));
    cudaMalloc(&d_a7, N * sizeof(double));
    cudaMalloc(&d_a8, N * sizeof(double));

    // Helper lambdas to extract omega components
    auto om1 = [&](int i, int d) { return h_omega1[i*2 + d]; };
    auto om2 = [&](int i, int d) { return h_omega2[i*2 + d]; };

    // Initialize G to zero
    memset(h_G, 0, (size_t)Ngrid * Ngrid * sizeof(double));

    // -----------------------------------------------------------------------
    // Row 0 (b=0): m = Omega1[0], coefficient 1 for Omega1 cols, 2 for Omega2
    // -----------------------------------------------------------------------
    {
        double m0 = om1(0,0), m1_ = om1(0,1);
        for (int a = 0; a < k1; a++) {
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om1(a,0),om1(a,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om1(a,0),om1(a,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[0*Ngrid+a] = (gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1; a < k1+k2; a++) {
            int ai = a - k1;
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om2(ai,0),om2(ai,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[0*Ngrid+a] = 2*(gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1+k2; a < k1+2*k2; a++) {
            int ai = a - k1 - k2;
            f2_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a2,N,om2(ai,0),om2(ai,1),m0,m1_);
            f6_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a6,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[0*Ngrid+a] = 2*(gpu_sum(d_a2,N) + gpu_sum(d_a6,N)) * dd;
        }
    }

    // -----------------------------------------------------------------------
    // Rows 1..k1-1 (b=1..k1-1): m = Omega1[b], coefficient 2 / 4 / 4
    // -----------------------------------------------------------------------
    for (int b = 1; b < k1; b++) {
        double m0 = om1(b,0), m1_ = om1(b,1);
        for (int a = 0; a < k1; a++) {
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om1(a,0),om1(a,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om1(a,0),om1(a,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 2*(gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1; a < k1+k2; a++) {
            int ai = a - k1;
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om2(ai,0),om2(ai,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 4*(gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1+k2; a < k1+2*k2; a++) {
            int ai = a - k1 - k2;
            f2_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a2,N,om2(ai,0),om2(ai,1),m0,m1_);
            f6_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a6,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 4*(gpu_sum(d_a2,N) + gpu_sum(d_a6,N)) * dd;
        }
    }

    // -----------------------------------------------------------------------
    // Rows k1+j2 for j2=0..k2-1 (real Omega2 rows): m = Omega2[j2]
    // Matches R: for (m in 1:k2) G[m+k1, k] <- ...  (all k2 rows filled)
    // -----------------------------------------------------------------------
    for (int j2 = 0; j2 < k2; j2++) {
        int b = j2 + k1;
        double m0 = om2(j2,0), m1_ = om2(j2,1);
        for (int a = 0; a < k1; a++) {
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om1(a,0),om1(a,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om1(a,0),om1(a,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = (gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1; a < k1+k2; a++) {
            int ai = a - k1;
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,om2(ai,0),om2(ai,1),m0,m1_);
            f5_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a5,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 2*(gpu_sum(d_a1,N) + gpu_sum(d_a5,N)) * dd;
        }
        for (int a = k1+k2; a < k1+2*k2; a++) {
            int ai = a - k1 - k2;
            f2_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a2,N,om2(ai,0),om2(ai,1),m0,m1_);
            f6_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a6,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 2*(gpu_sum(d_a2,N) + gpu_sum(d_a6,N)) * dd;
        }
    }

    // -----------------------------------------------------------------------
    // Rows k1+k2+j2 for j2=0..k2-1 (imaginary Omega2 rows): m = Omega2[j2]
    // Matches R: for (m in 1:k2) G[m+k1+k2, k] <- ...  (all k2 rows filled)
    // Bottom-right block: R uses -2*Phi1 + 2*Phi8 (note negative Phi1).
    // -----------------------------------------------------------------------
    for (int j2 = 0; j2 < k2; j2++) {
        int b = j2 + k1 + k2;
        double m0 = om2(j2,0), m1_ = om2(j2,1);
        for (int a = 0; a < k1; a++) {
            f3_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a3,N,om1(a,0),om1(a,1),m0,m1_);
            f7_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a7,N,om1(a,0),om1(a,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = (gpu_sum(d_a3,N) + gpu_sum(d_a7,N)) * dd;
        }
        for (int a = k1; a < k1+k2; a++) {
            int ai = a - k1;
            f3_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a3,N,om2(ai,0),om2(ai,1),m0,m1_);
            f7_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a7,N,om2(ai,0),om2(ai,1),m0,m1_);
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 2*(gpu_sum(d_a3,N) + gpu_sum(d_a7,N)) * dd;
        }
        for (int a = k1+k2; a < k1+2*k2; a++) {
            int ai = a - k1 - k2;
            // R: -2*Phi1(Omega2[m],Omega2[col]) + 2*Phi8(...)
            // k/m args swapped (R passes Omega2[m] as k, Omega2[col] as m).
            f1_kernel<<<BLOCKS,THREADS>>>(d_z1,d_a1,N,m0,m1_,om2(ai,0),om2(ai,1));
            f8_kernel<<<BLOCKS,THREADS>>>(d_z2,d_a8,N,m0,m1_,om2(ai,0),om2(ai,1));
            cudaDeviceSynchronize();
            h_G[b*Ngrid+a] = 2*(-gpu_sum(d_a1,N) + gpu_sum(d_a8,N)) * dd;
        }
    }

    cudaFree(d_a1); cudaFree(d_a2); cudaFree(d_a3);
    cudaFree(d_a5); cudaFree(d_a6); cudaFree(d_a7); cudaFree(d_a8);
}

// ===========================================================================
// g_matrix_gpu_fast — replace 160,000 serialised kernel+sync+reduce calls
// with 1 precompute kernel + 10 DGEMMs.
//
// Mathematics:
//   Every entry G[b, a] = D_row[b] * delta^2 * D_col[a] * sum_i Phi_b(z_i) * W_a(z_i)
//   where Phi_b is the row basis and W_a is the column weight.
//
//   Coefficients:
//     D_row: 1 for b=0, 2 for b=1..k1-1 (Omega1), 1 for Omega2 rows
//     D_col: 1 for Omega1 cols, 2 for Omega2 cols (real and imag)
//
//   Row bases  (k×N, freq×point):
//     PhiCos1[b, i] = cos(2π z_i · om1_b)   b ∈ Omega1
//     PhiCos2[b, i] = cos(2π z_i · om2_b)   b ∈ Omega2
//     PhiSin2[b, i] = sin(2π z_i · om2_b)   b ∈ Omega2
//
//   Col weights (k×N):
//     WCol1 [a,i] = [2π(v-dK)·ω]*sin(z·ω) − 4π²|ω|²K·cos(z·ω)   ω=om1_a
//     WCol2r[a,i] = same formula with ω=om2_a
//     WCol2i[a,i] = −(2π(v+dK)·ω)·cos(z·ω) + (−4π²|ω|²K)·sin(z·ω)  ω=om2_a
//
//   Special block G_33 (b∈Omega2_imag, a∈Omega2_imag):
//     Uses *row* weights instead of Phi rows (k/m args are swapped in original):
//     RowW1[j2,i] = −2π(v·mb)·sin(z·mb)
//     RowW2[j2,i] = (−4π²|mb|²K)·sin(z·mb) − 2π(dK·mb)·cos(z·mb)
//     G_33 = delta^2 * (RowW1 × PhiCos2^T + RowW2 × PhiSin2^T)
//     (then D_col ×2 applies to the Omega2_imag cols, giving coeff=2)
//
//   Output: h_G row-major (same layout as g_matrix_gpu).
// ===========================================================================
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Precompute kernel: all row bases and col weights in one pass.
// Total threads = N*(k1+k2).
// ---------------------------------------------------------------------------
static __global__ void g_precompute_kernel(
    const double* __restrict__ z1,     // N×4 row-major: x,y,vx,vy
    const double* __restrict__ z2,     // N×5 row-major: x,y,K,dKx,dKy
    const double* __restrict__ om1,    // k1×2 row-major
    const double* __restrict__ om2,    // k2×2 row-major
    int N, int k1, int k2,
    double* PhiCos1,   // k1×N col-major
    double* PhiCos2,   // k2×N col-major
    double* PhiSin2,   // k2×N col-major
    double* WCol1,     // k1×N col-major
    double* WCol2r,    // k2×N col-major
    double* WCol2i,    // k2×N col-major
    double* RowW1,     // k2×N col-major  (G33 special row weight)
    double* RowW2      // k2×N col-major  (G33 special row weight)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ---- Omega1 section ------------------------------------------------
    if (tid < N * k1) {
        int i = tid / k1, f = tid % k1;
        double x = z1[i*4], y = z1[i*4+1], vx = z1[i*4+2], vy = z1[i*4+3];
        double K  = z2[i*5+2], dKx = z2[i*5+3], dKy = z2[i*5+4];
        double ka1 = om1[f*2], ka2 = om1[f*2+1];
        double arg = 2.0*M_PI*(x*ka1 + y*ka2);
        double s = sin(arg), c = cos(arg);
        double Av  =  2.0*M_PI*(vx*ka1  + vy*ka2);
        double AgK =  2.0*M_PI*(dKx*ka1 + dKy*ka2);
        double AK  = -4.0*M_PI*M_PI*(ka1*ka1 + ka2*ka2)*K;
        PhiCos1[f + (size_t)i*k1] = c;
        WCol1  [f + (size_t)i*k1] = (Av - AgK)*s + AK*c;
        return;
    }

    // ---- Omega2 section ------------------------------------------------
    tid -= N * k1;
    if (tid < N * k2) {
        int i = tid / k2, f = tid % k2;
        double x = z1[i*4], y = z1[i*4+1], vx = z1[i*4+2], vy = z1[i*4+3];
        double K  = z2[i*5+2], dKx = z2[i*5+3], dKy = z2[i*5+4];
        double ka1 = om2[f*2], ka2 = om2[f*2+1];
        double arg = 2.0*M_PI*(x*ka1 + y*ka2);
        double s = sin(arg), c = cos(arg);
        double Av  =  2.0*M_PI*(vx*ka1  + vy*ka2);
        double AgK =  2.0*M_PI*(dKx*ka1 + dKy*ka2);
        double AK  = -4.0*M_PI*M_PI*(ka1*ka1 + ka2*ka2)*K;
        PhiCos2[f + (size_t)i*k2] = c;
        PhiSin2[f + (size_t)i*k2] = s;
        WCol2r [f + (size_t)i*k2] = (Av - AgK)*s + AK*c;   // same formula as Omega1
        WCol2i [f + (size_t)i*k2] = -(Av + AgK)*c + AK*s;  // f2+f6 formula for imag cols
        RowW1  [f + (size_t)i*k2] = -Av * s;                // G33 row weight 1
        RowW2  [f + (size_t)i*k2] = AK*s - AgK*c;          // G33 row weight 2
    }
}

// Scale element at (row, col) in Ngrid×Ngrid col-major matrix by factor
static __global__ void g_scale_rows(double* G, int Ngrid, int row_start, int row_end,
                                     int col_start, int col_end, double factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nrows = row_end - row_start, ncols = col_end - col_start;
    if (idx >= nrows * ncols) return;
    int r = idx % nrows + row_start;
    int c = idx / nrows + col_start;
    G[r + (size_t)c * Ngrid] *= factor;
}

// Transpose col-major (Ngrid×Ngrid) to row-major h_G[b*Ngrid+a] = G[b,a]
// col-major G[b,a] is at offset b + a*Ngrid.
// row-major h_G[b*Ngrid+a]. Since b+a*Ngrid ≠ b*Ngrid+a in general, need explicit transpose.
static __global__ void g_transpose_d2h(const double* __restrict__ G_cm,
                                        double* __restrict__ G_rm, int Ngrid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ngrid * Ngrid) return;
    int b = idx / Ngrid;  // row in output (row-major)
    int a = idx % Ngrid;  // col in output
    G_rm[idx] = G_cm[b + (size_t)a * Ngrid];  // G_cm[b, a] in col-major
}

void g_matrix_gpu_fast(const double* d_z1, const double* d_z2,
                        const double* h_omega1, const double* h_omega2,
                        int N, int k1, int k2, double delta, double* h_G)
{
    const int Ngrid = k1 + 2*k2;
    const double dd = delta * delta;

    cublasHandle_t blas;
    cublasCreate(&blas);

    // Upload omega arrays to GPU
    double *d_om1 = nullptr, *d_om2 = nullptr;
    cudaMalloc(&d_om1, k1*2*sizeof(double));
    cudaMalloc(&d_om2, k2*2*sizeof(double));
    cudaMemcpy(d_om1, h_omega1, k1*2*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_om2, h_omega2, k2*2*sizeof(double), cudaMemcpyHostToDevice);

    // Allocate precomputed matrices (freq × N, col-major)
    double *PhiCos1, *PhiCos2, *PhiSin2;
    double *WCol1, *WCol2r, *WCol2i;
    double *RowW1, *RowW2;
    cudaMalloc(&PhiCos1, (size_t)k1*N*sizeof(double));
    cudaMalloc(&PhiCos2, (size_t)k2*N*sizeof(double));
    cudaMalloc(&PhiSin2, (size_t)k2*N*sizeof(double));
    cudaMalloc(&WCol1,   (size_t)k1*N*sizeof(double));
    cudaMalloc(&WCol2r,  (size_t)k2*N*sizeof(double));
    cudaMalloc(&WCol2i,  (size_t)k2*N*sizeof(double));
    cudaMalloc(&RowW1,   (size_t)k2*N*sizeof(double));
    cudaMalloc(&RowW2,   (size_t)k2*N*sizeof(double));

    // Launch precompute kernel
    int total = N * (k1 + k2);
    g_precompute_kernel<<<(total+255)/256, 256>>>(
        d_z1, d_z2, d_om1, d_om2, N, k1, k2,
        PhiCos1, PhiCos2, PhiSin2,
        WCol1, WCol2r, WCol2i, RowW1, RowW2);
    cudaDeviceSynchronize();

    // Allocate output G (Ngrid × Ngrid col-major) — initialise to 0
    double *d_G = nullptr;
    cudaMalloc(&d_G, (size_t)Ngrid*Ngrid*sizeof(double));
    cudaMemset(d_G, 0, (size_t)Ngrid*Ngrid*sizeof(double));

    // Helper: submatrix pointer in col-major d_G at (row_start, col_start)
    auto blk = [&](int rs, int cs) -> double* {
        return d_G + rs + (size_t)cs * Ngrid;
    };

    double one = 1.0, zero = 0.0;

    // ----- 8 standard blocks (Phi × WCol^T) -----
    // G_11  [0:k1,     0:k1    ]  col-major ldc=Ngrid
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k1, k1, N,
                &dd, PhiCos1, k1, WCol1, k1, &zero, blk(0, 0), Ngrid);
    // G_12  [0:k1,     k1:k1+k2]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k1, k2, N,
                &dd, PhiCos1, k1, WCol2r, k2, &zero, blk(0, k1), Ngrid);
    // G_13  [0:k1,     k1+k2:Ngrid]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k1, k2, N,
                &dd, PhiCos1, k1, WCol2i, k2, &zero, blk(0, k1+k2), Ngrid);
    // G_21  [k1:k1+k2, 0:k1    ]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k1, N,
                &dd, PhiCos2, k2, WCol1, k1, &zero, blk(k1, 0), Ngrid);
    // G_22  [k1:k1+k2, k1:k1+k2]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k2, N,
                &dd, PhiCos2, k2, WCol2r, k2, &zero, blk(k1, k1), Ngrid);
    // G_23  [k1:k1+k2, k1+k2:Ngrid]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k2, N,
                &dd, PhiCos2, k2, WCol2i, k2, &zero, blk(k1, k1+k2), Ngrid);
    // G_31  [k1+k2:Ngrid, 0:k1  ]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k1, N,
                &dd, PhiSin2, k2, WCol1, k1, &zero, blk(k1+k2, 0), Ngrid);
    // G_32  [k1+k2:Ngrid, k1:k1+k2]
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k2, N,
                &dd, PhiSin2, k2, WCol2r, k2, &zero, blk(k1+k2, k1), Ngrid);

    // ----- G_33 special block -----
    // Row-weight matrices (RowW1, RowW2) play the role of Phi here.
    // G_33 = dd * (RowW1 × PhiCos2^T + RowW2 × PhiSin2^T)
    // After D_col ×2: coeff becomes 2*dd.
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k2, N,
                &dd, RowW1, k2, PhiCos2, k2, &zero, blk(k1+k2, k1+k2), Ngrid);
    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, k2, k2, N,
                &dd, RowW2, k2, PhiSin2, k2, &one,  blk(k1+k2, k1+k2), Ngrid);

    cudaDeviceSynchronize();

    // ----- Apply D_col: multiply Omega2 columns (k1..Ngrid-1) by 2 -----
    {
        int nelems = Ngrid * (Ngrid - k1);
        g_scale_rows<<<(nelems+255)/256, 256>>>(
            d_G, Ngrid, 0, Ngrid, k1, Ngrid, 2.0);
    }

    // ----- Apply D_row: multiply Omega1 rows 1..k1-1 by 2 -----
    {
        int nelems = (k1-1) * Ngrid;
        if (nelems > 0)
            g_scale_rows<<<(nelems+255)/256, 256>>>(
                d_G, Ngrid, 1, k1, 0, Ngrid, 2.0);
    }

    cudaDeviceSynchronize();

    // ----- Transpose col-major d_G → row-major for h_G -----
    double *d_G_rm = nullptr;
    cudaMalloc(&d_G_rm, (size_t)Ngrid*Ngrid*sizeof(double));
    g_transpose_d2h<<<(Ngrid*Ngrid+255)/256, 256>>>(d_G, d_G_rm, Ngrid);
    cudaDeviceSynchronize();
    cudaMemcpy(h_G, d_G_rm, (size_t)Ngrid*Ngrid*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(PhiCos1); cudaFree(PhiCos2); cudaFree(PhiSin2);
    cudaFree(WCol1);   cudaFree(WCol2r);  cudaFree(WCol2i);
    cudaFree(RowW1);   cudaFree(RowW2);
    cudaFree(d_om1);   cudaFree(d_om2);
    cudaFree(d_G);     cudaFree(d_G_rm);
    cublasDestroy(blas);
}

// ===========================================================================
// Standalone verifier — compiled only when -DG_VERIFY_MAIN is set
// ===========================================================================
#ifdef G_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[])
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s z1.csv z2.csv omega1.csv omega2.csv [out_G.csv] [ref_G.csv]\n\n"
            "  z1.csv     — Nr²×4 velocity field (x y vx vy)\n"
            "  z2.csv     — Nr²×5 diffusivity field (x y K dKx dKy)\n"
            "  omega1.csv — k1×2 first-set frequencies\n"
            "  omega2.csv — k2×2 second-set frequencies\n"
            "  out_G.csv  — optional: write GPU G here\n"
            "  ref_G.csv  — optional: reference G from R for comparison\n",
            argv[0]);
        return 1;
    }

    const char* z1_path     = argv[1];
    const char* z2_path     = argv[2];
    const char* om1_path    = argv[3];
    const char* om2_path    = argv[4];
    const char* out_path    = (argc >= 6) ? argv[5] : nullptr;   // write GPU G here
    const char* ref_path    = (argc >= 7) ? argv[6] : nullptr;   // optional R reference

    // ---- Read inputs -------------------------------------------------------
    int nz1r = 0, nz1c = 0, nz2r = 0, nz2c = 0;
    int no1r = 0, no1c = 0, no2r = 0, no2c = 0;

    std::vector<double> h_z1, h_z2, h_om1, h_om2;
    try {
        h_z1  = read_csv_flat(z1_path,  nz1r, nz1c, true);
        h_z2  = read_csv_flat(z2_path,  nz2r, nz2c, true);
        h_om1 = read_csv_flat(om1_path, no1r, no1c, true);
        h_om2 = read_csv_flat(om2_path, no2r, no2c, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading inputs: %s\n", e.what());
        return 1;
    }

    if (nz1c != 4 || nz2c != 5 || no1c != 2 || no2c != 2) {
        fprintf(stderr, "ERROR: unexpected column counts z1=%d z2=%d om1=%d om2=%d\n",
                nz1c, nz2c, no1c, no2c);
        return 1;
    }
    int N  = nz1r;
    int k1 = no1r, k2 = no2r;
    int Nr = (int)round(sqrt((double)N));
    double delta = 1.0 / Nr;
    int Ngrid = k1 + 2 * k2;

    printf("N=%d  Nr=%d  k1=%d  k2=%d  Ngrid=%d\n", N, Nr, k1, k2, Ngrid);

    // ---- Upload z1, z2 to device ------------------------------------------
    double *d_z1 = nullptr, *d_z2 = nullptr;
    cudaMalloc(&d_z1, N * 4 * sizeof(double));
    cudaMalloc(&d_z2, N * 5 * sizeof(double));
    cudaMemcpy(d_z1, h_z1.data(), N * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, h_z2.data(), N * 5 * sizeof(double), cudaMemcpyHostToDevice);

    // ---- Run GPU G computation --------------------------------------------
    std::vector<double> h_G(Ngrid * Ngrid, 0.0);
    printf("Computing G matrix on GPU (%d × %d)...\n", Ngrid, Ngrid);

    g_matrix_gpu(d_z1, d_z2, h_om1.data(), h_om2.data(),
                 N, k1, k2, delta, h_G.data());

    cudaFree(d_z1);
    cudaFree(d_z2);

    printf("G[0][0]=%.8f  G[220][220]=%.8f\n",
           h_G[0*Ngrid+0], h_G[220*Ngrid+220]);

    // ---- Optionally write output ------------------------------------------
    if (out_path) {
        try {
            write_csv_flat(out_path, h_G.data(), Ngrid, Ngrid);
            printf("Written: %s\n", out_path);
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR writing output: %s\n", e.what());
        }
    }

    // ---- Read reference G (optional) --------------------------------------
    if (!ref_path) {
        printf("\nNo reference G provided — skipping comparison.\n");
        printf("(Pass a ref_G.csv as 5th argument to compare GPU vs R.)\n");
        return 0;
    }

    int refr = 0, refc = 0;
    std::vector<double> h_ref;
    try {
        h_ref = read_csv_flat(ref_path, refr, refc, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading reference '%s': %s\n", ref_path, e.what());
        return 1;
    }
    if (refr != Ngrid || refc != Ngrid) {
        fprintf(stderr, "ERROR: reference expected %d×%d, got %d×%d\n",
                Ngrid, Ngrid, refr, refc);
        return 1;
    }

    // ---- Field-by-field comparison ----------------------------------------
    int    total = Ngrid * Ngrid;
    int    compared = 0;
    double max_diff = 0.0, sum_diff = 0.0;
    int    gt_1e6 = 0;

    struct BadCell { int i, j; double diff, gpu, ref; };
    std::vector<BadCell> bad;
    bad.reserve(total);

    for (int i = 0; i < total; i++) {
        bool gn = std::isnan(h_G[i]);
        bool rn = std::isnan(h_ref[i]);
        if (gn || rn) continue;
        double d = std::fabs(h_G[i] - h_ref[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
        ++compared;
        if (d > 1e-6) ++gt_1e6;
        bad.push_back({i / Ngrid, i % Ngrid, d, h_G[i], h_ref[i]});
    }
    double mean_diff = (compared > 0) ? sum_diff / compared : 0.0;

    std::sort(bad.begin(), bad.end(),
              [](const BadCell& a, const BadCell& b){ return a.diff > b.diff; });

    printf("\n=== G Matrix Verification ===\n");
    printf("Total cells        : %d\n",   total);
    printf("Cells compared     : %d\n",   compared);
    printf("Max  |GPU - R|     : %.3e\n", max_diff);
    printf("Mean |GPU - R|     : %.3e\n", mean_diff);
    printf("Cells > 1e-6       : %d\n",   gt_1e6);

    int show = (int)std::min((size_t)5, bad.size());
    if (show > 0 && bad[0].diff > 1e-8) {
        printf("\nTop-%d worst cells (row col GPU R diff):\n", show);
        for (int i = 0; i < show; i++)
            printf("  [%3d,%3d]  GPU=%.10g  R=%.10g  |diff|=%.3e\n",
                   bad[i].i, bad[i].j, bad[i].gpu, bad[i].ref, bad[i].diff);
    }

    bool pass = (max_diff < 1e-6);
    printf("\nResult : %s\n", pass ? "PASS" : "FAIL");
    if (!pass)
        printf("  Reason: max |diff| = %.3e >= 1e-6\n", max_diff);
    return pass ? 0 : 1;
}

#endif // G_VERIFY_MAIN
