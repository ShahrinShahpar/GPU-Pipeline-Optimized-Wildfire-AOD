// gibbs_sampler.cu  —  GPU Gibbs FFBS2 sampler, OPTIMIZED v3.
//
// All computation in FP64; no mixed precision.
//
// New optimizations over v2:
//
// OPT-G: Exploit G's right-half identity structure.
//   From main_pipeline.cu:
//     expG.topRightCorner(Ngrid, Ngrid)    = I;
//     expG.bottomRightCorner(Ngrid, Ngrid) = I;
//   So G = [G_left | I ; G_left | I] where G_left is dense 800×400.
//   (More precisely, G[:, half:n_state] = [I_{half}; I_{half}])
//
//   C_prd = G * C_flt * G^T + W breaks into:
//     Z   = G_left * C_flt[0:half,:]  +  [ C_flt[half:n_state,:] ]  (broadcast add)
//                                         [ C_flt[half:n_state,:] ]
//     C_prd = Z[:,0:half] * G_left^T  +  [ Z[:,half:] | Z[:,half:] ] + W
//   Each step uses DGEMM(n_state, n_state, half=400) instead of DGEMM(800,800,800).
//   FLOPs: 2 × (800×800×400) = 1024M  vs  2 × (800×800×800) = 2048M → 2× speedup.
//
// OPT-9: Precompute smoother-gain B_t = C_flt[t] * G^T * C_prd[t+1]^{-1} during
//   the forward pass. Store G*C_flt[t] (Z_t) and B_t in FP64 per time step.
//   Backward then uses 1 DGEMM (B_t × Z_t) to get H_t, eliminating:
//     - potrf(C_prd) per backward step  (moved to forward, same total work)
//     - 1 large potrs per backward step  (same move)
//     - 1 DGEMM + 1 geam per backward step  (genuine savings)
//
// All previous optimizations retained:
//   OPT-1  Cholesky replaces LU for Q and C_prd (SPD matrices)
//   OPT-2  Custom diagonal kernel (no Vt matrix copy)
//   OPT-3  Pre-upload Ftt and yt once before Gibbs loop
//   OPT-4  Half-state DGEMM for Ftt-based products (Ftt right half = 0)
//   OPT-5  No unnecessary cudaDeviceSynchronize inside filterStep
//   OPT-7  Batched residuals: 1 D2H sync for all T steps

#include "gibbs_sampler.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

using namespace std;
using Clock = chrono::high_resolution_clock;
using Sec   = chrono::duration<double>;

// --------------------------------------------------------------------------
// CUDA event-based timer (µs resolution, GPU-side)
// --------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t t0, t1;
    double accum_ms = 0.0;
    GpuTimer() { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~GpuTimer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void start() { cudaEventRecord(t0); }
    void stop()  { cudaEventRecord(t1); }
    void sync_and_add() {
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        accum_ms += ms;
    }
};

#define GS_CUDA_CHECK(call) do { \
    cudaError_t _e = call; \
    if (_e != cudaSuccess) { \
        cerr << "CUDA error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " - " << cudaGetErrorString(_e) << endl; exit(1); } \
} while(0)
#define GS_CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = call; \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        cerr << "cuBLAS error [gibbs] " << __FILE__ << ":" << __LINE__ << endl; exit(1); } \
} while(0)
#define GS_CUSOLVER_CHECK(call) do { \
    cusolverStatus_t _s = call; \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        cerr << "cuSOLVER error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " code=" << _s << endl; exit(1); } \
} while(0)
#define GS_CURAND_CHECK(call) do { \
    curandStatus_t _s = call; \
    if (_s != CURAND_STATUS_SUCCESS) { \
        cerr << "cuRAND error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " code=" << _s << endl; exit(1); } \
} while(0)

// ---------------------------------------------------------------------------
// CPU Wishart sampler (unchanged)
// ---------------------------------------------------------------------------
static MatrixXd rwish_cpu(mt19937& rng, int nu, const MatrixXd& Phi) {
    int p = Phi.rows();
    double phi_scale = Phi(0, 0);
    normal_distribution<double> nd(0.0, 1.0);
    MatrixXd Z(p, nu);
    for (int j = 0; j < nu; j++)
        for (int i = 0; i < p; i++)
            Z(i, j) = nd(rng);
    MatrixXd W(p, p);
    W.setZero();
    W.selfadjointView<Eigen::Lower>().rankUpdate(Z, phi_scale);
    W = W.selfadjointView<Eigen::Lower>();
    return W;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------
__global__ void gs_add_diag_reg(double* M, int n, double reg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) M[i*n + i] += reg;
}

__global__ void gs_squared_norm(const double* v, double* out, int n) {
    __shared__ double sh[256];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x + tid;
    sh[tid] = (idx < n) ? v[idx]*v[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

__global__ void gs_add_sigma_diag(double* Q, int n, int n1, double s1, double s2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Q[i*n + i] += (i < n1) ? s1 : s2;
}

// ---------------------------------------------------------------------------
// CPU sampling helpers
// ---------------------------------------------------------------------------
static double rinvgamma(mt19937& gen, double alpha, double beta) {
    gamma_distribution<double> gd(alpha, 1.0/beta);
    return 1.0 / gd(gen);
}

static MatrixXd riwish(mt19937& gen, int nu, const MatrixXd& Phi) {
    int p = Phi.rows();
    MatrixXd A = MatrixXd::Zero(p, p);
    normal_distribution<double> nd(0.0, 1.0);
    chi_squared_distribution<double> cs(nu);
    for (int i = 0; i < p; i++) {
        A(i, i) = sqrt(cs(gen));
        for (int j = 0; j < i; j++) A(i, j) = nd(gen);
    }
    MatrixXd X = A.triangularView<Eigen::Lower>()
                  .solve(MatrixXd::Identity(p, p));
    Eigen::LLT<MatrixXd> llt(Phi);
    MatrixXd Z = llt.matrixL().solve(X);
    MatrixXd W(p, p);
    W.setZero();
    W.selfadjointView<Eigen::Lower>().rankUpdate(Z.transpose());
    return W.selfadjointView<Eigen::Lower>();
}

// ---------------------------------------------------------------------------
// GibbsSamplerPipeline
// ---------------------------------------------------------------------------
class GibbsSamplerPipeline {
public:
    cublasHandle_t     cublas;
    cusolverDnHandle_t solver;
    curandGenerator_t  curand_gen;

    int n_state, max_obs, max_T, half;

    // Working buffers (FP64)
    double *d_G, *d_W;
    double *d_m_flt, *d_C_flt, *d_m_prd, *d_C_prd;
    double *d_Ftt, *d_yt;
    double *d_Q, *d_K_gain, *d_innov;
    double *d_theta, *d_ht, *d_Ht, *d_z;
    double *d_residual;
    double *d_state_rsd;
    double *d_tmp1, *d_tmp2, *d_tmp3, *d_tmp4, *d_tmp5;
    int    *d_info, *d_ipiv;
    double *d_work;
    int lwork;

    // OPT-7: persistent squared-norm accumulators
    double *d_y1ss, *d_y2ss;

    // All-time-step snapshots
    vector<double*> d_m_flt_all, d_C_flt_all, d_m_prd_all, d_C_prd_all;

    // OPT-3: preloaded observation data
    vector<double*> d_Ftt_preload, d_yt_preload;
    vector<int>     preload_nobs;

    // OPT-9 (FP64): smoother-gain B_t and G*C_flt[t] per time step
    //   d_Bt_all[t] = B_t = C_flt[t] * G^T * C_prd[t+1]^{-1} (n_state × n_state)
    //   d_Z_all[t]  = Z_t = G * C_flt[t]                      (n_state × n_state)
    //   Both allocated for t = 0..T-2 (T-1 backward steps).
    vector<double*> d_Bt_all, d_Z_all;

    // OPT-10: CUDA stream parallelism — B_t chain and K-gain chain are independent
    // after C_prd is computed. Run them concurrently on separate streams.
    //
    //   stream_Bt  : potrf(C_prd) + potrs(Z) + geam(B_t) + save
    //   stream_K   : innov + Q DGEMMs + potrf(Q) + potrs(K) + geam(K^T)
    //
    // After fork, main thread does cudaStreamSynchronize(stream_K), then
    // proceeds with m_flt update and C_flt DGEMMs on the default stream.
    // B_t save on stream_Bt completes asynchronously before the outer
    // cudaDeviceSynchronize() at the end of the forward loop.
    //
    // Conflict analysis: B_t uses d_tmp3/tmp4/tmp5; K-gain uses d_kQ/d_kKg.
    // No buffer conflicts between the two streams.
    cudaStream_t       stream_Bt, stream_K;
    cublasHandle_t     cublas_Bt, cublas_K;      // per-stream cuBLAS handles
    cusolverDnHandle_t solver_Bt, solver_K;       // per-stream cuSolver handles
    double *d_kQ;     // K-stream: Q matrix copy for potrf (replaces d_tmp3 in K chain)
    double *d_kKg;    // K-stream: K_gain transpose buffer  (replaces d_tmp4 in K chain)
    double *d_work_Bt, *d_work_K;  // per-stream cuSolver workspaces
    int    *d_info_Bt, *d_info_K;  // per-stream cuSolver info
    // Event barrier: non-blocking streams must wait for PRED (default stream) to finish
    // before reading d_C_prd / d_m_prd / d_tmp1.
    cudaEvent_t event_pred_done;

    GibbsSamplerPipeline(int n_state_, int max_obs_, int max_T_,
                         unsigned long long curand_seed=42)
        : n_state(n_state_), max_obs(max_obs_), max_T(max_T_), half(n_state_/2)
    {
        GS_CUBLAS_CHECK(cublasCreate(&cublas));
        GS_CUSOLVER_CHECK(cusolverDnCreate(&solver));
        GS_CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        GS_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen, curand_seed));

        // OPT-10: create non-blocking streams and per-stream handles
        GS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_Bt, cudaStreamNonBlocking));
        GS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_K,  cudaStreamNonBlocking));
        GS_CUBLAS_CHECK(cublasCreate(&cublas_Bt));
        GS_CUBLAS_CHECK(cublasSetStream(cublas_Bt, stream_Bt));
        GS_CUBLAS_CHECK(cublasCreate(&cublas_K));
        GS_CUBLAS_CHECK(cublasSetStream(cublas_K, stream_K));
        GS_CUSOLVER_CHECK(cusolverDnCreate(&solver_Bt));
        GS_CUSOLVER_CHECK(cusolverDnSetStream(solver_Bt, stream_Bt));
        GS_CUSOLVER_CHECK(cusolverDnCreate(&solver_K));
        GS_CUSOLVER_CHECK(cusolverDnSetStream(solver_K, stream_K));
        GS_CUDA_CHECK(cudaEventCreateWithFlags(&event_pred_done, cudaEventDisableTiming));

        auto alloc = [&](double** p, size_t n){
            GS_CUDA_CHECK(cudaMalloc(p, n*sizeof(double)));
        };

        alloc(&d_G,  (size_t)n_state*n_state);
        alloc(&d_W,  (size_t)n_state*n_state);
        alloc(&d_m_flt, n_state);
        alloc(&d_C_flt, (size_t)n_state*n_state);
        alloc(&d_m_prd, n_state);
        alloc(&d_C_prd, (size_t)n_state*n_state);
        alloc(&d_Ftt,   (size_t)max_obs*n_state);
        alloc(&d_yt,    max_obs);
        alloc(&d_Q,     (size_t)max_obs*max_obs);
        alloc(&d_K_gain,(size_t)n_state*max_obs);
        int mobs = max(max_obs, n_state);
        alloc(&d_innov, mobs);
        alloc(&d_theta, (size_t)max_T*n_state);
        alloc(&d_ht,    n_state);
        alloc(&d_Ht,    (size_t)n_state*n_state);
        alloc(&d_z,     n_state);
        alloc(&d_residual, mobs);
        alloc(&d_state_rsd,(size_t)n_state*(max_T>1?max_T-1:1));
        size_t wmax = max((size_t)n_state*n_state, (size_t)mobs*(size_t)mobs);
        alloc(&d_tmp1, wmax);
        alloc(&d_tmp2, wmax);
        alloc(&d_tmp3, wmax);
        alloc(&d_tmp4, (size_t)n_state*mobs);
        alloc(&d_tmp5, (size_t)n_state*n_state);
        GS_CUDA_CHECK(cudaMalloc(&d_y1ss, sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_y2ss, sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        GS_CUDA_CHECK(cudaMalloc(&d_ipiv, max(max_obs,n_state)*sizeof(int)));

        int lw1, lw2, lw3;
        GS_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                                      max_obs, d_Q,    max_obs, &lw1));
        GS_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                                      n_state, d_tmp3, n_state, &lw2));
        GS_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                                      n_state, d_Ht,   n_state, &lw3));
        lwork = max({lw1, lw2, lw3});
        GS_CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(double)));

        // OPT-10: extra per-stream buffers (non-overlapping with B_t buffers)
        size_t kbuf = max((size_t)max_obs*(size_t)max_obs,
                          (size_t)n_state*(size_t)max_obs);
        GS_CUDA_CHECK(cudaMalloc(&d_kQ,      kbuf*sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_kKg,     (size_t)n_state*max_obs*sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_work_Bt, lwork*sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_work_K,  lwork*sizeof(double)));
        GS_CUDA_CHECK(cudaMalloc(&d_info_Bt, sizeof(int)));
        GS_CUDA_CHECK(cudaMalloc(&d_info_K,  sizeof(int)));

        for (int t = 0; t <= max_T; t++) {
            double *pm, *pC, *pa, *pb;
            GS_CUDA_CHECK(cudaMalloc(&pm, n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pC, (size_t)n_state*n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pa, n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pb, (size_t)n_state*n_state*sizeof(double)));
            d_m_flt_all.push_back(pm);
            d_C_flt_all.push_back(pC);
            d_m_prd_all.push_back(pa);
            d_C_prd_all.push_back(pb);
        }
    }

    ~GibbsSamplerPipeline() {
        cudaFree(d_G); cudaFree(d_W);
        cudaFree(d_m_flt); cudaFree(d_C_flt); cudaFree(d_m_prd); cudaFree(d_C_prd);
        cudaFree(d_Ftt); cudaFree(d_yt);
        cudaFree(d_Q); cudaFree(d_K_gain); cudaFree(d_innov);
        cudaFree(d_theta); cudaFree(d_ht); cudaFree(d_Ht); cudaFree(d_z);
        cudaFree(d_residual); cudaFree(d_state_rsd);
        cudaFree(d_tmp1); cudaFree(d_tmp2); cudaFree(d_tmp3);
        cudaFree(d_tmp4); cudaFree(d_tmp5);
        cudaFree(d_y1ss); cudaFree(d_y2ss);
        cudaFree(d_info); cudaFree(d_ipiv); cudaFree(d_work);
        // OPT-10: stream parallelism resources
        cudaFree(d_kQ); cudaFree(d_kKg);
        cudaFree(d_work_Bt); cudaFree(d_work_K);
        cudaFree(d_info_Bt); cudaFree(d_info_K);
        cublasDestroy(cublas_Bt); cublasDestroy(cublas_K);
        cusolverDnDestroy(solver_Bt); cusolverDnDestroy(solver_K);
        cudaStreamDestroy(stream_Bt); cudaStreamDestroy(stream_K);
        cudaEventDestroy(event_pred_done);
        for (auto p : d_m_flt_all)  cudaFree(p);
        for (auto p : d_C_flt_all)  cudaFree(p);
        for (auto p : d_m_prd_all)  cudaFree(p);
        for (auto p : d_C_prd_all)  cudaFree(p);
        for (auto p : d_Ftt_preload) cudaFree(p);
        for (auto p : d_yt_preload)  cudaFree(p);
        for (auto p : d_Bt_all)  cudaFree(p);
        for (auto p : d_Z_all)   cudaFree(p);
        cublasDestroy(cublas);
        cusolverDnDestroy(solver);
        curandDestroyGenerator(curand_gen);
    }

    void preloadObservations(const vector<MatrixXd>& Ftt_all,
                             const vector<VectorXd>& yt_all) {
        int T = (int)Ftt_all.size();
        d_Ftt_preload.resize(T);
        d_yt_preload.resize(T);
        preload_nobs.resize(T);
        // OPT-9: allocate smoother-gain buffers for T-1 steps
        d_Bt_all.resize(T-1, nullptr);
        d_Z_all.resize(T-1, nullptr);
        for (int t = 0; t < T; t++) {
            int nobs = (int)yt_all[t].size();
            preload_nobs[t] = nobs;
            GS_CUDA_CHECK(cudaMalloc(&d_Ftt_preload[t],
                                     (size_t)nobs*n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&d_yt_preload[t],
                                     (size_t)nobs*sizeof(double)));
            GS_CUDA_CHECK(cudaMemcpy(d_Ftt_preload[t], Ftt_all[t].data(),
                                     (size_t)nobs*n_state*sizeof(double),
                                     cudaMemcpyHostToDevice));
            GS_CUDA_CHECK(cudaMemcpy(d_yt_preload[t], yt_all[t].data(),
                                     (size_t)nobs*sizeof(double),
                                     cudaMemcpyHostToDevice));
        }
        for (int t = 0; t < T-1; t++) {
            GS_CUDA_CHECK(cudaMalloc(&d_Bt_all[t],
                                     (size_t)n_state*n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&d_Z_all[t],
                                     (size_t)n_state*n_state*sizeof(double)));
        }
    }

    void setGW(const MatrixXd& G, const MatrixXd& W) {
        GS_CUDA_CHECK(cudaMemcpy(d_G, G.data(),
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_W, W.data(),
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyHostToDevice));
    }

    void initialize(const VectorXd& m0, const MatrixXd& C0) {
        GS_CUDA_CHECK(cudaMemcpy(d_m_flt_all[0], m0.data(),
                                 n_state*sizeof(double), cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt_all[0], C0.data(),
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyHostToDevice));
    }

    // OPT-G helper: compute Z = G * C_flt and store in d_out (n_state × n_state).
    //   Exploits G = [G_left | I; G_left | I] where G_left = G[:, 0:half].
    //   Z = G_left * C_flt[0:half, :] + [C_flt[half:, :]; C_flt[half:, :]]
    //   Uses 1 DGEMM(n_state, n_state, half) + 2 dgeam  (2× fewer FLOPs than full DGEMM).
    void compute_G_times_C(const double* d_C, double* d_out) {
        double one = 1.0, zero = 0.0;
        // d_tmp1 = G_left * C[0:half, :]   (n_state × n_state, k=half)
        // G_left = d_G[:, 0:half]: pointer d_G, lda=n_state, m=n_state, k=half
        // C[0:half, :] = d_C (first half rows), pointer d_C, lda=n_state, k=half, n=n_state
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state, half,
                                    &one, d_G, n_state, d_C, n_state,
                                    &zero, d_out, n_state));
        // d_out[0:half, :] += C[half:n_state, :]
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    half, n_state,
                                    &one, d_out,            n_state,
                                    &one, d_C + half,       n_state,
                                    d_out, n_state));
        // d_out[half:n_state, :] += C[half:n_state, :]
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    half, n_state,
                                    &one, d_out + half,     n_state,
                                    &one, d_C + half,       n_state,
                                    d_out + half, n_state));
    }

    // OPT-G helper: compute C_prd = Z * G^T + W, store in d_C_prd.
    //   G^T = [G_left^T; I_ext^T] where I_ext^T = [I | I].
    //   Z * G^T = Z[:,0:half] * G_left^T + [Z[:,half:] | Z[:,half:]]
    //   Uses 1 DGEMM(n_state, n_state, half) + 3 dgeam.
    void compute_Z_times_GT_plus_W(const double* d_Z, double* d_C_prd_out) {
        double one = 1.0, zero = 0.0;
        // C_prd = Z_left * G_left^T   (G_left^T via OP_T on d_G with k=half)
        // Z_left = d_Z[:, 0:half]: pointer d_Z, lda=n_state, m=n_state, k=half
        // G_left^T: d_G, OP_T, lda=n_state, k=half
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                    n_state, n_state, half,
                                    &one, d_Z, n_state, d_G, n_state,
                                    &zero, d_C_prd_out, n_state));
        // C_prd[:, 0:half] += Z[:, half:n_state]
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, half,
                                    &one, d_C_prd_out,                   n_state,
                                    &one, d_Z + (size_t)half*n_state,    n_state,
                                    d_C_prd_out, n_state));
        // C_prd[:, half:n_state] += Z[:, half:n_state]
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, half,
                                    &one, d_C_prd_out + (size_t)half*n_state, n_state,
                                    &one, d_Z + (size_t)half*n_state,         n_state,
                                    d_C_prd_out + (size_t)half*n_state, n_state));
        // C_prd += W
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state,
                                    &one, d_C_prd_out, n_state,
                                    &one, d_W,         n_state,
                                    d_C_prd_out, n_state));
    }

    // Forward filter step t.
    // OPT-G: C_prd via G-structure DGEMMs (half the FLOPs).
    // OPT-9 + OPT-10: precompute B_{t-1} on stream_Bt while computing K-gain on stream_K.
    // Both chains are independent after C_prd is ready:
    //   stream_Bt : potrf(C_prd) → potrs(Z) → geam(B_t^T) → save  [uses d_tmp3/4/5]
    //   stream_K  : innov + Q DGEMMs + potrf(Q) + potrs(K) + geam(K^T) [uses d_kQ/d_kKg]
    // Join: cudaStreamSynchronize(stream_K) before m_flt / C_flt update.
    void filterStep(int t, double sigma1, double sigma2, int n1) {
        int n_obs = preload_nobs[t];

        GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_flt_all[t],
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt, d_C_flt_all[t],
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyDeviceToDevice));

        double one=1.0, zero=0.0, neg1=-1.0;

        // ---- PREDICTION (default stream) ----
        // m_prd = G * m_flt
        GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                    &one, d_G, n_state, d_m_flt, 1,
                                    &zero, d_m_prd, 1));

        // OPT-G: Z = G * C_flt using G's identity-block right half
        compute_G_times_C(d_C_flt, d_tmp1);  // d_tmp1 = Z

        // OPT-9: save Z for backward
        if (t > 0 && t-1 < (int)d_Z_all.size() && d_Z_all[t-1] != nullptr) {
            GS_CUDA_CHECK(cudaMemcpy(d_Z_all[t-1], d_tmp1,
                                     (size_t)n_state*n_state*sizeof(double),
                                     cudaMemcpyDeviceToDevice));
        }

        // OPT-G: C_prd = Z * G^T + W
        compute_Z_times_GT_plus_W(d_tmp1, d_C_prd);

        GS_CUDA_CHECK(cudaMemcpy(d_m_prd_all[t], d_m_prd,
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_prd_all[t], d_C_prd,
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyDeviceToDevice));

        // ---- BARRIER: record end-of-PRED on default stream, then make both
        //              non-blocking streams wait before touching d_C_prd / d_m_prd / d_tmp1.
        GS_CUDA_CHECK(cudaEventRecord(event_pred_done));  // on default (NULL) stream
        GS_CUDA_CHECK(cudaStreamWaitEvent(stream_Bt, event_pred_done, 0));
        GS_CUDA_CHECK(cudaStreamWaitEvent(stream_K,  event_pred_done, 0));

        // ---- FORK: launch B_t chain (stream_Bt) and K-gain chain (stream_K) ----

        bool do_Bt = (t > 0 && t-1 < (int)d_Bt_all.size() && d_Bt_all[t-1] != nullptr);

        // --- stream_Bt: B_{t-1} = (C_prd^{-1} * Z)^T ---
        if (do_Bt) {
            GS_CUDA_CHECK(cudaMemcpyAsync(d_tmp3, d_C_prd,
                                          (size_t)n_state*n_state*sizeof(double),
                                          cudaMemcpyDeviceToDevice, stream_Bt));
            GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver_Bt, CUBLAS_FILL_MODE_LOWER,
                                               n_state, d_tmp3, n_state,
                                               d_work_Bt, lwork, d_info_Bt));
            GS_CUDA_CHECK(cudaMemcpyAsync(d_tmp4, d_tmp1,
                                          (size_t)n_state*n_state*sizeof(double),
                                          cudaMemcpyDeviceToDevice, stream_Bt));
            GS_CUSOLVER_CHECK(cusolverDnDpotrs(solver_Bt, CUBLAS_FILL_MODE_LOWER,
                                               n_state, n_state, d_tmp3, n_state,
                                               d_tmp4, n_state, d_info_Bt));
            // B_{t-1} = d_tmp4^T
            GS_CUBLAS_CHECK(cublasDgeam(cublas_Bt, CUBLAS_OP_T, CUBLAS_OP_N,
                                        n_state, n_state,
                                        &one, d_tmp4, n_state,
                                        &zero, d_tmp5, n_state, d_tmp5, n_state));
            GS_CUDA_CHECK(cudaMemcpyAsync(d_Bt_all[t-1], d_tmp5,
                                          (size_t)n_state*n_state*sizeof(double),
                                          cudaMemcpyDeviceToDevice, stream_Bt));
        }

        // --- stream_K: K-gain = C_prd * Ftt^T * Q^{-1} ---
        // innovation = y - Ftt * m_prd
        GS_CUDA_CHECK(cudaMemcpyAsync(d_innov, d_yt_preload[t],
                                      n_obs*sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_K));
        GS_CUBLAS_CHECK(cublasDgemv(cublas_K, CUBLAS_OP_N, n_obs, n_state,
                                    &neg1, d_Ftt_preload[t], n_obs, d_m_prd, 1,
                                    &one, d_innov, 1));
        // Q = Ftt * C_prd * Ftt^T + sigma  (k=half, OPT-4)
        GS_CUBLAS_CHECK(cublasDgemm(cublas_K, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_obs, n_state, half,
                                    &one,  d_Ftt_preload[t], n_obs,
                                           d_C_prd, n_state,
                                    &zero, d_tmp2, n_obs));
        GS_CUBLAS_CHECK(cublasDgemm(cublas_K, CUBLAS_OP_N, CUBLAS_OP_T,
                                    n_obs, n_obs, half,
                                    &one,  d_tmp2, n_obs,
                                           d_Ftt_preload[t], n_obs,
                                    &zero, d_Q, n_obs));
        { int blk = (n_obs+255)/256;
          gs_add_sigma_diag<<<blk, 256, 0, stream_K>>>(d_Q, n_obs, n1, sigma1, sigma2);
          gs_add_diag_reg  <<<blk, 256, 0, stream_K>>>(d_Q, n_obs, 1e-8); }
        // K-gain: potrf(Q) + potrs
        GS_CUDA_CHECK(cudaMemcpyAsync(d_kQ, d_Q,
                                      (size_t)n_obs*n_obs*sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_K));
        GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver_K, CUBLAS_FILL_MODE_LOWER,
                                           n_obs, d_kQ, n_obs,
                                           d_work_K, lwork, d_info_K));
        GS_CUDA_CHECK(cudaMemcpyAsync(d_K_gain, d_tmp2,
                                      (size_t)n_obs*n_state*sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_K));
        GS_CUSOLVER_CHECK(cusolverDnDpotrs(solver_K, CUBLAS_FILL_MODE_LOWER,
                                           n_obs, n_state, d_kQ, n_obs,
                                           d_K_gain, n_obs, d_info_K));
        // Transpose K_gain: (n_obs × n_state) → (n_state × n_obs)
        GS_CUBLAS_CHECK(cublasDgeam(cublas_K, CUBLAS_OP_T, CUBLAS_OP_N,
                                    n_state, n_obs,
                                    &one, d_K_gain, n_obs,
                                    &zero, d_kKg, n_state, d_kKg, n_state));
        GS_CUDA_CHECK(cudaMemcpyAsync(d_K_gain, d_kKg,
                                      (size_t)n_state*n_obs*sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_K));

        // ---- JOIN: wait for both streams before reusing shared buffers ----
        // stream_K is the critical path (longer); stream_Bt should already be done.
        GS_CUDA_CHECK(cudaStreamSynchronize(stream_K));
        if (do_Bt) GS_CUDA_CHECK(cudaStreamSynchronize(stream_Bt));

        // ---- UPDATE (default stream): uses d_K_gain (ready), d_C_prd (still valid) ----
        // m_flt = m_prd + K_gain * innov
        GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_prd,
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_obs,
                                    &one, d_K_gain, n_state, d_innov, 1,
                                    &one, d_m_flt, 1));

        // OPT-4: C_flt = C_prd - K * Ftt_L * C_prd  (k=half)
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, half, n_obs,
                                    &one,  d_K_gain,           n_state,
                                           d_Ftt_preload[t],   n_obs,
                                    &zero, d_tmp1, n_state));
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state, half,
                                    &one,  d_tmp1, n_state, d_C_prd, n_state,
                                    &zero, d_tmp3, n_state));
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state,
                                    &one, d_C_prd, n_state,
                                    &neg1, d_tmp3, n_state,
                                    d_C_flt, n_state));

        GS_CUDA_CHECK(cudaMemcpy(d_m_flt_all[t+1], d_m_flt,
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt_all[t+1], d_C_flt,
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyDeviceToDevice));
        // NOTE: stream_Bt (B_t save) may still be running; it completes before
        // the outer cudaDeviceSynchronize() after the forward loop.
    }

    // OPT-9: Backward smoother using precomputed B_t (FP64) and Z_t = G*C_flt[t].
    //   Per step: h_t = m_flt[t] + B_t * diff            (replaces potrf+potrs for h_t)
    //             H_t = C_flt[t] - B_t * Z_t             (replaces potrf+2 potrs + DGEMM)
    //             theta[t] ~ N(h_t, H_t)                 (same as before)
    void backwardSample(int T) {
        double one=1.0, zero=0.0, neg1=-1.0;

        // Initial sample theta[T-1] ~ N(m_flt[T], C_flt[T])
        GS_CUDA_CHECK(cudaMemcpy(d_Ht, d_C_flt_all[T],
                                 (size_t)n_state*n_state*sizeof(double),
                                 cudaMemcpyDeviceToDevice));
        { int blk = (n_state+255)/256;
          gs_add_diag_reg<<<blk,256>>>(d_Ht, n_state, 1e-8); }
        GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER,
                                           n_state, d_Ht, n_state,
                                           d_work, lwork, d_info));
        GS_CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_z, n_state, 0.0, 1.0));
        GS_CUDA_CHECK(cudaMemcpy(d_ht, d_m_flt_all[T],
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUBLAS_CHECK(cublasDtrmv(cublas, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                    n_state, d_Ht, n_state, d_z, 1));
        GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &one, d_z, 1, d_ht, 1));
        GS_CUDA_CHECK(cudaMemcpy(d_theta+(T-1)*n_state, d_ht,
                                 n_state*sizeof(double), cudaMemcpyDeviceToDevice));

        for (int t = T-2; t >= 0; t--) {
            // Load filter state at time t: m_flt[t] = d_m_flt_all[t+1]
            //                              C_flt[t] = d_C_flt_all[t+1]
            GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_flt_all[t+1],
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUDA_CHECK(cudaMemcpy(d_m_prd, d_m_prd_all[t+1],
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));

            // diff = theta[t+1] - m_prd[t+1]
            GS_CUDA_CHECK(cudaMemcpy(d_innov, d_theta+(t+1)*n_state,
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &neg1, d_m_prd, 1, d_innov, 1));

            // OPT-9: h_t = m_flt[t] + B_t * diff  (1 DGEMV)
            GS_CUDA_CHECK(cudaMemcpy(d_ht, d_m_flt,
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                        &one, d_Bt_all[t], n_state, d_innov, 1,
                                        &one, d_ht, 1));

            // OPT-9: H_t = C_flt[t] - B_t * Z_t  (1 DGEMM, FP64)
            //   H_t = C_flt[t] - B_t * G*C_flt[t]
            //   Mathematically = C_flt[t] - B_t * C_prd[t+1] * B_t^T (smoother variance)
            GS_CUDA_CHECK(cudaMemcpy(d_Ht, d_C_flt_all[t+1],
                                     (size_t)n_state*n_state*sizeof(double),
                                     cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n_state, n_state, n_state,
                                        &one,  d_Bt_all[t], n_state,
                                               d_Z_all[t],  n_state,
                                        &zero, d_tmp3, n_state));
            GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n_state, n_state,
                                        &one, d_Ht, n_state,
                                        &neg1, d_tmp3, n_state,
                                        d_Ht, n_state));
            { int blk = (n_state+255)/256;
              gs_add_diag_reg<<<blk,256>>>(d_Ht, n_state, 1e-8); }

            // Factor H_t and sample theta[t] ~ N(h_t, H_t)
            GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER,
                                               n_state, d_Ht, n_state,
                                               d_work, lwork, d_info));
            GS_CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_z, n_state, 0.0, 1.0));
            GS_CUBLAS_CHECK(cublasDtrmv(cublas, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                        n_state, d_Ht, n_state, d_z, 1));
            GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &one, d_z, 1, d_ht, 1));
            GS_CUDA_CHECK(cudaMemcpy(d_theta+t*n_state, d_ht,
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }

    // OPT-7: Batched residuals (1 GPU sync for all T steps)
    void computeResiduals(int T,
                          const vector<int>& n1_per_t,
                          double& y1_ss, double& y1_n,
                          double& y2_ss, double& y2_n,
                          MatrixXd& state_rsd)
    {
        double one=1.0, neg1=-1.0;
        y1_ss = y1_n = y2_ss = y2_n = 0.0;

        GS_CUDA_CHECK(cudaMemset(d_y1ss, 0, sizeof(double)));
        GS_CUDA_CHECK(cudaMemset(d_y2ss, 0, sizeof(double)));

        for (int t = 0; t < T; t++) {
            int n1   = n1_per_t[t];
            int nobs = preload_nobs[t];
            int n2   = nobs - n1;

            if (n1 > 0) {
                GS_CUDA_CHECK(cudaMemcpy(d_residual, d_yt_preload[t],
                                         n1*sizeof(double), cudaMemcpyDeviceToDevice));
                GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n1, half,
                                            &neg1, d_Ftt_preload[t], nobs,
                                            d_theta + t*n_state, 1,
                                            &one, d_residual, 1));
                gs_squared_norm<<<(n1+255)/256, 256>>>(d_residual, d_y1ss, n1);
                y1_n += n1;
            }
            if (n2 > 0) {
                GS_CUDA_CHECK(cudaMemcpy(d_residual, d_yt_preload[t] + n1,
                                         n2*sizeof(double), cudaMemcpyDeviceToDevice));
                GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n2, half,
                                            &neg1, d_Ftt_preload[t] + n1, nobs,
                                            d_theta + t*n_state, 1,
                                            &one, d_residual, 1));
                gs_squared_norm<<<(n2+255)/256, 256>>>(d_residual, d_y2ss, n2);
                y2_n += n2;
            }
        }

        GS_CUDA_CHECK(cudaDeviceSynchronize());
        GS_CUDA_CHECK(cudaMemcpy(&y1_ss, d_y1ss, sizeof(double), cudaMemcpyDeviceToHost));
        GS_CUDA_CHECK(cudaMemcpy(&y2_ss, d_y2ss, sizeof(double), cudaMemcpyDeviceToHost));

        double one2 = 1.0;
        state_rsd.resize(n_state, T > 1 ? T-1 : 1);
        vector<double> buf((size_t)n_state*(T > 1 ? T-1 : 1));
        for (int t = 0; t < T-1; t++) {
            GS_CUDA_CHECK(cudaMemcpy(d_state_rsd + t*n_state,
                                     d_theta + (t+1)*n_state,
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                        &neg1, d_G, n_state, d_theta + t*n_state, 1,
                                        &one2, d_state_rsd + t*n_state, 1));
        }
        GS_CUDA_CHECK(cudaMemcpy(buf.data(), d_state_rsd,
                                 buf.size()*sizeof(double), cudaMemcpyDeviceToHost));
        for (int t = 0; t < T-1; t++)
            for (int i = 0; i < n_state; i++)
                state_rsd(i, t) = buf[t*n_state + i];
    }

    void getMflt(int T, vector<VectorXd>& out) {
        out.resize(T);
        vector<double> buf(n_state);
        for (int t = 0; t < T; t++) {
            GS_CUDA_CHECK(cudaMemcpy(buf.data(), d_m_flt_all[t+1],
                                     n_state*sizeof(double), cudaMemcpyDeviceToHost));
            out[t] = Eigen::Map<VectorXd>(buf.data(), n_state);
        }
    }
};

// ---------------------------------------------------------------------------
// run_gibbs_pipeline
// ---------------------------------------------------------------------------
void run_gibbs_pipeline(
    int T, int N_sample, int seed,
    const MatrixXd& expG,
    const VectorXd& m0,
    const MatrixXd& C0,
    const vector<MatrixXd>& Ftt_all,
    const vector<VectorXd>& yt_all,
    const vector<MatrixXd>& Vt_all,
    const vector<int>& n1_per_t,
    int max_obs,
    vector<VectorXd>& m_flt,
    vector<VectorXd>& m_flt_gibbs)
{
    cout << "=== Gibbs FFBS2 Pipeline (OPTIMIZED v3 — G-struct + OPT-9 FP64) ===" << endl;
    cout << "T=" << T << "  N_sample=" << N_sample << "  n_state=" << m0.size() << endl;

    mt19937 gen(seed);
    int n_state = (int)m0.size();

    const double alpha1_init=1.0, alpha2_init=1.0;
    const double beta1_init=0.1,  beta2_init=0.1;
    const int nu_init = C0.rows();
    MatrixXd Phi_init = 0.01 * MatrixXd::Identity(C0.rows(), C0.cols());

    double sigma1 = beta1_init / (alpha1_init + 1.0);
    double sigma2 = beta2_init / (alpha2_init + 1.0);
    cout << "  Sampling initial W ~ Wishart(" << nu_init << ", Phi)..." << flush;
    MatrixXd W = rwish_cpu(gen, nu_init, Phi_init);
    cout << " done." << endl;

    double alpha1=alpha1_init, alpha2=alpha2_init;
    double beta1=beta1_init,   beta2=beta2_init;
    int nu = nu_init;
    MatrixXd Phi = Phi_init;

    GibbsSamplerPipeline sampler(n_state, max_obs, T, (unsigned long long)seed);

    cout << "  Pre-uploading observations to GPU..." << flush;
    sampler.preloadObservations(Ftt_all, yt_all);
    cout << " done." << endl;

    double t_fwd=0, t_bwd=0, t_res=0, t_cpu=0;
    auto t_start = Clock::now();

    for (int k = 0; k < N_sample; k++) {
        if (k % max(1, N_sample/10) == 0)
            cout << "  Iteration " << k+1 << "/" << N_sample
                 << "  sigma1=" << sigma1 << "  sigma2=" << sigma2 << endl;

        sampler.setGW(expG, W);
        sampler.initialize(m0, C0);

        auto ta = Clock::now();
        for (int t = 0; t < T; t++)
            sampler.filterStep(t, sigma1, sigma2, n1_per_t[t]);
        GS_CUDA_CHECK(cudaDeviceSynchronize());
        t_fwd += Sec(Clock::now() - ta).count();

        auto tb = Clock::now();
        sampler.backwardSample(T);
        GS_CUDA_CHECK(cudaDeviceSynchronize());
        t_bwd += Sec(Clock::now() - tb).count();

        auto tc = Clock::now();
        double y1_ss, y1_n, y2_ss, y2_n;
        MatrixXd srsd;
        sampler.computeResiduals(T, n1_per_t, y1_ss, y1_n, y2_ss, y2_n, srsd);
        t_res += Sec(Clock::now() - tc).count();

        auto td = Clock::now();
        alpha1 += y1_n/2.0;  beta1 += y1_ss/2.0;
        alpha2 += y2_n/2.0;  beta2 += y2_ss/2.0;
        nu += (T-1);
        Phi += srsd * srsd.transpose();
        sigma1 = rinvgamma(gen, alpha1, beta1);
        sigma2 = rinvgamma(gen, alpha2, beta2);
        W = riwish(gen, nu, Phi);
        t_cpu += Sec(Clock::now() - td).count();
    }

    auto t_end = Clock::now();
    double secs = Sec(t_end - t_start).count();
    cout << "Gibbs done: " << secs << " s  (" << secs/N_sample << " s/iter)" << endl;
    cout << "  Per-stage breakdown (total " << N_sample << " iters):" << endl;
    cout << "    Forward filter : " << t_fwd << " s" << endl;
    cout << "    Backward sample: " << t_bwd << " s" << endl;
    cout << "    Residuals      : " << t_res << " s" << endl;
    cout << "    CPU (riwish)   : " << t_cpu << " s" << endl;
    cout << "  Final params: sigma1=" << sigma1 << "  sigma2=" << sigma2 << endl;

    sampler.getMflt(T, m_flt_gibbs);

    sampler.setGW(expG, W);
    sampler.initialize(m0, C0);
    for (int t = 0; t < T; t++)
        sampler.filterStep(t, sigma1, sigma2, n1_per_t[t]);
    sampler.getMflt(T, m_flt);
}
