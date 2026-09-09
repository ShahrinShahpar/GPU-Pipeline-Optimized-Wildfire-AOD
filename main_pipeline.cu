/*
 * main_pipeline.cu — GPU AOD pipeline: regrid → KNN → OF → smooth → K → G → Gibbs.
 *
 * Part of gpuimplementation/ — pure GPU pipeline, zero R.
 * Data stays on the GPU between regrid and KNN; no host round-trip
 * within the KNN pass.  Optical flow uses the host-wrapper interface
 * (optical_flow_gpu takes host pointers internally) so KNN outputs
 * are accumulated on the host before the OF pass.
 *
 * Pipeline (mirrors application.R exactly):
 *   1. Regrid (GPU)           — scatter → 60×60 grid
 *   2. KNN imputation (GPU)   — fill NaN cells, bnstruct::knn.impute replica
 *   3. Optical flow (GPU)     — SpatialVx::OF(W=15) for 20 G16 pairs
 *   4. Average + smooth       — velocity(dat.velocity,1,19) + image.smooth
 *   5. K diffusivity (GPU)    — Smagorinsky model: v_a → K → K.smooth → dK
 *   6. G matrix (GPU)         — Fourier basis Omega(N=20) → G_ad → expm(G)
 *   7. Gibbs FFBS2 (GPU)      — F matrix, UDS obs, obs assembly, Kalman filter,
 *                                backward sampling → m_flt filter means
 *
 * OF, smooth, K, G and Gibbs are only run for G16+G17 (matching application.R).
 *
 * Usage:
 *   ./aod_pipeline [manifest_G16] [manifest_G17] [output_dir] [k]
 *
 * Compiled with -DSAVE_INTERMEDIATES (default): writes per-step CSVs for
 *   KNN:    <output_dir>/G16_knn/img_NNN.csv
 *   OF:     <output_dir>/G16_of/speed_NNN.csv / angle_NNN.csv
 *   Smooth: <output_dir>/G16_smooth/speed_smooth.csv / angle_smooth.csv
 *   K:      <output_dir>/G16_k/v_a.csv   (long,lat,v_x,v_y)
 *           <output_dir>/G16_k/K_ifm.csv (long,lat,K,dK_dx,dK_dy)
 *   G:      <output_dir>/G16_g/G_gen.csv (400×400)
 *           <output_dir>/G16_g/G.csv     (400×400)
 *   Gibbs:  <output_dir>/G16_gibbs/m_flt.csv (time_step,coef_idx,value)
 *           <output_dir>/G16_gibbs/obs_summary.csv (t,n1,n2 per step)
 */

#include "regrid_gpu.cuh"
#include "knn_impute.cuh"
#include "optical_flow.cuh"
#include "image_smooth.cuh"
#include "fourier_basis.cuh"
#include "newgtry_gpu.cuh"
#include "matrix_expm.cuh"
#include "obs_assembly.cuh"
#include "gibbs_sampler.cuh"
#include "csv_io.cuh"

#include <Eigen/Dense>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <limits>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */
#define NR         60          /* grid side length — 60×60 = 3600 cells */
#define N_GRID     (NR * NR)   /* 3600 */
#define K_KNN      10          /* nearest neighbours, bnstruct::knn.impute */
#define N_OF_PAIRS 20          /* OF pairs: (1,2)..(20,21) */
#define N_OF_AVG   19          /* pairs to average: 1..19 (R: velocity(1,19)) */
#define W_OF       15          /* OF window size */
#define N_FOURIER  20          /* N for Fourier basis — Function_Omega(20) */
#define T_GIBBS    20          /* time steps used by Gibbs (R: 1..20) */
#define N_UDS      20          /* UDS downsampling resolution (20×20 = 400 selected pixels) */
#define N_GIBBS_SAMPLE 10      /* Gibbs iterations (10=R default, 100=convergence test) */
#define GIBBS_SEED     222     /* RNG seed matching R's set.seed(222) */

/* -------------------------------------------------------------------------
 * Error check macro
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/* -------------------------------------------------------------------------
 * Grid coordinate helpers (identical grid used by application.R).
 *   lon[i] = -123.98 + i * 0.04,  i = 0 .. NR-1
 *   lat[j] =  35.02  + j * 0.04,  j = 0 .. NR-1
 * ---------------------------------------------------------------------- */
static const double LON0  = -123.98;
static const double LAT0  =   35.02;
static const double GSTEP =    0.04;   /* degrees per cell */
/* DELTA = GSTEP * 111.0 km per cell — computed inline in device kernels */

/* -------------------------------------------------------------------------
 * Smagorinsky diffusivity kernel.
 *   Replicates application.R lines 111-119 exactly.
 *   Grid is normalised to [0,1]: delta = 1/Nr.
 *   Matrix layout: M[lon_idx * Nr + lat_idx]  (row = lon, col = lat).
 *
 *   Dividing by delta (=1/Nr) is equivalent to multiplying by Nr.
 *   col.dif.x = rowDiffs(vx)/(1/Nr) → p1[i,j] = (vx[i,j+1]-vx[i,j])*Nr   j<Nr-1 else NaN
 *   row.dif.y = colDiffs(vy)/(1/Nr) → p2[i,j] = (vy[i+1,j]-vy[i,j])*Nr   i<Nr-1 else NaN
 *   row.dif.x = colDiffs(vx)/(1/Nr) → p3[i,j] = (vx[i+1,j]-vx[i,j])*Nr   i<Nr-1 else NaN
 *   col.dif.y = rowDiffs(vy)/(1/Nr) → p4[i,j] = (vy[i,j+1]-vy[i,j])*Nr   j<Nr-1 else NaN
 *   K = 0.28 * (1/Nr)² * sqrt((p1-p2)² + (p3+p4)²)
 * ---------------------------------------------------------------------- */
__global__ void k_smagorinsky(const double* __restrict__ vx,
                               const double* __restrict__ vy,
                               double* __restrict__ K,
                               int Nr)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;   /* lon index */
    int j = blockIdx.x * blockDim.x + threadIdx.x;   /* lat index */
    if (i >= Nr || j >= Nr) return;

    const double GPU_NAN = __longlong_as_double(0x7FF8000000000000LL);
    const double Nr_d    = (double)Nr;
    const double delta   = 1.0 / Nr_d;   /* normalised grid: 1/60 */

    if (i == Nr-1 || j == Nr-1) { K[i*Nr + j] = GPU_NAN; return; }

    /* All four diffs divide by delta (= multiply by Nr) */
    double p1 = (vx[i*Nr + j+1]   - vx[i*Nr + j])   * Nr_d;  /* rowDiffs(vx)/delta */
    double p2 = (vy[(i+1)*Nr + j] - vy[i*Nr + j])   * Nr_d;  /* colDiffs(vy)/delta */
    double p3 = (vx[(i+1)*Nr + j] - vx[i*Nr + j])   * Nr_d;  /* colDiffs(vx)/delta */
    double p4 = (vy[i*Nr + j+1]   - vy[i*Nr + j])   * Nr_d;  /* rowDiffs(vy)/delta */

    double d12 = p1 - p2, d34 = p3 + p4;
    K[i*Nr + j] = 0.28 * delta * delta * sqrt(d12*d12 + d34*d34);
}

/* -------------------------------------------------------------------------
 * Gradient-of-K kernel.
 *   Replicates application.R lines 122-125 (normalised grid, delta = 1/Nr):
 *   col.dif = rowDiffs(K)/(1/Nr)  → D.K.x: finite diff in lat dir × Nr
 *   row.dif = colDiffs(K)/(1/Nr)  → D.K.y: finite diff in lon dir × Nr
 *   Boundary: repeat the last valid finite difference (not pad with zero).
 *     D.K.x[i,Nr-1] = D.K.x[i,Nr-2]   (cbind repeats col 59)
 *     D.K.y[Nr-1,j] = D.K.y[Nr-2,j]   (rbind repeats row 59)
 * ---------------------------------------------------------------------- */
__global__ void k_grad_K(const double* __restrict__ K_smooth,
                          double* __restrict__ dKdx,
                          double* __restrict__ dKdy,
                          int Nr)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nr || j >= Nr) return;

    const double Nr_d = (double)Nr;   /* dividing by 1/Nr = multiplying by Nr */

    /* dKdx (lat direction): use j or j-1 for the repeat at the last column */
    int jj = (j < Nr-1) ? j : Nr-2;
    dKdx[i*Nr + j] = (K_smooth[i*Nr + jj+1] - K_smooth[i*Nr + jj]) * Nr_d;

    /* dKdy (lon direction): use i or i-1 for the repeat at the last row */
    int ii = (i < Nr-1) ? i : Nr-2;
    dKdy[i*Nr + j] = (K_smooth[(ii+1)*Nr + j] - K_smooth[ii*Nr + j]) * Nr_d;
}

/* -------------------------------------------------------------------------
 * Transpose kernel: (lat × lon) → (lon × lat).
 *   src[r * N + c] → dst[c * N + r]
 * Converts regrid's lat-row-major output into knn_impute_gpu's expected
 * lon-row-major input (matches R's matrix(AOD_vector, Nr, Nr) layout).
 * ---------------------------------------------------------------------- */
__global__ void k_transpose(const double* __restrict__ src,
                                   double* __restrict__ dst,
                             int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N)
        dst[c * N + r] = src[r * N + c];
}

static void transpose_gpu(const double* d_src, double* d_dst, int N)
{
    dim3 block(8, 8);
    dim3 grid((N + 7) / 8, (N + 7) / 8);
    k_transpose<<<grid, block>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */
struct ImageEntry { int index; std::string path; };

static std::vector<ImageEntry> load_manifest(const char* path)
{
    std::vector<ImageEntry> v;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open manifest: %s\n", path);
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ImageEntry e;
        ss >> e.index >> e.path;
        v.push_back(e);
    }
    return v;
}

static bool read_bin(const std::string& path,
                     std::vector<double>& lats,
                     std::vector<double>& lons,
                     std::vector<double>& aods)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { fprintf(stderr, "  Cannot open %s\n", path.c_str()); return false; }
    int n = 0;
    if (fread(&n, sizeof(int), 1, fp) != 1 || n <= 0) { fclose(fp); return false; }
    lats.resize(n); lons.resize(n); aods.resize(n);
    bool ok = (fread(lats.data(), sizeof(double), n, fp) == (size_t)n) &&
              (fread(lons.data(), sizeof(double), n, fp) == (size_t)n) &&
              (fread(aods.data(), sizeof(double), n, fp) == (size_t)n);
    fclose(fp);
    return ok;
}

static int count_nan(const double* v, int n)
{
    int cnt = 0;
    for (int i = 0; i < n; i++) if (std::isnan(v[i])) ++cnt;
    return cnt;
}

/* Write one 60×60 matrix as a V1..V60 CSV at full float64 precision.
 * Used for KNN, OF (speed/angle), and smooth outputs. */
static void write_matrix_csv(const char* path, const double* data, int Nr)
{
    std::ofstream f(path);
    f << std::setprecision(17);
    for (int j = 0; j < Nr; j++) { if (j > 0) f << ','; f << 'V' << (j+1); }
    f << '\n';
    for (int i = 0; i < Nr; i++) {
        for (int j = 0; j < Nr; j++) {
            if (j > 0) f << ',';
            double v = data[i * Nr + j];
            if (std::isnan(v)) f << "NaN"; else f << v;
        }
        f << '\n';
    }
}

/* -------------------------------------------------------------------------
 * Pass 1: regrid → transpose → KNN for every image in the manifest.
 * Returns a map from time-step index (1-based) to 3600-element host vector.
 *
 * If raw_out != nullptr, also captures the pre-KNN, pre-transpose regrid
 * output (d_means layout: lat_i * NR + lon_i) for time steps 1..T_GIBBS.
 * These raw images are needed by Layer 7 (Gibbs UDS downsampling).
 * Layout: h_raw[lat_i * NR + lon_i] — NaN for cells with no obs.
 * ---------------------------------------------------------------------- */
static std::map<int, std::vector<double>>
process_satellite_knn(const char*  sat_name,
                      const char*  manifest_path,
                      const char*  knn_dir,
                      int          k,
                      double*& d_lats,
                      double*& d_lons,
                      double*& d_aods_buf,
                      int&     alloc_pts,
                      double*  d_means,
                      double*  d_knn,
                      std::map<int, std::vector<double>>* raw_out = nullptr)
{
    auto manifest = load_manifest(manifest_path);
    printf("\n[%s] KNN pass: %zu images...\n", sat_name, manifest.size());

#ifdef SAVE_INTERMEDIATES
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", knn_dir);
        (void)system(cmd);
    }
#endif

    using _Clock = std::chrono::steady_clock;
    using _Sec   = std::chrono::duration<double>;
    std::map<int, std::vector<double>> all_knn;
    int n_done = 0, n_skip = 0;
    double t_regrid_acc = 0.0, t_knn_acc = 0.0;   /* per-satellite accumulators */

    for (const auto& img : manifest) {
        printf("  [%02d] ", img.index);
        fflush(stdout);

        std::vector<double> h_lats, h_lons, h_aods;
        if (!read_bin(img.path, h_lats, h_lons, h_aods)) {
            printf("SKIP (read error)\n");
            ++n_skip;
            continue;
        }

        int n_pts = (int)h_lats.size();
        printf("%5d pts  ", n_pts);
        fflush(stdout);

        if (n_pts > alloc_pts) {
            if (d_lats) { cudaFree(d_lats); cudaFree(d_lons); cudaFree(d_aods_buf); }
            CUDA_CHECK(cudaMalloc(&d_lats,     n_pts * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_lons,     n_pts * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_aods_buf, n_pts * sizeof(double)));
            alloc_pts = n_pts;
        }

        CUDA_CHECK(cudaMemcpy(d_lats,     h_lats.data(), n_pts*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lons,     h_lons.data(), n_pts*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_aods_buf, h_aods.data(), n_pts*sizeof(double), cudaMemcpyHostToDevice));

        { auto _t = _Clock::now(); regrid_gpu(d_lats, d_lons, d_aods_buf, n_pts, d_means);
          cudaDeviceSynchronize();
          t_regrid_acc += _Sec(_Clock::now() - _t).count(); }

        /* Capture raw regrid image (pre-transpose, pre-KNN) for Gibbs UDS step.
         * d_means layout: h_raw[lat_i * NR + lon_i], NaN for empty cells. */
        if (raw_out && img.index >= 1 && img.index <= T_GIBBS) {
            std::vector<double> h_raw(N_GRID);
            CUDA_CHECK(cudaMemcpy(h_raw.data(), d_means,
                                  N_GRID * sizeof(double), cudaMemcpyDeviceToHost));
            (*raw_out)[img.index] = std::move(h_raw);
        }

        { auto _t = _Clock::now(); transpose_gpu(d_means, d_knn, NR);
          knn_impute_gpu(d_knn, NR, k);
          cudaDeviceSynchronize();
          t_knn_acc += _Sec(_Clock::now() - _t).count(); }

        /* Copy KNN result to host and store by index */
        std::vector<double> h_knn(N_GRID);
        CUDA_CHECK(cudaMemcpy(h_knn.data(), d_knn, N_GRID*sizeof(double), cudaMemcpyDeviceToHost));

        printf("knn OK  NaN_remain=%d", count_nan(h_knn.data(), N_GRID));

#ifdef SAVE_INTERMEDIATES
        char csv_path[512];
        snprintf(csv_path, sizeof(csv_path), "%s/img_%03d.csv", knn_dir, img.index);
        write_matrix_csv(csv_path, h_knn.data(), NR);
        printf("  -> %s\n", csv_path);
#else
        printf("\n");
#endif

        all_knn[img.index] = std::move(h_knn);
        ++n_done;
    }

    printf("[%s] KNN done: %d processed, %d skipped.\n", sat_name, n_done, n_skip);
    printf("[%s]   regrid only : %.3f s  (%.4f s/img)\n", sat_name,
           t_regrid_acc, n_done > 0 ? t_regrid_acc/n_done : 0.0);
    printf("[%s]   KNN only    : %.3f s  (%.4f s/img)\n", sat_name,
           t_knn_acc,    n_done > 0 ? t_knn_acc/n_done    : 0.0);
    return all_knn;
}

/* -------------------------------------------------------------------------
 * Pass 2 (G16 only): optical flow → average → image smooth.
 *
 * Mirrors application.R exactly:
 *   for i in 1..20: OF(final=knn[i+1], xhat=knn[i], W=15)
 *   velocity(dat.velocity, 1, 19):
 *     speed.avg = mean of pairs 1..19
 *     angle.avg = mean of pairs 1..19 (in radians)
 *     speed.avg.smooth = image.smooth(speed.avg)$z / Nr
 *     angle.avg.smooth = image.smooth(angle.avg)$z
 * ---------------------------------------------------------------------- */
static void run_of_smooth(const char* sat_name,
                           const std::map<int, std::vector<double>>& all_knn,
                           const char* of_dir,
                           const char* smooth_dir,
                           std::vector<double>& h_speed_smooth_out,
                           std::vector<double>& h_angle_smooth_out)
{
    /* Check that we have at least steps 1..21 */
    int n_avail = 0;
    for (int i = 1; i <= N_OF_PAIRS + 1; i++)
        if (all_knn.count(i)) ++n_avail;

    if (n_avail < N_OF_PAIRS + 1) {
        printf("[%s] Only %d/%d steps available for OF — skipping OF pass.\n",
               sat_name, n_avail, N_OF_PAIRS + 1);
        h_speed_smooth_out.clear();
        h_angle_smooth_out.clear();
        return;
    }

#ifdef SAVE_INTERMEDIATES
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", of_dir);
        (void)system(cmd);
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", smooth_dir);
        (void)system(cmd);
    }
#endif

    printf("\n[%s] OF pass: %d pairs (W=%d)...\n", sat_name, N_OF_PAIRS, W_OF);

    /* Run OF for all 20 pairs — store speed (pixels/step) and angle (degrees) */
    std::vector<std::vector<double>> sp(N_OF_PAIRS), ang(N_OF_PAIRS);
    for (int i = 0; i < N_OF_PAIRS; i++) {
        int idx1 = i + 1, idx2 = i + 2;   /* 1-based time-step indices */
        sp[i].resize(N_GRID); ang[i].resize(N_GRID);
        optical_flow_gpu(all_knn.at(idx1).data(),
                         all_knn.at(idx2).data(),
                         NR, W_OF,
                         sp[i].data(), ang[i].data());
        printf("  [%s] OF pair (%d,%d) done\n", sat_name, idx1, idx2);

#ifdef SAVE_INTERMEDIATES
        char path[512];
        snprintf(path, sizeof(path), "%s/speed_%03d.csv", of_dir, i + 1);
        write_matrix_csv(path, sp[i].data(), NR);
        snprintf(path, sizeof(path), "%s/angle_%03d.csv", of_dir, i + 1);
        write_matrix_csv(path, ang[i].data(), NR);
#endif
    }

    /* Average pairs 0..N_OF_AVG-1 (R uses velocity(dat.velocity, 1, 19)).
     * Angle is converted to radians before averaging (matches dat.velocity). */
    std::vector<double> speed_avg(N_GRID, 0.0), angle_avg_rad(N_GRID, 0.0);
    for (int cell = 0; cell < N_GRID; cell++) {
        double sum_s = 0.0, sum_a = 0.0;
        bool any_nan = false;
        for (int p = 0; p < N_OF_AVG; p++) {
            double s = sp[p][cell];
            double a = ang[p][cell];
            if (std::isnan(s) || std::isnan(a)) { any_nan = true; break; }
            sum_s += s;
            sum_a += a * M_PI / 180.0;   /* degrees → radians, matching dat.velocity */
        }
        speed_avg[cell]     = any_nan ? (double)NAN : sum_s / N_OF_AVG;
        angle_avg_rad[cell] = any_nan ? (double)NAN : sum_a / N_OF_AVG;
    }

    /* Upload averaged fields to device for cuFFT-based image smooth */
    double *d_speed_avg   = nullptr, *d_angle_avg   = nullptr;
    double *d_speed_smooth = nullptr, *d_angle_smooth = nullptr;
    CUDA_CHECK(cudaMalloc(&d_speed_avg,    N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_angle_avg,    N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_speed_smooth, N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_angle_smooth, N_GRID * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_speed_avg, speed_avg.data(),
                          N_GRID*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_angle_avg, angle_avg_rad.data(),
                          N_GRID*sizeof(double), cudaMemcpyHostToDevice));

    /* Smooth both fields (default aRange=1, tol=1e-8, matching R defaults) */
    image_smooth_gpu(d_speed_avg,   d_speed_smooth, NR);
    image_smooth_gpu(d_angle_avg,   d_angle_smooth, NR);
    printf("[%s] image_smooth done.\n", sat_name);

    std::vector<double> h_speed_smooth(N_GRID), h_angle_smooth(N_GRID);
    CUDA_CHECK(cudaMemcpy(h_speed_smooth.data(), d_speed_smooth,
                          N_GRID*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_angle_smooth.data(), d_angle_smooth,
                          N_GRID*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_speed_avg); cudaFree(d_angle_avg);
    cudaFree(d_speed_smooth); cudaFree(d_angle_smooth);

    /* Divide speed by Nr (matches R: speed.avg.smooth <- image.smooth(...)$z / N_r) */
    for (int i = 0; i < N_GRID; i++)
        if (!std::isnan(h_speed_smooth[i]))
            h_speed_smooth[i] /= NR;

#ifdef SAVE_INTERMEDIATES
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/speed_smooth.csv", smooth_dir);
        write_matrix_csv(path, h_speed_smooth.data(), NR);
        snprintf(path, sizeof(path), "%s/angle_smooth.csv", smooth_dir);
        write_matrix_csv(path, h_angle_smooth.data(), NR);
        printf("[%s] Smooth outputs -> %s/\n", sat_name, smooth_dir);
    }
#endif

    h_speed_smooth_out = std::move(h_speed_smooth);
    h_angle_smooth_out = std::move(h_angle_smooth);
}

/* -------------------------------------------------------------------------
 * Pass 3 (G16 only): K diffusivity — Smagorinsky model.
 *
 * Mirrors application.R lines 78-80 and 111-126 exactly.
 *   Grid normalised to [0,1]: delta = 1/Nr = 1/60.
 *   v_x = speed_smooth * cos(angle_smooth)   (no unit conversion)
 *   v_y = speed_smooth * sin(angle_smooth)
 *   K   = 0.28 * (1/Nr)² * sqrt((p1-p2)² + (p3+p4)²)
 *         where all four differences divide by 1/Nr (= multiply by Nr)
 *   K.smooth via image.smooth (cuFFT)
 *   D.K.x: diff in lat dir ×Nr, last col repeats col Nr-2
 *   D.K.y: diff in lon dir ×Nr, last row repeats row Nr-2
 *
 * Saves (column-major, lon varies fastest — matches R's c(matrix) order):
 *   v_a.csv   : long,lat,v_x,v_y
 *   K_ifm.csv : long,lat,K,dK_dx,dK_dy
 * ---------------------------------------------------------------------- */
static void run_k_diffusivity(const char* sat_name,
                               const std::vector<double>& h_speed_smooth,
                               const std::vector<double>& h_angle_smooth,
                               const char* k_dir,
                               std::vector<double>& h_vx_out,
                               std::vector<double>& h_vy_out,
                               std::vector<double>& h_Ks_out,
                               std::vector<double>& h_dKdx_out,
                               std::vector<double>& h_dKdy_out)
{
    if (h_speed_smooth.empty()) {
        printf("[%s] Smooth outputs unavailable — skipping K pass.\n", sat_name);
        return;
    }

#ifdef SAVE_INTERMEDIATES
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", k_dir);
        (void)system(cmd);
    }
#endif

    printf("\n[%s] K diffusivity pass...\n", sat_name);

    /* 1. Compute v_x, v_y on host.
     *    Matches application.R lines 78-80: NO unit conversion.
     *    v_x = speed_smooth * cos(angle_smooth)
     *    v_y = speed_smooth * sin(angle_smooth)
     *    Speed stays in normalised units; the grid is [0,1] so delta=1/Nr.   */
    std::vector<double> h_vx(N_GRID), h_vy(N_GRID);
    for (int k = 0; k < N_GRID; k++) {
        double sp  = h_speed_smooth[k];
        double ang = h_angle_smooth[k];
        if (std::isnan(sp) || std::isnan(ang)) {
            h_vx[k] = h_vy[k] = std::numeric_limits<double>::quiet_NaN();
        } else {
            h_vx[k] = sp * std::cos(ang);
            h_vy[k] = sp * std::sin(ang);
        }
    }

    /* 2. Upload v_x, v_y to device; compute K; smooth K; compute dK */
    double *d_vx = nullptr, *d_vy = nullptr;
    double *d_K  = nullptr, *d_Ks = nullptr;
    double *d_dKdx = nullptr, *d_dKdy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vx,   N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vy,   N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K,    N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ks,   N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dKdx, N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dKdy, N_GRID * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), N_GRID*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), N_GRID*sizeof(double), cudaMemcpyHostToDevice));

    dim3 blk(8, 8), grd((NR+7)/8, (NR+7)/8);
    k_smagorinsky<<<grd, blk>>>(d_vx, d_vy, d_K, NR);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    image_smooth_gpu(d_K, d_Ks, NR);   /* cuFFT-based, matches fields::image.smooth */

    k_grad_K<<<grd, blk>>>(d_Ks, d_dKdx, d_dKdy, NR);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    /* 3. Download results */
    std::vector<double> h_Ks(N_GRID), h_dKdx(N_GRID), h_dKdy(N_GRID);
    CUDA_CHECK(cudaMemcpy(h_Ks.data(),   d_Ks,   N_GRID*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dKdx.data(), d_dKdx, N_GRID*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dKdy.data(), d_dKdy, N_GRID*sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_K);
    cudaFree(d_Ks); cudaFree(d_dKdx); cudaFree(d_dKdy);

    printf("[%s] K diffusivity done.\n", sat_name);

#ifdef SAVE_INTERMEDIATES
    /* Write in column-major order (lon varies fastest) — matches R's c(matrix). */
    auto write_val = [](std::ofstream& f, double v) {
        if (std::isnan(v)) f << "NA"; else f << v;
    };

    /* v_a.csv: long,lat,v_x,v_y */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/v_a.csv", k_dir);
        std::ofstream f(path);
        f << std::setprecision(17);
        f << "long,lat,v_x,v_y\n";
        for (int j = 0; j < NR; j++) {            /* lat (col) — outer */
            double lat_v = LAT0 + j * GSTEP;
            for (int i = 0; i < NR; i++) {         /* lon (row) — inner */
                double lon_v = LON0 + i * GSTEP;
                f << lon_v << ',' << lat_v << ',';
                write_val(f, h_vx[i*NR + j]); f << ',';
                write_val(f, h_vy[i*NR + j]); f << '\n';
            }
        }
        printf("[%s] -> %s\n", sat_name, path);
    }

    /* K_ifm.csv: long,lat,K,dK_dx,dK_dy */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/K_ifm.csv", k_dir);
        std::ofstream f(path);
        f << std::setprecision(17);
        f << "long,lat,K,dK_dx,dK_dy\n";
        for (int j = 0; j < NR; j++) {
            double lat_v = LAT0 + j * GSTEP;
            for (int i = 0; i < NR; i++) {
                double lon_v = LON0 + i * GSTEP;
                f << lon_v << ',' << lat_v << ',';
                write_val(f, h_Ks[i*NR + j]);   f << ',';
                write_val(f, h_dKdx[i*NR + j]); f << ',';
                write_val(f, h_dKdy[i*NR + j]); f << '\n';
            }
        }
        printf("[%s] -> %s\n", sat_name, path);
    }
#endif

    /* Export to caller for downstream G matrix pass.
     * Done after SAVE_INTERMEDIATES so the vectors are still valid for CSV writes. */
    h_vx_out   = std::move(h_vx);
    h_vy_out   = std::move(h_vy);
    h_Ks_out   = std::move(h_Ks);
    h_dKdx_out = std::move(h_dKdx);
    h_dKdy_out = std::move(h_dKdy);
}

/* -------------------------------------------------------------------------
 * Write an N×N matrix as a CSV with header V1..VN, full float64 precision.
 * Row-major storage: data[i*N + j] is row i, col j.
 * ---------------------------------------------------------------------- */
static void write_square_csv(const char* path, const double* data, int N)
{
    std::ofstream f(path);
    f << std::setprecision(17);
    for (int j = 0; j < N; j++) { if (j > 0) f << ','; f << 'V' << (j+1); }
    f << '\n';
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j > 0) f << ',';
            double v = data[i * N + j];
            if (std::isnan(v)) f << "NA"; else f << v;
        }
        f << '\n';
    }
}

/* -------------------------------------------------------------------------
 * Pass 4 (G16 only): Fourier basis Omega → G_ad generator → expm(G).
 *
 * Mirrors application.R lines 156-180:
 *   N     <- 20
 *   Omega <- Function_Omega(N)
 *   G_gen <- G_ad(1/Nr, v.a, K.ifm, Omega)
 *   G     <- expm(G_gen)
 *
 * z1 and z2 are built in column-major order (lon/x varies fastest),
 * matching R's expand.grid(x, y) which is what G_ad feeds to cbind().
 * Index mapping for flat index k = 0..N_GRID-1:
 *   lon_idx = k % NR,  lat_idx = k / NR
 *   x = lon_idx / NR,  y = lat_idx / NR
 *   h_vx[lon_idx*NR + lat_idx]  — column-major access to row-major array
 *
 * Saves (row-major, each row of G_gen/G written as a CSV row):
 *   G_gen.csv : 400×400 generator matrix
 *   G.csv     : 400×400 transition matrix = expm(G_gen)
 * ---------------------------------------------------------------------- */
static void run_g_matrix(const char* sat_name,
                          const std::vector<double>& h_vx,
                          const std::vector<double>& h_vy,
                          const std::vector<double>& h_Ks,
                          const std::vector<double>& h_dKdx,
                          const std::vector<double>& h_dKdy,
                          const char* g_dir,
                          std::vector<double>* h_G_out = nullptr)
{
    if (h_vx.empty()) {
        printf("[%s] K outputs unavailable — skipping G pass.\n", sat_name);
        return;
    }

    printf("\n[%s] G matrix pass (N_FOURIER=%d)...\n", sat_name, N_FOURIER);

    /* 1. Generate Omega (exact replica of R's Function_Omega(N)) */
    std::vector<double> o1_k1, o1_k2, o2_k1, o2_k2;
    generate_omega(N_FOURIER, o1_k1, o1_k2, o2_k1, o2_k2);
    int n_om1 = (int)o1_k1.size();   /* k1 = 4  */
    int n_om2 = (int)o2_k1.size();   /* k2 = 198 */
    int Ngrid = n_om1 + 2 * n_om2;   /* 400 */
    printf("[%s] Ngrid=%d  k1=%d  k2=%d\n", sat_name, Ngrid, n_om1, n_om2);

    /* Pack omega as interleaved [k1,k2] pairs expected by g_matrix_gpu */
    std::vector<double> h_omega1(n_om1 * 2), h_omega2(n_om2 * 2);
    for (int i = 0; i < n_om1; i++) {
        h_omega1[i*2]   = o1_k1[i];
        h_omega1[i*2+1] = o1_k2[i];
    }
    for (int i = 0; i < n_om2; i++) {
        h_omega2[i*2]   = o2_k1[i];
        h_omega2[i*2+1] = o2_k2[i];
    }

    /* 2. Build z1 (N_GRID×4) and z2 (N_GRID×5) in column-major order.
     *    Row k: lon_idx=k%NR, lat_idx=k/NR → x=lon/NR, y=lat/NR.
     *    Access h_vx[lon_idx*NR + lat_idx] (row-major → column-major read).
     *    This matches R's expand.grid(x,y) where x=lon varies fastest. */
    const double Nr_d = (double)NR;
    std::vector<double> h_z1(N_GRID * 4), h_z2(N_GRID * 5);
    for (int k = 0; k < N_GRID; k++) {
        int lon_idx = k % NR;
        int lat_idx = k / NR;
        int idx     = lon_idx * NR + lat_idx;   /* row-major index */
        double x    = lon_idx / Nr_d;
        double y    = lat_idx / Nr_d;
        h_z1[k*4+0] = x;
        h_z1[k*4+1] = y;
        h_z1[k*4+2] = h_vx[idx];
        h_z1[k*4+3] = h_vy[idx];
        h_z2[k*5+0] = x;
        h_z2[k*5+1] = y;
        h_z2[k*5+2] = h_Ks[idx];
        h_z2[k*5+3] = h_dKdx[idx];
        h_z2[k*5+4] = h_dKdy[idx];
    }

    /* 3. Upload z1, z2 to device */
    double *d_z1 = nullptr, *d_z2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_z1, N_GRID * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z2, N_GRID * 5 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_z1, h_z1.data(), N_GRID*4*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z2, h_z2.data(), N_GRID*5*sizeof(double), cudaMemcpyHostToDevice));

    /* 4. Compute G_gen on GPU (Phi integrals — fast batched DGEMM version) */
    std::vector<double> h_G_gen((size_t)Ngrid * Ngrid, 0.0);
    printf("[%s] Computing G_gen (%dx%d) — fast DGEMM path...\n", sat_name, Ngrid, Ngrid);
    g_matrix_gpu_fast(d_z1, d_z2, h_omega1.data(), h_omega2.data(),
                      N_GRID, n_om1, n_om2, 1.0 / NR, h_G_gen.data());
    cudaFree(d_z1); cudaFree(d_z2);
    printf("[%s] G_gen done.\n", sat_name);

    /* 5. Compute G = expm(G_gen) on GPU (Padé-13 + scaling-and-squaring) */
    double *d_G_gen = nullptr, *d_G = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G_gen, (size_t)Ngrid * Ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G,     (size_t)Ngrid * Ngrid * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_G_gen, h_G_gen.data(),
                          (size_t)Ngrid * Ngrid * sizeof(double), cudaMemcpyHostToDevice));
    printf("[%s] Computing expm(G_gen)...\n", sat_name);
    matrix_expm_gpu(d_G_gen, d_G, Ngrid);

    std::vector<double> h_G((size_t)Ngrid * Ngrid);
    CUDA_CHECK(cudaMemcpy(h_G.data(), d_G,
                          (size_t)Ngrid * Ngrid * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_G_gen); cudaFree(d_G);
    printf("[%s] expm done.\n", sat_name);

    /* Return h_G to caller for downstream Gibbs pass */
    if (h_G_out) *h_G_out = h_G;

#ifdef SAVE_INTERMEDIATES
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", g_dir);
        (void)system(cmd);

        char path[512];
        snprintf(path, sizeof(path), "%s/G_gen.csv", g_dir);
        write_square_csv(path, h_G_gen.data(), Ngrid);
        printf("[%s] -> %s\n", sat_name, path);

        snprintf(path, sizeof(path), "%s/G.csv", g_dir);
        write_square_csv(path, h_G.data(), Ngrid);
        printf("[%s] -> %s\n", sat_name, path);
    }
#endif
}

/* -------------------------------------------------------------------------
 * UDS pixel selection — replicates R's UDS(datset, n.slt=20).
 *
 * R: lat.slt = seq(lat.range[1], lat.range[2], 0.04)[60/n.slt*(1:n.slt)]
 *    long.slt = same formula for lon
 *
 * seq(..., 0.04) produces NR=60 values (0-based indices 0..59).
 * 60/20*(1:20) = 3.0, 6.0, ..., 60.0  (exact integers — step=3).
 * R truncates non-integer indices: v[3.0] → v[3] (1-based) → C index 2.
 *
 * Returns 20 selected 0-based grid indices for lat (and identically for lon):
 * {2, 5, 8, ..., 59}.
 * ---------------------------------------------------------------------- */
static std::vector<int> compute_uds_sel_indices(int Nr, int n_slt)
{
    std::vector<int> sel;
    double step = (double)Nr / n_slt;
    for (int k = 1; k <= n_slt; k++) {
        int r_idx_1based = (int)(step * k);   /* R truncation of step*k */
        sel.push_back(r_idx_1based - 1);      /* convert R 1-based → C 0-based */
    }
    return sel;
}

/* -------------------------------------------------------------------------
 * Pass 7: Observation assembly + Gibbs FFBS2.
 *
 * Mirrors application.R lines 183-229:
 *   F       <- Function_F(Nr, N, Omega)            # 3600×400 design matrix
 *   G17_ds  <- UDS(G17.aod.raw, 25)               # 25×25 downsampled obs
 *   G16_ds  <- UDS(G16.aod.raw, 25)
 *   y1[[t]] <- G17_ds[[t]]$AOD  (t=1..20)
 *   y2[[t]] <- G16_ds[[t]]$AOD  (t=1..20)
 *   obs.ccl <- OBS_ccl2(y1, y2, F)               # non-NaN extraction
 *   G_aug   <- rbind(cbind(G,I), cbind(0,I))     # 800×800 augmented G
 *   fit     <- Gibbs_FFBS2(obs.ccl, G, m0, C0, N.sample=10)
 *
 * Inputs:
 *   raw_g17 : raw regrid images for G17 (time 1..T_GIBBS), lat_i*NR+lon_i layout
 *   raw_g16 : same for G16
 *   h_G     : 400×400 GPU transition matrix G = expm(G_gen), row-major
 *   gibbs_dir: output directory for m_flt.csv / obs_summary.csv
 * ---------------------------------------------------------------------- */
static void run_gibbs_layer(const char* sat_name,
                             const std::map<int,std::vector<double>>& raw_g17,
                             const std::map<int,std::vector<double>>& raw_g16,
                             const std::vector<double>& h_G,
                             const char* gibbs_dir)
{
    if (h_G.empty()) {
        printf("[%s] G matrix unavailable — skipping Gibbs pass.\n", sat_name);
        return;
    }
    if ((int)raw_g17.size() < T_GIBBS || (int)raw_g16.size() < T_GIBBS) {
        printf("[%s] Raw images unavailable — skipping Gibbs pass.\n", sat_name);
        return;
    }

    printf("\n[%s] Gibbs FFBS2 pass (T=%d, N_sample=%d)...\n",
           sat_name, T_GIBBS, N_GIBBS_SAMPLE);

    /* 1. UDS selection indices */
    std::vector<int> sel = compute_uds_sel_indices(NR, N_UDS);  /* 25 lat/lon indices */
    /* 625 selected flat indices: h_raw[lat_i*NR + lon_i] */
    std::vector<int> sel_flat;
    sel_flat.reserve(N_UDS * N_UDS);
    for (int j = 0; j < N_UDS; j++)          /* lat index */
        for (int i = 0; i < N_UDS; i++)      /* lon index */
            sel_flat.push_back(sel[j] * NR + sel[i]);

    /* 2. Build F matrix on GPU */
    FourierBasisGPU fbasis;
    fbasis.build(NR, N_FOURIER);
    int Ngrid = fbasis.n_cols;               /* 400 = N^2 */
    int n_state = 2 * Ngrid;                 /* 800 */
    printf("[%s] F matrix: %d×%d\n", sat_name, N_GRID, Ngrid);

    /* 3. Obs assembly device buffers */
    double *d_y = nullptr, *d_Ft = nullptr, *d_yc = nullptr;
    int    *d_id = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y,  N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ft, (size_t)N_GRID * Ngrid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_yc, N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_id, N_GRID * sizeof(int)));

    /* 4. For each time step: apply UDS → obs assembly → build Eigen matrices */
    std::vector<MatrixXd> Ftt_all(T_GIBBS);
    std::vector<VectorXd> yt_all(T_GIBBS);
    std::vector<int> n1_per_t(T_GIBBS);
    int max_obs = 0;

    /* Using row-major Eigen alias for direct download from GPU row-major buffers */
    using RowMajorMatXd = Eigen::Matrix<double,
                                        Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>;

    for (int t = 1; t <= T_GIBBS; t++) {
        /* Apply UDS to G17 (y1): keep selected pixels, rest = NaN */
        std::vector<double> h_y1(N_GRID, std::numeric_limits<double>::quiet_NaN());
        if (raw_g17.count(t)) {
            const auto& raw = raw_g17.at(t);
            for (int idx : sel_flat)
                h_y1[idx] = raw[idx];
        }
        /* Apply UDS to G16 (y2): keep selected pixels, rest = NaN */
        std::vector<double> h_y2(N_GRID, std::numeric_limits<double>::quiet_NaN());
        if (raw_g16.count(t)) {
            const auto& raw = raw_g16.at(t);
            for (int idx : sel_flat)
                h_y2[idx] = raw[idx];
        }

        /* Obs assembly for G17 → F1t (n1×Ngrid), y1c (n1) */
        CUDA_CHECK(cudaMemcpy(d_y, h_y1.data(), N_GRID*sizeof(double),
                              cudaMemcpyHostToDevice));
        int n1 = 0;
        obs_assembly_gpu(fbasis.d_F, d_y, N_GRID, Ngrid, d_Ft, d_yc, d_id, n1);

        RowMajorMatXd F1t_rm(n1, Ngrid);
        VectorXd y1c(n1);
        CUDA_CHECK(cudaMemcpy(F1t_rm.data(), d_Ft,
                              (size_t)n1*Ngrid*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y1c.data(), d_yc,
                              n1*sizeof(double), cudaMemcpyDeviceToHost));
        MatrixXd F1t = F1t_rm;   /* row-major → column-major conversion */

        /* Obs assembly for G16 → F2t (n2×Ngrid), y2c (n2) */
        CUDA_CHECK(cudaMemcpy(d_y, h_y2.data(), N_GRID*sizeof(double),
                              cudaMemcpyHostToDevice));
        int n2 = 0;
        obs_assembly_gpu(fbasis.d_F, d_y, N_GRID, Ngrid, d_Ft, d_yc, d_id, n2);

        RowMajorMatXd F2t_rm(n2, Ngrid);
        VectorXd y2c(n2);
        CUDA_CHECK(cudaMemcpy(F2t_rm.data(), d_Ft,
                              (size_t)n2*Ngrid*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y2c.data(), d_yc,
                              n2*sizeof(double), cudaMemcpyDeviceToHost));
        MatrixXd F2t = F2t_rm;

        /* Build combined Ftt = [F1t|0 ; F2t|0]  — n_obs × 2*Ngrid, col-major */
        int n_obs = n1 + n2;
        MatrixXd Ftt = MatrixXd::Zero(n_obs, n_state);
        Ftt.topRows(n1).leftCols(Ngrid)    = F1t;
        Ftt.bottomRows(n2).leftCols(Ngrid) = F2t;

        VectorXd yt(n_obs);
        yt.head(n1) = y1c;
        yt.tail(n2) = y2c;

        Ftt_all[t-1]  = std::move(Ftt);
        yt_all[t-1]   = std::move(yt);
        n1_per_t[t-1] = n1;
        max_obs = std::max(max_obs, n_obs);

        printf("[%s] t=%02d  n1(G17)=%d  n2(G16)=%d  total=%d\n",
               sat_name, t, n1, n2, n_obs);
    }

    cudaFree(d_y); cudaFree(d_Ft); cudaFree(d_yc); cudaFree(d_id);

    /* 5. Build augmented G_aug (800×800, column-major Eigen)
     *    G_aug = [[G, I], [0, I]]  matching R's Gibbs_FFBS2 G augmentation */
    MatrixXd expG = MatrixXd::Zero(n_state, n_state);
    for (int i = 0; i < Ngrid; i++)
        for (int j = 0; j < Ngrid; j++)
            expG(i, j) = h_G[i * Ngrid + j];          /* row-major h_G */
    expG.topRightCorner(Ngrid, Ngrid)    = MatrixXd::Identity(Ngrid, Ngrid);
    expG.bottomRightCorner(Ngrid, Ngrid) = MatrixXd::Identity(Ngrid, Ngrid);

    /* 6. Initial state: m0 = rep(0.1, 800), C0 = 0.01 * I_800 */
    VectorXd m0 = VectorXd::Constant(n_state, 0.1);
    MatrixXd C0 = 0.01 * MatrixXd::Identity(n_state, n_state);

    /* 7. Run Gibbs FFBS2 */
    printf("[%s] Starting Gibbs (n_state=%d, max_obs=%d)...\n",
           sat_name, n_state, max_obs);
    std::vector<VectorXd> m_flt;
    std::vector<VectorXd> m_flt_gibbs; /* last Gibbs iteration (structured alpha2) */
    std::vector<MatrixXd> Vt_dummy;   /* not used by run_gibbs_pipeline internals */
    run_gibbs_pipeline(T_GIBBS, N_GIBBS_SAMPLE, GIBBS_SEED,
                       expG, m0, C0,
                       Ftt_all, yt_all, Vt_dummy,
                       n1_per_t, max_obs, m_flt, m_flt_gibbs);

#ifdef SAVE_INTERMEDIATES
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", gibbs_dir);
        (void)system(cmd);

        /* Save m_flt: time_step (1-based, matching R's exports/m_flt.csv),
         * coef_idx (1-based), value.
         * getMflt returns m_flt[t] = filtered mean after time step t+1,
         * so we write time_step = t+1 to match R's indexing convention. */
        char path[512];
        snprintf(path, sizeof(path), "%s/m_flt.csv", gibbs_dir);
        {
            std::ofstream f(path);
            f << std::setprecision(17);
            f << "time_step,coef_idx,value\n";
            for (int t = 0; t < (int)m_flt.size(); t++) {
                for (int i = 0; i < (int)m_flt[t].size(); i++) {
                    double v = m_flt[t](i);
                    f << (t+1) << ',' << (i+1) << ',';
                    if (std::isnan(v)) f << "NA"; else f << v;
                    f << '\n';
                }
            }
        }
        printf("[%s] -> %s  (%zu time steps × %d coefs)\n",
               sat_name, path, m_flt.size(),
               m_flt.empty() ? 0 : (int)m_flt[0].size());

        /* Save m_flt_gibbs: last Gibbs iteration's forward-filtered means.
         * alpha2 (bias) block retains spatial structure here because W used
         * has off-diagonal coupling built through backward-sample residuals. */
        snprintf(path, sizeof(path), "%s/m_flt_gibbs.csv", gibbs_dir);
        {
            std::ofstream f(path);
            f << std::setprecision(17);
            f << "time_step,coef_idx,value\n";
            for (int t = 0; t < (int)m_flt_gibbs.size(); t++) {
                for (int i = 0; i < (int)m_flt_gibbs[t].size(); i++) {
                    double v = m_flt_gibbs[t](i);
                    f << (t+1) << ',' << (i+1) << ',';
                    if (std::isnan(v)) f << "NA"; else f << v;
                    f << '\n';
                }
            }
        }
        printf("[%s] -> %s  (last Gibbs iter, structured alpha2)\n", sat_name, path);

        /* Save obs summary: t,n1(G17 obs count),n2(G16 obs count) */
        snprintf(path, sizeof(path), "%s/obs_summary.csv", gibbs_dir);
        {
            std::ofstream f(path);
            f << "t,n1,n2\n";
            for (int t = 0; t < T_GIBBS; t++)
                f << (t+1) << ',' << n1_per_t[t] << ','
                  << ((int)yt_all[t].size() - n1_per_t[t]) << '\n';
        }
        printf("[%s] -> %s\n", sat_name, path);
    }
#endif
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    const char* manifest_g16 = (argc > 1) ? argv[1] : "manifest_G16.txt";
    const char* manifest_g17 = (argc > 2) ? argv[2] : "manifest_G17.txt";
    const char* output_dir   = (argc > 3) ? argv[3] : "output";
    int         k            = (argc > 4) ? atoi(argv[4]) : K_KNN;

    printf("=== GPU AOD Pipeline: regrid -> KNN -> OF -> smooth -> K -> G -> Gibbs ===\n");
    printf("manifest_G16 : %s\n", manifest_g16);
    printf("manifest_G17 : %s\n", manifest_g17);
    printf("output_dir   : %s\n", output_dir);
    printf("k (KNN)      : %d\n", k);
#ifdef SAVE_INTERMEDIATES
    printf("SAVE_INTERMEDIATES : ON\n");
#else
    printf("SAVE_INTERMEDIATES : OFF\n");
#endif

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    using Clock = std::chrono::steady_clock;
    using Sec   = std::chrono::duration<double>;
    auto t_total_start = Clock::now();
    double t_knn_s = 0, t_vkifm_s = 0, t_g_s = 0, t_gibbs_s = 0;

    double *d_means = nullptr, *d_knn = nullptr;
    CUDA_CHECK(cudaMalloc(&d_means, N_GRID * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_knn,   N_GRID * sizeof(double)));

    double *d_lats = nullptr, *d_lons = nullptr, *d_aods_buf = nullptr;
    int     alloc_pts = 0;

    char g16_knn_dir[512], g17_knn_dir[512];
    char g16_of_dir[512],  g16_smooth_dir[512], g16_k_dir[512];
    char g16_g_dir[512],   g16_gibbs_dir[512];
    snprintf(g16_knn_dir,    sizeof(g16_knn_dir),    "%s/G16_knn",    output_dir);
    snprintf(g17_knn_dir,    sizeof(g17_knn_dir),    "%s/G17_knn",    output_dir);
    snprintf(g16_of_dir,     sizeof(g16_of_dir),     "%s/G16_of",     output_dir);
    snprintf(g16_smooth_dir, sizeof(g16_smooth_dir), "%s/G16_smooth", output_dir);
    snprintf(g16_k_dir,      sizeof(g16_k_dir),      "%s/G16_k",      output_dir);
    snprintf(g16_g_dir,      sizeof(g16_g_dir),      "%s/G16_g",      output_dir);
    snprintf(g16_gibbs_dir,  sizeof(g16_gibbs_dir),  "%s/G16_gibbs",  output_dir);

    /* Pass 1: KNN for both satellites; also capture raw regrid images for Gibbs */
    std::map<int, std::vector<double>> raw_g16, raw_g17;
    {
        auto t0 = Clock::now();
        auto knn16_tmp = process_satellite_knn("G16", manifest_g16, g16_knn_dir, k,
                                               d_lats, d_lons, d_aods_buf, alloc_pts,
                                               d_means, d_knn, &raw_g16);
        auto knn17_tmp = process_satellite_knn("G17", manifest_g17, g17_knn_dir, k,
                                               d_lats, d_lons, d_aods_buf, alloc_pts,
                                               d_means, d_knn, &raw_g17);
        t_knn_s = Sec(Clock::now() - t0).count();
        printf("[TIMING] KNN (G16+G17): %.3f s\n", t_knn_s);

        cudaFree(d_means); cudaFree(d_knn);
        if (d_lats) { cudaFree(d_lats); cudaFree(d_lons); cudaFree(d_aods_buf); }

        /* Pass 2: OF + smooth for G16 only (matches application.R) */
        std::vector<double> g16_speed_smooth, g16_angle_smooth;
        {
            auto t1 = Clock::now();
            run_of_smooth("G16", knn16_tmp, g16_of_dir, g16_smooth_dir,
                          g16_speed_smooth, g16_angle_smooth);

            /* Pass 3: K diffusivity for G16 */
            std::vector<double> g16_vx, g16_vy, g16_Ks, g16_dKdx, g16_dKdy;
            run_k_diffusivity("G16", g16_speed_smooth, g16_angle_smooth, g16_k_dir,
                              g16_vx, g16_vy, g16_Ks, g16_dKdx, g16_dKdy);
            t_vkifm_s = Sec(Clock::now() - t1).count();
            printf("[TIMING] v + K.ifm (OF+smooth+K): %.3f s\n", t_vkifm_s);

            /* Pass 4: G matrix for G16 */
            std::vector<double> h_G_for_gibbs;
            {
                auto t2 = Clock::now();
                run_g_matrix("G16", g16_vx, g16_vy, g16_Ks, g16_dKdx, g16_dKdy,
                             g16_g_dir, &h_G_for_gibbs);
                t_g_s = Sec(Clock::now() - t2).count();
                printf("[TIMING] G matrix (Fourier+G_ad+expm): %.3f s\n", t_g_s);
            }

            /* Pass 5: Gibbs FFBS2 */
            {
                auto t3 = Clock::now();
                run_gibbs_layer("G16", raw_g17, raw_g16, h_G_for_gibbs, g16_gibbs_dir);
                t_gibbs_s = Sec(Clock::now() - t3).count();
                printf("[TIMING] Gibbs FFBS2 (T=%d, N=%d): %.3f s\n",
                       T_GIBBS, N_GIBBS_SAMPLE, t_gibbs_s);
            }
        }
    }

    double t_total_s = Sec(Clock::now() - t_total_start).count();
    printf("\n========== GPU TIMING SUMMARY ==========\n");
    printf("  %-30s : %.3f s\n",  "KNN (G16+G17)",           t_knn_s);
    printf("  %-30s : %.3f s\n",  "v + K.ifm (OF+smooth+K)", t_vkifm_s);
    printf("  %-30s : %.3f s\n",  "G matrix (G_ad + expm)",  t_g_s);
    printf("  %-30s : %.3f s\n",  "Gibbs FFBS2",             t_gibbs_s);
    printf("  %-30s : %.3f s\n",  "Total pipeline",          t_total_s);
    printf("=========================================\n");

    /* Save timing CSV — same structure as R's timing_summary.csv */
    {
        char tpath[512];
        snprintf(tpath, sizeof(tpath), "%s/timing_gpu.csv", output_dir);
        std::ofstream tf(tpath);
        tf << std::fixed << std::setprecision(6);
        tf << "section,unit,value\n";
        tf << "KNN_G16_G17,secs,"    << t_knn_s    << "\n";
        tf << "v_and_Kifm,secs,"     << t_vkifm_s  << "\n";
        tf << "G_computation,secs,"  << t_g_s      << "\n";
        tf << "Gibbs,secs,"          << t_gibbs_s  << "\n";
        tf << "total_run,secs,"      << t_total_s  << "\n";
        printf("[TIMING] Saved: %s\n", tpath);
    }

    printf("\n=== Pipeline complete. ===\n");
    return 0;
}
