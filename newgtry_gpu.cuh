#pragma once

// GPU-only G matrix computation — exact replica of util.R::G_ad()
//
// Computes the 400×400 transition matrix G using Phi integral approximations.
// This is a GPU-only port of newgtry.cu (CPU code removed).
//
// Algorithm mirrors G_ad(delta, v, K.ifm, Omega) in util.R:
//   G has Ngrid = k1 + 2*k2 rows and columns (= 4 + 2*198 = 400)
//   Phi integrals (f1-f8 kernels) are evaluated over Nr²=3600 spatial points.
//
// d_z1    : device, N × 4 row-major (x, y, vx, vy)      where N = Nr²
// d_z2    : device, N × 5 row-major (x, y, K, dKx, dKy)
// h_omega1: host,   k1 × 2  (first-set frequencies)
// h_omega2: host,   k2 × 2  (second-set frequencies)
// N       : Nr * Nr (= 3600)
// k1      : nrow(Omega1) = 4
// k2      : nrow(Omega2) = 198
// delta   : spatial step = 1/Nr (= 1/60)
// h_G     : host output, Ngrid × Ngrid doubles (= 400 × 400), row-major
void g_matrix_gpu(const double* d_z1,
                  const double* d_z2,
                  const double* h_omega1,
                  const double* h_omega2,
                  int N, int k1, int k2, double delta,
                  double* h_G);

// Fast version: replaces 160,000 serialised kernel+sync+reduce calls with
// 1 precompute kernel + 10 DGEMMs. Produces identical results.
void g_matrix_gpu_fast(const double* d_z1,
                       const double* d_z2,
                       const double* h_omega1,
                       const double* h_omega2,
                       int N, int k1, int k2, double delta,
                       double* h_G);
