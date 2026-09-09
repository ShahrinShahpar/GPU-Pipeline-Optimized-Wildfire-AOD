# Makefile for gpuimplementation/ GPU AOD pipeline
#
# Targets:
#   make regrid_standalone   — standalone regrid verifier (reads .bin, writes CSV)
#   make aod_pipeline        — full pipeline: regrid → KNN → OF → smooth → K → G → Gibbs
#   make knn_impute_verify   — standalone KNN verifier (reads CSV, compares vs R ref)
#   make of_verify           — standalone OF verifier (one pair)
#   make smooth_verify       — standalone image-smooth verifier (one matrix)
#   make g_matrix_verify     — standalone G matrix verifier (z1,z2,omega → G_gen)
#   make obs_assembly_verify — standalone obs assembly verifier
#   make all                 — builds all targets (default)
#   make ARCH=sm_80 ...      — override arch (sm_70=V100, sm_80=A100, sm_86=RTX3090)
#   make clean

NVCC     ?= nvcc
ARCH     ?= $(shell python3 -c \
              "import subprocess, re; \
               out = subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], \
                 capture_output=True, text=True); \
               cc = out.stdout.strip().split('\n')[0].replace('.',''); \
               print('sm_' + cc)" 2>/dev/null || echo sm_70)

# Eigen 3.4.0 — header-only, bundled locally
EIGEN_INC = -I$(dir $(abspath $(lastword $(MAKEFILE_LIST))))eigen-3.4.0

CUDA_INC   = /mnt/sdcard/cuda-13.2/targets/sbsa-linux/include
CUDA_LIB   = /mnt/sdcard/cuda-13.2/targets/sbsa-linux/lib

NVCCFLAGS = -O3 -std=c++17 --generate-code arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH) \
            -I$(CUDA_INC) -L$(CUDA_LIB) -L$(CUDA_LIB)/stubs -L/usr/local/cuda-stubs -Wno-deprecated-gpu-targets

.PHONY: all clean regrid_standalone aod_pipeline knn_impute_verify of_verify \
        smooth_verify g_matrix_verify obs_assembly_verify

all: regrid_standalone aod_pipeline knn_impute_verify of_verify smooth_verify \
     g_matrix_verify obs_assembly_verify
	@echo "Built all targets  (arch=$(ARCH))"

# Standalone regrid verifier — exact replica of original regrid_gpu binary.
# Run: ./regrid_standalone manifest_G16.txt output/G16
regrid_standalone: regrid_gpu.cu regrid_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -DREGRID_STANDALONE_MAIN -o $@ regrid_gpu.cu
	@echo "Built: regrid_standalone  (arch=$(ARCH))"

# Full pipeline: .bin → GPU regrid → KNN → OF(G16) → smooth(G16) → K → G → Gibbs → CSV
# Run: ./aod_pipeline
aod_pipeline: main_pipeline.cu regrid_gpu.cu regrid_gpu.cuh \
              knn_impute.cu knn_impute.cuh \
              optical_flow.cu optical_flow.cuh \
              image_smooth.cu image_smooth.cuh \
              fourier_basis.cu fourier_basis.cuh \
              newgtry_gpu.cu newgtry_gpu.cuh \
              matrix_expm.cu matrix_expm.cuh \
              obs_assembly.cu obs_assembly.cuh \
              gibbs_sampler.cu gibbs_sampler.cuh \
              csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) $(EIGEN_INC) -DSAVE_INTERMEDIATES -o $@ \
		main_pipeline.cu regrid_gpu.cu knn_impute.cu \
		optical_flow.cu image_smooth.cu \
		fourier_basis.cu newgtry_gpu.cu matrix_expm.cu \
		obs_assembly.cu gibbs_sampler.cu \
		csv_io.cu \
		-lcufft -lcublas -lcublasLt -lcusolver -lcurand
	@echo "Built: aod_pipeline  (arch=$(ARCH))"

# Standalone KNN verifier — feed any 60×60 input CSV and R reference CSV.
# Run: ./knn_impute_verify input.csv ref.csv output.csv
knn_impute_verify: knn_impute.cu knn_impute.cuh csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) -DKNN_VERIFY_MAIN -I. -o $@ knn_impute.cu csv_io.cu
	@echo "Built: knn_impute_verify  (arch=$(ARCH))"

# Standalone OF verifier — test one image pair.
# Run: ./of_verify initial.csv final.csv out_speed.csv out_angle.csv ref_speed.csv ref_angle.csv
of_verify: optical_flow.cu optical_flow.cuh csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) -DOF_VERIFY_MAIN -I. -o $@ optical_flow.cu csv_io.cu
	@echo "Built: of_verify  (arch=$(ARCH))"

# Standalone image-smooth verifier.
# Run: ./smooth_verify input.csv ref.csv output.csv
smooth_verify: image_smooth.cu image_smooth.cuh csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) -DSMOOTH_VERIFY_MAIN -I. -o $@ image_smooth.cu csv_io.cu -lcufft
	@echo "Built: smooth_verify  (arch=$(ARCH))"

# Standalone G matrix verifier.
# Run: ./g_matrix_verify z1.csv z2.csv omega1.csv omega2.csv out_G.csv ref_G.csv
g_matrix_verify: newgtry_gpu.cu newgtry_gpu.cuh fourier_basis.cu fourier_basis.cuh \
                 csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) -DG_VERIFY_MAIN -I. -o $@ \
		newgtry_gpu.cu fourier_basis.cu csv_io.cu
	@echo "Built: g_matrix_verify  (arch=$(ARCH))"

# Standalone obs assembly verifier.
# Run: ./obs_assembly_verify F.csv y.csv ref_Ft.csv ref_yc.csv ref_id.csv
obs_assembly_verify: obs_assembly.cu obs_assembly.cuh csv_io.cu csv_io.cuh
	$(NVCC) $(NVCCFLAGS) -DOBS_VERIFY_MAIN -I. -o $@ obs_assembly.cu csv_io.cu
	@echo "Built: obs_assembly_verify  (arch=$(ARCH))"

clean:
	rm -f regrid_standalone aod_pipeline knn_impute_verify of_verify smooth_verify \
	      g_matrix_verify obs_assembly_verify *.o
