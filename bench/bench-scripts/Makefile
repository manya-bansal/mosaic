SHELL := /bin/bash


all: kick-tires

all-benchmarks: run-fig13 draw-fig13 run-fig14 draw-fig14 run-fig15 draw-fig15 \
	 run-fig14 draw-fig14 run-fig16 draw-fig16 run-fig17 draw-fig17 \
	 run-fig18 draw-fig18 run-fig19 draw-fig19 run-fig20 draw-fig20 \
	 run-fig21 draw-fig21 run-fig22

kick-tires: run-unit-test run-fig18 draw-fig18

run-unit-test:
	$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/build/bin/taco-test interface


# Each of these scripts inlcudes a list of systems that the benchmark runs in.
# Please change the list of systems to correspond with the systems that are
# currently downloaded on your computer.


run-fig13:
	/gemv/bench.sh

run-fig14:
	/symmgemv/bench.sh

run-fig15:
	./spmv/bench.sh

run-fig16:
	./sddmm_varSparsity/bench.sh

run-fig17:
	./blockedSparse_4T_5/bench.sh

run-fig18:
	./mmadd/bench.sh

run-fig19:
	./sddmm/bench.sh

run-fig20:
	./blockedSparse_4T/bench.sh

run-fig21:
	./ttv/bench.sh

# To draw the figures for the benchmarks that can be run on your machine, 
# please change the --systems flag to only include the systems that the benchmark
# ran on.

draw-fig13:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/gemv/result \
	--type=vary_dim --systems=mkl,taco,tblis,gsl,blas \
	--name=fig13 --start_dim=100 --step_dim=250
	cp ./gemv/result/fig13.pdf figs/
	

draw-fig14:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/symmgemv/result \
	--type=vary_dim --systems=mkl,taco,tblis,gsl,blas \
	--name=fig14 --start_dim=100 --step_dim=250
	cp ./symmgemv/result/fig14.pdf figs/


draw-fig15:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/spmv/result \
	--type=vary_dim --systems=mkl,taco,cuda --stardust=SpMV\
	--name=fig15 --start_dim=250 --step_dim=250
	cp ./spmv/result/fig15.pdf figs/

draw-fig16:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/sddmm_varSparsity/result \
	--type=vary_sparse --systems=blas,taco,dot_mkl,gemv_gsl,gemv_mkl,mkl,tblis \
	--name=fig16 --sparsity=0.00078125,0.0015625,0.003125,0.00625,0.0125,0.025,0.05,0.1
	cp ./sddmm_varSparsity/result/fig16.pdf figs/

draw-fig17:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/blockedSparse_4T_5/result \
	--type=vary_dim --systems=mkl,gsl,taco,blas \
	--name=fig17 --start_dim=10 --step_dim=10
	cp ./blockedSparse_4T_5/result/fig17.pdf figs/

draw-fig18:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/mmadd/result \
	--type=vary_sparse --systems=mkl,taco,bug_mkl --stardust=Plus2CSR \
	--name=fig18 --sparsity=0.00625,0.0125,0.025,0.05,0.1,0.2,0.4,0.8,1.0
	cp ./mmadd/result/fig18.pdf figs/


draw-fig19:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/sddmm/result \
	--type=vary_dim --systems=mkl,gsl,dot_blas,taco,tblis,gemv_blas,blas \
	--name=fig19 --start_dim=100 --step_dim=200
	cp ./sddmm/result/fig19.pdf figs/

draw-fig20:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/blockedSparse_4T/result \
	--type=vary_dim --systems=mkl,gsl,taco,blas \
	--name=fig20 --start_dim=10 --step_dim=10
	cp ./blockedSparse_4T/result/fig20.pdf figs/

draw-fig21:
	python3 $(PATH_TO_MOSAIC_ARTIFACT)/mosaic-benchmarks/run-figure.py \
	--data_dir=$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/bench/bench-scripts/ttv/result \
	--type=vary_dim --systems=blas,taco,mkl,tblis,gsl \
	--name=fig21 --start_dim=10 --step_dim=100
	cp ./ttv/result/fig21.pdf figs/

run-fig22:
	python3 ./smt_query_time/SMT_avx.py
	python3 ./smt_query_time/SMT_stardust.py
	$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/build/bin/taco-test testLinearAVX
	$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/build/bin/taco-test testRandomAVX
	$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/build/bin/taco-test testLinearStardust
	$(PATH_TO_MOSAIC_ARTIFACT)/mosaic/build/bin/taco-test testRandomStardust

draw-fig22:
	@echo "No graph to generate here, pls call make run-fig22 to look at the command directly."
