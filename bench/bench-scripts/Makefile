

all: run-fig13 draw-fig13 run-fig14 draw-fig14 run-fig15 draw-fig15

run-fig13:
	/gemv/bench.sh

run-fig14:
	/symmgemv/bench.sh

run-fig15:
	./spmv/bench.sh

run-fig16:
	{ time ./sddmm_varSparsity/bench.sh ; } > ./sddmm_varSparsity/time.txt

run-fig17:
	{ time ./blockedSparse_4T_5/bench.sh ; } > ./blockedSparse_4T_5/time.txt

run-fig18:
	./mmadd/bench.sh

run-fig19:
	./sddmm/bench.sh

run-fig20:
	{ time ./blockedSparse_4T/bench.sh ; } > ./blockedSparse_4T/time.txt

run-fig21:
	./ttv/bench.sh

run-fig22:
	python3 ./smt_query_time/SMT_avx.py
	python3 ./smt_query_time/SMT_stardust.py


draw-fig13:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/gemv/result \
	--type=vary_dim --systems=mkl,taco,tblis,gsl,blas \
	--name=fig13 --start_dim=100 --step_dim=250
	cp ./gemv/result/fig13.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/gemv/result*
	

draw-fig14:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/symmgemv/result \
	--type=vary_dim --systems=mkl,taco,tblis,gsl,blas \
	--name=fig14 --start_dim=100 --step_dim=250
	cp ./symmgemv/result/fig14.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/symmgemv/result*


draw-fig15:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/spmv/result \
	--type=vary_dim --systems=mkl,taco,cuda --stardust=SpMV\
	--name=fig15 --start_dim=250 --step_dim=250
	cp ./spmv/result/fig15.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/spmv/result*


draw-fig16:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/sddmm_varSparsity/result \
	--type=vary_sparse --systems=blas,taco,dot_mkl,gemv_gsl,gemv_mkl,mkl,tblis \
	--name=fig16 --sparsity=0.00078125,0.0015625,0.003125,0.00625,0.0125,0.025,0.05,0.1
	cp ./sddmm_varSparsity/result/fig16.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/sddmm_varSparsity/result*


draw-fig18:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/mmadd/result \
	--type=vary_sparse --systems=mkl,taco --stardust=Plus2CSR \
	--name=fig18 --sparsity=0.00625,0.0125,0.025,0.05,0.1,0.2,0.4,0.8,1.0
	cp ./mmadd/result/fig18.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/mmadd/result*


draw-fig19:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/sddmm/result \
	--type=vary_dim --systems=mkl,gsl,dot_blas,taco,tblis,gemv_blas,blas \
	--name=fig19 --start_dim=100 --step_dim=200
	cp ./sddmm/result/fig19.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/sddmm/result*


draw-fig21:
	python3 /home/reviewer/mosaic-benchmarks/run-figure.py \
	--data_dir=/home/reviewer/mosaic/bench/bench-scripts/ttv/result \
	--type=vary_dim --systems=blas,taco,mkl,tblis,gsl \
	--name=fig21 --start_dim=10 --step_dim=100
	cp ./ttv/result/fig21.pdf figs/
	#rm -v /home/reviewer/mosaic/bench/bench-scripts/ttv/result*
