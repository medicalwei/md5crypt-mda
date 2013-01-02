build: md5crypt-mda.cu
	nvcc --ptxas-options=-v -lcudart -arch sm_30 -o md5crypt-mda -Xcompiler -fopenmp md5crypt-mda.cu
debug: md5crypt-mda.cu
	nvcc -g -G -lineinfo --ptxas-options=-v -lcudart -arch sm_30 -o md5crypt-mda -Xcompiler -fopenmp md5crypt-mda.cu
clean: md5crypt-mda
	rm md5crypt-mda
