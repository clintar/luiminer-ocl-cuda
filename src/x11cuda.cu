#include "x11cuda.h"
#include "miner.h"

inline void __checkCudaErrors( cudaError err, const char *cFile, const int iLine ) {
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s\n", cFile, iLine, (int)err, cudaGetErrorString( err ) );
		system("pause");
		exit(-1);
	}
}

/* -------------------------------------------------- */

int cuGetNumDevs(void) {
	cudaError_t err;
	int version;
	int numGpu = 0;

	err = cudaDriverGetVersion(&version);
	if (err != cudaSuccess) {
		applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
		exit(-1);
	}

	checkCudaErrors( cudaGetDeviceCount(&numGpu) );
	if (numGpu > MAX_GPU)
		numGpu = MAX_GPU;
	if (numGpu == 0) {
		printf("Unable to query number of CUDA devices! Are you sure that you have an nVidia videocards?\n");
		exit(-1);
	}

	return numGpu;
}

/* -------------------------------------------------- */

GPU_cuda* cuInitGpu(uint32_t id, enum CC requiredCapability) {
	int threadsInBlock;
	int versionCC;
	cudaDeviceProp deviceProp;

	applog(LOG_DEBUG, "[GPU%u] Init", id);
	GPU_cuda* gpu = (GPU_cuda*)calloc(1, sizeof(GPU_cuda));
	gpu->numDev = id;

	checkCudaErrors( cudaGetDeviceProperties(&deviceProp, gpu->numDev) );
	versionCC = (deviceProp.major * 0x10 + deviceProp.minor);
	if (versionCC < requiredCapability) {
		fprintf(stderr, "[GPU %d] device compute capability under then need\n", gpu->numDev);
		return NULL;
	}
	printf("[GPU %d] CUDA device is READY. ('%s' , CC:%02x)\n\n", gpu->numDev, deviceProp.name, versionCC);

	gpu->threads = deviceProp.maxThreadsPerBlock;
	threadsInBlock = deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
	gpu->blocks = deviceProp.multiProcessorCount * threadsInBlock;

	gpu->output = (uint64_t*)malloc(gpu->blocks * gpu->threads * sizeof(uint64_t));
	checkCudaErrors( cudaMalloc((void**)&gpu->output_dev, gpu->blocks * gpu->threads * sizeof(uint64_t) ) );

	checkCudaErrors( cudaMalloc((void**)&gpu->input_dev, 88 * sizeof(uint8_t)) );

	checkCudaErrors( cudaMalloc((void**)&gpu->workHash_dev, 64 * sizeof(uint8_t) * gpu->blocks * gpu->threads ) );

	gpu->curnonce = 0;

	return gpu;
}

/* -------------------------------------------------- */

#include "helper.h"

#include "blake.cu"
#include "bmw.cu"
#include "groestl.cu"
#include "skein.cu"
#include "jh.cu"
#include "keccak.cu"
#include "luffa.cu"
#include "cubehash.cu"
#include "aes_helper.cu"
#include "shavite.cu"
#include "simd.cu"
#include "echo.cu"

int scanhash_x11_jsonrpc_2_cuda(int thr_id, GPU_cuda *gpu, struct work *curwork, uint64_t max_nonce, unsigned long *hashes_done) {
	int i;
	int gpuFound;
	uint64_t start, end, step, j;

    uint64_t n = gpu->curnonce;
    const uint32_t first_nonce = n;
    uint32_t hash[32 / 4];
	uint8_t tmpblock[81];
	
	memcpy(tmpblock, curwork->data, 81);
	curwork->noncecnt = 0;

	//checkCudaErrors( cudaMemcpy(gpu->input_dev, curwork->data, 88, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpyToSymbol(data_cmem, curwork->data, 81) );

    do {
		gpuFound = 0;

    	if (opt_work_size + n > max_nonce)
    		break;

		start = n;
		end = start + opt_work_size;
		step = gpu->blocks * gpu->threads;

		for (j = start; j < end; j+=step) {
			
			checkCudaErrors( cudaMemset(gpu->output_dev, 0, gpu->blocks * gpu->threads * sizeof(uint64_t)) );

			kernelBlake512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev, j);

			kernelBmw512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);

			kernelGroestl512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelSkein512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelJh512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelKeccak512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelLuffa512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelCubehash512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelShavite512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);
			
			kernelSimd512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev);

			kernelEcho512<<< gpu->blocks, gpu->threads>>>(gpu->workHash_dev, gpu->output_dev, j, curwork->target[7]);

			checkCudaErrors( cudaMemcpy(gpu->output, gpu->output_dev, gpu->blocks * gpu->threads * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

			for(i = 0; i < gpu->blocks * gpu->threads; i++) {
				if (gpu->output[i] != 0) {
					gpuFound++;
					uint8_t tmpbuf[10];
					uint64_t testN = gpu->output[i];
			
					tmpbuf[0] = 0x01;
					tmpbuf[1] = testN;
					tmpbuf[2] = testN>>8;
					tmpbuf[3] = testN>>16;
					tmpbuf[4] = testN>>24;
					tmpbuf[5] = testN>>32;
					tmpbuf[6] = testN>>40;
					tmpbuf[7] = testN>>48;
					tmpbuf[8] = testN>>56;

					memcpy(((unsigned char *)tmpblock), tmpbuf, 9);

					x11_hash((uint8_t*)hash, 81, tmpblock);
					if(hash[7] < curwork->target[7])
						curwork->nonces[curwork->noncecnt++] = gpu->output[i];
					else
						applog(LOG_ERR, "[GPU%u] share doesn't validate on CPU, hash=%08x, target=%08x", gpu->numDev, hash[7], curwork->target[7]);
				}
			}
			if(opt_debug) 
				applog(LOG_DEBUG, "%d winning nonces found!", gpuFound);

		}

		if(curwork->nonces) {
			gpu->curnonce = n + opt_work_size;
			*hashes_done = n - first_nonce + opt_work_size;
            return true;
        }
				
    	n += opt_work_size;

    } while (likely((n < max_nonce && !work_restart[thr_id].restart)));

    *hashes_done = n - first_nonce;
    gpu->curnonce = n;
    return 0;
}