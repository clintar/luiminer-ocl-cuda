#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "helper.h"

#ifdef __cplusplus
extern "C" {
#endif

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

enum CC {
	ALL = 0,
	ARCH_10 = 0x10,
	ARCH_11 = 0x11,
	ARCH_12 = 0x12,
	ARCH_13 = 0x13,
	ARCH_20 = 0x20,
	ARCH_30 = 0x30,
	ARCH_35 = 0x35,
	ARCH_37 = 0x37,
	ARCH_50 = 0x50,
	ARCH_52 = 0x52,
};

typedef struct {
	int numDev;
	int blocks;
	int threads;

	uint8_t *input_dev;
	uint8_t *workHash_dev;
	uint64_t *output;
	unsigned long long *output_dev;
	//uint32_t threadNumber;
	uint64_t curnonce;
}GPU_cuda;

int cuGetNumDevs(void) ;
GPU_cuda* cuInitGpu(uint32_t id, enum CC requiredCapability = ALL);

int scanhash_x11_jsonrpc_2_cuda(int thr_id, GPU_cuda *gpu, struct work *curwork, uint64_t max_nonce, unsigned long *hashes_done) ;

#ifdef __cplusplus
} /* extern "C" */
#endif
