#ifndef __GPU_H__
#define __GPU_H__

//#define OUTPUT_SIZE 16
#define OUTPUT_SIZE	0x100
#define MAX_GPU 32
#define MAX_WORK_SIZE 1024*1024

//#include "compat/stdbool.h"
#include <stdio.h>
#include <stdint.h>
#include <CL/cl.h>

typedef struct {
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
	cl_kernel kernel[11];

	cl_program program;
	cl_mem inputBuffer;
	cl_mem outputBuffer;
	cl_mem stateBuffer;

	uint32_t type;
	uint64_t *output;
	uint32_t threadNumber;
	uint64_t curnonce;
}GPU_ocl;

int scanhash_x11_jsonrpc_2_ocl(int thr_id, GPU_ocl *gpu_ocl, struct work *curwork, uint64_t max_nonce, unsigned long *hashes_done) ;

GPU_ocl* clInitGpu(uint32_t id, uint32_t type);
void runGPU(GPU_ocl* gpu_ocl, uint32_t work_size, size_t offset, cl_ulong target);
void releaseGPU(GPU_ocl* gpu_ocl);
void CopyBufferToDevice(cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t offset, size_t size);
void CopyBufferToHost  (cl_command_queue queue, cl_mem buffer, void* h_Buffer, size_t offset, size_t size);

#endif /* __MINER_H__ */
