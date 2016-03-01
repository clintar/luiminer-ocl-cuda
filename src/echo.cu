/*
	by Pecho <pecho@2ba.su>
	draft version
	based on Christian Buchner's & Christian H.'s (Germany) CUDA implementation
*/

//#include "helper.h"

//#include "aes_helper.cu"

__device__ __forceinline__ void AES_2ROUND(
	uint32 &x0, uint32 &x1, uint32 &x2, uint32 &x3,
	uint32 &k0, uint32 &k1, uint32 &k2, uint32 &k3)
{
	uint32 y0, y1, y2, y3;

	aes_roundK(
		x0, x1, x2, x3,
		k0,
		y0, y1, y2, y3);

	aes_round(
		y0, y1, y2, y3,
		x0, x1, x2, x3);

	k0++;
}

__device__ void echo_round(uint32 &k0, uint32 &k1, uint32 &k2, uint32 &k3, uint32 *W, int round) {
	// Big Sub Words
#pragma unroll 16
	for(int i=0;i<16;i++)
	{
		int idx = i<<2; // *4
		AES_2ROUND(W[idx+0], W[idx+1], W[idx+2], W[idx+3],
					k0, k1, k2, k3);
	}

	// Shift Rows
#pragma unroll 4
	for(int i=0;i<4;i++)
	{
		uint32 t;

		/// 1, 5, 9, 13
		t = W[4 + i];
		W[4 + i] = W[20 + i];
		W[20 + i] = W[36 + i];
		W[36 + i] = W[52 + i];
		W[52 + i] = t;

		// 2, 6, 10, 14
		t = W[8 + i];
		W[8 + i] = W[40 + i];
		W[40 + i] = t;
		t = W[24 + i];
		W[24 + i] = W[56 + i];
		W[56 + i] = t;

		// 15, 11, 7, 3
		t = W[60 + i];
		W[60 + i] = W[44 + i];
		W[44 + i] = W[28 + i];
		W[28 + i] = W[12 + i];
		W[12 + i] = t;
	}

	// Mix Columns
#pragma unroll 4
	for(int i=0;i<4;i++)
	{
#pragma unroll 4
		for(int j=0;j<4;j++)
		{
			int idx = j<<2; // j*4

			uint32 a = W[ ((idx + 0)<<2) + i];
			uint32 b = W[ ((idx + 1)<<2) + i];
			uint32 c = W[ ((idx + 2)<<2) + i];
			uint32 d = W[ ((idx + 3)<<2) + i];

			uint32 ab = a ^ b;
			uint32 bc = b ^ c;
			uint32 cd = c ^ d;

			uint32 t;
			t = ((ab & 0x80808080) >> 7);
			uint32 abx = t<<4 ^ t<<3 ^ t<<1 ^ t;
			t = ((bc & 0x80808080) >> 7);
			uint32 bcx = t<<4 ^ t<<3 ^ t<<1 ^ t;
			t = ((cd & 0x80808080) >> 7);
			uint32 cdx = t<<4 ^ t<<3 ^ t<<1 ^ t;

			abx ^= ((ab & 0x7F7F7F7F) << 1);
			bcx ^= ((bc & 0x7F7F7F7F) << 1);
			cdx ^= ((cd & 0x7F7F7F7F) << 1);

			W[ ((idx + 0)<<2) + i] = abx ^ bc ^ d;
			W[ ((idx + 1)<<2) + i] = bcx ^ a ^ cd;
			W[ ((idx + 2)<<2) + i] = cdx ^ ab ^ d;
			W[ ((idx + 3)<<2) + i] = abx ^ bcx ^ cdx ^ ab ^ c;
		}
	}
}

__global__ void kernelEcho512(const unsigned char *hashes, unsigned long long *dst, uint64 n, uint32 target) {
	int i;
	uint32 W[64];

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64 currentNonce = n + tid;

	uint32 k0 = 512, k1 = 0, k2 = 0, k3 = 0; // K0 = bitlen
	/* Init */
#pragma unroll 8
	for(int i=0;i<32;i+=4)
	{
		W[i + 0] = 512;
		W[i + 1] = 0;
		W[i + 2] = 0;
		W[i + 3] = 0;
	}
#pragma unroll 16
	for (i = 0; i < 16; i++) 
		W[i+32] = dec32le2(hashes + (tid*64) + (i << 2));
	W[48] = 0x80; // fest
#pragma unroll 10
	for(int i=49;i<59;i++)
		W[i] = 0;
	W[59] = 0x02000000; // fest
	W[60] = k0; // bitlen
	W[61] = k1;
	W[62] = k2;
	W[63] = k3;

	for(i=0;i<10;i++)
		echo_round(k0, k1, k2, k3, W, i);

#pragma unroll 8
	for(int i=0;i<32;i+=4)
	{
		W[i  ] ^= W[32 + i    ] ^ 512;
		W[i+1] ^= W[32 + i + 1];
		W[i+2] ^= W[32 + i + 2];
		W[i+3] ^= W[32 + i + 3];
	}

#pragma unroll 16
	for(int i=0;i<16;i++)
		W[i] ^= dec32le2(hashes + (tid*64) + (i << 2));

	W[27] ^= 0x02000000;
	W[28] ^= k0;

	//for (i = 0; i < 16; i++)
		//enc32le2(dst + (tid*64) + (i << 2), W[i]);

	
	dst[tid] = 0;
	if (W[7] < target)
		dst[tid] = currentNonce;

	//__syncthreads();
}
