/*
	by Pecho <pecho@2ba.su>
	draft version
	based on Christian Buchner's & Christian H.'s (Germany) CUDA implementation
*/

//#include "helper.h"

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

#define ROTATEUPWARDS7(a) (((a) << 7) | ((a) >> 25))
#define ROTATEUPWARDS11(a) (((a) << 11) | ((a) >> 21))
#define SWAP(a,b) { uint32 u = a; a = b; b = u; }

__constant__ uint32 IV_512[32] = {
	0x2AEA2A61, 0x50F494D4, 0x2D538B8B,
	0x4167D83E, 0x3FEE2313, 0xC701CF8C,
	0xCC39968E, 0x50AC5695, 0x4D42C787,
	0xA647A8B3, 0x97CF0BEF, 0x825B4537,
	0xEEF864D2, 0xF22090C4, 0xD0E5CD33,
	0xA23911AE, 0xFCD398D9, 0x148FE485,
	0x1B017BEF, 0xB6444532, 0x6A536159,
	0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
	0xD65C8A2B, 0xA5A70E75, 0xB1C62456,
	0xBC796576, 0x1921C8F7, 0xE7989AF1,
	0x7795D246, 0xD43E3B44
};

__device__ void rrounds(uint32 x[2][2][2][2][2])
{
	int r;
	int j;
	int k;
	int l;
	int m;

	//#pragma unroll 16
	for (r = 0;r < CUBEHASH_ROUNDS;++r) {

		/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (l = 0;l < 2;++l)
#pragma unroll 2
					for (m = 0;m < 2;++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (l = 0;l < 2;++l)
#pragma unroll 2
					for (m = 0;m < 2;++m)
						x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

		/* "swap x_00klm with x_01klm" */
#pragma unroll 2
		for (k = 0;k < 2;++k)
#pragma unroll 2
			for (l = 0;l < 2;++l)
#pragma unroll 2
				for (m = 0;m < 2;++m)
					SWAP(x[0][0][k][l][m],x[0][1][k][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0;j < 2;++j)
#pragma unroll 2
						for (k = 0;k < 2;++k)
#pragma unroll 2
							for (l = 0;l < 2;++l)
#pragma unroll 2
								for (m = 0;m < 2;++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (m = 0;m < 2;++m)
					SWAP(x[1][j][k][0][m],x[1][j][k][1][m])

					/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for (j = 0;j < 2;++j)
#pragma unroll 2
						for (k = 0;k < 2;++k)
#pragma unroll 2
							for (l = 0;l < 2;++l)
#pragma unroll 2
								for (m = 0;m < 2;++m)
									x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (l = 0;l < 2;++l)
#pragma unroll 2
					for (m = 0;m < 2;++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (l = 0;l < 2;++l)
#pragma unroll 2
				for (m = 0;m < 2;++m)
					SWAP(x[0][j][0][l][m],x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0;j < 2;++j)
#pragma unroll 2
						for (k = 0;k < 2;++k)
#pragma unroll 2
							for (l = 0;l < 2;++l)
#pragma unroll 2
								for (m = 0;m < 2;++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (l = 0;l < 2;++l)
					SWAP(x[1][j][k][l][0],x[1][j][k][l][1])

	}
}

__device__ void block_tox(uint32 block[16], uint32 x[2][2][2][2][2])
{
	int k;
	int l;
	int m;
	uint32 *in = block;

#pragma unroll 2
	for (k = 0;k < 2;++k)
#pragma unroll 2
		for (l = 0;l < 2;++l)
#pragma unroll 2
			for (m = 0;m < 2;++m)
				x[0][0][k][l][m] ^= *in++;
}

__device__ void hash_fromx(uint32 hash[16], uint32 x[2][2][2][2][2])
{
	int j;
	int k;
	int l;
	int m;
	uint32 *out = hash;

#pragma unroll 2
	for (j = 0;j < 2;++j)
#pragma unroll 2
		for (k = 0;k < 2;++k)
#pragma unroll 2
			for (l = 0;l < 2;++l)
#pragma unroll 2
				for (m = 0;m < 2;++m)
					*out++ = x[0][j][k][l][m];
}

__global__ void kernelCubehash512(unsigned char *hashes) {
	uint32 x[2][2][2][2][2];

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Init(x);
	uint32 *iv = IV_512;
	int i,j,k,l,m;
#pragma unroll 2
	for (i = 0;i < 2;++i)
#pragma unroll 2
		for (j = 0;j < 2;++j)
#pragma unroll 2
			for (k = 0;k < 2;++k)
#pragma unroll 2
				for (l = 0;l < 2;++l)
#pragma unroll 2
					for (m = 0;m < 2;++m)
						x[i][j][k][l][m] = *iv++;

	uint32 message[8];
	for (i = 0; i < 8; i++) 
		message[i] = dec32le2(hashes + (tid*64) + (i << 2));
	block_tox(message, x);
	rrounds(x);

	for (i = 0; i < 8; i++) 
		message[i] = dec32le2(hashes + (tid*64) +32+ (i << 2));
	block_tox(message, x);
	rrounds(x);

	// Padding Block
	uint32 last[8];
	last[0] = 0x80;
	for (i=1; i < 8; i++) 
		last[i] = 0;
	block_tox(last, x);
	rrounds(x);

	/* Final */
	/* "the integer 1 is xored into the last state word x_11111" */
	x[1][1][1][1][1] ^= 1;

	/* "the state is then transformed invertibly through 10r identical rounds" */
#pragma unroll 10
	for (i = 0;i < 10;++i) 
		rrounds(x);

	uint32 resultHash[16];
	/* "output the first h/8 bytes of the state" */
	hash_fromx(resultHash, x);
	for (i = 0; i < 16; i++)
		enc32le2(hashes + (tid*64) + (i << 2), resultHash[i]);

	//__syncthreads();
}