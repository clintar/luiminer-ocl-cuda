/*
	by Pecho <pecho@2ba.su>
	draft version
	based on Christian Buchner's & Christian H.'s (Germany) CUDA implementation
*/

//#include "helper.h"

#define MULT2(a,j)\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;

#define TWEAK(a0,a1,a2,a3,j)\
	a0 = (a0<<(j))|(a0>>(32-j));\
	a1 = (a1<<(j))|(a1>>(32-j));\
	a2 = (a2<<(j))|(a2>>(32-j));\
	a3 = (a3<<(j))|(a3>>(32-j));

#define STEP(c0,c1)\
	SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
	SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
	MIXWORD(chainv[0],chainv[4]);\
	MIXWORD(chainv[1],chainv[5]);\
	MIXWORD(chainv[2],chainv[6]);\
	MIXWORD(chainv[3],chainv[7]);\
	ADD_CONSTANT(chainv[0],chainv[4],c0,c1);

#define SUBCRUMB(a0,a1,a2,a3,a4)\
	a4  = a0;\
	a0 |= a1;\
	a2 ^= a3;\
	a1  = ~a1;\
	a0 ^= a3;\
	a3 &= a4;\
	a1 ^= a3;\
	a3 ^= a2;\
	a2 &= a0;\
	a0  = ~a0;\
	a2 ^= a1;\
	a1 |= a3;\
	a4 ^= a1;\
	a3 ^= a2;\
	a2 &= a1;\
	a1 ^= a0;\
	a0  = a4;

#define MIXWORD(a0,a4)\
	a4 ^= a0;\
	a0  = (a0<<2) | (a0>>(30));\
	a0 ^= a4;\
	a4  = (a4<<14) | (a4>>(18));\
	a4 ^= a0;\
	a0  = (a0<<10) | (a0>>(22));\
	a0 ^= a4;\
	a4  = (a4<<1) | (a4>>(31));

#define ADD_CONSTANT(a0,b0,c0,c1)\
	a0 ^= c0;\
	b0 ^= c1;

/* initial values of chaining variables */
__constant__ uint32 IV[40] = {
	0x6d251e69,0x44b051e0,0x4eaa6fb4,0xdbf78465,
	0x6e292011,0x90152df4,0xee058139,0xdef610bb,
	0xc3b44b95,0xd9d2f256,0x70eee9a0,0xde099fa3,
	0x5d9b0557,0x8fc944b3,0xcf1ccf0e,0x746cd581,
	0xf7efc89d,0x5dba5781,0x04016ce5,0xad659c05,
	0x0306194f,0x666d1836,0x24aa230a,0x8b264ae7,
	0x858075d5,0x36d79cce,0xe571f7d7,0x204b1f67,
	0x35870c6a,0x57e9e923,0x14bcb808,0x7cde72ce,
	0x6c68e9be,0x5ec41e22,0xc825b7c7,0xaffb4363,
	0xf5df3999,0x0fc688f1,0xb07224cc,0x03e86cea
};

__constant__ uint32 CNS[80] = {
	0x303994a6,0xe0337818,0xc0e65299,0x441ba90d,
	0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f,
	0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4,
	0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
	0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4,
	0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28,
	0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b,
	0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
	0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72,
	0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7,
	0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719,
	0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
	0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91,
	0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be,
	0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5,
	0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
	0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab,
	0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0,
	0x78602649,0x29131ab6,0x8edae952,0x0fc053c3,
	0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
};

__device__ void rnd512(uint32 *state, uint32 *buffer)
{
	int i,j;
	uint32 t[40];
	uint32 chainv[8];
	uint32 tmp;

#pragma unroll 8
	for(i=0;i<8;i++) {
		t[i]=0;
#pragma unroll 5
		for(j=0;j<5;j++) {
			t[i] ^= state[i+8*j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			state[i+8*j] ^= t[i];
		}
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			t[i+8*j] = state[i+8*j];
		}
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
		MULT2(state, j);
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			state[8*j+i] ^= t[8*((j+1)%5)+i];
		}
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			t[i+8*j] = state[i+8*j];
		}
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
		MULT2(state, j);
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			state[8*j+i] ^= t[8*((j+4)%5)+i];
		}
	}

#pragma unroll 5
	for(j=0;j<5;j++) {
#pragma unroll 8
		for(i=0;i<8;i++) {
			state[i+8*j] ^= buffer[i];
		}
		MULT2(buffer, 0);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		chainv[i] = state[i];
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		STEP(CNS[(2*i)],CNS[(2*i)+1]);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		state[i] = chainv[i];
		chainv[i] = state[i+8];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

#pragma unroll 8
	for(i=0;i<8;i++) {
		STEP(CNS[(2*i)+16],CNS[(2*i)+16+1]);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		state[i+8] = chainv[i];
		chainv[i] = state[i+16];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

#pragma unroll 8
	for(i=0;i<8;i++) {
		STEP(CNS[(2*i)+32],CNS[(2*i)+32+1]);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		state[i+16] = chainv[i];
		chainv[i] = state[i+24];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

#pragma unroll 8
	for(i=0;i<8;i++) {
		STEP(CNS[(2*i)+48],CNS[(2*i)+48+1]);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		state[i+24] = chainv[i];
		chainv[i] = state[i+32];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

#pragma unroll 8
	for(i=0;i<8;i++) {
		STEP(CNS[(2*i)+64],CNS[(2*i)+64+1]);
	}

#pragma unroll 8
	for(i=0;i<8;i++) {
		state[i+32] = chainv[i];
	}
}


__global__ void kernelLuffa512(unsigned char *hashes) {
	int i, j;

	uint32 buffer[8]; /* Buffer to be hashed */
	uint32 chainv[40];   /* Chaining values */
	
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for(i=0;i<40;i++) 
		chainv[i] = IV[i];
	
	for(i=0;i<8;i++) 
		buffer[i] = 0;

	for (i = 0; i < 8; i++) 
		buffer[i] = dec32be2(hashes + (tid*64) + (i << 2));
	rnd512(chainv, buffer);
	
	for (i = 0; i < 8; i++) 
		buffer[i] = dec32be2(hashes + (tid*64) +32+ (i << 2));
	rnd512(chainv, buffer);

	/* final */
	buffer[0] = 0x80000000;
	for(i=1;i<8;i++) 
		buffer[i] = 0;
	rnd512(chainv, buffer);

	/*---- blank round with m=0 ----*/
	for(i=0;i<8;i++) 
		buffer[i] =0;
	rnd512(chainv, buffer);

	uint32 b[16];
	for(i=0;i<8;i++) {
		b[i] = 0;
		for(j=0;j<5;j++) {
			b[i] ^= chainv[i+8*j];
		}
	}
	for(i=0;i<8;i++) 
		buffer[i]=0;
	rnd512(chainv, buffer);

	for(i=0;i<8;i++) {
		b[8+i] = 0;
		for(j=0;j<5;j++) {
			b[8+i] ^= chainv[i+8*j];
		}
	}
	for (i = 0; i < 16; i++)
		enc32be2(hashes + (tid*64) + (i << 2), b[i]);

	//__syncthreads();
}