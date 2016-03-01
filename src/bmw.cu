/*
	by Pecho <pecho@2ba.su>
	draft version
	based on SPH
*/

//#include "helper.h"

#define SHL(x, n) ((x) << (n))
#define SHR(x, n) ((x) >> (n))

#define CONST_EXP2 q[i+0] + ROTL64(q[i+1], 5) + q[i+2] + ROTL64(q[i+3], 11) + \
  q[i+4] + ROTL64(q[i+5], 27) + q[i+6] + ROTL64(q[i+7], 32) + \
  q[i+8] + ROTL64(q[i+9], 37) + q[i+10] + ROTL64(q[i+11], 43) + \
  q[i+12] + ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__global__ void kernelBmw512(unsigned char *hashes) {
	int i;
	uint64 BMW_H[16];

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	BMW_H[0] = 0x8081828384858687ULL;
	BMW_H[1] = 0x88898A8B8C8D8E8FULL;
	BMW_H[2] = 0x9091929394959697ULL;
	BMW_H[3] = 0x98999A9B9C9D9E9FULL;
	BMW_H[4] = 0xA0A1A2A3A4A5A6A7ULL; 
	BMW_H[5] = 0xA8A9AAABACADAEAFULL;
	BMW_H[6] = 0xB0B1B2B3B4B5B6B7ULL; 
	BMW_H[7] = 0xB8B9BABBBCBDBEBFULL;
	BMW_H[8] = 0xC0C1C2C3C4C5C6C7ULL; 
	BMW_H[9] = 0xC8C9CACBCCCDCECFULL;
	BMW_H[10] = 0xD0D1D2D3D4D5D6D7ULL; 
	BMW_H[11] = 0xD8D9DADBDCDDDEDFULL;
	BMW_H[12] = 0xE0E1E2E3E4E5E6E7ULL; 
	BMW_H[13] = 0xE8E9EAEBECEDEEEFULL;
	BMW_H[14] = 0xF0F1F2F3F4F5F6F7ULL; 
	BMW_H[15] = 0xF8F9FAFBFCFDFEFFULL;

	uint64 mv[16],q[32];
    uint64 tmp;

	mv[0] = dec64le2(hashes + (tid*64) +   0);
    mv[1] = dec64le2(hashes + (tid*64) +   8);
    mv[2] = dec64le2(hashes + (tid*64) +  16);
    mv[3] = dec64le2(hashes + (tid*64) +  24);
    mv[4] = dec64le2(hashes + (tid*64) +  32);
    mv[5] = dec64le2(hashes + (tid*64) +  40);
    mv[6] = dec64le2(hashes + (tid*64) +  48);
    mv[7] = dec64le2(hashes + (tid*64) +  56);
	mv[8] = 0x80;
	mv[9] = 0;
	mv[10] = 0;
	mv[11] = 0;
	mv[12] = 0;
	mv[13] = 0;
	mv[14] = 0;
	mv[15] = 0x200;

	tmp = (mv[5] ^ BMW_H[5]) - (mv[7] ^ BMW_H[7]) + (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]) + (mv[14] ^ BMW_H[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[1];
	tmp = (mv[6] ^ BMW_H[6]) - (mv[8] ^ BMW_H[8]) + (mv[11] ^ BMW_H[11]) + (mv[14] ^ BMW_H[14]) - (mv[15] ^ BMW_H[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[2];
	tmp = (mv[0] ^ BMW_H[0]) + (mv[7] ^ BMW_H[7]) + (mv[9] ^ BMW_H[9]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[3];
	tmp = (mv[0] ^ BMW_H[0]) - (mv[1] ^ BMW_H[1]) + (mv[8] ^ BMW_H[8]) - (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[4];
	tmp = (mv[1] ^ BMW_H[1]) + (mv[2] ^ BMW_H[2]) + (mv[9] ^ BMW_H[9]) - (mv[11] ^ BMW_H[11]) - (mv[14] ^ BMW_H[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + BMW_H[5];
	tmp = (mv[3] ^ BMW_H[3]) - (mv[2] ^ BMW_H[2]) + (mv[10] ^ BMW_H[10]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[6];
	tmp = (mv[4] ^ BMW_H[4]) - (mv[0] ^ BMW_H[0]) - (mv[3] ^ BMW_H[3]) - (mv[11] ^ BMW_H[11]) + (mv[13] ^ BMW_H[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[7];
	tmp = (mv[1] ^ BMW_H[1]) - (mv[4] ^ BMW_H[4]) - (mv[5] ^ BMW_H[5]) - (mv[12] ^ BMW_H[12]) - (mv[14] ^ BMW_H[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[8];
	tmp = (mv[2] ^ BMW_H[2]) - (mv[5] ^ BMW_H[5]) - (mv[6] ^ BMW_H[6]) + (mv[13] ^ BMW_H[13]) - (mv[15] ^ BMW_H[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[9];
	tmp = (mv[0] ^ BMW_H[0]) - (mv[3] ^ BMW_H[3]) + (mv[6] ^ BMW_H[6]) - (mv[7] ^ BMW_H[7]) + (mv[14] ^ BMW_H[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + BMW_H[10];
	tmp = (mv[8] ^ BMW_H[8]) - (mv[1] ^ BMW_H[1]) - (mv[4] ^ BMW_H[4]) - (mv[7] ^ BMW_H[7]) + (mv[15] ^ BMW_H[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[11];
	tmp = (mv[8] ^ BMW_H[8]) - (mv[0] ^ BMW_H[0]) - (mv[2] ^ BMW_H[2]) - (mv[5] ^ BMW_H[5]) + (mv[9] ^ BMW_H[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[12];
	tmp = (mv[1] ^ BMW_H[1]) + (mv[3] ^ BMW_H[3]) - (mv[6] ^ BMW_H[6]) - (mv[9] ^ BMW_H[9]) + (mv[10] ^ BMW_H[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[13];
	tmp = (mv[2] ^ BMW_H[2]) + (mv[4] ^ BMW_H[4]) + (mv[7] ^ BMW_H[7]) + (mv[10] ^ BMW_H[10]) + (mv[11] ^ BMW_H[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[14];
	tmp = (mv[3] ^ BMW_H[3]) - (mv[5] ^ BMW_H[5]) + (mv[8] ^ BMW_H[8]) - (mv[11] ^ BMW_H[11]) - (mv[12] ^ BMW_H[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + BMW_H[15];
	tmp = (mv[12] ^ BMW_H[12]) - (mv[4] ^ BMW_H[4]) - (mv[6] ^ BMW_H[6]) - (mv[9] ^ BMW_H[9]) + (mv[13] ^ BMW_H[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[0];

	for(i=0;i<2;i++)
	{
		q[i+16] =
		(SHR(q[i], 1) ^ SHL(q[i], 2) ^ ROTL64(q[i], 13) ^ ROTL64(q[i], 43)) +
		(SHR(q[i+1], 2) ^ SHL(q[i+1], 1) ^ ROTL64(q[i+1], 19) ^ ROTL64(q[i+1], 53)) +
		(SHR(q[i+2], 2) ^ SHL(q[i+2], 2) ^ ROTL64(q[i+2], 28) ^ ROTL64(q[i+2], 59)) +
		(SHR(q[i+3], 1) ^ SHL(q[i+3], 3) ^ ROTL64(q[i+3], 4) ^ ROTL64(q[i+3], 37)) +
		(SHR(q[i+4], 1) ^ SHL(q[i+4], 2) ^ ROTL64(q[i+4], 13) ^ ROTL64(q[i+4], 43)) +
		(SHR(q[i+5], 2) ^ SHL(q[i+5], 1) ^ ROTL64(q[i+5], 19) ^ ROTL64(q[i+5], 53)) +
		(SHR(q[i+6], 2) ^ SHL(q[i+6], 2) ^ ROTL64(q[i+6], 28) ^ ROTL64(q[i+6], 59)) +
		(SHR(q[i+7], 1) ^ SHL(q[i+7], 3) ^ ROTL64(q[i+7], 4) ^ ROTL64(q[i+7], 37)) +
		(SHR(q[i+8], 1) ^ SHL(q[i+8], 2) ^ ROTL64(q[i+8], 13) ^ ROTL64(q[i+8], 43)) +
		(SHR(q[i+9], 2) ^ SHL(q[i+9], 1) ^ ROTL64(q[i+9], 19) ^ ROTL64(q[i+9], 53)) +
		(SHR(q[i+10], 2) ^ SHL(q[i+10], 2) ^ ROTL64(q[i+10], 28) ^ ROTL64(q[i+10], 59)) +
		(SHR(q[i+11], 1) ^ SHL(q[i+11], 3) ^ ROTL64(q[i+11], 4) ^ ROTL64(q[i+11], 37)) +
		(SHR(q[i+12], 1) ^ SHL(q[i+12], 2) ^ ROTL64(q[i+12], 13) ^ ROTL64(q[i+12], 43)) +
		(SHR(q[i+13], 2) ^ SHL(q[i+13], 1) ^ ROTL64(q[i+13], 19) ^ ROTL64(q[i+13], 53)) +
		(SHR(q[i+14], 2) ^ SHL(q[i+14], 2) ^ ROTL64(q[i+14], 28) ^ ROTL64(q[i+14], 59)) +
		(SHR(q[i+15], 1) ^ SHL(q[i+15], 3) ^ ROTL64(q[i+15], 4) ^ ROTL64(q[i+15], 37)) +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
	}

	for(i=2;i<6;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
	}

	for(i=6;i<9;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i+7]);
	}

	for(i=9;i<13;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
	}

	for(i=13;i<16;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i-13], (i-13)+1) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
	}

	uint64 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint64 XH64 = XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];

	BMW_H[0] = (SHL(XH64, 5) ^ SHR(q[16],5) ^ mv[0]) + ( XL64 ^ q[24] ^ q[0]);
	BMW_H[1] = (SHR(XH64, 7) ^ SHL(q[17],8) ^ mv[1]) + ( XL64 ^ q[25] ^ q[1]);
	BMW_H[2] = (SHR(XH64, 5) ^ SHL(q[18],5) ^ mv[2]) + ( XL64 ^ q[26] ^ q[2]);
	BMW_H[3] = (SHR(XH64, 1) ^ SHL(q[19],5) ^ mv[3]) + ( XL64 ^ q[27] ^ q[3]);
	BMW_H[4] = (SHR(XH64, 3) ^ q[20] ^ mv[4]) + ( XL64 ^ q[28] ^ q[4]);
	BMW_H[5] = (SHL(XH64, 6) ^ SHR(q[21],6) ^ mv[5]) + ( XL64 ^ q[29] ^ q[5]);
	BMW_H[6] = (SHR(XH64, 4) ^ SHL(q[22],6) ^ mv[6]) + ( XL64 ^ q[30] ^ q[6]);
	BMW_H[7] = (SHR(XH64,11) ^ SHL(q[23],2) ^ mv[7]) + ( XL64 ^ q[31] ^ q[7]);

	BMW_H[8] = ROTL64(BMW_H[4], 9) + ( XH64 ^ q[24] ^ mv[8]) + (SHL(XL64,8) ^ q[23] ^ q[8]);
	BMW_H[9] = ROTL64(BMW_H[5],10) + ( XH64 ^ q[25] ^ mv[9]) + (SHR(XL64,6) ^ q[16] ^ q[9]);
	BMW_H[10] = ROTL64(BMW_H[6],11) + ( XH64 ^ q[26] ^ mv[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
	BMW_H[11] = ROTL64(BMW_H[7],12) + ( XH64 ^ q[27] ^ mv[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
	BMW_H[12] = ROTL64(BMW_H[0],13) + ( XH64 ^ q[28] ^ mv[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
	BMW_H[13] = ROTL64(BMW_H[1],14) + ( XH64 ^ q[29] ^ mv[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
	BMW_H[14] = ROTL64(BMW_H[2],15) + ( XH64 ^ q[30] ^ mv[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	BMW_H[15] = ROTL64(BMW_H[3],16) + ( XH64 ^ q[31] ^ mv[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);

	/* final */
	for(i=0;i<16;i++)
	{
		mv[i] = BMW_H[i];
		BMW_H[i] = 0xaaaaaaaaaaaaaaa0ul + (uint64)i;
	}

	tmp = (mv[5] ^ BMW_H[5]) - (mv[7] ^ BMW_H[7]) + (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]) + (mv[14] ^ BMW_H[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[1];
	tmp = (mv[6] ^ BMW_H[6]) - (mv[8] ^ BMW_H[8]) + (mv[11] ^ BMW_H[11]) + (mv[14] ^ BMW_H[14]) - (mv[15] ^ BMW_H[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[2];
	tmp = (mv[0] ^ BMW_H[0]) + (mv[7] ^ BMW_H[7]) + (mv[9] ^ BMW_H[9]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[3];
	tmp = (mv[0] ^ BMW_H[0]) - (mv[1] ^ BMW_H[1]) + (mv[8] ^ BMW_H[8]) - (mv[10] ^ BMW_H[10]) + (mv[13] ^ BMW_H[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[4];
	tmp = (mv[1] ^ BMW_H[1]) + (mv[2] ^ BMW_H[2]) + (mv[9] ^ BMW_H[9]) - (mv[11] ^ BMW_H[11]) - (mv[14] ^ BMW_H[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + BMW_H[5];
	tmp = (mv[3] ^ BMW_H[3]) - (mv[2] ^ BMW_H[2]) + (mv[10] ^ BMW_H[10]) - (mv[12] ^ BMW_H[12]) + (mv[15] ^ BMW_H[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[6];
	tmp = (mv[4] ^ BMW_H[4]) - (mv[0] ^ BMW_H[0]) - (mv[3] ^ BMW_H[3]) - (mv[11] ^ BMW_H[11]) + (mv[13] ^ BMW_H[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[7];
	tmp = (mv[1] ^ BMW_H[1]) - (mv[4] ^ BMW_H[4]) - (mv[5] ^ BMW_H[5]) - (mv[12] ^ BMW_H[12]) - (mv[14] ^ BMW_H[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[8];
	tmp = (mv[2] ^ BMW_H[2]) - (mv[5] ^ BMW_H[5]) - (mv[6] ^ BMW_H[6]) + (mv[13] ^ BMW_H[13]) - (mv[15] ^ BMW_H[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[9];
	tmp = (mv[0] ^ BMW_H[0]) - (mv[3] ^ BMW_H[3]) + (mv[6] ^ BMW_H[6]) - (mv[7] ^ BMW_H[7]) + (mv[14] ^ BMW_H[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + BMW_H[10];
	tmp = (mv[8] ^ BMW_H[8]) - (mv[1] ^ BMW_H[1]) - (mv[4] ^ BMW_H[4]) - (mv[7] ^ BMW_H[7]) + (mv[15] ^ BMW_H[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[11];
	tmp = (mv[8] ^ BMW_H[8]) - (mv[0] ^ BMW_H[0]) - (mv[2] ^ BMW_H[2]) - (mv[5] ^ BMW_H[5]) + (mv[9] ^ BMW_H[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + BMW_H[12];
	tmp = (mv[1] ^ BMW_H[1]) + (mv[3] ^ BMW_H[3]) - (mv[6] ^ BMW_H[6]) - (mv[9] ^ BMW_H[9]) + (mv[10] ^ BMW_H[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + BMW_H[13];
	tmp = (mv[2] ^ BMW_H[2]) + (mv[4] ^ BMW_H[4]) + (mv[7] ^ BMW_H[7]) + (mv[10] ^ BMW_H[10]) + (mv[11] ^ BMW_H[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + BMW_H[14];
	tmp = (mv[3] ^ BMW_H[3]) - (mv[5] ^ BMW_H[5]) + (mv[8] ^ BMW_H[8]) - (mv[11] ^ BMW_H[11]) - (mv[12] ^ BMW_H[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + BMW_H[15];
	tmp = (mv[12] ^ BMW_H[12]) - (mv[4] ^ BMW_H[4]) - (mv[6] ^ BMW_H[6]) - (mv[9] ^ BMW_H[9]) + (mv[13] ^ BMW_H[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + BMW_H[0];

	for(i=0;i<2;i++)
	{
		q[i+16] =
		(SHR(q[i], 1) ^ SHL(q[i], 2) ^ ROTL64(q[i], 13) ^ ROTL64(q[i], 43)) +
		(SHR(q[i+1], 2) ^ SHL(q[i+1], 1) ^ ROTL64(q[i+1], 19) ^ ROTL64(q[i+1], 53)) +
		(SHR(q[i+2], 2) ^ SHL(q[i+2], 2) ^ ROTL64(q[i+2], 28) ^ ROTL64(q[i+2], 59)) +
		(SHR(q[i+3], 1) ^ SHL(q[i+3], 3) ^ ROTL64(q[i+3], 4) ^ ROTL64(q[i+3], 37)) +
		(SHR(q[i+4], 1) ^ SHL(q[i+4], 2) ^ ROTL64(q[i+4], 13) ^ ROTL64(q[i+4], 43)) +
		(SHR(q[i+5], 2) ^ SHL(q[i+5], 1) ^ ROTL64(q[i+5], 19) ^ ROTL64(q[i+5], 53)) +
		(SHR(q[i+6], 2) ^ SHL(q[i+6], 2) ^ ROTL64(q[i+6], 28) ^ ROTL64(q[i+6], 59)) +
		(SHR(q[i+7], 1) ^ SHL(q[i+7], 3) ^ ROTL64(q[i+7], 4) ^ ROTL64(q[i+7], 37)) +
		(SHR(q[i+8], 1) ^ SHL(q[i+8], 2) ^ ROTL64(q[i+8], 13) ^ ROTL64(q[i+8], 43)) +
		(SHR(q[i+9], 2) ^ SHL(q[i+9], 1) ^ ROTL64(q[i+9], 19) ^ ROTL64(q[i+9], 53)) +
		(SHR(q[i+10], 2) ^ SHL(q[i+10], 2) ^ ROTL64(q[i+10], 28) ^ ROTL64(q[i+10], 59)) +
		(SHR(q[i+11], 1) ^ SHL(q[i+11], 3) ^ ROTL64(q[i+11], 4) ^ ROTL64(q[i+11], 37)) +
		(SHR(q[i+12], 1) ^ SHL(q[i+12], 2) ^ ROTL64(q[i+12], 13) ^ ROTL64(q[i+12], 43)) +
		(SHR(q[i+13], 2) ^ SHL(q[i+13], 1) ^ ROTL64(q[i+13], 19) ^ ROTL64(q[i+13], 53)) +
		(SHR(q[i+14], 2) ^ SHL(q[i+14], 2) ^ ROTL64(q[i+14], 28) ^ ROTL64(q[i+14], 59)) +
		(SHR(q[i+15], 1) ^ SHL(q[i+15], 3) ^ ROTL64(q[i+15], 4) ^ ROTL64(q[i+15], 37)) +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
	}

	for(i=2;i<6;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i+10], i+11) ) ^ BMW_H[i+7]);
	}

	for(i=6;i<9;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i+7]);
	}

	for(i=9;i<13;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i+3], i+4) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
	}

	for(i=13;i<16;i++)
	{
		q[i+16] = CONST_EXP2 +
		(( ((i+16)*(0x0555555555555555ul)) + ROTL64(mv[i], i+1) +
		ROTL64(mv[i-13], (i-13)+1) - ROTL64(mv[i-6], (i-6)+1) ) ^ BMW_H[i-9]);
	}

	XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	XH64 = XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];

	BMW_H[0] = (SHL(XH64, 5) ^ SHR(q[16],5) ^ mv[0]) + ( XL64 ^ q[24] ^ q[0]);
	BMW_H[1] = (SHR(XH64, 7) ^ SHL(q[17],8) ^ mv[1]) + ( XL64 ^ q[25] ^ q[1]);
	BMW_H[2] = (SHR(XH64, 5) ^ SHL(q[18],5) ^ mv[2]) + ( XL64 ^ q[26] ^ q[2]);
	BMW_H[3] = (SHR(XH64, 1) ^ SHL(q[19],5) ^ mv[3]) + ( XL64 ^ q[27] ^ q[3]);
	BMW_H[4] = (SHR(XH64, 3) ^ q[20] ^ mv[4]) + ( XL64 ^ q[28] ^ q[4]);
	BMW_H[5] = (SHL(XH64, 6) ^ SHR(q[21],6) ^ mv[5]) + ( XL64 ^ q[29] ^ q[5]);
	BMW_H[6] = (SHR(XH64, 4) ^ SHL(q[22],6) ^ mv[6]) + ( XL64 ^ q[30] ^ q[6]);
	BMW_H[7] = (SHR(XH64,11) ^ SHL(q[23],2) ^ mv[7]) + ( XL64 ^ q[31] ^ q[7]);

	BMW_H[8] = ROTL64(BMW_H[4], 9) + ( XH64 ^ q[24] ^ mv[8]) + (SHL(XL64,8) ^ q[23] ^ q[8]);
	BMW_H[9] = ROTL64(BMW_H[5],10) + ( XH64 ^ q[25] ^ mv[9]) + (SHR(XL64,6) ^ q[16] ^ q[9]);
	BMW_H[10] = ROTL64(BMW_H[6],11) + ( XH64 ^ q[26] ^ mv[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
	BMW_H[11] = ROTL64(BMW_H[7],12) + ( XH64 ^ q[27] ^ mv[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
	BMW_H[12] = ROTL64(BMW_H[0],13) + ( XH64 ^ q[28] ^ mv[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
	BMW_H[13] = ROTL64(BMW_H[1],14) + ( XH64 ^ q[29] ^ mv[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
	BMW_H[14] = ROTL64(BMW_H[2],15) + ( XH64 ^ q[30] ^ mv[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	BMW_H[15] = ROTL64(BMW_H[3],16) + ( XH64 ^ q[31] ^ mv[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);

	enc64le2(hashes + (tid*64) + 0, BMW_H[8]);
	enc64le2(hashes + (tid*64) + 8, BMW_H[9]);
	enc64le2(hashes + (tid*64) + 16, BMW_H[10]);
	enc64le2(hashes + (tid*64) + 24, BMW_H[11]);
	enc64le2(hashes + (tid*64) + 32, BMW_H[12]);
	enc64le2(hashes + (tid*64) + 40, BMW_H[13]);
	enc64le2(hashes + (tid*64) + 48, BMW_H[14]);
	enc64le2(hashes + (tid*64) + 56, BMW_H[15]);

	//__syncthreads();
}