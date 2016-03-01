#pragma once

#define uint8 unsigned char
#define uint32 unsigned int
#define uint64 unsigned long long

#define T32(x)    ((x) & 0xFFFFFFFF)
#define ROTL32(x, n)   T32(((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR32(x, n)   ROTL32(x, (32 - (n)))

#define T64(x)    ((x) & 0xFFFFFFFFFFFFFFFF)
#define ROTL64(x, n)   T64(((x) << (n)) | ((x) >> (64 - (n))))
#define ROTR64(x, n)   ROTL64(x, (64 - (n)))

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

__device__ unsigned long long dec64be2(const unsigned char *src) {
	return ((unsigned long long)(((const unsigned char *)src)[0]) << 56)
		| ((unsigned long long)(((const unsigned char *)src)[1]) << 48)
		| ((unsigned long long)(((const unsigned char *)src)[2]) << 40)
		| ((unsigned long long)(((const unsigned char *)src)[3]) << 32)
		| ((unsigned long long)(((const unsigned char *)src)[4]) << 24)
		| ((unsigned long long)(((const unsigned char *)src)[5]) << 16)
		| ((unsigned long long)(((const unsigned char *)src)[6]) << 8)
		| (unsigned long long)(((const unsigned char *)src)[7]);
}

__device__ unsigned long long dec64le2(const void *src) {
	return (unsigned long long)(((const unsigned char *)src)[0])
		| ((unsigned long long)(((const unsigned char *)src)[1]) << 8)
		| ((unsigned long long)(((const unsigned char *)src)[2]) << 16)
		| ((unsigned long long)(((const unsigned char *)src)[3]) << 24)
		| ((unsigned long long)(((const unsigned char *)src)[4]) << 32)
		| ((unsigned long long)(((const unsigned char *)src)[5]) << 40)
		| ((unsigned long long)(((const unsigned char *)src)[6]) << 48)
		| ((unsigned long long)(((const unsigned char *)src)[7]) << 56);
}

__device__ void enc64be2(void *dst, unsigned long long val) {
	((unsigned char *)dst)[0] = (val >> 56);
	((unsigned char *)dst)[1] = (val >> 48);
	((unsigned char *)dst)[2] = (val >> 40);
	((unsigned char *)dst)[3] = (val >> 32);
	((unsigned char *)dst)[4] = (val >> 24);
	((unsigned char *)dst)[5] = (val >> 16);
	((unsigned char *)dst)[6] = (val >> 8);
	((unsigned char *)dst)[7] = val;
}

__device__ void enc64le2(void *dst, unsigned long long val) {
	((unsigned char *)dst)[0] = val;
	((unsigned char *)dst)[1] = (val >> 8);
	((unsigned char *)dst)[2] = (val >> 16);
	((unsigned char *)dst)[3] = (val >> 24);
	((unsigned char *)dst)[4] = (val >> 32);
	((unsigned char *)dst)[5] = (val >> 40);
	((unsigned char *)dst)[6] = (val >> 48);
	((unsigned char *)dst)[7] = (val >> 56);
}

__device__ unsigned int dec32le2(const void *src) {
	return (unsigned int)(((const unsigned char *)src)[0])
		| ((unsigned int)(((const unsigned char *)src)[1]) << 8)
		| ((unsigned int)(((const unsigned char *)src)[2]) << 16)
		| ((unsigned int)(((const unsigned char *)src)[3]) << 24);
}

__device__ unsigned int dec32be2(const void *src) {
	return ((unsigned int)(((const unsigned char *)src)[0]) << 24)
		| ((unsigned int)(((const unsigned char *)src)[1]) << 16)
		| ((unsigned int)(((const unsigned char *)src)[2]) << 8)
		| (unsigned int)(((const unsigned char *)src)[3]);
}

__device__ void enc32le2(void *dst, unsigned int val) {
	((unsigned char *)dst)[0] = val;
	((unsigned char *)dst)[1] = (val >> 8);
	((unsigned char *)dst)[2] = (val >> 16);
	((unsigned char *)dst)[3] = (val >> 24);
}

__device__ void enc32be2(void *dst, unsigned int val) {
	((unsigned char *)dst)[0] = (val >> 24);
	((unsigned char *)dst)[1] = (val >> 16);
	((unsigned char *)dst)[2] = (val >> 8);
	((unsigned char *)dst)[3] = val;
}

__device__ unsigned int bytePerm2(unsigned int a, unsigned int b, unsigned int slct)
{
	unsigned int i0 = (slct >>  0) & 0x7;
	unsigned int i1 = (slct >>  4) & 0x7;
	unsigned int i2 = (slct >>  8) & 0x7;
	unsigned int i3 = (slct >> 12) & 0x7;

	return (((((i0 < 4) ? (a >> (i0*8)) : (b >> ((i0-4)*8))) & 0xff) <<  0) +
		((((i1 < 4) ? (a >> (i1*8)) : (b >> ((i1-4)*8))) & 0xff) <<  8) +
		((((i2 < 4) ? (a >> (i2*8)) : (b >> ((i2-4)*8))) & 0xff) << 16) +
		((((i3 < 4) ? (a >> (i3*8)) : (b >> ((i3-4)*8))) & 0xff) << 24));
}
