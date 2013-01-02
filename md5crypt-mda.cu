/*
 * MD5 crypt massive dictionary attack (md5crypt-mda)
 * CUDA implementation based on Benjamin Vernoux's MD5 cracker
 * Derived from the RSA Data Security, Inc. MD5 Message Digest Algorithm
 *
 * Yao Wei, mwei@lxde.org
 *
 * ====
 * optimizations planned:
 * - memory coalscing
 * - sorted password list. same password length in single warp.
 *   (which has exactly same iterations without branch diversion)
 * - bank deconflicting
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define HASH_LENGTH 16 /* hash size. may do with bank deconflicting. */
#define SALT_MAX_LENGTH 8
#define PASSWORD_MAX_LENGTH 16
#define DICTIONARY_MAX_SIZE 100000000
#define SHADOW_MAX_SIZE 20000
#define GPU_COUNT 1

#define BLOCK_SIZE 512

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (uint)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

#define x0 0x67452301
#define y0 0xEFCDAB89
#define z0 0x98BADCFE
#define w0 0x10325476

struct md5_ctx {
  unsigned int input[16];
  unsigned int inputSize;
  uint4 hash;
};

__device__ void md5_init(struct md5_ctx *ctx) {
  ctx->inputSize = 0;
  ctx->hash.x = x0;
  ctx->hash.y = y0;
  ctx->hash.z = z0;
  ctx->hash.w = w0;
}

/* md5 update script. must filled up with 128 bytes. */
inline __device__ void md5_calc(struct md5_ctx *ctx) {
  uint4 nhash;

  nhash.x = ctx->hash.x;
  nhash.y = ctx->hash.y;
  nhash.z = ctx->hash.z;
  nhash.w = ctx->hash.w;

	/* Round 1 */
	FF ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[0],  S11, 3614090360); /* 1 */
  FF ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[1],  S12, 3905402710); /* 2 */
  FF ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[2],  S13,  606105819); /* 3 */
  FF ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[3],  S14, 3250441966); /* 4 */
  FF ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[4],  S11, 4118548399); /* 5 */
  FF ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[5],  S12, 1200080426); /* 6 */
  FF ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[6],  S13, 2821735955); /* 7 */
  FF ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[7],  S14, 4249261313); /* 8 */
  FF ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[8],  S11, 1770035416); /* 9 */
  FF ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[9],  S12, 2336552879); /* 10 */
  FF ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[10], S13, 4294925233); /* 11 */
  FF ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[11], S14, 2304563134); /* 12 */
  FF ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[12], S11, 1804603682); /* 13 */
  FF ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[13], S12, 4254626195); /* 14 */
  FF ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[14], S13, 2792965006); /* 15 */
  FF ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[15], S14, 1236535329); /* 16 */

  /* Round 2 */
  GG ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[1],  S21, 4129170786); /* 17 */
  GG ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[6],  S22, 3225465664); /* 18 */
  GG ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[11], S23,  643717713); /* 19 */
  GG ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[0],  S24, 3921069994); /* 20 */
  GG ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[5],  S21, 3593408605); /* 21 */
  GG ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[10], S22,   38016083); /* 22 */
  GG ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[15], S23, 3634488961); /* 23 */
  GG ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[4],  S24, 3889429448); /* 24 */
  GG ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[9],  S21,  568446438); /* 25 */
  GG ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[14], S22, 3275163606); /* 26 */
  GG ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[3],  S23, 4107603335); /* 27 */
  GG ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[8],  S24, 1163531501); /* 28 */
  GG ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[13], S21, 2850285829); /* 29 */
  GG ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[2],  S22, 4243563512); /* 30 */
  GG ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[7],  S23, 1735328473); /* 31 */
  GG ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[12], S24, 2368359562); /* 32 */

  /* Round 3 */
  HH ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[5],  S31, 4294588738); /* 33 */
  HH ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[8],  S32, 2272392833); /* 34 */
  HH ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[11], S33, 1839030562); /* 35 */
  HH ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[14], S34, 4259657740); /* 36 */
  HH ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[1],  S31, 2763975236); /* 37 */
  HH ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[4],  S32, 1272893353); /* 38 */
  HH ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[7],  S33, 4139469664); /* 39 */
  HH ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[10], S34, 3200236656); /* 40 */
  HH ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[13], S31,  681279174); /* 41 */
  HH ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[0],  S32, 3936430074); /* 42 */
  HH ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[3],  S33, 3572445317); /* 43 */
  HH ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[6],  S34,   76029189); /* 44 */
  HH ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[9],  S31, 3654602809); /* 45 */
  HH ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[12], S32, 3873151461); /* 46 */
  HH ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[15], S33,  530742520); /* 47 */
  HH ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[2],  S34, 3299628645); /* 48 */

  /* Round 4 */
  II ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[0],  S41, 4096336452); /* 49 */
  II ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[7],  S42, 1126891415); /* 50 */
  II ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[14], S43, 2878612391); /* 51 */
  II ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[5],  S44, 4237533241); /* 52 */
  II ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[12], S41, 1700485571); /* 53 */
  II ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[3],  S42, 2399980690); /* 54 */
  II ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[10], S43, 4293915773); /* 55 */
  II ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[1],  S44, 2240044497); /* 56 */
  II ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[8],  S41, 1873313359); /* 57 */
  II ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[15], S42, 4264355552); /* 58 */
  II ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[6],  S43, 2734768916); /* 59 */
  II ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[13], S44, 1309151649); /* 60 */
  II ( nhash.x, nhash.y, nhash.z, nhash.w, ctx->input[4],  S41, 4149444226); /* 61 */
  II ( nhash.w, nhash.x, nhash.y, nhash.z, ctx->input[11], S42, 3174756917); /* 62 */
  II ( nhash.z, nhash.w, nhash.x, nhash.y, ctx->input[2],  S43,  718787259); /* 63 */
  II ( nhash.y, nhash.z, nhash.w, nhash.x, ctx->input[9],  S44, 3951481745); /* 64 */

	ctx->hash.x += nhash.x;
	ctx->hash.y += nhash.y;
	ctx->hash.z += nhash.z;
	ctx->hash.w += nhash.w;
}

__device__ void md5_update(struct md5_ctx *ctx, const char *in, int size) {
  char * inputChar = (char *) ctx->input;

  for(int i=0; i<size; i++){
    inputChar[ctx->inputSize++] = in[i];
    if (ctx->inputSize % 64 == 0) md5_calc(ctx);
  }
}

__device__ void md5_final(char * final, struct md5_ctx *ctx){
  char * inputChar = (char *) ctx->input;
  unsigned int realSize = ctx->inputSize;

  inputChar[ctx->inputSize++] = 0x80;
  if (ctx->inputSize % 64 == 0) md5_calc(ctx);

  while(ctx->inputSize % 64 != 56){
    inputChar[ctx->inputSize++] = 0x00;
    if (ctx->inputSize % 64 == 0) md5_calc(ctx);
  }

  ctx->input[14] = realSize << 3;
  ctx->input[15] = realSize >> 29;

  md5_calc(ctx);
  memcpy(final, &(ctx->hash), 16);
}

__device__ unsigned char itoa64[] = /* 0 ... 63 => ascii - 64 */
  "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

__device__ bool to64(char *s, u_int32_t v, int n) {
  while (--n >= 0) {
    if (*s != itoa64[v&0x3f]){
      return false;
    }
    v >>= 6;
    s++;
  }
  return true;
}


__device__ bool md5crypt(const char *pw, const char *salt, const unsigned char saltlength, const char *target) {
	struct md5_ctx ctx;
	char final[16];
  unsigned int pwlen;

  for(pwlen = 0; pwlen < 16 && pw[pwlen] != '\0'; pwlen++);

  /* get first "final" from password+salt+password */
	md5_init(&ctx);
	md5_update(&ctx, pw, pwlen);
	md5_update(&ctx, salt, saltlength);
	md5_update(&ctx, pw, pwlen);
	md5_final(final, &ctx);

  /* get second "final" from password+"$1$"+salt+fractions_of_final+weird_things */
	md5_init(&ctx);
	md5_update(&ctx, pw, pwlen);
	md5_update(&ctx, "$1$", 3);
	md5_update(&ctx, salt, saltlength);

	for (char pl = pwlen; pl > 0; pl -= 16) {
		md5_update(&ctx, final, pl>16 ? 16 : pl);
  }

  memset(final, 0, 16);

	for (char i = pwlen; i != 0; i >>= 1) {
		if(i&1) {
		  md5_update(&ctx, final, 1);
		} else {
		  md5_update(&ctx, pw, 1);
    }
  }

	md5_final(final, &ctx);

	/* 1000 iterations that runs slow in _Pentium 60 MHz_ >w< */
	for(unsigned short i=0;i<1000;i++) {
		md5_init(&ctx);

		if(i & 1) {
      md5_update(&ctx, pw, pwlen);
    } else {
      md5_update(&ctx, final, 16);
    }

		if(i % 3) {
      md5_update(&ctx, salt, saltlength);
    }

		if(i % 7) {
      md5_update(&ctx, pw, pwlen);
    }

		if(i & 1) {
			md5_update(&ctx, final, 16);
    } else {
			md5_update(&ctx, pw, pwlen);
    }

		md5_final(final, &ctx);
	}

  #pragma unroll
  for(int i=0; i<16; i++){
    if (final[i] != target[i]) return false;
  }
  return true;
}

__global__ void md5_crypt_dictionary_attack(const char *t, const char *salt, const unsigned char saltlength, const char *dictionary, const unsigned int * dictionary_index, unsigned int dictionary_index_size, unsigned int *result) {
  if (*result != UINT_MAX) return;
  /* get dictionary word from global memory */
  unsigned int dictionary_id = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ char target[17];
  if (threadIdx.x < 16)
    target[threadIdx.x] = t[threadIdx.x];

  /* size check */
  if (dictionary_id >= dictionary_index_size){
    return;
  }

  const char *password = &(dictionary[dictionary_index[dictionary_id]]);

  /* attack and examine result */
  bool a = md5crypt(password, salt, saltlength, target);
  if (a == true){
    *result = dictionary_id;
  }
}

unsigned char a64toi[128];
void from64_prepare(){
  int c = 0;
  a64toi['.'] = c++;
  a64toi['/'] = c++;
  for (int i='0'; i<='9'; i++){
    a64toi[i] = c++;
  }
  for (int i='A'; i<='Z'; i++){
    a64toi[i] = c++;
  }
  for (int i='a'; i<='z'; i++){
    a64toi[i] = c++;
  }
}
void from64(char* from, char* to){
  /* 12, 6, 0, 13, 7, 1, 14, 8, 2, 15, 9, 3, 5, 10, 4, 11 */
  int x;
  x = a64toi[from[3]] << 18 | a64toi[from[2]] << 12 | a64toi[from[1]] << 6 | a64toi[from[0]];
  to[12] = x & 0x000000FF; x >>= 8;
  to[6] = x & 0x000000FF; x >>= 8;
  to[0] = x & 0x000000FF; x >>= 8;
  x = a64toi[from[7]] << 18 | a64toi[from[6]] << 12 | a64toi[from[5]] << 6 | a64toi[from[4]];
  to[13] = x & 0x000000FF; x >>= 8;
  to[7] = x & 0x000000FF; x >>= 8;
  to[1] = x & 0x000000FF; x >>= 8;
  x = a64toi[from[11]] << 18 | a64toi[from[10]] << 12 | a64toi[from[9]] << 6 | a64toi[from[8]];
  to[14] = x & 0x000000FF; x >>= 8;
  to[8] = x & 0x000000FF; x >>= 8;
  to[2] = x & 0x000000FF; x >>= 8;
  x = a64toi[from[15]] << 18 | a64toi[from[14]] << 12 | a64toi[from[13]] << 6 | a64toi[from[12]];
  to[15] = x & 0x000000FF; x >>= 8;
  to[9] = x & 0x000000FF; x >>= 8;
  to[3] = x & 0x000000FF; x >>= 8;
  x = a64toi[from[19]] << 18 | a64toi[from[18]] << 12 | a64toi[from[17]] << 6 | a64toi[from[16]];
  to[5] = x & 0x000000FF; x >>= 8;
  to[10] = x & 0x000000FF; x >>= 8;
  to[4] = x & 0x000000FF; x >>= 8;
  x = a64toi[from[21]] << 6 | a64toi[from[20]];
  to[11] = x & 0x000000FF; x >>= 8;
}

void read_file(FILE * file, char** storage, unsigned int* storage_size, unsigned int** index, unsigned int* index_size){
  fseek(file, 0L, SEEK_END);
  *storage_size = ftell(file);
  rewind(file);

  *storage = (char *) malloc(*storage_size+1);
  fread(*storage, *storage_size, 1, file);

  *index_size = 0;
  for(int i=0; i<=*storage_size; i++){
    if ((*storage)[i] == '\n'){
      (*storage)[i] = '\0';
      *index_size += 1;
    }
  }

  *index = (unsigned int *) malloc(sizeof(unsigned int)*(*index_size));

  unsigned int state = 0;
  unsigned int index_number = 0;
  for(unsigned int i=0; i<=*storage_size; i++){
    if ((*storage)[i] != '\0' && state == 0){
      state = 1;
      (*index)[index_number] = i;
      index_number++;
    } else if ((*storage)[i] == '\0' && state == 1){
      state = 0;
    }
  }
}

int main(int argc, char** argv){
  omp_set_num_threads(GPU_COUNT);

  if (argc < 2 || strcmp(argv[1], "-h") == 0){
    printf("%s [dictionary] [shadow]\n", argv[0]);
  }

  FILE *dictionary = fopen(argv[1], "r");
  FILE *shadow = fopen(argv[2], "r");

  if (!dictionary || !shadow){
    fprintf(stderr, "Error opening file. Check if the file correct or not.");
    return -1;
  }

  char* dictionary_mem;
  unsigned int dictionary_size;
  unsigned int* dictionary_index;
  unsigned int dictionary_index_size;
  read_file(dictionary, &dictionary_mem, &dictionary_size, &dictionary_index, &dictionary_index_size);
  printf("dictionary_index_size: %d\n", dictionary_index_size);

  char hash[23];
  char salt[9];
  char hashArray[SHADOW_MAX_SIZE][23];
  char tArray[SHADOW_MAX_SIZE][16];
  char saltArray[SHADOW_MAX_SIZE][9];
  char line[100];
  unsigned int shadow_count = 0;

  from64_prepare();
  while(fgets(line, 100, shadow)){
    if (sscanf(line, "$1$%[^$]$%[^\n]", &salt, &hash) <= 0){
      continue;
    }
    memcpy(hashArray[shadow_count], hash, sizeof(char) * 23);
    from64(hash, tArray[shadow_count]);
    memcpy(saltArray[shadow_count], salt, sizeof(char) * 9);
    shadow_count += 1;
    if (shadow_count >= SHADOW_MAX_SIZE) break;
  }
  fclose(shadow);
  fclose(dictionary);

  unsigned int grid_size = dictionary_size / BLOCK_SIZE;
  if (dictionary_size % BLOCK_SIZE != 0) grid_size += 1;

  #pragma omp parallel
  {
    int gid = omp_get_thread_num();
    cudaSetDevice(gid);
    cudaFuncSetCacheConfig( md5_crypt_dictionary_attack, cudaFuncCachePreferL1 );

    char *d_dictionary;
    unsigned int* d_dictionary_index;
    cudaMalloc((void**)&d_dictionary, sizeof(char)*dictionary_size);
    cudaMalloc((void**)&d_dictionary_index, sizeof(unsigned int)*dictionary_index_size);
    cudaMemcpy(d_dictionary, dictionary_mem, sizeof(char)*dictionary_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dictionary_index, dictionary_index, sizeof(unsigned int)*dictionary_index_size, cudaMemcpyHostToDevice);

    char *d_t, *d_salt;
    cudaMalloc((void**)&d_t, sizeof(char)*16);
    cudaMalloc((void**)&d_salt, sizeof(char)*9);

    unsigned int *result;
    cudaHostAlloc((void**) &result, sizeof(int), cudaHostAllocDefault);

    #pragma omp for
    for(int i=0; i<shadow_count; i++){
      *result = UINT_MAX;

      cudaMemcpy(d_t, tArray[i], sizeof(char)*16, cudaMemcpyHostToDevice);
      cudaMemcpy(d_salt, saltArray[i], sizeof(char)*9, cudaMemcpyHostToDevice);

      md5_crypt_dictionary_attack<<<grid_size,BLOCK_SIZE>>>(d_t, d_salt, strlen(saltArray[i]), d_dictionary, d_dictionary_index, dictionary_index_size, result);
      cudaDeviceSynchronize();

      printf("[%d]", gid);
      if (*result != UINT_MAX){
        printf("$1$%s$%s = %s\n", saltArray[i], hashArray[i], dictionary_mem+dictionary_index[*result]);
      } else {
        printf("$1$%s$%s = not found\n", saltArray[i], hashArray[i]);
      }
    }

    cudaFreeHost(result);
    cudaFree(d_t);
    cudaFree(d_salt);
    cudaFree(d_dictionary);
    cudaFree(d_dictionary_index);
  }

  free(dictionary_mem);
  free(dictionary_index);
}
