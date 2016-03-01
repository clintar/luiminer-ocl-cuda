#ifndef __MINER_H__
#define __MINER_H__

#define PROGRAM_NAME "miner_x11"
/* Define to the full name of this package. */
#define PACKAGE_NAME "cpuminer"
/* Define to the version of this package. */
#define PACKAGE_VERSION "2.3.3.1"
/* Define to the full name and version of this package. */
#define PACKAGE_STRING "cpuminer 2.3.3.1"

#include <stdint.h>
#ifdef WIN32
#include "compat/sys/time.h"
#endif

#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#include "compat.h"


# include <stdlib.h>
# include <stddef.h>

#if HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif

#if HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
#  ifdef  __cplusplus
extern "C"
#  endif
    void *alloca (size_t);
# endif
#endif

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#define HAVE_GETOPT_LONG 1

#define false   0
#define true    1

#define bool int

#ifdef __INTELLISENSE__
/* should be in stdint.h but... */
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int8_t;
typedef unsigned __int16 uint8_t;

typedef unsigned __int32 time_t;
typedef char *  va_list;
#endif

#if HAVE_SYSLOG_H
#include <syslog.h>
#else
enum {
    LOG_ERR,
    LOG_WARNING,
    LOG_NOTICE,
    LOG_INFO,
    LOG_DEBUG,
};
#endif

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
    | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

/* FreeBSD uses superpages if vm.pmap.pg_ps_enabled=1 (default on on x86_64 FreeBSD 10 */
#if defined(__linux__)
#  define MINERD_WANT_MMAP 1
#  define MINERD_MMAP_FLAGS (MAP_HUGETLB | MAP_POPULATE)
#else
#  define MINERD_WANT_MMAP 0
#endif

#ifdef _MSC_VER
#define INLINE __inline
#else
#define INLINE inline
#endif

static INLINE uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
    return __builtin_bswap32(v);
#else
    return bswap_32(v);
#endif
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static INLINE uint32_t be32dec(const void *pp)
{
    const uint8_t *p = (uint8_t const *)pp;
    return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
        ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static INLINE uint32_t le32dec(const void *pp)
{
    const uint8_t *p = (uint8_t const *)pp;
    return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
        ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static INLINE void be32enc(void *pp, uint32_t x)
{
    uint8_t *p = (uint8_t *)pp;
    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static INLINE void le32enc(void *pp, uint32_t x)
{
    uint8_t *p = (uint8_t *)pp;
    p[0] = x & 0xff;
    p[1] = (x >> 8) & 0xff;
    p[2] = (x >> 16) & 0xff;
    p[3] = (x >> 24) & 0xff;
}
#endif

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads((str), 0, (err_ptr))
#else
#define JSON_LOADS(str, err_ptr) json_loads((str), (err_ptr))
#endif

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

#include "x11opencl.h"
#include "x11cuda.h"

struct thr_info {
    int		id;
    pthread_t	pth;
    struct thread_q	*q;
	GPU_ocl *gpu_ocl;
	GPU_cuda *gpu_cuda;
	uint32_t shares_accepted;
	uint32_t shares_rejected;
};

struct work_restart {
    volatile unsigned long	restart;
    char			padding[128 - sizeof(unsigned long)];
};

extern int opt_work_size;

extern bool opt_debug;
extern bool opt_protocol;
extern bool opt_redirect;
extern int opt_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool want_stratum;
extern bool have_stratum;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern bool use_syslog;
extern pthread_mutex_t applog_lock;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern struct work_restart *work_restart;
extern bool jsonrpc_2;
extern char rpc2_id[65];



extern volatile bool stratum_have_work;
extern volatile bool need_to_rerequest_job;

#define JSON_RPC_LONGPOLL	(1 << 0)
#define JSON_RPC_QUIET_404	(1 << 1)
#define JSON_RPC_IGNOREERR  (1 << 2)

/* util.c */

extern void applog(int prio, const char *fmt, ...);
extern json_t *json_rpc_call(CURL *curl, const char *url, const char *userpass,
                             const char *rpc_req, int *curl_err, int flags);
extern char *bin2hex(const unsigned char *p, size_t len);
extern bool hex2bin(unsigned char *p, const char *hexstr, size_t len);
extern size_t hex2bin_len(unsigned char *p, const char *hexstr, size_t len);
extern int timeval_subtract(struct timeval *result, struct timeval *x,
struct timeval *y);
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
extern void diff_to_target(uint32_t *target, double diff);


struct work {
    uint32_t data[32];
    uint32_t target[8];
    uint32_t job_len;
	uint32_t noncecnt;
	uint64_t nonces[0xFF];
	
    char *job_id;
    size_t xnonce2_len;
    unsigned char *xnonce2;
};

struct stratum_job {
    char *job_id;
    unsigned char prevhash[32];
    size_t coinbase_size;
    unsigned char *coinbase;
    unsigned char *xnonce2;
    size_t merkle_count;
    unsigned char **merkle;
    unsigned char version[4];
    unsigned char nbits[4];
    unsigned char ntime[4];
    bool clean;
    double diff;
};

struct stratum_ctx {
    char *url;

    CURL *curl;
    char *curl_url;
    char curl_err_str[CURL_ERROR_SIZE];
    curl_socket_t sock;
    size_t sockbuf_size;
    char *sockbuf;
    pthread_mutex_t sock_lock;

    double next_diff;

    char *session_id;
    size_t xnonce1_size;
    unsigned char *xnonce1;
    size_t xnonce2_size;
    struct stratum_job job;
    struct work work;
    pthread_mutex_t work_lock;
};

/* x11cpu.c */
extern "C" void x11_hash(uint8_t* output, size_t len, const uint8_t* input);
extern "C" int scanhash_x11_jsonrpc_2(int thr_id, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, uint64_t *hashes_done);


bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);

extern bool stratum_request_job(struct stratum_ctx *sctx);

extern bool rpc2_job_decode(const json_t *job, struct work *work);
extern bool rpc2_login_decode(const json_t *val);

struct thread_q;

extern struct thread_q *tq_new(void);
extern void tq_free(struct thread_q *tq);
extern bool tq_push(struct thread_q *tq, void *data);
extern void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
extern void tq_freeze(struct thread_q *tq);
extern void tq_thaw(struct thread_q *tq);

#endif /* __MINER_H__ */
