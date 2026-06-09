#include <platform/platform.h>
#include <base/types.h>
#include <base/buddy.h>

#ifdef PLATFORM_HOSTED
#include <sys/mman.h>
#endif

// =============================================================================
// == Linux (x86_64) Implementation
// =============================================================================

// Compiler-runtime helpers required by the GNU/Clang toolchain in -nostdlib
// mode on Linux.
//
// Even with -fno-builtin, Clang/GCC emit implicit calls to memcpy() and
// memset() for things like struct assignment, passing/returning structs by
// value, and aggregate initialization. Without libc (or compiler-rt /
// libgcc), the link step fails with "undefined reference to `memcpy'" /
// "`memset'".
//
// Other backends do not need this:
//   * macOS:   libSystem.dylib provides memcpy/memset.
//   * Windows: MSVC with /kernel inlines them as intrinsics.
//   * WASM:    Clang lowers them to the native memory.copy / memory.fill
//              instructions.
//
// The implementations are tight byte loops. At -O3 -flto clang's
// loop-idiom-recognition pass DOES recognise these as memcpy/memset
// patterns and would rewrite the loop body as a call to memcpy/memset
// itself — which, in a freestanding build where these very functions
// are the only memcpy/memset definitions in the binary, results in
// unbounded recursion (stack overflow / SIGSEGV) the first time
// anything calls memset.
//
// Per-function workaround: insert a volatile asm barrier in each
// iteration. Volatile asm with a "memory" clobber has side effects
// the optimiser must preserve, which prevents loop-idiom-recognition
// from treating the loop body as a pure byte-copy or byte-fill
// pattern. Tested against clang 20 -O3 -flto on x86_64-linux-gnu:
// the generated machine code for memcpy/memset is a vectorised byte
// loop with no call instruction inside.
//
// They are marked weak so a higher layer (for example a C standard
// library subset that wraps memcpy()/memset() around
// base_memcpy()/base_memset()) can provide its own strong definitions
// without colliding with these at link time. Code that only links
// against corec gets these unconditionally.
//
// Hosted LFortran (PLATFORM_HOSTED) must not define these: weak shims can
// interact badly with glibc IFUNC/GOT resolution when linked into a normal
// libc-hosted binary.
#if !defined(PLATFORM_HOSTED)
__attribute__((weak))
void* memcpy(void* dest, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
        __asm__ volatile("" ::: "memory");
    }
    return dest;
}

__attribute__((weak))
void* memset(void* s, int c, size_t n) {
    unsigned char* p = (unsigned char*)s;
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)c;
        __asm__ volatile("" ::: "memory");
    }
    return s;
}
#endif

// Syscall numbers for x86_64
#define SYS_READ 0
#define SYS_OPEN 2
#define SYS_CLOSE 3
#define SYS_LSEEK 8
#define SYS_MMAP 9
#define SYS_MPROTECT 10
#define SYS_MUNMAP 11
#define SYS_READV 19
#define SYS_WRITEV 20
#define SYS_DUP 32
#define SYS_DUP2 33
#define SYS_EXIT 60
#define SYS_FCNTL 72
#define SYS_OPENAT 257

// AT_FDCWD: special value meaning "current working directory" for openat
#define AT_FDCWD -100

// fcntl commands
#define F_DUPFD 0

// mmap flags
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_FAILED ((void*)-1)

// Our emulated heap state for Linux
static uint8_t* linux_heap_base = NULL;
static size_t committed_pages = 0;
static const size_t RESERVED_SIZE = 1ULL << 32; // Reserve 4GB of virtual address space

// Command line arguments storage
static int stored_argc = 0;
static char** stored_argv = NULL;

typedef struct {
    void* addr;
    size_t size;
    bool in_use;
} MmapHandle;

#define MMAP_HANDLE_CAP 10
static MmapHandle g_mmap_handles[MMAP_HANDLE_CAP] = {0};

// Helper function to make a raw syscall.
static inline long syscall(long n, long a1, long a2, long a3, long a4, long a5, long a6) {
    long ret;
    register long r10 __asm__("r10") = a4;
    register long r8 __asm__("r8") = a5;
    register long r9 __asm__("r9") = a6;
    __asm__ volatile (
        "syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9)
        : "rcx", "r11", "memory"
    );
    return ret;
}

// Implementation of `fd_write` using the `writev` syscall.
uint32_t platform_fd_write(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten) {
    ssize_t ret = syscall(SYS_WRITEV, (long)fd, (long)iovs, (long)iovs_len, 0, 0, 0);
    if (ret < 0) {
        *nwritten = 0;
        return (uint32_t)-ret; // Return errno-like value
    }
    *nwritten = ret;
    return 0; // Success
}

void platform_exit(int status) {
    syscall(SYS_EXIT, (long)status, 0, 0, 0, 0, 0);
    __builtin_unreachable();
}

// Initializes the heap using mmap. We reserve large chunk of virtual
// address space but don't commit any physical memory to it initially.
static void ensure_heap_initialized() {
    if (linux_heap_base == NULL) {
#ifdef PLATFORM_HOSTED
        // Linked into a normal libc-hosted binary (e.g. LFortran + mlir API):
        // prefer libc mmap/mprotect for portability across toolchains.
        void *p = mmap(NULL, RESERVED_SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            linux_heap_base = NULL;
        } else {
            linux_heap_base = (uint8_t *)p;
        }
#else
        long mmap_ret = syscall(
            SYS_MMAP,
            (long)NULL,          // address hint
            (long)RESERVED_SIZE,       // size
            (long)0,                   // protection (none)
            (long)(MAP_PRIVATE | MAP_ANONYMOUS), // flags
            (long)-1,                  // file descriptor
            (long)0                    // offset
        );
        if (mmap_ret < 0) {
            linux_heap_base = NULL;
        } else {
            linux_heap_base = (uint8_t*)mmap_ret;
        }
#endif
    }
}

void* platform_heap_base() {
    return linux_heap_base;
}


// Implementation of platform_heap_size(). Returns committed page count.
size_t platform_heap_size() {
    return committed_pages * PLATFORM_WASM_PAGE_SIZE;
}

static inline uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}

// Implementation of platform_heap_grow(). Commits pages using `mprotect`.
void* platform_heap_grow(size_t num_bytes) {
    size_t num_pages = align(num_bytes, PLATFORM_WASM_PAGE_SIZE) / PLATFORM_WASM_PAGE_SIZE;
    if (linux_heap_base == NULL) {
        return NULL;
    }

    size_t new_total_pages = committed_pages + num_pages;
    if ((new_total_pages * PLATFORM_WASM_PAGE_SIZE) > RESERVED_SIZE) {
        return NULL; // Cannot grow beyond reserved size
    }

    void *region = linux_heap_base + (committed_pages * PLATFORM_WASM_PAGE_SIZE);
    size_t region_len = num_pages * PLATFORM_WASM_PAGE_SIZE;
#ifdef PLATFORM_HOSTED
    if (mprotect(region, region_len, PROT_READ | PROT_WRITE) != 0) {
        return NULL;
    }
#else
    long ret = syscall(
        SYS_MPROTECT,
        (long)region,
        (long)region_len,
        (long)(PROT_READ | PROT_WRITE),
        0, 0, 0
    );
    if (ret != 0) {
        return NULL;
    }
#endif

    void* old_top = linux_heap_base + (committed_pages * PLATFORM_WASM_PAGE_SIZE);
    committed_pages = new_total_pages;
    return old_top;
}

// Math functions using x86_64 SSE instructions
double fast_sqrt(double x) {
    double result;
    __asm__("sqrtsd %1, %0" : "=x"(result) : "x"(x));
    return result;
}

float fast_sqrtf(float x) {
    float result;
    __asm__("sqrtss %1, %0" : "=x"(result) : "x"(x));
    return result;
}

// Public initialization function for hosts that provide their own entry
// point (PLATFORM_SKIP_ENTRY); the default _start path below calls this
// itself.
void platform_init(int argc, char** argv) {
    stored_argc = argc;
    stored_argv = argv;
    ensure_heap_initialized();
    buddy_init();
}

// Linux open() flags
#define O_RDONLY   0x0000
#define O_WRONLY   0x0001
#define O_RDWR     0x0002
#define O_CREAT    0x0040  // Octal 0100 = 0x40
#define O_TRUNC    0x0200  // Octal 01000 = 0x200

// File I/O implementations
platform_fd_t platform_path_open(const char* path, size_t path_len, uint64_t rights, int oflags) {
    // Extract access mode from rights
    int os_flags = 0;
    int has_read = (rights & PLATFORM_RIGHT_FD_READ) != 0;
    int has_write = (rights & PLATFORM_RIGHT_FD_WRITE) != 0;

    if (has_read && has_write) {
        os_flags |= O_RDWR;
    } else if (has_write) {
        os_flags |= O_WRONLY;
    } else {
        os_flags |= O_RDONLY;
    }

    // Map oflags to Linux creation flags
    if (oflags & PLATFORM_O_CREAT) os_flags |= O_CREAT;
    if (oflags & PLATFORM_O_TRUNC) os_flags |= O_TRUNC;

    // Use openat syscall with AT_FDCWD (current directory)
    // Default mode for created files (0644)
    long result = syscall(SYS_OPENAT, (long)AT_FDCWD, (long)path, (long)os_flags, (long)0644, 0, 0);

    // On Linux, open() can return 0, 1, or 2 if those file descriptors were closed.
    // This would collide with stdin/stdout/stderr. To prevent this, we use fcntl()
    // with F_DUPFD to find the lowest available FD >= 3, then close the original low FD.
    // See: https://man7.org/linux/man-pages/man2/open.2.html
    // "The file descriptor returned by a successful call will be the lowest-numbered
    //  file descriptor not currently open for the process."
    // See: https://man7.org/linux/man-pages/man2/fcntl.2.html
    // "F_DUPFD: Find the lowest numbered available file descriptor greater than or
    //  equal to arg and make it be a copy of fd."
    if (result >= 0 && result <= PLATFORM_STDERR_FD) {
        // Use fcntl with F_DUPFD to get lowest available FD >= 3
        long new_fd = syscall(SYS_FCNTL, result, (long)F_DUPFD, (long)3, 0, 0, 0);
        if (new_fd < 0) {
            // fcntl failed, close the original FD and return error
            syscall(SYS_CLOSE, result, 0, 0, 0, 0, 0);
            return -1;
        }
        // Verify new_fd is not a reserved FD (should be > 2)
        if (new_fd <= PLATFORM_STDERR_FD) {
            // This should never happen, but if it does, close both and fail
            syscall(SYS_CLOSE, new_fd, 0, 0, 0, 0, 0);
            syscall(SYS_CLOSE, result, 0, 0, 0, 0, 0);
            return -1;
        }
        syscall(SYS_CLOSE, result, 0, 0, 0, 0, 0);
        result = new_fd;
    }

    return (platform_fd_t)result;
}

int platform_fd_close(platform_fd_t fd) {
    long result = syscall(SYS_CLOSE, (long)fd, 0, 0, 0, 0, 0);
    return (result < 0) ? (int)(-result) : 0;
}

int platform_fd_read(platform_fd_t fd, const iovec_t* iovs, size_t iovs_len, size_t* nread) {
    long result = syscall(SYS_READV, (long)fd, (long)iovs, (long)iovs_len, 0, 0, 0);
    if (result < 0) {
        *nread = 0;
        return (int)(-result);  // Return errno
    }
    *nread = (size_t)result;
    return 0;  // Success
}

int platform_fd_seek(platform_fd_t fd, int64_t offset, int whence, uint64_t* newoffset) {
    long result = syscall(SYS_LSEEK, (long)fd, (long)offset, (long)whence, 0, 0, 0);
    if (result < 0) {
        *newoffset = 0;
        return (int)(-result);  // Return errno
    }
    *newoffset = (uint64_t)result;
    return 0;  // Success
}

int platform_fd_tell(platform_fd_t fd, uint64_t* offset) {
    long result = syscall(SYS_LSEEK, (long)fd, 0, (long)PLATFORM_SEEK_CUR, 0, 0, 0);
    if (result < 0) {
        *offset = 0;
        return (int)(-result);  // Return errno
    }
    *offset = (uint64_t)result;
    return 0;  // Success
}

// Command line arguments implementation
int platform_args_sizes_get(size_t* argc, size_t* argv_buf_size) {
    *argc = (size_t)stored_argc;

    // Calculate total buffer size needed
    size_t total_size = 0;
    for (int i = 0; i < stored_argc; i++) {
        const char* arg = stored_argv[i];
        while (*arg++) total_size++;  // strlen
        total_size++;  // null terminator
    }
    *argv_buf_size = total_size;
    return 0;
}

bool platform_read_file_mmap(const char *filename, uint64_t *out_handle, void **out_data, size_t *out_size) {
    if (!filename || !out_handle || !out_data || !out_size) return false;
    *out_handle = 0;
    *out_data = NULL;
    *out_size = 0;

    long fd = syscall(SYS_OPENAT, (long)AT_FDCWD, (long)filename, (long)O_RDONLY, 0, 0, 0);
    if (fd < 0) {
        return false;
    }

    long end = syscall(SYS_LSEEK, fd, 0, PLATFORM_SEEK_END, 0, 0, 0);
    if (end < 0) {
        syscall(SYS_CLOSE, fd, 0, 0, 0, 0, 0);
        return false;
    }
    size_t file_size = (size_t)end;

    // Reset to start
    if (syscall(SYS_LSEEK, fd, 0, PLATFORM_SEEK_SET, 0, 0, 0) < 0) {
        syscall(SYS_CLOSE, fd, 0, 0, 0, 0, 0);
        return false;
    }

    if (file_size == 0) {
        syscall(SYS_CLOSE, fd, 0, 0, 0, 0, 0);
        *out_size = 0;
        return true;
    }

    void* addr = (void*)syscall(
        SYS_MMAP,
        (long)NULL,
        (long)file_size,
        (long)(PROT_READ | PROT_WRITE),
        (long)MAP_PRIVATE,
        (long)fd,
        (long)0
    );
    syscall(SYS_CLOSE, fd, 0, 0, 0, 0, 0);

    if (addr == MAP_FAILED) {
        return false;
    }

    int slot = -1;
    for (int i = 0; i < MMAP_HANDLE_CAP; i++) {
        if (!g_mmap_handles[i].in_use) {
            slot = i;
            break;
        }
    }
    if (slot == -1) {
        syscall(SYS_MUNMAP, (long)addr, (long)file_size, 0, 0, 0, 0);
        return false;
    }

    g_mmap_handles[slot].addr = addr;
    g_mmap_handles[slot].size = file_size;
    g_mmap_handles[slot].in_use = true;

    *out_handle = (uint64_t)(slot + 1);
    *out_data = addr;
    *out_size = file_size;
    return true;
}

void platform_file_unmap(uint64_t handle) {
    if (handle == 0) return;
    uint64_t idx = handle - 1;
    if (idx >= MMAP_HANDLE_CAP) return;
    MmapHandle *h = &g_mmap_handles[idx];
    if (!h->in_use) return;
    if (h->addr && h->size > 0) {
        syscall(SYS_MUNMAP, (long)h->addr, (long)h->size, 0, 0, 0, 0);
    }
    h->addr = NULL;
    h->size = 0;
    h->in_use = false;
}

int platform_args_get(char** argv, char* argv_buf) {
    char* buf_ptr = argv_buf;
    for (int i = 0; i < stored_argc; i++) {
        argv[i] = buf_ptr;
        const char* src = stored_argv[i];
        while (*src) {
            *buf_ptr++ = *src++;
        }
        *buf_ptr++ = '\0';
    }
    return 0;
}

#ifndef PLATFORM_SKIP_ENTRY
// Forward declaration for application entry point (only when platform provides entry)
int app_main();

// Initialize the platform and call the application
static int platform_init_and_run(int argc, char** argv) {
    platform_init(argc, argv);
    int status = app_main();
    return status;
}

// The entry point for a -nostdlib Linux program is `_start`.
// The kernel enters with RSP % 16 == 0, but the ABI requires RSP % 16 == 8
// before a call instruction (so after the call pushes return address, it's aligned).
// We need to ensure proper 16-byte stack alignment for functions using SSE instructions.
// On entry, the stack layout is:
//   rsp+0: argc
//   rsp+8: argv[0]
//   rsp+16: argv[1]
//   ...
__attribute__((naked))
void _start() {
    __asm__ volatile (
        "xor %rbp, %rbp\n"           // Clear frame pointer as per ABI
        "mov (%rsp), %rdi\n"         // argc from stack to first argument (rdi)
        "lea 8(%rsp), %rsi\n"        // argv from stack+8 to second argument (rsi)
        "andq $-16, %rsp\n"          // Align stack to 16 bytes
        "call _start_c\n"            // Call the C portion
        "mov %eax, %edi\n"           // Move return value to exit code
        "mov $60, %eax\n"            // SYS_EXIT
        "syscall\n"                  // Exit
        "hlt\n"                      // Should never reach here
    );
}

// The actual C entry point
int _start_c(int argc, char** argv) {
    return platform_init_and_run(argc, argv);
}
#endif
