// =============================================================================
// platform_wasm.c — WebAssembly (WASI) backend for the Core C platform layer.
// =============================================================================
//
// This file is the wasm-target counterpart of platform_linux.c,
// platform_macos.c, and platform_windows.c. The other three backends call
// directly into the host operating system through compiler/syscall
// mechanisms. WASI is different: it is itself a portable system interface
// that sits between this `.wasm` and whatever embeds it. So this file is
// thin — most of it is just forwarding the platform.h API onto the
// matching wasi_snapshot_preview1 imports declared below:
//
//     fd_write, fd_read, fd_close, fd_seek, fd_tell,
//     path_open,
//     args_sizes_get, args_get,
//     proc_exit
//
// "Imports" here is the WASM term: these symbols are not defined in the
// `.wasm` module. The runtime that loads the module must supply them.
// We use exactly nine of them; everything else (including memory) is
// intrinsic via __builtin_wasm_memory_size / __builtin_wasm_memory_grow.
//
// What runs the produced .wasm?
//
//   * `wasmtime` (used by `pixi run -e wasm test_wasm`) — a standalone
//     WASI runtime that talks directly to the OS.
//   * Node.js, via examples/js/run_node.js.
//   * Any modern browser, via examples/js/index.html.
//
// The two JS hosts share platform/js/wasi.js, which is the JavaScript
// counterpart of *this* file — it implements the same nine WASI imports
// in pure JS and exposes a host-friendly API that run_node.js and
// index.html plug into. See platform/js/wasi.js for details.
// =============================================================================

#include <platform/platform.h>
#include <base/types.h>
#include <base/buddy.h>

#define WASI(name) __attribute__((__import_module__("wasi_snapshot_preview1"), __import_name__(#name))) name

uint32_t WASI(fd_write)(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten);
void WASI(proc_exit)(int status);
int WASI(path_open)(int dirfd, int dirflags, const char* path, size_t path_len, int oflags, uint64_t fs_rights_base, uint64_t fs_rights_inheriting, int fdflags, int* fd);
int WASI(fd_close)(int fd);
int WASI(fd_read)(int fd, const iovec_t* iovs, size_t iovs_len, size_t* nread);
int WASI(fd_seek)(int fd, int64_t offset, int whence, uint64_t* newoffset);
int WASI(fd_tell)(int fd, uint64_t* offset);
int WASI(args_sizes_get)(size_t* argc, size_t* argv_buf_size);
int WASI(args_get)(char** argv, char* argv_buf);

#undef WASI

// =============================================================================
// == WebAssembly (WASI) Implementation
// =============================================================================

// __heap_base is a special symbol provided by the wasm-ld linker. It marks
// the end of the static data section and the beginning of the linear memory
// heap that we can manage. It is declared as an external variable.
//extern uint8_t __heap_base;

// Wrapper around the `memory.size` WASM instruction.
// The argument `0` is required for the current memory space.
// Returns the pointer to the last allocated byte plus one.
size_t platform_heap_size() {
    return PLATFORM_WASM_PAGE_SIZE * __builtin_wasm_memory_size(0)
        - (size_t)platform_heap_base();
}

static inline uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}

// Wrapper around the `memory.grow` WASM instruction.
// Attempts to grow the linear memory by `num_pages`.
// Returns the previous size in pages on success, or -1 on failure.
void* platform_heap_grow(size_t num_bytes) {
    size_t num_pages = align(num_bytes, PLATFORM_WASM_PAGE_SIZE) / PLATFORM_WASM_PAGE_SIZE;
    size_t prev_size = __builtin_wasm_memory_grow(0, num_pages);
    if (prev_size == (size_t)(-1)) {
        return NULL;
    }
    return (void*)(prev_size * PLATFORM_WASM_PAGE_SIZE);
}


extern uint8_t* __heap_base;

void* platform_heap_base() {
    return &__heap_base;
}

void platform_exit(int status) {
    proc_exit(status);
}

uint32_t platform_fd_write(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten) {
    return fd_write(fd, iovs, iovs_len, nwritten);
}

// File I/O implementations
platform_fd_t platform_path_open(const char* path, size_t path_len, uint64_t rights, int oflags) {
    // WASI requires path_open to be called with a directory fd (use 3 for preopen)
    // We simplify by using the preopen directory
    int fd = -1;

    // Both rights and oflags are passed through directly (no translation needed)
    int ret = path_open(
        3,           // dirfd (preopen)
        0,           // dirflags
        path,
        path_len,
        oflags,      // passed through directly
        rights,      // passed through directly
        0,           // inheriting rights
        0,           // fdflags
        &fd
    );

    // Note: The WASI specification reserves file descriptors 0, 1, and 2 for
    // stdin, stdout, and stderr. The path_open function will never return these
    // values, so there's no collision risk with standard streams.
    // See: https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md
    // "File descriptors 0, 1, and 2 are always reserved for stdin, stdout, and stderr."
    return (ret == 0) ? fd : -1;
}

int platform_fd_close(platform_fd_t fd) {
    return fd_close(fd);
}

int platform_fd_read(platform_fd_t fd, const iovec_t* iovs, size_t iovs_len, size_t* nread) {
    return fd_read(fd, iovs, iovs_len, nread);
}

int platform_fd_seek(platform_fd_t fd, int64_t offset, int whence, uint64_t* newoffset) {
    return fd_seek(fd, offset, whence, newoffset);
}

int platform_fd_tell(platform_fd_t fd, uint64_t* offset) {
    return fd_tell(fd, offset);
}

// Command line arguments implementation
int platform_args_sizes_get(size_t* argc, size_t* argv_buf_size) {
    return args_sizes_get(argc, argv_buf_size);
}

int platform_args_get(char** argv, char* argv_buf) {
    return args_get(argv, argv_buf);
}

void ensure_heap_initialized() {
}

// Math functions using WASM builtins
double fast_sqrt(double x) {
    return __builtin_sqrt(x);
}

float fast_sqrtf(float x) {
    return __builtin_sqrtf(x);
}

bool platform_read_file_mmap(const char *filename, uint64_t *out_handle, void **out_data, size_t *out_size) {
    (void)filename;
    if (out_handle) *out_handle = 0;
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;
    return false;
}

void platform_file_unmap(uint64_t handle) {
    (void)handle;
}

// Buddy allocator hooks exported to the JS host. The JS counterpart
// (platform/js/wasi.js and the example hosts) can use these to allocate
// memory inside this module's linear memory before calling into the
// `.wasm`. Native backends do not need this — they share the host's
// address space with the embedder.
__attribute__((export_name("wasm_buddy_alloc")))
void *wasm_buddy_alloc(size_t size) {
    return buddy_alloc(size, NULL);
}

__attribute__((export_name("wasm_buddy_free")))
void wasm_buddy_free(void *ptr) {
    buddy_free(ptr);
}

// Public initialization function for hosts that provide their own entry
// point (PLATFORM_SKIP_ENTRY); the default _start path below calls this
// itself.
void platform_init(int argc, char** argv) {
    buddy_init();
}

#ifndef PLATFORM_SKIP_ENTRY
// Forward declaration for application entry point (only when platform provides entry)
int app_main();

// Initialize the platform and call the application
static void platform_init_and_run() {
    platform_init(0, NULL);  // WASM doesn't receive argc/argv in _start
    int status = app_main();
    platform_exit(status);
}

// For WASI, the entry point is `_start`
void _start() {
    platform_init_and_run();
}
#endif
