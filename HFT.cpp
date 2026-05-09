/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║         HIGH FREQUENCY TRADING ENGINE — Ultra Low Latency   ║
 * ║         C++20 | Lock-Free | NUMA-Aware | Kernel Bypass      ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Compile:
 *   g++ -O3 -march=native -mtune=native -std=c++20 \
 *       -fno-exceptions -fno-rtti \
 *       -falign-functions=64 -falign-loops=64 \
 *       -funroll-loops -fprefetch-loop-arrays \
 *       -lrt -lpthread -o hft_engine hft_engine.cpp
 *
 * Run with CPU isolation (recommended):
 *   taskset -c 2,3 numactl --cpunodebind=0 --membind=0 ./hft_engine
 */

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt,lzcnt")

#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <chrono>
#include <immintrin.h>   // AVX2 intrinsics
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <numa.h>        // libnuma — apt install libnuma-dev
#include <x86intrin.h>   // __rdtsc, __rdtscp, _mm_pause
#include <cstdio>
#include <cstdlib>
#include <new>

// ─── Compile-time configuration ──────────────────────────────────────────────
static constexpr int    CORE_MARKET_DATA  = 2;   // CPU core: market data thread
static constexpr int    CORE_STRATEGY     = 3;   // CPU core: strategy thread
static constexpr int    CORE_ORDER        = 4;   // CPU core: order management
static constexpr size_t RING_BUFFER_SIZE  = 1 << 17;   // 131,072 slots (power-of-2!)
static constexpr size_t CACHE_LINE        = 64;
static constexpr int    MAX_ORDER_LEVELS  = 10;
static constexpr double TICK_SIZE         = 0.01;
static constexpr int    INSTRUMENT_COUNT  = 256;

// ─── Branch prediction hints ─────────────────────────────────────────────────
#define LIKELY(x)    __builtin_expect(!!(x), 1)
#define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#define HOT          __attribute__((hot, optimize("O3")))
#define COLD         __attribute__((cold))
#define FORCE_INLINE __attribute__((always_inline)) inline
#define NO_INLINE    __attribute__((noinline))
#define ALIGNED(n)   __attribute__((aligned(n)))
#define PACKED       __attribute__((packed))

// ─── Cacheline-aligned allocator ─────────────────────────────────────────────
template<typename T, size_t Align = CACHE_LINE>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0)
            throw std::bad_alloc{};
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, size_t) noexcept { free(ptr); }
};

// ─── TSC (Time Stamp Counter) — nanosecond-precision clock ───────────────────
// Avoid syscalls: use RDTSC directly (requires invariant TSC on modern CPUs)
struct TSC {
    static uint64_t frequency_hz;

    FORCE_INLINE static uint64_t now() noexcept {
        uint32_t aux;
        return __rdtscp(&aux);  // serialising read
    }

    static void calibrate() noexcept {
        // Measure TSC ticks per second against CLOCK_REALTIME
        auto t0 = now();
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_REALTIME, &ts0);
        // spin ~100ms
        volatile uint64_t dummy = 0;
        for (volatile int i = 0; i < 300'000'000; ++i) dummy += i;
        clock_gettime(CLOCK_REALTIME, &ts1);
        auto t1 = now();
        uint64_t ns = (ts1.tv_sec - ts0.tv_sec) * 1'000'000'000ULL
                    + (ts1.tv_nsec - ts0.tv_nsec);
        frequency_hz = (t1 - t0) * 1'000'000'000ULL / ns;
    }

    FORCE_INLINE static uint64_t to_ns(uint64_t ticks) noexcept {
        return ticks * 1'000'000'000ULL / frequency_hz;
    }
};
uint64_t TSC::frequency_hz = 3'000'000'000ULL; // default 3 GHz, calibrated at runtime

// ─── Price representation — integer fixed-point (avoids FP rounding) ─────────
// 1 unit = 0.01 (2 decimal places). $123.45 → 12345
using Price = int64_t;
using Qty   = int32_t;
using Side  = uint8_t;

static constexpr Side BID = 0;
static constexpr Side ASK = 1;

FORCE_INLINE Price double_to_price(double d) noexcept {
    return static_cast<Price>(d * 100.0 + 0.5);
}
FORCE_INLINE double price_to_double(Price p) noexcept {
    return static_cast<double>(p) * 0.01;
}

// ─── Market data tick ─────────────────────────────────────────────────────────
struct alignas(CACHE_LINE) Tick {
    uint64_t tsc_recv;       // TSC at reception
    uint32_t instrument_id;
    Price    bid_prices[MAX_ORDER_LEVELS];
    Price    ask_prices[MAX_ORDER_LEVELS];
    Qty      bid_qtys  [MAX_ORDER_LEVELS];
    Qty      ask_qtys  [MAX_ORDER_LEVELS];
    uint8_t  levels;
    uint8_t  _pad[7];
};
static_assert(sizeof(Tick) % CACHE_LINE == 0, "Tick must be cacheline-aligned");

// ─── Order ────────────────────────────────────────────────────────────────────
struct alignas(32) Order {
    uint64_t order_id;
    uint64_t tsc_sent;
    Price    price;
    Qty      qty;
    uint32_t instrument_id;
    Side     side;
    uint8_t  type;      // 0=limit 1=market 2=cancel
    uint8_t  _pad[2];
};

// ─── Lock-free SPSC ring buffer (Single-Producer Single-Consumer) ─────────────
// Zero-allocation, zero-lock hot path. Uses cache-line padding to avoid
// false sharing between producer and consumer head/tail indices.
template<typename T, size_t N>
struct alignas(CACHE_LINE) SPSCQueue {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    static constexpr size_t MASK = N - 1;

    struct alignas(CACHE_LINE) { std::atomic<size_t> v{0}; } head;
    struct alignas(CACHE_LINE) { std::atomic<size_t> v{0}; } tail;

    alignas(CACHE_LINE) std::array<T, N> buffer;

    // Producer side — called from a single thread only
    FORCE_INLINE bool try_push(const T& item) noexcept {
        const size_t h = head.v.load(std::memory_order_relaxed);
        const size_t next = (h + 1) & MASK;
        if (UNLIKELY(next == tail.v.load(std::memory_order_acquire)))
            return false;  // full
        buffer[h] = item;
        head.v.store(next, std::memory_order_release);
        return true;
    }

    // Consumer side — called from a single thread only
    FORCE_INLINE bool try_pop(T& item) noexcept {
        const size_t t = tail.v.load(std::memory_order_relaxed);
        if (UNLIKELY(t == head.v.load(std::memory_order_acquire)))
            return false;  // empty
        item = buffer[t];
        tail.v.store((t + 1) & MASK, std::memory_order_release);
        return true;
    }

    FORCE_INLINE size_t size() const noexcept {
        return (head.v.load(std::memory_order_relaxed) -
                tail.v.load(std::memory_order_relaxed)) & MASK;
    }
};

// ─── Order Book (per instrument) ─────────────────────────────────────────────
// Maintains top-N price levels. Update is O(levels) — acceptable for L2 books.
struct alignas(CACHE_LINE) OrderBook {
    Price bid_prices[MAX_ORDER_LEVELS] = {};
    Price ask_prices[MAX_ORDER_LEVELS] = {};
    Qty   bid_qtys  [MAX_ORDER_LEVELS] = {};
    Qty   ask_qtys  [MAX_ORDER_LEVELS] = {};
    uint8_t levels = 0;
    uint32_t instrument_id = 0;
    uint64_t last_update_tsc = 0;

    HOT void apply(const Tick& t) noexcept {
        levels = t.levels;
        last_update_tsc = t.tsc_recv;
        const size_t bytes = levels * sizeof(Price);
        std::memcpy(bid_prices, t.bid_prices, bytes);
        std::memcpy(ask_prices, t.ask_prices, bytes);
        std::memcpy(bid_qtys,   t.bid_qtys,   levels * sizeof(Qty));
        std::memcpy(ask_qtys,   t.ask_qtys,   levels * sizeof(Qty));
    }

    FORCE_INLINE Price best_bid() const noexcept { return bid_prices[0]; }
    FORCE_INLINE Price best_ask() const noexcept { return ask_prices[0]; }
    FORCE_INLINE Price mid_price() const noexcept {
        return (bid_prices[0] + ask_prices[0]) >> 1;
    }
    FORCE_INLINE Price spread() const noexcept {
        return ask_prices[0] - bid_prices[0];
    }
    FORCE_INLINE bool is_valid() const noexcept {
        return levels > 0 && bid_prices[0] > 0 && ask_prices[0] > 0
            && ask_prices[0] > bid_prices[0];
    }
};

// ─── Signal type ──────────────────────────────────────────────────────────────
enum class Signal : uint8_t { NONE=0, BUY=1, SELL=2, CLOSE=3 };

// ─── Strategy state (per instrument) ─────────────────────────────────────────
struct alignas(CACHE_LINE) StrategyState {
    // EMA parameters (integer arithmetic, multiplied by 2^16 for precision)
    static constexpr int64_t EMA_SCALE  = 1 << 16;
    static constexpr int64_t ALPHA_FAST = static_cast<int64_t>(0.1 * EMA_SCALE); // ~10-period EMA
    static constexpr int64_t ALPHA_SLOW = static_cast<int64_t>(0.02 * EMA_SCALE); // ~50-period EMA

    int64_t ema_fast    = 0;
    int64_t ema_slow    = 0;
    int64_t vol_sum_sq  = 0;  // for volatility
    Price   prev_mid    = 0;
    int32_t position    = 0;
    int32_t tick_count  = 0;
    bool    initialized = false;

    // Maximum absolute position
    static constexpr int32_t MAX_POS = 1000;

    HOT Signal on_tick(const OrderBook& book) noexcept {
        if (UNLIKELY(!book.is_valid())) return Signal::NONE;

        const Price mid = book.mid_price();
        const int64_t mid_scaled = static_cast<int64_t>(mid) * EMA_SCALE;

        if (UNLIKELY(!initialized)) {
            ema_fast = mid_scaled;
            ema_slow = mid_scaled;
            prev_mid = mid;
            initialized = true;
            return Signal::NONE;
        }

        // EMA update: ema = ema + alpha * (x - ema) — integer approximation
        ema_fast += (ALPHA_FAST * (mid_scaled - ema_fast)) >> 16;
        ema_slow += (ALPHA_SLOW * (mid_scaled - ema_slow)) >> 16;

        // Volatility proxy: sum of |Δmid|
        const int64_t delta = static_cast<int64_t>(mid - prev_mid);
        vol_sum_sq = (vol_sum_sq * 127 + delta * delta) >> 7; // decaying average
        prev_mid = mid;
        ++tick_count;

        // Suppress signals during high volatility (avoid being picked off)
        const int64_t vol_threshold = 10LL * EMA_SCALE;  // ~10 ticks
        if (vol_sum_sq > vol_threshold * vol_threshold)
            return Signal::NONE;

        // Require minimum spread (1 tick) to avoid toxic flow
        if (book.spread() < double_to_price(TICK_SIZE))
            return Signal::NONE;

        // Crossover signal
        const int64_t diff = ema_fast - ema_slow;
        const int64_t threshold = 2 * EMA_SCALE; // ~$0.02 crossover band

        if (diff > threshold && position < MAX_POS)  return Signal::BUY;
        if (diff < -threshold && position > -MAX_POS) return Signal::SELL;
        if (position > 0 && diff < 0)                return Signal::CLOSE;
        if (position < 0 && diff > 0)                return Signal::CLOSE;

        return Signal::NONE;
    }
};

// ─── Order ID generator — lock-free monotonic counter ────────────────────────
struct OrderIDGen {
    alignas(CACHE_LINE) std::atomic<uint64_t> counter{1};
    FORCE_INLINE uint64_t next() noexcept {
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
};

// ─── Latency statistics — ring buffer of measurements ────────────────────────
struct LatencyStats {
    static constexpr size_t N = 1 << 20; // 1M samples
    alignas(CACHE_LINE) std::atomic<size_t> idx{0};
    uint32_t samples[N]; // nanoseconds, fits in uint32 (up to ~4.29s)

    void record(uint64_t tsc_start, uint64_t tsc_end) noexcept {
        uint64_t ns = TSC::to_ns(tsc_end - tsc_start);
        size_t i = idx.fetch_add(1, std::memory_order_relaxed) & (N - 1);
        samples[i] = static_cast<uint32_t>(ns < 0xFFFFFFFF ? ns : 0xFFFFFFFF);
    }

    NO_INLINE void print_percentiles() const noexcept {
        size_t count = std::min(idx.load(std::memory_order_relaxed), N);
        if (count == 0) { printf("No samples.\n"); return; }
        // Copy & sort
        uint32_t* tmp = new uint32_t[count];
        std::memcpy(tmp, samples, count * sizeof(uint32_t));
        // Insertion sort for small counts, otherwise partial selection
        // (full sort is fine here — this is reporting, not hot path)
        std::sort(tmp, tmp + count);
        printf("┌─────────────────────────────────────────┐\n");
        printf("│   Tick→Signal Latency  (ns)  n=%-7zu  │\n", count);
        printf("├─────────────────┬───────────────────────┤\n");
        printf("│ P50             │ %21u │\n", tmp[count * 50 / 100]);
        printf("│ P90             │ %21u │\n", tmp[count * 90 / 100]);
        printf("│ P99             │ %21u │\n", tmp[count * 99 / 100]);
        printf("│ P99.9           │ %21u │\n", tmp[count * 999 / 1000]);
        printf("│ P99.99          │ %21u │\n", tmp[count * 9999 / 10000]);
        printf("│ Max             │ %21u │\n", tmp[count - 1]);
        printf("└─────────────────┴───────────────────────┘\n");
        delete[] tmp;
    }
};

// ─── HFT Engine — top-level orchestrator ─────────────────────────────────────
class HFTEngine {
public:
    // Queues between threads
    SPSCQueue<Tick,  RING_BUFFER_SIZE> md_queue;    // market data → strategy
    SPSCQueue<Order, RING_BUFFER_SIZE> order_queue; // strategy → OMS

    alignas(CACHE_LINE) OrderBook  books[INSTRUMENT_COUNT];
    alignas(CACHE_LINE) StrategyState strategies[INSTRUMENT_COUNT];
    OrderIDGen id_gen;
    LatencyStats latency;

    std::atomic<bool> running{true};
    uint64_t ticks_processed = 0;
    uint64_t signals_generated = 0;
    uint64_t orders_sent = 0;

    // ── Market-data simulation (replaces real feed handler / kernel-bypass NIC)
    // In production: use DPDK / Solarflare / Exanic for kernel-bypass UDP/TCP
    HOT void market_data_thread() noexcept {
        pin_to_core(CORE_MARKET_DATA);
        set_realtime_priority(99);
        prefault_stack();

        uint64_t seq = 0;
        while (LIKELY(running.load(std::memory_order_relaxed))) {
            // --- In production this is replaced by: ---
            // ef_vi_receive_init() / recvmmsg() / DPDK rte_eth_rx_burst()
            Tick t;
            t.tsc_recv     = TSC::now();
            t.instrument_id = seq % INSTRUMENT_COUNT;
            t.levels        = MAX_ORDER_LEVELS;

            // Simulate a realistic bid/ask ladder
            Price base = double_to_price(100.0 + (seq % 200) * 0.01);
            for (int i = 0; i < MAX_ORDER_LEVELS; ++i) {
                t.bid_prices[i] = base - i * double_to_price(TICK_SIZE);
                t.ask_prices[i] = base + (i + 1) * double_to_price(TICK_SIZE);
                t.bid_qtys[i]   = 100 * (i + 1);
                t.ask_qtys[i]   = 100 * (i + 1);
            }

            // Spin-wait (not sleep!) if queue is full — preserves timing
            while (UNLIKELY(!md_queue.try_push(t))) {
                _mm_pause(); // x86 PAUSE: reduces power + pipeline hazards
            }
            ++seq;
        }
    }

    // ── Strategy thread — tick → signal → order, target < 500ns end-to-end
    HOT void strategy_thread() noexcept {
        pin_to_core(CORE_STRATEGY);
        set_realtime_priority(99);
        prefault_stack();

        Tick t;
        while (LIKELY(running.load(std::memory_order_relaxed))) {
            // Busy-poll — no condition variables, no mutexes, no syscalls
            if (UNLIKELY(!md_queue.try_pop(t))) {
                _mm_pause();
                continue;
            }

            const uint64_t t_signal_start = TSC::now();

            const uint32_t id = t.instrument_id;
            OrderBook& book   = books[id];
            book.apply(t);
            ++ticks_processed;

            StrategyState& st = strategies[id];
            Signal sig = st.on_tick(book);

            if (LIKELY(sig == Signal::NONE)) continue;
            ++signals_generated;

            // Build order
            Order o;
            o.order_id      = id_gen.next();
            o.instrument_id = id;
            o.tsc_sent      = TSC::now();
            o.qty           = 100;
            o.type          = 0; // limit order

            switch (sig) {
                case Signal::BUY:
                    o.side  = BID;
                    o.price = book.best_bid() + double_to_price(TICK_SIZE); // post at BBO+1 tick
                    st.position += o.qty;
                    break;
                case Signal::SELL:
                    o.side  = ASK;
                    o.price = book.best_ask() - double_to_price(TICK_SIZE);
                    st.position -= o.qty;
                    break;
                case Signal::CLOSE:
                    o.side  = (st.position > 0) ? ASK : BID;
                    o.price = (st.position > 0) ? book.best_ask() : book.best_bid();
                    o.qty   = std::abs(st.position);
                    st.position = 0;
                    break;
                default: continue;
            }

            while (UNLIKELY(!order_queue.try_push(o))) _mm_pause();

            const uint64_t t_signal_end = TSC::now();
            latency.record(t.tsc_recv, t_signal_end); // end-to-end: receive → signal
        }
    }

    // ── Order Management System — send to exchange gateway
    // In production: FIX/OUCH/ITCH over Solarflare / kernel-bypass TCP
    HOT void oms_thread() noexcept {
        pin_to_core(CORE_ORDER);
        set_realtime_priority(98);
        prefault_stack();

        Order o;
        while (LIKELY(running.load(std::memory_order_relaxed))) {
            if (UNLIKELY(!order_queue.try_pop(o))) {
                _mm_pause();
                continue;
            }
            ++orders_sent;

            // Simulate wire serialisation (replace with actual FIX/binary codec)
            char buf[128];
            int len = snprintf(buf, sizeof(buf),
                "OID=%lu INS=%u SIDE=%c PX=%.2f QTY=%d TYPE=%u",
                o.order_id, o.instrument_id,
                (o.side == BID) ? 'B' : 'S',
                price_to_double(o.price), o.qty, o.type);
            (void)len; // in production: send via sendmsg() / ef_vi_transmit()
        }
    }

    // ── Thread utilities ──────────────────────────────────────────────────────
    static void pin_to_core(int core) noexcept {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    static void set_realtime_priority(int prio) noexcept {
        sched_param sp{.sched_priority = prio};
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    }

    // Pre-fault stack pages to avoid page-fault latency on hot path
    static void prefault_stack() noexcept {
        volatile char buf[65536];
        std::memset(const_cast<char*>(buf), 0, sizeof(buf));
    }

    // Lock all current+future memory pages — prevent swapping
    static void lock_memory() noexcept {
        mlockall(MCL_CURRENT | MCL_FUTURE);
    }

    // Disable CPU frequency scaling for deterministic latency
    static void set_performance_governor() noexcept {
        // Run as root: echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
        // This is a reminder — cannot do from userspace without root
    }

    void print_stats() const noexcept {
        printf("\n");
        printf("══════════════════════════════════════════\n");
        printf("  HFT Engine Statistics\n");
        printf("══════════════════════════════════════════\n");
        printf("  Ticks processed : %lu\n", ticks_processed);
        printf("  Signals generated: %lu\n", signals_generated);
        printf("  Orders sent      : %lu\n", orders_sent);
        printf("  Signal ratio     : %.4f%%\n",
               ticks_processed > 0 ? 100.0 * signals_generated / ticks_processed : 0.0);
        printf("\n");
        latency.print_percentiles();
    }
};

// ─── Huge-page backed allocator for the engine ───────────────────────────────
// Reduces TLB misses on hot data structures
void* alloc_huge(size_t size) noexcept {
    void* ptr = mmap(nullptr, size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                     -1, 0);
    if (ptr == MAP_FAILED) {
        // Fallback to regular pages
        ptr = mmap(nullptr, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    }
    return ptr;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  HFT Engine — Ultra-Low Latency C++20   ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // System setup
    HFTEngine::lock_memory();
    TSC::calibrate();
    printf("[init] TSC frequency: %.3f GHz\n", TSC::frequency_hz / 1e9);

    // Allocate engine on huge pages if possible
    void* mem = alloc_huge(sizeof(HFTEngine));
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }
    HFTEngine* engine = new(mem) HFTEngine();

    // Spawn threads
    pthread_t t_md, t_strat, t_oms;
    pthread_create(&t_md,    nullptr, [](void* e) -> void* {
        static_cast<HFTEngine*>(e)->market_data_thread(); return nullptr; }, engine);
    pthread_create(&t_strat, nullptr, [](void* e) -> void* {
        static_cast<HFTEngine*>(e)->strategy_thread();    return nullptr; }, engine);
    pthread_create(&t_oms,   nullptr, [](void* e) -> void* {
        static_cast<HFTEngine*>(e)->oms_thread();         return nullptr; }, engine);

    printf("[init] Threads started. Running for 5 seconds...\n\n");

    // Run for 5 seconds then stop
    struct timespec ts{.tv_sec = 5, .tv_nsec = 0};
    nanosleep(&ts, nullptr);

    engine->running.store(false, std::memory_order_seq_cst);

    pthread_join(t_md,    nullptr);
    pthread_join(t_strat, nullptr);
    pthread_join(t_oms,   nullptr);

    engine->print_stats();

    engine->~HFTEngine();
    munmap(mem, sizeof(HFTEngine));
    return 0;
}