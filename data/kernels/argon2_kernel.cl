/* C compatibility For dumb IDEs: */
#ifndef __OPENCL_VERSION__
#ifndef __cplusplus
typedef int bool;
#endif
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef size_t uintptr_t;
typedef ptrdiff_t intptr_t;
#ifndef __kernel
#define __kernel
#endif
#ifndef __global
#define __global
#endif
#ifndef __private
#define __private
#endif
#ifndef __local
#define __local
#endif
#ifndef __constant
#define __constant const
#endif
#endif /* __OPENCL_VERSION__ */

#define ARGON2_D 0
#define ARGON2_I 1

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

#ifndef ARGON2_VERSION
#define ARGON2_VERSION ARGON2_VERSION_13
#endif

#ifndef ARGON2_TYPE
#define ARGON2_TYPE ARGON2_I
#endif

/*
 * BLAMKA ROUND:
 *  G is sequential (atomic)
 *  4 G's in parallel (2 passes)
 *
 * FILL BLOCK:
 *  1. some moving and xoring...
 *  2. 8 x BLAMKA ROUND in parallel
 *  3. 8 x BLAMKA ROUND in parallel (on diff. distributed data)
 *  4. some moving and xoring again...
 *
 * Conclusion:
 *  BLAMKA ROUND == 4 threads
 *  FILL BLOCK == 32 threads
 */

#define MASK_32 0xFFFFFFFFUL

#define F(x, y) ((x) + (y) + 2 * upsample( \
    mul_hi((uint)(x), (uint)(y)), \
    (uint)(x) * (uint)(y) \
    ))

#define rotr64(x, n) rotate(x, (ulong)(64 - (n)))

void g(__local uint *v_lo, __local uint *v_hi, uint4 i)
{
    ulong a, b, c, d;
    a = upsample(v_hi[i.s0], v_lo[i.s0]);
    b = upsample(v_hi[i.s1], v_lo[i.s1]);
    c = upsample(v_hi[i.s2], v_lo[i.s2]);
    d = upsample(v_hi[i.s3], v_lo[i.s3]);

    a = F(a, b);
    d = rotr64(d ^ a, 32);
    c = F(c, d);
    b = rotr64(b ^ c, 24);
    a = F(a, b);
    d = rotr64(d ^ a, 16);
    c = F(c, d);
    b = rotr64(b ^ c, 63);

    v_lo[i.s0] = (uint)a;
    v_lo[i.s1] = (uint)b;
    v_lo[i.s2] = (uint)c;
    v_lo[i.s3] = (uint)d;

    v_hi[i.s0] = (uint)(a >> 32);
    v_hi[i.s1] = (uint)(b >> 32);
    v_hi[i.s2] = (uint)(c >> 32);
    v_hi[i.s3] = (uint)(d >> 32);
}

struct block_g {
    ulong data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_l {
    uint lo[ARGON2_QWORDS_IN_BLOCK];
    uint hi[ARGON2_QWORDS_IN_BLOCK];
};

void shuffle_block(__local struct block_l *block, size_t thread)
{
    /* |  x   x   x  |   x   x   | */
    /* |  subblock   | hash_lane | */
    size_t subblock = (thread >> 2) & 0x7U;
    uint hash_lane = (thread >> 0) & 0x3U;
    uint4 u0123 = (uint4)(0U, 1U, 2U, 3U);
    uint4 index0 = 4 * u0123 + hash_lane;
    uint4 index1 = 4 * u0123 + (u0123 + hash_lane) % 4;

    __local ulong *v_lo, *v_hi;

    g(block->lo + 16 * subblock, block->hi + 16 * subblock, index0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g(block->lo + 16 * subblock, block->hi + 16 * subblock, index1);

    barrier(CLK_LOCAL_MEM_FENCE);

    index0 = 16 * (index0 / 2) + index0 % 2;
    index1 = 16 * (index1 / 2) + index1 % 2;

    g(block->lo + 2 * subblock, block->hi + 2 * subblock, index0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g(block->lo + 2 * subblock, block->hi + 2 * subblock, index1);
}

void fill_block(__global const struct block_g *restrict ref_block,
                __local struct block_l *restrict prev_block,
                __local struct block_l *restrict next_block,
                size_t thread)
{
    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        ulong in = ref_block->data[pos];
        next_block->lo[pos] = prev_block->lo[pos] ^= (uint)in;
        next_block->hi[pos] = prev_block->hi[pos] ^= (uint)(in >> 32);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->lo[pos] ^= prev_block->lo[pos];
        next_block->hi[pos] ^= prev_block->hi[pos];
    }
}

#if ARGON2_VERSION != ARGON2_VERSION_10
void fill_block_xor(__global const struct block_g *restrict ref_block,
                    __local struct block_l *restrict prev_block,
                    __local struct block_l *restrict next_block,
                    size_t thread)
{
    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        ulong in = ref_block->data[pos];
        next_block->lo[pos] ^= prev_block->lo[pos] ^= (uint)in;
        next_block->hi[pos] ^= prev_block->hi[pos] ^= (uint)(in >> 32);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->lo[pos] ^= prev_block->lo[pos];
        next_block->hi[pos] ^= prev_block->hi[pos];
    }
}
#endif

#if ARGON2_TYPE == ARGON2_I
void next_addresses(uint thread_input,
                    __local struct block_l *restrict addr,
                    __local struct block_l *restrict tmp,
                    size_t thread)
{
    addr->lo[thread] = thread_input;
    addr->hi[thread] = 0;
    for (size_t i = 1; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        addr->hi[pos] = addr->lo[pos] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    tmp->lo[thread] = addr->lo[thread] ^= thread_input;
    tmp->hi[thread] = addr->hi[thread];
    for (size_t i = 1; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        tmp->lo[pos] = addr->lo[pos];
        tmp->hi[pos] = addr->hi[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        addr->lo[pos] ^= tmp->lo[pos];
        addr->hi[pos] ^= tmp->hi[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

/* 1 fill_segment call == throughput of 1 fill_block call */
/* 1 fill_block call == 64 threads */
/* CONCLUSION: we need 64 threads per fill_segment (i. e. per lane) */

#if ARGON2_TYPE == ARGON2_I
#define SHARED_BLOCKS 3
#else
#define SHARED_BLOCKS 2
#endif
/*
 * how to map thread ids to position/sub-tasks?
 *
 * GLOBAL ID == (job, lane, thread)
 * GROUP SIZE == (1, lanes, 32)
 */
__kernel void argon2_kernel(
        __global struct block_g *memory, __local struct block_l *shared,
        uint passes, uint lanes, uint segment_blocks)
{
    size_t job_id = get_global_id(0);
    size_t lane = get_global_id(1);
    size_t thread = get_global_id(2);

    size_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select lane's shared memory buffer: */
    shared += lane * SHARED_BLOCKS;

    __local struct block_l *restrict curr = &shared[0];
    __local struct block_l *restrict prev = &shared[1];
#if ARGON2_TYPE == ARGON2_I
    __local struct block_l *restrict addr = &shared[2];

    uint thread_input;
    switch (thread) {
    case 1:
        thread_input = lane;
        break;
    case 2:
        thread_input = segment_blocks == 2 ? 1 : 0;
        break;
    case 3:
        thread_input = lanes * lane_blocks;
        break;
    case 4:
        thread_input = passes;
        break;
    case 5:
        thread_input = ARGON2_I;
        break;
    default:
        thread_input = 0;
    }

    if (segment_blocks > 2) {
        if (thread == 6) {
            ++thread_input;
        }
        next_addresses(thread_input, addr, curr, thread);
    }
#endif

    __global struct block_g *mem_lane = memory + lane * lane_blocks;
    __global struct block_g *mem_prev = mem_lane + 1;
    __global struct block_g *mem_curr = mem_lane + 2;

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        ulong in = mem_prev->data[pos];
        prev->lo[pos] = (uint)in;
        prev->hi[pos] = (uint)(in >> 32);
    }
    for (uint pass = 0; pass < passes; ++pass) {
        for (size_t index = pass == 0 ? 2 : 0; index < lane_blocks; ++index) {
            uint offset = index % segment_blocks;
            uint slice = index / segment_blocks;

            uint pseudo_rand_lo, pseudo_rand_hi;
#if ARGON2_TYPE == ARGON2_I
            size_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
            if (addr_index == 0) {
                if (thread == 6) {
                    ++thread_input;
                }
                next_addresses(thread_input, addr, curr, thread);
            }
            pseudo_rand_lo = addr->lo[addr_index];
            pseudo_rand_hi = addr->hi[addr_index];
#else
            pseudo_rand_lo = prev->lo[0];
            pseudo_rand_hi = prev->hi[0];
#endif

            uint ref_lane = pseudo_rand_hi % lanes;

            uint base;
            if (pass != 0) {
                base = lane_blocks - segment_blocks;
            } else {
                if (slice == 0) {
                    ref_lane = lane;
                }
                base = slice * segment_blocks;
            }

            uint ref_area_size = base + offset - 1;
            if (ref_lane != lane) {
                ref_area_size = min(ref_area_size, base);
            }

            uint ref_index = pseudo_rand_lo;
            ref_index = mul_hi(ref_index, ref_index);
            ref_index = ref_area_size - 1 - mul_hi(ref_area_size, ref_index);

            if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
                ref_index += (slice + 1) * segment_blocks;
                ref_index %= lane_blocks;
            }

            __global struct block_g *mem_ref = memory +
                    ref_lane * lane_blocks + ref_index;

            /* NOTE: no need to wrap fill_block in barriers, since
             * it starts & ends in 'nicely parallel' memory operations
             * like we do in this loop (IOW: this thread only depends on
             * its own data w.r.t. these boundaries) */
#if ARGON2_VERSION == ARGON2_VERSION_10
            fill_block(mem_ref, prev, curr, thread);
#else
            if (pass != 0) {
                for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
                    size_t pos = i * THREADS_PER_LANE + thread;
                    ulong in = mem_curr->data[pos];
                    curr->lo[pos] = (uint)in;
                    curr->hi[pos] = (uint)(in >> 32);
                }

                fill_block_xor(mem_ref, prev, curr, thread);
            } else {
                fill_block(mem_ref, prev, curr, thread);
            }
#endif

            for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
                size_t pos = i * THREADS_PER_LANE + thread;
                mem_curr->data[pos] = upsample(curr->hi[pos], curr->lo[pos]);
            }

            /* swap curr and prev buffers: */
            __local struct block_l *tmp = curr;
            curr = prev;
            prev = tmp;

            if ((index + 1) % segment_blocks == 0) {
                barrier(CLK_GLOBAL_MEM_FENCE);
                if (thread == 2) {
                    ++thread_input;
                }
                if (thread == 6) {
                    thread_input = 0UL;
                }
            }

            ++mem_curr;
        }
        if (thread == 0) {
            ++thread_input;
        }
        if (thread == 2) {
            thread_input = 0UL;
        }
        mem_curr = mem_lane;
    }
}
