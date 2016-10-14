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

#define G(a, b, c, d) \
    do { \
        a = F(a, b); \
        d = rotr64(d ^ a, 32); \
        c = F(c, d); \
        b = rotr64(b ^ c, 24); \
        a = F(a, b); \
        d = rotr64(d ^ a, 16); \
        c = F(c, d); \
        b = rotr64(b ^ c, 63); \
    } while (0)

struct block {
    ulong data[ARGON2_QWORDS_IN_BLOCK];
};

void shuffle_block(__local struct block *block, size_t thread)
{
    /* |  x   x   x  |   x   x   | */
    /* |  subblock   | hash_lane | */
    size_t subblock = (thread >> 2) & 0x7U;
    uint hash_lane = (thread >> 0) & 0x3U;
    uint4 u0123 = (uint4)(0U, 1U, 2U, 3U);
    uint4 index0 = 4 * u0123 + hash_lane;
    uint4 index1 = 4 * u0123 + (u0123 + hash_lane) % 4;

    __local ulong *v;

    v = block->data + 16 * subblock;
    G(v[index0.s0], v[index0.s1], v[index0.s2], v[index0.s3]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(v[index1.s0], v[index1.s1], v[index1.s2], v[index1.s3]);
    barrier(CLK_LOCAL_MEM_FENCE);

    index0 = 16 * (index0 / 2) + index0 % 2;
    index1 = 16 * (index1 / 2) + index1 % 2;

    v = block->data + 2 * subblock;
    G(v[index0.s0], v[index0.s1], v[index0.s2], v[index0.s3]);
    barrier(CLK_LOCAL_MEM_FENCE);
    G(v[index1.s0], v[index1.s1], v[index1.s2], v[index1.s3]);
}

void fill_block(__global const struct block *restrict ref_block,
                __local struct block *restrict prev_block,
                __local struct block *restrict next_block,
                size_t thread)
{
    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->data[pos] = prev_block->data[pos] ^= ref_block->data[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->data[pos] ^= prev_block->data[pos];
    }
}

#if ARGON2_VERSION != ARGON2_VERSION_10
void fill_block_xor(__global const struct block *restrict ref_block,
                    __local struct block *restrict prev_block,
                    __local struct block *restrict next_block,
                    size_t thread)
{
    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->data[pos] ^= prev_block->data[pos] ^= ref_block->data[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        next_block->data[pos] ^= prev_block->data[pos];
    }
}
#endif

#if ARGON2_TYPE == ARGON2_I
void next_addresses(ulong thread_input,
                    __local struct block *restrict addr,
                    __local struct block *restrict tmp,
                    size_t thread)
{
    addr->data[thread] = thread_input;
    for (size_t i = 1; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        addr->data[pos] = 0UL;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    tmp->data[thread] = addr->data[thread] ^= thread_input;
    for (size_t i = 1; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        tmp->data[pos] = addr->data[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        addr->data[pos] ^= tmp->data[pos];
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
        __global struct block *memory, __local struct block *shared,
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

    __local struct block *restrict curr = &shared[0];
    __local struct block *restrict prev = &shared[1];
#if ARGON2_TYPE == ARGON2_I
    __local struct block *restrict addr = &shared[2];

    ulong thread_input;
    switch (thread) {
    case 1:
        thread_input = lane;
        break;
    case 2:
        thread_input = segment_blocks == 2 ? 1UL : 0UL;
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
        thread_input = 0UL;
    }

    if (segment_blocks > 2) {
        if (thread == 6) {
            ++thread_input;
        }
        next_addresses(thread_input, addr, curr, thread);
    }
#endif

    __global struct block *mem_lane = memory + lane * lane_blocks;
    __global struct block *mem_prev = mem_lane + 1;
    __global struct block *mem_curr = mem_lane + 2;

    for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
        size_t pos = i * THREADS_PER_LANE + thread;
        prev->data[pos] = mem_prev->data[pos];
    }
    for (uint pass = 0; pass < passes; ++pass) {
        for (size_t index = pass == 0 ? 2 : 0; index < lane_blocks; ++index) {
            uint offset = index % segment_blocks;
            uint slice = index / segment_blocks;

            ulong pseudo_rand;
#if ARGON2_TYPE == ARGON2_I
            size_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
            if (addr_index == 0) {
                if (thread == 6) {
                    ++thread_input;
                }
                next_addresses(thread_input, addr, curr, thread);
            }
            pseudo_rand = addr->data[addr_index];
#else
            pseudo_rand = prev->data[0];
#endif

            uint ref_lane = (uint)(pseudo_rand >> 32) % lanes;

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

            uint ref_index = (uint)pseudo_rand;
            ref_index = mul_hi(ref_index, ref_index);
            ref_index = ref_area_size - 1 - mul_hi(ref_area_size, ref_index);

            if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
                ref_index += (slice + 1) * segment_blocks;
                ref_index %= lane_blocks;
            }

            __global struct block *mem_ref = memory +
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
                    curr->data[pos] = mem_curr->data[pos];
                }

                fill_block_xor(mem_ref, prev, curr, thread);
            } else {
                fill_block(mem_ref, prev, curr, thread);
            }
#endif

            for (size_t i = 0; i < QWORDS_PER_THREAD; i++) {
                size_t pos = i * THREADS_PER_LANE + thread;
                mem_curr->data[pos] = curr->data[pos];
            }

            /* swap curr and prev buffers: */
            __local struct block *tmp = curr;
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
