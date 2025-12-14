using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.InferenceOptimization.CpuOptimization
{
    /// <summary>
    /// Provides CPU cache optimization utilities including prefetching and cache-aware algorithms
    /// </summary>
    public static class CacheOptimizer
    {
        /// <summary>
        /// Gets the optimal block size for the L1 cache
        /// </summary>
        public static int L1BlockSize => 64; // 64 floats = 256 bytes, typical L1 cache line

        /// <summary>
        /// Gets the optimal block size for the L2 cache
        /// </summary>
        public static int L2BlockSize => 512; // Tuned for typical L2 cache

        /// <summary>
        /// Gets the optimal block size for the L3 cache
        /// </summary>
        public static int L3BlockSize => 2048; // Tuned for typical L3 cache

        /// <summary>
        /// Prefetch data for reading (temporal locality)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Prefetch(void* address)
        {
            // This hints the CPU to fetch data into cache
            // Note: .NET JIT may or may not honor this depending on platform
            System.Runtime.Intrinsics.X86.Sse.Prefetch0(address);
        }

        /// <summary>
        /// Prefetch data with low temporal locality (won't pollute cache)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void PrefetchNonTemporal(void* address)
        {
            System.Runtime.Intrinsics.X86.Sse.PrefetchNonTemporal(address);
        }

        /// <summary>
        /// Computes optimal tiling parameters for a 2D operation
        /// </summary>
        public static (int tileM, int tileN, int tileK) ComputeOptimalTiling(
            int m, int n, int k,
            int elementSize = 4) // 4 bytes for float
        {
            var caps = PlatformDetector.Capabilities;
            int l1Size = caps.L1CacheSize;
            int l2Size = caps.L2CacheSize;

            // We want tiles to fit in L1 cache
            // For matrix multiplication: tileM * tileK + tileK * tileN + tileM * tileN elements
            // Simplified: aim for sqrt(L1Size / (3 * elementSize)) per dimension

            int maxTileSize = (int)Math.Sqrt(l1Size / (3.0 * elementSize));

            // Round down to nearest power of 2 for better memory alignment
            int tileSize = 1;
            while (tileSize * 2 <= maxTileSize)
            {
                tileSize *= 2;
            }

            // Ensure minimum tile size
            tileSize = Math.Max(tileSize, 16);

            // Adjust based on actual matrix dimensions
            int tileM = Math.Min(tileSize, m);
            int tileN = Math.Min(tileSize, n);
            int tileK = Math.Min(tileSize, k);

            return (tileM, tileN, tileK);
        }

        /// <summary>
        /// Cache-aware transpose of a 2D array
        /// </summary>
        public static unsafe void TransposeBlocked(float* src, float* dst, int rows, int cols)
        {
            const int blockSize = 32; // Tuned for cache line size

            for (int i = 0; i < rows; i += blockSize)
            {
                for (int j = 0; j < cols; j += blockSize)
                {
                    int iMax = Math.Min(i + blockSize, rows);
                    int jMax = Math.Min(j + blockSize, cols);

                    // Transpose block
                    for (int ii = i; ii < iMax; ii++)
                    {
                        for (int jj = j; jj < jMax; jj++)
                        {
                            dst[jj * rows + ii] = src[ii * cols + jj];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Cache-aware copying with prefetching
        /// </summary>
        public static unsafe void CopyWithPrefetch(float* src, float* dst, int length)
        {
            const int prefetchDistance = 64; // Prefetch 64 elements ahead

            int i = 0;

            // Main loop with prefetching
            for (; i < length - prefetchDistance; i++)
            {
                Prefetch(src + i + prefetchDistance);
                dst[i] = src[i];
            }

            // Remaining elements without prefetch
            for (; i < length; i++)
            {
                dst[i] = src[i];
            }
        }

        /// <summary>
        /// Z-order (Morton order) indexing for better cache locality in 2D access patterns
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int MortonEncode(int x, int y)
        {
            return (Part1By1(y) << 1) | Part1By1(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Part1By1(int n)
        {
            n &= 0x0000ffff;
            n = (n ^ (n << 8)) & 0x00ff00ff;
            n = (n ^ (n << 4)) & 0x0f0f0f0f;
            n = (n ^ (n << 2)) & 0x33333333;
            n = (n ^ (n << 1)) & 0x55555555;
            return n;
        }

        /// <summary>
        /// Converts Z-order index back to 2D coordinates
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (int x, int y) MortonDecode(int code)
        {
            return (Compact1By1(code), Compact1By1(code >> 1));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Compact1By1(int n)
        {
            n &= 0x55555555;
            n = (n ^ (n >> 1)) & 0x33333333;
            n = (n ^ (n >> 2)) & 0x0f0f0f0f;
            n = (n ^ (n >> 4)) & 0x00ff00ff;
            n = (n ^ (n >> 8)) & 0x0000ffff;
            return n;
        }

        /// <summary>
        /// Estimates the number of cache misses for a given access pattern
        /// </summary>
        public static double EstimateCacheMisses(int dataSize, int accessStride, int cacheSize, int cacheLineSize)
        {
            // Simple cache miss estimation model
            int elementsPerLine = cacheLineSize / sizeof(float);
            int totalLines = (dataSize + elementsPerLine - 1) / elementsPerLine;
            int cacheLinesAvailable = cacheSize / cacheLineSize;

            if (accessStride <= elementsPerLine)
            {
                // Sequential access - good cache behavior
                return totalLines * 0.1; // ~10% miss rate for sequential
            }
            else if (totalLines <= cacheLinesAvailable)
            {
                // Data fits in cache
                return totalLines * 0.05; // ~5% miss rate
            }
            else
            {
                // Poor cache behavior - strided access with cache thrashing
                return totalLines * 0.8; // ~80% miss rate
            }
        }
    }
}
