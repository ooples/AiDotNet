using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Helpers;

internal static class MatrixMultiplyHelper
{
    private const long DefaultBlasWorkThreshold = 128L * 128L * 128L;
    private const long DefaultBlockedWorkThreshold = 64L * 64L * 64L;
    private const long DefaultParallelThreshold = 16384;
    private static readonly int? BlockSizeOverride = ReadEnvInt("AIDOTNET_MATMUL_BLOCK_SIZE");
    private static readonly long? BlasWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLAS_THRESHOLD");
    private static readonly long? BlockedWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLOCKED_THRESHOLD");
    private static readonly long? ParallelThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_PARALLEL_THRESHOLD");

    internal static bool TryGemm<T>(ReadOnlyMemory<T> a, int aOffset, ReadOnlyMemory<T> b, int bOffset, Memory<T> c, int cOffset, int m, int k, int n)
    {
        long work = (long)m * k * n;
        if (work < GetBlasWorkThreshold())
        {
            return false;
        }

        long aLength = (long)m * k;
        long bLength = (long)k * n;
        long cLength = (long)m * n;
        if (aLength <= 0 || bLength <= 0 || cLength <= 0 ||
            aLength > int.MaxValue || bLength > int.MaxValue || cLength > int.MaxValue)
        {
            return false;
        }

        if (aOffset < 0 || bOffset < 0 || cOffset < 0 ||
            a.Length < aOffset + aLength ||
            b.Length < bOffset + bLength ||
            c.Length < cOffset + cLength)
        {
            return false;
        }

        bool copyBack = false;

        T[] aArray;
        int aStart;
        if (TryGetArraySegment(a, out var aSegment, out var aBaseOffset))
        {
            aArray = aSegment;
            aStart = aBaseOffset + aOffset;
        }
        else
        {
            aArray = a.Span.Slice(aOffset, (int)aLength).ToArray();
            aStart = 0;
        }

        T[] bArray;
        int bStart;
        if (TryGetArraySegment(b, out var bSegment, out var bBaseOffset))
        {
            bArray = bSegment;
            bStart = bBaseOffset + bOffset;
        }
        else
        {
            bArray = b.Span.Slice(bOffset, (int)bLength).ToArray();
            bStart = 0;
        }

        T[] cArray;
        int cStart;
        if (TryGetArraySegment(c, out var cSegment, out var cBaseOffset))
        {
            cArray = cSegment;
            cStart = cBaseOffset + cOffset;
        }
        else
        {
            cArray = new T[(int)cLength];
            cStart = 0;
            copyBack = true;
        }

        bool used = TryGemmFromArray(aArray, aStart, bArray, bStart, cArray, cStart, m, k, n);
        if (used && copyBack)
        {
            cArray.AsSpan(0, (int)cLength).CopyTo(c.Span.Slice(cOffset, (int)cLength));
        }

        return used;
    }

    internal static bool ShouldUseBlocked<T>(int m, int k, int n)
    {
        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            return false;
        }

        long work = (long)m * k * n;
        return work >= GetBlockedWorkThreshold();
    }

    internal static void MultiplyBlocked<T>(
        INumericOperations<T> numOps,
        ReadOnlyMemory<T> a,
        ReadOnlyMemory<T> b,
        Memory<T> c,
        int m,
        int k,
        int n,
        int aStride,
        int bStride,
        int cStride,
        int aOffset = 0,
        int bOffset = 0,
        int cOffset = 0,
        bool allowParallel = true)
    {
        int block = GetBlockSize<T>();
        int numRowBlocks = (m + block - 1) / block;
        bool parallel = allowParallel &&
            (long)m * n >= GetParallelThreshold() &&
            Environment.ProcessorCount > 1;

        Action<int> multiplyBlock = iiBlock =>
        {
            int iStart = iiBlock * block;
            int iEnd = Math.Min(iStart + block, m);
            var aSpan = a.Span;
            var bSpan = b.Span;
            var cSpan = c.Span;

            for (int kk = 0; kk < k; kk += block)
            {
                int kLen = Math.Min(block, k - kk);
                for (int jj = 0; jj < n; jj += block)
                {
                    int nLen = Math.Min(block, n - jj);
                    for (int i = iStart; i < iEnd; i++)
                    {
                        int aRowOffset = aOffset + (i * aStride) + kk;
                        int cRowOffset = cOffset + (i * cStride) + jj;

                        for (int kIndex = 0; kIndex < kLen; kIndex++)
                        {
                            T aik = aSpan[aRowOffset + kIndex];
                            int bRowOffset = bOffset + ((kk + kIndex) * bStride) + jj;
                            var cBlock = cSpan.Slice(cRowOffset, nLen);
                            var bBlock = bSpan.Slice(bRowOffset, nLen);
                            numOps.MultiplyAdd(cBlock, bBlock, aik, cBlock);
                        }
                    }
                }
            }
        };

        if (parallel)
        {
            Parallel.For(0, numRowBlocks, multiplyBlock);
            return;
        }

        for (int iiBlock = 0; iiBlock < numRowBlocks; iiBlock++)
        {
            multiplyBlock(iiBlock);
        }
    }

    private static bool TryGetArraySegment<T>(ReadOnlyMemory<T> memory, out T[] array, out int offset)
    {
        if (MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array != null)
        {
            array = segment.Array;
            offset = segment.Offset;
            return true;
        }

        array = Array.Empty<T>();
        offset = 0;
        return false;
    }

    private static bool TryGetArraySegment<T>(Memory<T> memory, out T[] array, out int offset)
        => TryGetArraySegment((ReadOnlyMemory<T>)memory, out array, out offset);

    private static bool TryGemmFromArray<T>(T[] a, int aOffset, T[] b, int bOffset, T[] c, int cOffset, int m, int k, int n)
    {
        if (typeof(T) == typeof(float))
        {
            if (a is float[] af && b is float[] bf && c is float[] cf)
            {
                return BlasProvider.TryGemm(m, n, k, af, aOffset, k, bf, bOffset, n, cf, cOffset, n);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            if (a is double[] ad && b is double[] bd && c is double[] cd)
            {
                return BlasProvider.TryGemm(m, n, k, ad, aOffset, k, bd, bOffset, n, cd, cOffset, n);
            }
        }

        return false;
    }

    private static int GetBlockSize<T>()
    {
        if (BlockSizeOverride.HasValue)
        {
            return Clamp(BlockSizeOverride.Value, 16, 128);
        }

        int elementSize = typeof(T) == typeof(double) ? 8 : 4;
        int l1 = PlatformDetector.Capabilities.L1CacheSize;
        if (l1 <= 0)
        {
            l1 = 32 * 1024;
        }

        int block = (int)Math.Sqrt(l1 / (2.0 * elementSize));
        if (block < 16)
        {
            block = 16;
        }
        else if (block > 128)
        {
            block = 128;
        }

        return block;
    }

    private static long GetBlasWorkThreshold()
    {
        if (BlasWorkThresholdOverride.HasValue && BlasWorkThresholdOverride.Value > 0)
        {
            return BlasWorkThresholdOverride.Value;
        }

        return DefaultBlasWorkThreshold;
    }

    private static long GetBlockedWorkThreshold()
    {
        if (BlockedWorkThresholdOverride.HasValue && BlockedWorkThresholdOverride.Value > 0)
        {
            return BlockedWorkThresholdOverride.Value;
        }

        return DefaultBlockedWorkThreshold;
    }

    private static long GetParallelThreshold()
    {
        if (ParallelThresholdOverride.HasValue && ParallelThresholdOverride.Value > 0)
        {
            return ParallelThresholdOverride.Value;
        }

        return DefaultParallelThreshold;
    }

    private static int Clamp(int value, int min, int max)
    {
        if (value < min)
        {
            return min;
        }

        return value > max ? max : value;
    }

    private static int? ReadEnvInt(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return int.TryParse(raw, out var value) && value > 0 ? value : null;
    }

    private static long? ReadEnvLong(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return long.TryParse(raw, out var value) && value > 0 ? value : null;
    }
}
