using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Helpers;

internal static class MatrixMultiplyHelper
{
    private const long DefaultBlasWorkThreshold = 128L * 128L * 128L;
    private const long DefaultBlockedWorkThreshold = 64L * 64L * 64L;
    private const long DefaultParallelThreshold = 16384;
    private const int DefaultPackedMaxDim = 128;
    private const long DefaultPackedMaxElements = 1_048_576;
    private static readonly int? BlockSizeOverride = ReadEnvInt("AIDOTNET_MATMUL_BLOCK_SIZE");
    private static readonly long? BlasWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLAS_THRESHOLD");
    private static readonly long? BlockedWorkThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_BLOCKED_THRESHOLD");
    private static readonly long? ParallelThresholdOverride = ReadEnvLong("AIDOTNET_MATMUL_PARALLEL_THRESHOLD");
    private static readonly int? PackedMaxDimOverride = ReadEnvInt("AIDOTNET_MATMUL_PACKED_MAX_DIM");
    private static readonly long? PackedMaxElementsOverride = ReadEnvLong("AIDOTNET_MATMUL_PACKED_MAX_ELEMENTS");
    private static readonly bool TraceEnabled = ReadEnvBool("AIDOTNET_MATMUL_TRACE");

    internal static bool TryGemm<T>(ReadOnlyMemory<T> a, int aOffset, ReadOnlyMemory<T> b, int bOffset, Memory<T> c, int cOffset, int m, int k, int n)
    {
        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            return false;
        }

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

    internal static bool TryMultiplyPacked<T>(
        INumericOperations<T> numOps,
        ReadOnlyMemory<T> a,
        MatrixBase<T> bMatrix,
        Memory<T> c,
        int m,
        int k,
        int n)
    {
        if (!ShouldUsePacked<T>(m, k, n))
        {
            return false;
        }

        var packed = PackedMatrixCache<T>.GetOrCreate(bMatrix, k, n, n);
        if (packed == null)
        {
            return false;
        }

        var aSpan = a.Span;
        var cSpan = c.Span;
        T[] packedData = packed.Data;

        for (int i = 0; i < m; i++)
        {
            var aRow = aSpan.Slice(i * k, k);
            int cRowOffset = i * n;
            for (int j = 0; j < n; j++)
            {
                var bCol = new ReadOnlySpan<T>(packedData, j * k, k);
                cSpan[cRowOffset + j] = numOps.Dot(aRow, bCol);
            }
        }

        return true;
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

    internal static void TraceMatmul(string path, int m, int n, int k)
    {
        if (!TraceEnabled)
        {
            return;
        }

        Console.WriteLine($"[MATMUL-TRACE CPU] {m}x{n}x{k} {path}");
    }

    private static bool ShouldUsePacked<T>(int m, int k, int n)
    {
        if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            return false;
        }

        int maxDim = GetPackedMaxDim();
        if (m > maxDim || n > maxDim || k > maxDim)
        {
            return false;
        }

        long packedElements = (long)k * n;
        if (packedElements <= 0)
        {
            return false;
        }

        return packedElements <= GetPackedMaxElements();
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
        if (typeof(T) == typeof(float) && a is float[] af && b is float[] bf && c is float[] cf)
        {
            return BlasProvider.TryGemm(m, n, k, af, aOffset, k, bf, bOffset, n, cf, cOffset, n);
        }
        else if (typeof(T) == typeof(double) && a is double[] ad && b is double[] bd && c is double[] cd)
        {
            return BlasProvider.TryGemm(m, n, k, ad, aOffset, k, bd, bOffset, n, cd, cOffset, n);
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
        int l1 = PlatformDetector.Capabilities?.L1CacheSize ?? 0;
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

    private static int GetPackedMaxDim()
    {
        if (PackedMaxDimOverride.HasValue && PackedMaxDimOverride.Value > 0)
        {
            return Clamp(PackedMaxDimOverride.Value, 8, 512);
        }

        return DefaultPackedMaxDim;
    }

    private static long GetPackedMaxElements()
    {
        if (PackedMaxElementsOverride.HasValue && PackedMaxElementsOverride.Value > 0)
        {
            return PackedMaxElementsOverride.Value;
        }

        return DefaultPackedMaxElements;
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

    private static bool ReadEnvBool(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
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

    private sealed class PackedMatrix<T>
    {
        public PackedMatrix(long version, int k, int n, T[] data)
        {
            Version = version;
            K = k;
            N = n;
            Data = data;
        }

        public long Version { get; }
        public int K { get; }
        public int N { get; }
        public T[] Data { get; }
    }

    private static class PackedMatrixCache<T>
    {
        private static readonly ConditionalWeakTable<MatrixBase<T>, PackedMatrix<T>> Cache = new();
        private static readonly object CacheLock = new object();

        public static PackedMatrix<T>? GetOrCreate(MatrixBase<T> matrix, int k, int n, int stride)
        {
            long version = matrix.Version;
            lock (CacheLock)
            {
                if (Cache.TryGetValue(matrix, out var packed) &&
                    packed.Version == version &&
                    packed.K == k &&
                    packed.N == n)
                {
                    return packed;
                }

                long packedSize = (long)k * n;
                if (packedSize <= 0 || packedSize > int.MaxValue)
                {
                    return null;
                }

                var data = new T[(int)packedSize];
                PackMatrix(matrix.AsMemory().Span, data, k, n, stride);

                packed = new PackedMatrix<T>(version, k, n, data);
                Cache.Remove(matrix);
                Cache.Add(matrix, packed);
                return packed;
            }
        }

        private static void PackMatrix(ReadOnlySpan<T> source, T[] destination, int k, int n, int stride)
        {
            for (int kk = 0; kk < k; kk++)
            {
                int srcOffset = kk * stride;
                int destOffset = kk;
                for (int j = 0; j < n; j++)
                {
                    destination[destOffset] = source[srcOffset + j];
                    destOffset += k;
                }
            }
        }
    }
}
