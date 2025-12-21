using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.InferenceOptimization.Kernels
{
    /// <summary>
    /// Optimized General Matrix Multiplication (GEMM) kernel
    /// Implements cache-aware blocked matrix multiplication with SIMD
    /// </summary>
    public class GemmKernel : ICustomOperator<float>
    {
        private const int BlockSize = 64; // Tuned for typical L1 cache
        private const int MinParallelSize = 256; // Minimum size for parallel execution

        public string Name => "GEMM";
        public string Version => "1.0.0";
        public int Priority => 100;

        public bool IsSupported()
        {
            // GEMM is always supported, but performance varies by platform
            return true;
        }

        public double EstimatedSpeedup()
        {
            var caps = PlatformDetector.Capabilities;
            if (caps.HasAVX2) return 3.0;
            if (caps.HasSSE42) return 2.0;
            if (caps.HasNeon) return 2.5;
            return 1.5;
        }

        public Tensor<float> Execute(params Tensor<float>[] inputs)
        {
            if (inputs == null || inputs.Length < 2)
                throw new ArgumentException("GEMM requires at least 2 input tensors");

            var a = inputs[0];
            var b = inputs[1];

            if (a.Shape.Length != 2 || b.Shape.Length != 2)
                throw new ArgumentException("GEMM requires 2D tensors (matrices)");

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            if (k != b.Shape[0])
                throw new ArgumentException($"Matrix dimensions incompatible: ({m}x{k}) * ({b.Shape[0]}x{n})");

            var result = new Tensor<float>(new[] { m, n });

            // Choose strategy based on matrix size
            if (m * n * k < MinParallelSize * MinParallelSize)
            {
                GemmBlocked(a.Data, b.Data, result.Data, m, n, k);
            }
            else
            {
                GemmParallel(a.Data, b.Data, result.Data, m, n, k);
            }

            return result;
        }

        /// <summary>
        /// Cache-blocked GEMM implementation
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GemmBlocked(float[] A, float[] B, float[] C, int M, int N, int K)
        {
            // Blocked algorithm for cache efficiency
            for (int i = 0; i < M; i += BlockSize)
            {
                int iMax = Math.Min(i + BlockSize, M);

                for (int j = 0; j < N; j += BlockSize)
                {
                    int jMax = Math.Min(j + BlockSize, N);
                    int spanLen = jMax - j;

                    for (int k = 0; k < K; k += BlockSize)
                    {
                        int kMax = Math.Min(k + BlockSize, K);

                        // Process block
                        for (int ii = i; ii < iMax; ii++)
                        {
                            for (int kk = k; kk < kMax; kk++)
                            {
                                float aVal = A[ii * K + kk];
                                var bRow = B.AsSpan(kk * N + j, spanLen);
                                var cRow = C.AsSpan(ii * N + j, spanLen);

                                // SIMD-optimized inner loop: cRow = cRow + aVal * bRow
                                SimdKernels.ScalarMultiplyAdd(cRow, bRow, aVal, cRow);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Parallel GEMM implementation for large matrices
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GemmParallel(float[] A, float[] B, float[] C, int M, int N, int K)
        {
            // Parallelize over rows of A
            Parallel.For(0, (M + BlockSize - 1) / BlockSize, iBlock =>
            {
                int i = iBlock * BlockSize;
                int iMax = Math.Min(i + BlockSize, M);

                for (int j = 0; j < N; j += BlockSize)
                {
                    int jMax = Math.Min(j + BlockSize, N);
                    int spanLen = jMax - j;

                    for (int k = 0; k < K; k += BlockSize)
                    {
                        int kMax = Math.Min(k + BlockSize, K);

                        for (int ii = i; ii < iMax; ii++)
                        {
                            for (int kk = k; kk < kMax; kk++)
                            {
                                float aVal = A[ii * K + kk];
                                var bRow = B.AsSpan(kk * N + j, spanLen);
                                var cRow = C.AsSpan(ii * N + j, spanLen);

                                SimdKernels.ScalarMultiplyAdd(cRow, bRow, aVal, cRow);
                            }
                        }
                    }
                }
            });
        }

        /// <summary>
        /// Matrix multiplication with transpose B optimization (C = A * B^T)
        /// </summary>
        public Tensor<float> GemmTransposeB(Tensor<float> a, Tensor<float> b)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2)
                throw new ArgumentException("GemmTransposeB requires 2D tensors");

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[0]; // Note: B is transposed

            if (k != b.Shape[1])
                throw new ArgumentException("Matrix dimensions incompatible for transpose");

            var result = new Tensor<float>(new[] { m, n });

            Parallel.For(0, m, i =>
            {
                var rowA = a.Data.AsSpan(i * k, k);
                for (int j = 0; j < n; j++)
                {
                    var rowB = b.Data.AsSpan(j * k, k);
                    result.Data[i * n + j] = SimdKernels.DotProduct(rowA, rowB);
                }
            });

            return result;
        }
    }
}
