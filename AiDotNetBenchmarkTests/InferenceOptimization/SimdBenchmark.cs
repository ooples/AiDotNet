using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.InferenceOptimization;
using AiDotNet.Tensors.Engines.Simd;
using System;

namespace AiDotNetBenchmarkTests.InferenceOptimization
{
    /// <summary>
    /// Benchmarks for SIMD-optimized operations
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net80)]
    [MemoryDiagnoser]
    [CsvExporter]
    [HtmlExporter]
    public class SimdBenchmark
    {
        private float[] _arrayA;
        private float[] _arrayB;
        private float[] _result;

        [Params(1000, 10000, 100000, 1000000)]
        public int ArraySize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            OptimizationInitializer.Initialize(enableProfiling: false);

            var random = new Random(42);
            _arrayA = new float[ArraySize];
            _arrayB = new float[ArraySize];
            _result = new float[ArraySize];

            for (int i = 0; i < ArraySize; i++)
            {
                _arrayA[i] = (float)random.NextDouble();
                _arrayB[i] = (float)random.NextDouble();
            }
        }

        #region Vector Addition

        [Benchmark(Baseline = true)]
        public void VectorAdd_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = _arrayA[i] + _arrayB[i];
            }
        }

        [Benchmark]
        public unsafe void VectorAdd_SIMD()
        {
            fixed (float* pA = _arrayA, pB = _arrayB, pR = _result)
            {
                SimdKernels.VectorAdd(pA, pB, pR, ArraySize);
            }
        }

        #endregion

        #region Vector Multiplication

        [Benchmark]
        public void VectorMultiply_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = _arrayA[i] * _arrayB[i];
            }
        }

        [Benchmark]
        public unsafe void VectorMultiply_SIMD()
        {
            fixed (float* pA = _arrayA, pB = _arrayB, pR = _result)
            {
                SimdKernels.VectorMultiply(pA, pB, pR, ArraySize);
            }
        }

        #endregion

        #region Dot Product

        [Benchmark]
        public float DotProduct_Scalar()
        {
            float sum = 0.0f;
            for (int i = 0; i < ArraySize; i++)
            {
                sum += _arrayA[i] * _arrayB[i];
            }
            return sum;
        }

        [Benchmark]
        public unsafe float DotProduct_SIMD()
        {
            fixed (float* pA = _arrayA, pB = _arrayB)
            {
                return SimdKernels.DotProduct(pA, pB, ArraySize);
            }
        }

        #endregion

        #region ReLU Activation

        [Benchmark]
        public void ReLU_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = Math.Max(0.0f, _arrayA[i]);
            }
        }

        [Benchmark]
        public unsafe void ReLU_SIMD()
        {
            fixed (float* pA = _arrayA, pR = _result)
            {
                SimdKernels.ReLU(pA, pR, ArraySize);
            }
        }

        #endregion

        #region Sum Reduction

        [Benchmark]
        public float Sum_Scalar()
        {
            float sum = 0.0f;
            for (int i = 0; i < ArraySize; i++)
            {
                sum += _arrayA[i];
            }
            return sum;
        }

        [Benchmark]
        public unsafe float Sum_SIMD()
        {
            fixed (float* pA = _arrayA)
            {
                return SimdKernels.Sum(pA, ArraySize);
            }
        }

        #endregion
    }
}
