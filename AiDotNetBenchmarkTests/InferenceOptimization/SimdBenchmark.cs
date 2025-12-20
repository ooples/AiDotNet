using System;
using AiDotNet.InferenceOptimization;
using AiDotNet.Tensors.Engines.Simd;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.InferenceOptimization
{
    /// <summary>
    /// Benchmarks for SIMD-optimized operations
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net80)]
    [MemoryDiagnoser]
    [CsvExporter]
    [HtmlExporter]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
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

            _arrayA = new float[ArraySize];
            _arrayB = new float[ArraySize];
            _result = new float[ArraySize];

            for (int i = 0; i < ArraySize; i++)
            {
                _arrayA[i] = DeterministicValue(i);
                _arrayB[i] = DeterministicValue(i + 1_000_000);
            }
        }

        #region Vector Addition

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("VectorAdd")]
        public void VectorAdd_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = _arrayA[i] + _arrayB[i];
            }
        }

        [Benchmark]
        [BenchmarkCategory("VectorAdd")]
        public void VectorAdd_SIMD()
        {
            SimdKernels.VectorAdd(_arrayA, _arrayB, _result);
        }

        #endregion

        #region Vector Multiplication

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("VectorMultiply")]
        public void VectorMultiply_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = _arrayA[i] * _arrayB[i];
            }
        }

        [Benchmark]
        [BenchmarkCategory("VectorMultiply")]
        public void VectorMultiply_SIMD()
        {
            SimdKernels.VectorMultiply(_arrayA, _arrayB, _result);
        }

        #endregion

        #region Dot Product

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("DotProduct")]
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
        [BenchmarkCategory("DotProduct")]
        public float DotProduct_SIMD()
        {
            return SimdKernels.DotProduct(_arrayA, _arrayB);
        }

        #endregion

        #region ReLU Activation

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("ReLU")]
        public void ReLU_Scalar()
        {
            for (int i = 0; i < ArraySize; i++)
            {
                _result[i] = Math.Max(0.0f, _arrayA[i]);
            }
        }

        [Benchmark]
        [BenchmarkCategory("ReLU")]
        public void ReLU_SIMD()
        {
            SimdKernels.ReLU(_arrayA, _result);
        }

        #endregion

        #region Sum Reduction

        [Benchmark(Baseline = true)]
        [BenchmarkCategory("Sum")]
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
        [BenchmarkCategory("Sum")]
        public float Sum_SIMD()
        {
            return SimdKernels.Sum(_arrayA);
        }

        #endregion

        private static float DeterministicValue(int i)
        {
            unchecked
            {
                uint x = (uint)(i * 1664525 + 1013904223);
                return (x & 0x00FFFFFF) / 16777216f;
            }
        }
    }
}
