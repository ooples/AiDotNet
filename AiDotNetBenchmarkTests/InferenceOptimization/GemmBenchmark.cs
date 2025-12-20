using System;
using AiDotNet.InferenceOptimization;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.InferenceOptimization
{
    /// <summary>
    /// Benchmarks for GEMM (General Matrix Multiplication) kernel
    /// Tests optimized implementation against naive implementation
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net80)]
    [MemoryDiagnoser]
    [CsvExporter]
    [HtmlExporter]
    public class GemmBenchmark
    {
        private Tensor<float> _matrixA;
        private Tensor<float> _matrixB;
        private GemmKernel _gemmKernel;

        [Params(64, 128, 256, 512, 1024)]
        public int MatrixSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            OptimizationInitializer.Initialize(enableProfiling: false);

            _gemmKernel = new GemmKernel();

            // Initialize matrices with deterministic data (avoids security hotspot noise in analysis)
            _matrixA = new Tensor<float>(new[] { MatrixSize, MatrixSize });
            _matrixB = new Tensor<float>(new[] { MatrixSize, MatrixSize });

            for (int i = 0; i < _matrixA.Data.Length; i++)
            {
                _matrixA.Data[i] = DeterministicValue(i);
            }

            for (int i = 0; i < _matrixB.Data.Length; i++)
            {
                _matrixB.Data[i] = DeterministicValue(i + 1_000_000);
            }
        }

        private static float DeterministicValue(int i)
        {
            unchecked
            {
                uint x = (uint)(i * 1664525 + 1013904223);
                return (x & 0x00FFFFFF) / 16777216f;
            }
        }

        [Benchmark(Baseline = true)]
        public Tensor<float> NaiveGemm()
        {
            // Naive triple-nested loop implementation
            var result = new Tensor<float>(new[] { MatrixSize, MatrixSize });

            for (int i = 0; i < MatrixSize; i++)
            {
                for (int j = 0; j < MatrixSize; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < MatrixSize; k++)
                    {
                        sum += _matrixA.Data[i * MatrixSize + k] * _matrixB.Data[k * MatrixSize + j];
                    }
                    result.Data[i * MatrixSize + j] = sum;
                }
            }

            return result;
        }

        [Benchmark]
        public Tensor<float> OptimizedGemm()
        {
            return _gemmKernel.Execute(_matrixA, _matrixB);
        }

        [Benchmark]
        public Tensor<float> OptimizedGemmTranspose()
        {
            return _gemmKernel.GemmTransposeB(_matrixA, _matrixB);
        }
    }
}
