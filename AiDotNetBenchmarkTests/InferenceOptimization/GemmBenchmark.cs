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
        private Tensor<float> _matrixA = null!;
        private Tensor<float> _matrixB = null!;
        private GemmKernel _gemmKernel = null!;

        [Params(64, 128, 256, 512, 1024)]
        public int MatrixSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            OptimizationInitializer.Initialize(enableProfiling: false);

            _gemmKernel = new GemmKernel();

            // Initialize matrices with deterministic data (avoids security hotspot noise in analysis)
            var dataA = new float[MatrixSize * MatrixSize];
            var dataB = new float[MatrixSize * MatrixSize];

            for (int i = 0; i < dataA.Length; i++)
            {
                dataA[i] = DeterministicValue(i);
            }

            for (int i = 0; i < dataB.Length; i++)
            {
                dataB[i] = DeterministicValue(i + 1_000_000);
            }

            _matrixA = new Tensor<float>(dataA, new[] { MatrixSize, MatrixSize });
            _matrixB = new Tensor<float>(dataB, new[] { MatrixSize, MatrixSize });
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
            var resultData = new float[MatrixSize * MatrixSize];
            var matrixAData = _matrixA.ToArray();
            var matrixBData = _matrixB.ToArray();

            for (int i = 0; i < MatrixSize; i++)
            {
                for (int j = 0; j < MatrixSize; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < MatrixSize; k++)
                    {
                        sum += matrixAData[i * MatrixSize + k] * matrixBData[k * MatrixSize + j];
                    }
                    resultData[i * MatrixSize + j] = sum;
                }
            }

            return new Tensor<float>(resultData, new[] { MatrixSize, MatrixSize });
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
