using System;
using AiDotNet.InferenceOptimization;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.InferenceOptimization
{
    /// <summary>
    /// Benchmarks for fused attention kernel
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net80)]
    [MemoryDiagnoser]
    [CsvExporter]
    [HtmlExporter]
    public class AttentionBenchmark
    {
        private Tensor<float> _q;
        private Tensor<float> _k;
        private Tensor<float> _v;
        private AttentionKernel _attentionKernel;

        [Params(64, 128, 256)]
        public int SequenceLength { get; set; }

        [Params(32, 64)]
        public int FeatureDim { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            OptimizationInitializer.Initialize(enableProfiling: false);

            _attentionKernel = new AttentionKernel();

            // Initialize Q, K, V tensors
            _q = new Tensor<float>(new[] { 1, SequenceLength, FeatureDim });
            _k = new Tensor<float>(new[] { 1, SequenceLength, FeatureDim });
            _v = new Tensor<float>(new[] { 1, SequenceLength, FeatureDim });

            for (int i = 0; i < _q.Data.Length; i++)
            {
                _q.Data[i] = DeterministicValue(i);
            }

            for (int i = 0; i < _k.Data.Length; i++)
            {
                _k.Data[i] = DeterministicValue(i + 1_000_000);
            }

            for (int i = 0; i < _v.Data.Length; i++)
            {
                _v.Data[i] = DeterministicValue(i + 2_000_000);
            }
        }

        private static float DeterministicValue(int i)
        {
            // Stable deterministic value in [0, 1) without PRNG APIs (avoids security hotspot noise in analysis).
            unchecked
            {
                uint x = (uint)(i * 1664525 + 1013904223);
                return (x & 0x00FFFFFF) / 16777216f;
            }
        }

        [Benchmark(Baseline = true)]
        public Tensor<float> NaiveAttention()
        {
            // Naive implementation: QK^T, softmax, multiply by V
            float scale = 1.0f / MathF.Sqrt(FeatureDim);

            // Compute attention scores
            var scores = new float[SequenceLength * SequenceLength];

            for (int i = 0; i < SequenceLength; i++)
            {
                for (int j = 0; j < SequenceLength; j++)
                {
                    float score = 0.0f;
                    for (int k = 0; k < FeatureDim; k++)
                    {
                        score += _q.Data[i * FeatureDim + k] * _k.Data[j * FeatureDim + k];
                    }
                    scores[i * SequenceLength + j] = score * scale;
                }
            }

            // Apply softmax
            for (int i = 0; i < SequenceLength; i++)
            {
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < SequenceLength; j++)
                {
                    if (scores[i * SequenceLength + j] > maxVal)
                        maxVal = scores[i * SequenceLength + j];
                }

                float sum = 0.0f;
                for (int j = 0; j < SequenceLength; j++)
                {
                    scores[i * SequenceLength + j] = MathF.Exp(scores[i * SequenceLength + j] - maxVal);
                    sum += scores[i * SequenceLength + j];
                }

                for (int j = 0; j < SequenceLength; j++)
                {
                    scores[i * SequenceLength + j] /= sum;
                }
            }

            // Multiply by V
            var result = new Tensor<float>(new[] { 1, SequenceLength, FeatureDim });

            for (int i = 0; i < SequenceLength; i++)
            {
                for (int j = 0; j < FeatureDim; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < SequenceLength; k++)
                    {
                        sum += scores[i * SequenceLength + k] * _v.Data[k * FeatureDim + j];
                    }
                    result.Data[i * FeatureDim + j] = sum;
                }
            }

            return result;
        }

        [Benchmark]
        public Tensor<float> OptimizedAttention()
        {
            return _attentionKernel.Execute(_q, _k, _v);
        }

        [Benchmark]
        public Tensor<float> MultiHeadAttention()
        {
            return _attentionKernel.MultiHeadAttention(_q, _k, _v, numHeads: 8);
        }
    }
}
