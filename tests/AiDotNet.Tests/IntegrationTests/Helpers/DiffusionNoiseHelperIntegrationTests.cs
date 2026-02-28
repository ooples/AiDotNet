using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for DiffusionNoiseHelper:
/// SampleGaussian (Tensor and Vector), ScaleNoise, AddNoise,
/// ComputeTimestepEmbeddings, ComputeTimestepEmbedding,
/// ComputeSNR, LerpNoise, SlerpNoise.
/// </summary>
public class DiffusionNoiseHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region SampleGaussian - Tensor

    [Fact]
    public void SampleGaussian_CorrectShape()
    {
        var noise = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 2, 3 }, seed: 42);
        Assert.Equal(new[] { 2, 3 }, noise.Shape);
        Assert.Equal(6, noise.Length);
    }

    [Fact]
    public void SampleGaussian_Seeded_Reproducible()
    {
        var noise1 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 42);
        var noise2 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 42);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(noise1[i], noise2[i], Tolerance);
        }
    }

    [Fact]
    public void SampleGaussian_HasVariance()
    {
        var noise = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 100 }, seed: 42);
        bool hasPositive = false, hasNegative = false;
        for (int i = 0; i < noise.Length; i++)
        {
            if (noise[i] > 0) hasPositive = true;
            if (noise[i] < 0) hasNegative = true;
        }
        Assert.True(hasPositive, "Gaussian noise should have positive values");
        Assert.True(hasNegative, "Gaussian noise should have negative values");
    }

    [Fact]
    public void SampleGaussian_NullShape_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.SampleGaussian(null!, seed: 42));
    }

    [Fact]
    public void SampleGaussian_EmptyShape_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.SampleGaussian(Array.Empty<int>(), seed: 42));
    }

    #endregion

    #region SampleGaussianVector

    [Fact]
    public void SampleGaussianVector_CorrectLength()
    {
        var noise = DiffusionNoiseHelper<double>.SampleGaussianVector(50, seed: 42);
        Assert.Equal(50, noise.Length);
    }

    [Fact]
    public void SampleGaussianVector_Seeded_Reproducible()
    {
        var v1 = DiffusionNoiseHelper<double>.SampleGaussianVector(20, seed: 123);
        var v2 = DiffusionNoiseHelper<double>.SampleGaussianVector(20, seed: 123);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(v1[i], v2[i], Tolerance);
        }
    }

    [Fact]
    public void SampleGaussianVector_ZeroLength_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DiffusionNoiseHelper<double>.SampleGaussianVector(0, seed: 42));
    }

    #endregion

    #region ScaleNoise

    [Fact]
    public void ScaleNoise_DoublesValues()
    {
        var noise = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, -2.0, 0.5 }));
        var scaled = DiffusionNoiseHelper<double>.ScaleNoise(noise, 2.0);
        Assert.Equal(2.0, scaled[0], Tolerance);
        Assert.Equal(-4.0, scaled[1], Tolerance);
        Assert.Equal(1.0, scaled[2], Tolerance);
    }

    [Fact]
    public void ScaleNoise_ZeroScale_AllZero()
    {
        var noise = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 5.0, -3.0, 1.0 }));
        var scaled = DiffusionNoiseHelper<double>.ScaleNoise(noise, 0.0);
        for (int i = 0; i < scaled.Length; i++)
        {
            Assert.Equal(0.0, scaled[i], Tolerance);
        }
    }

    #endregion

    #region AddNoise

    [Fact]
    public void AddNoise_BlendCorrectly()
    {
        var signal = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1.0, 2.0 }));
        var noise = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 0.5, -0.5 }));
        // sqrtAlpha=0.9, sqrtOneMinusAlpha=0.1
        var noisy = DiffusionNoiseHelper<double>.AddNoise(signal, noise, 0.9, 0.1);

        // noisy[0] = 0.9 * 1.0 + 0.1 * 0.5 = 0.95
        Assert.Equal(0.95, noisy[0], Tolerance);
        // noisy[1] = 0.9 * 2.0 + 0.1 * (-0.5) = 1.75
        Assert.Equal(1.75, noisy[1], Tolerance);
    }

    [Fact]
    public void AddNoise_AllSignal_ReturnsSignal()
    {
        var signal = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 5.0, 10.0, 15.0 }));
        var noise = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 1.0, 1.0 }));
        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, 1.0, 0.0);
        Assert.Equal(5.0, result[0], Tolerance);
        Assert.Equal(10.0, result[1], Tolerance);
    }

    #endregion

    #region ComputeTimestepEmbeddings

    [Fact]
    public void ComputeTimestepEmbeddings_CorrectShape()
    {
        var embeddings = DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(new[] { 0, 100, 500 }, 16);
        Assert.Equal(new[] { 3, 16 }, embeddings.Shape);
    }

    [Fact]
    public void ComputeTimestepEmbeddings_DifferentTimesteps_DifferentEmbeddings()
    {
        var embeddings = DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(new[] { 0, 999 }, 8);
        bool differ = false;
        for (int i = 0; i < 8; i++)
        {
            if (Math.Abs(embeddings[0, i] - embeddings[1, i]) > 1e-10)
            {
                differ = true;
                break;
            }
        }
        Assert.True(differ, "Different timesteps should produce different embeddings");
    }

    [Fact]
    public void ComputeTimestepEmbeddings_NullTimesteps_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(null!, 8));
    }

    [Fact]
    public void ComputeTimestepEmbeddings_OddDimension_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(new[] { 0 }, 7));
    }

    #endregion

    #region ComputeTimestepEmbedding (single)

    [Fact]
    public void ComputeTimestepEmbedding_CorrectLength()
    {
        var embedding = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 16);
        Assert.Equal(16, embedding.Length);
    }

    [Fact]
    public void ComputeTimestepEmbedding_SinCosPattern()
    {
        var embedding = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 4);
        // First half should be sin, second half should be cos
        // sin and cos should be bounded [-1, 1]
        for (int i = 0; i < 4; i++)
        {
            Assert.True(embedding[i] >= -1.0 - Tolerance && embedding[i] <= 1.0 + Tolerance,
                $"Embedding value {embedding[i]} should be in [-1, 1]");
        }
    }

    #endregion

    #region ComputeSNR

    [Fact]
    public void ComputeSNR_HighAlpha_HighSNR()
    {
        // alpha = 0.99 -> SNR = 0.99 / 0.01 = 99
        var snr = DiffusionNoiseHelper<double>.ComputeSNR(0.99);
        Assert.Equal(99.0, snr, 0.01);
    }

    [Fact]
    public void ComputeSNR_LowAlpha_LowSNR()
    {
        // alpha = 0.01 -> SNR = 0.01 / 0.99 ≈ 0.0101
        var snr = DiffusionNoiseHelper<double>.ComputeSNR(0.01);
        Assert.True(snr < 0.02);
        Assert.True(snr > 0.0);
    }

    [Fact]
    public void ComputeSNR_AlphaOne_ReturnsLargeValue()
    {
        // alpha = 1.0 -> (1 - alpha) = 0 -> should return large value
        var snr = DiffusionNoiseHelper<double>.ComputeSNR(1.0);
        Assert.True(snr > 1e9, "SNR at alpha=1 should be very large");
    }

    #endregion

    #region LerpNoise

    [Fact]
    public void LerpNoise_AtZero_ReturnsNoise1()
    {
        var n1 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        var n2 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 10.0, 20.0, 30.0 }));
        var result = DiffusionNoiseHelper<double>.LerpNoise(n1, n2, 0.0);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void LerpNoise_AtOne_ReturnsNoise2()
    {
        var n1 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        var n2 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 10.0, 20.0, 30.0 }));
        var result = DiffusionNoiseHelper<double>.LerpNoise(n1, n2, 1.0);
        Assert.Equal(10.0, result[0], Tolerance);
        Assert.Equal(20.0, result[1], Tolerance);
    }

    [Fact]
    public void LerpNoise_Midpoint_IsAverage()
    {
        var n1 = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 0.0, 10.0 }));
        var n2 = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 10.0, 0.0 }));
        var result = DiffusionNoiseHelper<double>.LerpNoise(n1, n2, 0.5);
        Assert.Equal(5.0, result[0], Tolerance);
        Assert.Equal(5.0, result[1], Tolerance);
    }

    #endregion

    #region SlerpNoise

    [Fact]
    public void SlerpNoise_AtZero_ReturnsNoise1()
    {
        var n1 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 0.0, 0.0 }));
        var n2 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));
        var result = DiffusionNoiseHelper<double>.SlerpNoise(n1, n2, 0.0);
        Assert.Equal(1.0, result[0], 0.01);
    }

    [Fact]
    public void SlerpNoise_AtOne_ReturnsNoise2()
    {
        var n1 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 0.0, 0.0 }));
        var n2 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));
        var result = DiffusionNoiseHelper<double>.SlerpNoise(n1, n2, 1.0);
        Assert.Equal(1.0, result[1], 0.01);
    }

    [Fact]
    public void SlerpNoise_SameVectors_ReturnsOriginal()
    {
        var n1 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        var n2 = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        // Very close vectors should fall back to lerp
        var result = DiffusionNoiseHelper<double>.SlerpNoise(n1, n2, 0.5);
        Assert.Equal(1.0, result[0], 0.01);
        Assert.Equal(2.0, result[1], 0.01);
    }

    #endregion

    #region ParallelProcessingHelper

    [Fact]
    public async Task ParallelProcessing_Functions_ReturnsAllResults()
    {
        var functions = new List<Func<int>>
        {
            () => 1,
            () => 2,
            () => 3,
            () => 4,
            () => 5
        };

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(functions, maxDegreeOfParallelism: 2);
        Assert.Equal(5, results.Count);
        Assert.Contains(1, results);
        Assert.Contains(5, results);
    }

    [Fact]
    public async Task ParallelProcessing_Tasks_ReturnsAllResults()
    {
        var tasks = Enumerable.Range(1, 10).Select(i => Task.FromResult(i * 10)).ToList();
        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 3);
        Assert.Equal(10, results.Count);
        Assert.Contains(10, results);
        Assert.Contains(100, results);
    }

    [Fact]
    public async Task ParallelProcessing_EmptyFunctions_ReturnsEmpty()
    {
        var functions = new List<Func<int>>();
        var results = await ParallelProcessingHelper.ProcessTasksInParallel(functions);
        Assert.Empty(results);
    }

    [Fact]
    public async Task ParallelProcessing_EmptyTasks_ReturnsEmpty()
    {
        var tasks = new List<Task<int>>();
        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);
        Assert.Empty(results);
    }

    #endregion
}
