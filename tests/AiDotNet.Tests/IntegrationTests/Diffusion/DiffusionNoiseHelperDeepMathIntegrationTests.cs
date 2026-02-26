using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Deep math integration tests for DiffusionNoiseHelper.
/// Verifies Box-Muller Gaussian sampling, sinusoidal timestep embeddings,
/// SNR computation, linear interpolation (Lerp), and spherical linear
/// interpolation (Slerp) against hand-calculated expected values.
/// </summary>
public class DiffusionNoiseHelperDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double StatisticalTolerance = 0.1; // For statistical properties with finite samples

    private static Tensor<double> MakeTensor(params double[] data)
        => new(new[] { data.Length }, new Vector<double>(data));

    // ─── Box-Muller Gaussian Sampling ───────────────────────────────────

    [Fact]
    public void SampleGaussian_Tensor_ReturnsCorrectShape()
    {
        var shape = new[] { 3, 4 };
        var result = DiffusionNoiseHelper<double>.SampleGaussian(shape, seed: 42);

        Assert.Equal(2, result.Shape.Length);
        Assert.Equal(3, result.Shape[0]);
        Assert.Equal(4, result.Shape[1]);
    }

    [Fact]
    public void SampleGaussian_Tensor_MeanApproximatelyZero()
    {
        // Box-Muller generates N(0,1) samples. With enough samples, mean should be ~0.
        var result = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 1000 }, seed: 123);
        var span = result.AsSpan();

        double sum = 0;
        for (int i = 0; i < span.Length; i++)
            sum += span[i];
        double mean = sum / span.Length;

        Assert.True(Math.Abs(mean) < StatisticalTolerance,
            $"Mean of 1000 Gaussian samples should be ~0, got {mean}");
    }

    [Fact]
    public void SampleGaussian_Tensor_VarianceApproximatelyOne()
    {
        // Standard normal: variance should be ~1.
        var result = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 2000 }, seed: 456);
        var span = result.AsSpan();

        double sum = 0, sumSq = 0;
        for (int i = 0; i < span.Length; i++)
        {
            sum += span[i];
            sumSq += span[i] * span[i];
        }
        double mean = sum / span.Length;
        double variance = sumSq / span.Length - mean * mean;

        Assert.True(Math.Abs(variance - 1.0) < StatisticalTolerance,
            $"Variance of 2000 Gaussian samples should be ~1.0, got {variance}");
    }

    [Fact]
    public void SampleGaussian_SeededReproducibility()
    {
        // Same seed should produce identical samples
        var result1 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 99);
        var result2 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 99);
        var span1 = result1.AsSpan();
        var span2 = result2.AsSpan();

        for (int i = 0; i < span1.Length; i++)
            Assert.Equal(span1[i], span2[i], Tolerance);
    }

    [Fact]
    public void SampleGaussian_DifferentSeedsProduceDifferentSamples()
    {
        var result1 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 1);
        var result2 = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 10 }, seed: 2);
        var span1 = result1.AsSpan();
        var span2 = result2.AsSpan();

        bool anyDifferent = false;
        for (int i = 0; i < span1.Length; i++)
        {
            if (Math.Abs(span1[i] - span2[i]) > Tolerance)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Different seeds should produce different samples");
    }

    [Fact]
    public void SampleGaussian_OddLength_HandlesCorrectly()
    {
        // Box-Muller generates pairs; odd length needs special handling
        var result = DiffusionNoiseHelper<double>.SampleGaussian(new[] { 7 }, seed: 42);
        var span = result.AsSpan();

        Assert.Equal(7, span.Length);
        for (int i = 0; i < span.Length; i++)
            Assert.True(double.IsFinite(span[i]), $"Sample[{i}] = {span[i]} should be finite");
    }

    [Fact]
    public void SampleGaussianVector_Length_Correct()
    {
        var result = DiffusionNoiseHelper<double>.SampleGaussianVector(20, seed: 42);
        Assert.Equal(20, result.Length);
    }

    [Fact]
    public void SampleGaussianVector_MeanApproximatelyZero()
    {
        var result = DiffusionNoiseHelper<double>.SampleGaussianVector(1000, seed: 789);

        double sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];
        double mean = sum / result.Length;

        Assert.True(Math.Abs(mean) < StatisticalTolerance,
            $"Mean of 1000 Gaussian vector samples should be ~0, got {mean}");
    }

    [Fact]
    public void SampleGaussian_EmptyShape_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.SampleGaussian(Array.Empty<int>(), seed: 42));
    }

    [Fact]
    public void SampleGaussianVector_ZeroLength_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DiffusionNoiseHelper<double>.SampleGaussianVector(0, seed: 42));
    }

    // ─── ScaleNoise ─────────────────────────────────────────────────────

    [Fact]
    public void ScaleNoise_HandCalculated_ScaleByTwo()
    {
        // noise = [1.0, -2.0, 3.0], scale = 2.0
        // expected = [2.0, -4.0, 6.0]
        var noise = MakeTensor(1.0, -2.0, 3.0);
        var result = DiffusionNoiseHelper<double>.ScaleNoise(noise, 2.0);
        var span = result.AsSpan();

        Assert.Equal(2.0, span[0], Tolerance);
        Assert.Equal(-4.0, span[1], Tolerance);
        Assert.Equal(6.0, span[2], Tolerance);
    }

    [Fact]
    public void ScaleNoise_ScaleByZero_AllZeros()
    {
        var noise = MakeTensor(5.0, -3.0, 7.0);
        var result = DiffusionNoiseHelper<double>.ScaleNoise(noise, 0.0);
        var span = result.AsSpan();

        for (int i = 0; i < span.Length; i++)
            Assert.Equal(0.0, span[i], Tolerance);
    }

    [Fact]
    public void ScaleNoise_ScaleByNegative_FlipsSigns()
    {
        // noise = [2.0, -3.0], scale = -1.5
        // expected = [-3.0, 4.5]
        var noise = MakeTensor(2.0, -3.0);
        var result = DiffusionNoiseHelper<double>.ScaleNoise(noise, -1.5);
        var span = result.AsSpan();

        Assert.Equal(-3.0, span[0], Tolerance);
        Assert.Equal(4.5, span[1], Tolerance);
    }

    // ─── AddNoise (forward diffusion formula) ───────────────────────────

    [Fact]
    public void AddNoise_HandCalculated_Midpoint()
    {
        // signal = [1.0, 2.0], noise = [0.5, -1.0]
        // sqrtAlphaCumprod = 0.8, sqrtOneMinusAlphaCumprod = 0.6
        // (0.8^2 + 0.6^2 = 0.64 + 0.36 = 1.0, valid schedule)
        // expected[0] = 0.8*1.0 + 0.6*0.5 = 0.8 + 0.3 = 1.1
        // expected[1] = 0.8*2.0 + 0.6*(-1.0) = 1.6 - 0.6 = 1.0
        var signal = MakeTensor(1.0, 2.0);
        var noise = MakeTensor(0.5, -1.0);

        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, 0.8, 0.6);
        var span = result.AsSpan();

        Assert.Equal(1.1, span[0], Tolerance);
        Assert.Equal(1.0, span[1], Tolerance);
    }

    [Fact]
    public void AddNoise_PureSignal_SqrtAlphaOne()
    {
        // sqrtAlphaCumprod=1.0, sqrtOneMinusAlphaCumprod=0.0 => pure signal
        var signal = MakeTensor(5.0, -3.0, 7.0);
        var noise = MakeTensor(100.0, 200.0, -50.0);

        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, 1.0, 0.0);
        var span = result.AsSpan();

        Assert.Equal(5.0, span[0], Tolerance);
        Assert.Equal(-3.0, span[1], Tolerance);
        Assert.Equal(7.0, span[2], Tolerance);
    }

    [Fact]
    public void AddNoise_PureNoise_SqrtAlphaZero()
    {
        // sqrtAlphaCumprod=0.0, sqrtOneMinusAlphaCumprod=1.0 => pure noise
        var signal = MakeTensor(10.0, 20.0);
        var noise = MakeTensor(-1.0, 3.0);

        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, 0.0, 1.0);
        var span = result.AsSpan();

        Assert.Equal(-1.0, span[0], Tolerance);
        Assert.Equal(3.0, span[1], Tolerance);
    }

    [Fact]
    public void AddNoise_EnergyConservation()
    {
        // For valid schedule: sqrtAlpha^2 + sqrtOneMinusAlpha^2 = 1
        // Using orthogonal unit vectors: ||result||^2 = sqrtAlpha^2 + sqrtOneMinusAlpha^2 = 1
        double sqrtAlpha = Math.Sqrt(0.7);
        double sqrtOneMinusAlpha = Math.Sqrt(0.3);

        var signal = MakeTensor(1.0, 0.0, 0.0, 0.0); // unit vector [1,0,0,0]
        var noise = MakeTensor(0.0, 1.0, 0.0, 0.0);  // orthogonal unit vector [0,1,0,0]

        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, sqrtAlpha, sqrtOneMinusAlpha);
        var span = result.AsSpan();

        double normSq = 0;
        for (int i = 0; i < span.Length; i++)
            normSq += span[i] * span[i];

        Assert.Equal(1.0, normSq, Tolerance);
    }

    // ─── Sinusoidal Timestep Embeddings ─────────────────────────────────

    [Fact]
    public void ComputeTimestepEmbedding_HandCalculated_Dim4()
    {
        // embeddingDim=4, halfDim=2
        // logScale = log(10000) / (2-1) = log(10000)
        // For timestep=100:
        //   i=0: freq = exp(0) = 1.0, angle = 100
        //   i=1: freq = exp(-log(10000)) = 1/10000 = 0.0001, angle = 0.01
        var result = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 4);

        double logScale = Math.Log(10000.0) / (2 - 1);
        double freq0 = Math.Exp(0);
        double freq1 = Math.Exp(-1.0 * logScale);

        Assert.Equal(Math.Sin(100 * freq0), result[0], Tolerance);
        Assert.Equal(Math.Sin(100 * freq1), result[1], Tolerance);
        Assert.Equal(Math.Cos(100 * freq0), result[2], Tolerance);
        Assert.Equal(Math.Cos(100 * freq1), result[3], Tolerance);
    }

    [Fact]
    public void ComputeTimestepEmbedding_Dim6_VerifyFrequencies()
    {
        // embeddingDim=6, halfDim=3
        // logScale = log(10000) / (3-1) = log(10000)/2
        var result = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(50, 6);

        double logScale = Math.Log(10000.0) / 2.0;
        double[] freqs = {
            Math.Exp(0),
            Math.Exp(-1.0 * logScale),
            Math.Exp(-2.0 * logScale)
        };

        for (int i = 0; i < 3; i++)
        {
            double angle = 50 * freqs[i];
            Assert.Equal(Math.Sin(angle), result[i], Tolerance);
            Assert.Equal(Math.Cos(angle), result[i + 3], Tolerance);
        }
    }

    [Fact]
    public void ComputeTimestepEmbedding_Timestep0_AllSinZeroAllCosOne()
    {
        // At timestep=0, all angles are 0: sin(0) = 0, cos(0) = 1
        var result = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(0, 8);

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(0.0, result[i], Tolerance);
            Assert.Equal(1.0, result[i + 4], Tolerance);
        }
    }

    [Fact]
    public void ComputeTimestepEmbedding_UnitNorm_SinCosProperty()
    {
        // For each frequency i: sin^2(angle_i) + cos^2(angle_i) = 1
        var result = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(500, 10);
        int halfDim = 5;

        for (int i = 0; i < halfDim; i++)
        {
            double sinVal = result[i];
            double cosVal = result[i + halfDim];
            double sumSq = sinVal * sinVal + cosVal * cosVal;
            Assert.Equal(1.0, sumSq, Tolerance);
        }
    }

    [Fact]
    public void ComputeTimestepEmbeddings_Batch_MatchesSingleEmbedding()
    {
        int[] timesteps = { 0, 100, 500, 999 };
        int dim = 8;
        var batchResult = DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(timesteps, dim);
        var batchSpan = batchResult.AsSpan();

        for (int b = 0; b < timesteps.Length; b++)
        {
            var single = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(timesteps[b], dim);
            for (int i = 0; i < dim; i++)
                Assert.Equal(single[i], batchSpan[b * dim + i], Tolerance);
        }
    }

    [Fact]
    public void ComputeTimestepEmbedding_DifferentTimesteps_DifferentEmbeddings()
    {
        var emb100 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 8);
        var emb200 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(200, 8);

        bool anyDifferent = false;
        for (int i = 0; i < emb100.Length; i++)
        {
            if (Math.Abs(emb100[i] - emb200[i]) > Tolerance)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Different timesteps should produce different embeddings");
    }

    [Fact]
    public void ComputeTimestepEmbedding_HighFrequencyVariesFaster()
    {
        var emb0 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(0, 8);
        var emb1 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(1, 8);

        double changeHighFreq = Math.Abs(emb0[0] - emb1[0]); // highest freq, i=0
        double changeLowFreq = Math.Abs(emb0[3] - emb1[3]);   // lowest freq, i=3

        Assert.True(changeHighFreq > changeLowFreq,
            $"High freq change ({changeHighFreq}) should exceed low freq change ({changeLowFreq})");
    }

    [Fact]
    public void ComputeTimestepEmbedding_OddDim_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 5));
    }

    [Fact]
    public void ComputeTimestepEmbeddings_EmptyTimesteps_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            DiffusionNoiseHelper<double>.ComputeTimestepEmbeddings(Array.Empty<int>(), 4));
    }

    // ─── SNR Computation ────────────────────────────────────────────────

    [Fact]
    public void ComputeSNR_HandCalculated_AlphaCumprod0_9()
    {
        // SNR = alpha / (1 - alpha) = 0.9 / 0.1 = 9.0
        var result = DiffusionNoiseHelper<double>.ComputeSNR(0.9);
        Assert.Equal(9.0, result, Tolerance);
    }

    [Fact]
    public void ComputeSNR_HandCalculated_AlphaCumprod0_5()
    {
        // SNR = 0.5 / 0.5 = 1.0
        var result = DiffusionNoiseHelper<double>.ComputeSNR(0.5);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ComputeSNR_HandCalculated_AlphaCumprod0_1()
    {
        // SNR = 0.1 / 0.9 ≈ 0.11111
        var result = DiffusionNoiseHelper<double>.ComputeSNR(0.1);
        Assert.Equal(0.1 / 0.9, result, Tolerance);
    }

    [Fact]
    public void ComputeSNR_HighAlpha_HighSNR()
    {
        var result = DiffusionNoiseHelper<double>.ComputeSNR(0.999);
        Assert.True(result > 100, $"SNR for alpha=0.999 should be >100, got {result}");
    }

    [Fact]
    public void ComputeSNR_LowAlpha_LowSNR()
    {
        var result = DiffusionNoiseHelper<double>.ComputeSNR(0.001);
        Assert.True(result < 0.01, $"SNR for alpha=0.001 should be <0.01, got {result}");
    }

    [Fact]
    public void ComputeSNR_AlphaOne_ReturnsLargeValue()
    {
        // alpha=1.0 => 1/(1-1) => sentinel value 1e10
        var result = DiffusionNoiseHelper<double>.ComputeSNR(1.0);
        Assert.Equal(1e10, result, 1.0);
    }

    // ─── Lerp (Linear Interpolation) ────────────────────────────────────

    [Fact]
    public void LerpNoise_HandCalculated_MidpointInterpolation()
    {
        // noise1 = [2.0, 4.0], noise2 = [6.0, 8.0], t = 0.5
        // result = 0.5*[2,4] + 0.5*[6,8] = [4.0, 6.0]
        var noise1 = MakeTensor(2.0, 4.0);
        var noise2 = MakeTensor(6.0, 8.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 0.5);
        var span = result.AsSpan();

        Assert.Equal(4.0, span[0], Tolerance);
        Assert.Equal(6.0, span[1], Tolerance);
    }

    [Fact]
    public void LerpNoise_T0_ReturnsNoise1()
    {
        var noise1 = MakeTensor(1.0, 2.0, 3.0);
        var noise2 = MakeTensor(10.0, 20.0, 30.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 0.0);
        var span = result.AsSpan();

        Assert.Equal(1.0, span[0], Tolerance);
        Assert.Equal(2.0, span[1], Tolerance);
        Assert.Equal(3.0, span[2], Tolerance);
    }

    [Fact]
    public void LerpNoise_T1_ReturnsNoise2()
    {
        var noise1 = MakeTensor(1.0, 2.0);
        var noise2 = MakeTensor(10.0, 20.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 1.0);
        var span = result.AsSpan();

        Assert.Equal(10.0, span[0], Tolerance);
        Assert.Equal(20.0, span[1], Tolerance);
    }

    [Fact]
    public void LerpNoise_T0_25_HandCalculated()
    {
        // noise1 = [0.0, 8.0], noise2 = [4.0, 0.0], t = 0.25
        // result = 0.75*[0,8] + 0.25*[4,0] = [1.0, 6.0]
        var noise1 = MakeTensor(0.0, 8.0);
        var noise2 = MakeTensor(4.0, 0.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 0.25);
        var span = result.AsSpan();

        Assert.Equal(1.0, span[0], Tolerance);
        Assert.Equal(6.0, span[1], Tolerance);
    }

    [Fact]
    public void LerpNoise_ClampsTBelowZero()
    {
        // t < 0 clamped to 0 => returns noise1
        var noise1 = MakeTensor(5.0, 6.0);
        var noise2 = MakeTensor(100.0, 200.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, -0.5);
        var span = result.AsSpan();

        Assert.Equal(5.0, span[0], Tolerance);
        Assert.Equal(6.0, span[1], Tolerance);
    }

    [Fact]
    public void LerpNoise_ClampsTAboveOne()
    {
        // t > 1 clamped to 1 => returns noise2
        var noise1 = MakeTensor(5.0, 6.0);
        var noise2 = MakeTensor(100.0, 200.0);

        var result = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 1.5);
        var span = result.AsSpan();

        Assert.Equal(100.0, span[0], Tolerance);
        Assert.Equal(200.0, span[1], Tolerance);
    }

    // ─── Slerp (Spherical Linear Interpolation) ─────────────────────────

    [Fact]
    public void SlerpNoise_T0_ReturnsNoise1()
    {
        var noise1 = MakeTensor(1.0, 0.0, 0.0);
        var noise2 = MakeTensor(0.0, 1.0, 0.0);

        var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 0.0);
        var span = result.AsSpan();

        Assert.Equal(1.0, span[0], 1e-4);
        Assert.Equal(0.0, span[1], 1e-4);
        Assert.Equal(0.0, span[2], 1e-4);
    }

    [Fact]
    public void SlerpNoise_T1_ReturnsNoise2()
    {
        var noise1 = MakeTensor(1.0, 0.0, 0.0);
        var noise2 = MakeTensor(0.0, 1.0, 0.0);

        var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 1.0);
        var span = result.AsSpan();

        Assert.Equal(0.0, span[0], 1e-4);
        Assert.Equal(1.0, span[1], 1e-4);
        Assert.Equal(0.0, span[2], 1e-4);
    }

    [Fact]
    public void SlerpNoise_OrthogonalVectors_Midpoint_HandCalculated()
    {
        // noise1 = [1,0], noise2 = [0,1] (orthogonal, theta=pi/2)
        // t=0.5: scale1 = scale2 = sin(pi/4) / sin(pi/2) = sqrt(2)/2
        // result = [sqrt(2)/2, sqrt(2)/2]
        var noise1 = MakeTensor(1.0, 0.0);
        var noise2 = MakeTensor(0.0, 1.0);

        var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 0.5);
        var span = result.AsSpan();

        double expected = Math.Sqrt(2.0) / 2.0;
        Assert.Equal(expected, span[0], 1e-4);
        Assert.Equal(expected, span[1], 1e-4);
    }

    [Fact]
    public void SlerpNoise_PreservesNorm()
    {
        // Slerp between unit vectors should produce unit-norm results
        var noise1 = MakeTensor(1.0, 0.0, 0.0);
        var noise2 = MakeTensor(0.0, 0.0, 1.0);

        double[] tValues = { 0.0, 0.25, 0.5, 0.75, 1.0 };
        foreach (double t in tValues)
        {
            var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, t);
            var span = result.AsSpan();

            double normSq = 0;
            for (int i = 0; i < span.Length; i++)
                normSq += span[i] * span[i];

            Assert.Equal(1.0, Math.Sqrt(normSq), 1e-3);
        }
    }

    [Fact]
    public void SlerpNoise_ParallelVectors_FallsBackToLerp()
    {
        // Nearly parallel vectors (theta ≈ 0) => slerp falls back to lerp
        var noise1 = MakeTensor(1.0, 0.0);
        var noise2 = MakeTensor(1.0, 1e-8);

        var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 0.5);
        var span = result.AsSpan();

        Assert.True(Math.Abs(span[0] - 1.0) < 0.01,
            $"Parallel slerp at t=0.5: result[0] should be ~1.0, got {span[0]}");
    }

    [Fact]
    public void SlerpNoise_OrthogonalVectors_QuarterPoint()
    {
        // noise1 = [1,0], noise2 = [0,1] (theta=pi/2)
        // t=0.25: scale1 = sin(0.75*pi/2)/sin(pi/2) = sin(3pi/8)
        //         scale2 = sin(0.25*pi/2)/sin(pi/2) = sin(pi/8)
        var noise1 = MakeTensor(1.0, 0.0);
        var noise2 = MakeTensor(0.0, 1.0);

        var result = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 0.25);
        var span = result.AsSpan();

        double theta = Math.PI / 2.0;
        double expectedX = Math.Sin(0.75 * theta) / Math.Sin(theta);
        double expectedY = Math.Sin(0.25 * theta) / Math.Sin(theta);

        Assert.Equal(expectedX, span[0], 1e-4);
        Assert.Equal(expectedY, span[1], 1e-4);
    }

    // ─── Cross-Method Consistency ───────────────────────────────────────

    [Fact]
    public void Lerp_Slerp_ParallelVectors_SameResult()
    {
        // For parallel vectors, slerp falls back to lerp
        var noise1 = MakeTensor(1.0, 2.0, 3.0);
        var noise2 = MakeTensor(2.0, 4.0, 6.0);

        var lerp = DiffusionNoiseHelper<double>.LerpNoise(noise1, noise2, 0.5);
        var slerp = DiffusionNoiseHelper<double>.SlerpNoise(noise1, noise2, 0.5);

        for (int i = 0; i < lerp.AsSpan().Length; i++)
            Assert.Equal(lerp.AsSpan()[i], slerp.AsSpan()[i], 1e-3);
    }

    [Fact]
    public void AddNoise_SameAsManualComputation()
    {
        double sqrtA = 0.7;
        double sqrt1mA = 0.714;
        var signal = MakeTensor(1.5, -2.0, 0.5);
        var noise = MakeTensor(-0.5, 1.0, 2.0);

        var result = DiffusionNoiseHelper<double>.AddNoise(signal, noise, sqrtA, sqrt1mA);
        var span = result.AsSpan();
        var sigSpan = signal.AsSpan();
        var noiSpan = noise.AsSpan();

        for (int i = 0; i < 3; i++)
        {
            double expected = sqrtA * sigSpan[i] + sqrt1mA * noiSpan[i];
            Assert.Equal(expected, span[i], Tolerance);
        }
    }

    [Fact]
    public void ScaleNoise_ThenAddNoise_EquivalentToDirectScaling()
    {
        var signal = MakeTensor(1.0, 2.0);
        var noise = MakeTensor(0.5, -1.0);

        double sqrtAlpha = 0.9;
        double sqrtOneMinusAlpha = 0.436;
        double scale = 2.0;

        var scaledNoise = DiffusionNoiseHelper<double>.ScaleNoise(noise, scale);
        var result1 = DiffusionNoiseHelper<double>.AddNoise(signal, scaledNoise, sqrtAlpha, sqrtOneMinusAlpha);
        var result2 = DiffusionNoiseHelper<double>.AddNoise(signal, noise, sqrtAlpha, sqrtOneMinusAlpha * scale);

        for (int i = 0; i < result1.AsSpan().Length; i++)
            Assert.Equal(result1.AsSpan()[i], result2.AsSpan()[i], Tolerance);
    }

    // ─── Embedding Orthogonality and Properties ─────────────────────────

    [Fact]
    public void ComputeTimestepEmbedding_NearbyTimesteps_HigherSimilarity()
    {
        var emb100 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(100, 16);
        var emb101 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(101, 16);
        var emb500 = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(500, 16);

        double dotNear = 0, dotFar = 0;
        double norm100 = 0, norm101 = 0, norm500 = 0;
        for (int i = 0; i < 16; i++)
        {
            dotNear += emb100[i] * emb101[i];
            dotFar += emb100[i] * emb500[i];
            norm100 += emb100[i] * emb100[i];
            norm101 += emb101[i] * emb101[i];
            norm500 += emb500[i] * emb500[i];
        }
        double cosNear = dotNear / (Math.Sqrt(norm100) * Math.Sqrt(norm101));
        double cosFar = dotFar / (Math.Sqrt(norm100) * Math.Sqrt(norm500));

        Assert.True(cosNear > cosFar,
            $"Nearby timesteps should have higher cosine similarity ({cosNear}) than distant ({cosFar})");
    }

    [Fact]
    public void ComputeTimestepEmbedding_EmbeddingDim_AllValuesInRange()
    {
        var result = DiffusionNoiseHelper<double>.ComputeTimestepEmbedding(999, 32);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] >= -1.0 - Tolerance && result[i] <= 1.0 + Tolerance,
                $"Embedding value[{i}] = {result[i]} should be in [-1, 1]");
        }
    }
}
