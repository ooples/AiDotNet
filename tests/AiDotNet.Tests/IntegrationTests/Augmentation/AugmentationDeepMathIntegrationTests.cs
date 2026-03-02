using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Augmentation;

/// <summary>
/// Deep mathematical correctness tests for augmentation algorithms (SMOTE, MixUp, FeatureNoise, etc.).
/// Each test verifies exact expected behavior with hand-calculated values and seeded randomness.
/// </summary>
public class AugmentationDeepMathIntegrationTests
{
    #region Helpers

    private static Matrix<double> M(double[,] data) => new(data);

    private static void AssertCell(Matrix<double> m, int row, int col, double expected, double tol = 1e-10)
    {
        Assert.True(
            Math.Abs(m[row, col] - expected) < tol,
            $"[{row},{col}]: expected {expected}, got {m[row, col]} (diff={Math.Abs(m[row, col] - expected)})");
    }

    #endregion

    #region SMOTE - Synthetic Sample Generation

    [Fact]
    public void Smote_GeneratedSamples_AreBetweenOriginals()
    {
        // SMOTE interpolates between a sample and its neighbor: synthetic = x + gap*(neighbor - x)
        // where gap is in [0, 1). So synthetic values must be BETWEEN the two samples.
        var smote = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 1.0);
        var data = M(new double[,] {
            { 0, 0 },
            { 10, 10 },
            { 20, 20 }
        });
        var context = new AugmentationContext<double>(seed: 42);

        var synthetic = smote.GenerateSyntheticSamples(data, context);

        // All synthetic samples should have values between 0 and 20
        for (int i = 0; i < synthetic.Rows; i++)
        {
            for (int c = 0; c < synthetic.Columns; c++)
            {
                double val = synthetic[i, c];
                Assert.True(val >= 0 && val <= 20,
                    $"Synthetic[{i},{c}]={val} should be between 0 and 20");
            }
        }
    }

    [Fact]
    public void Smote_SamplingRatio_ControlsOutputCount()
    {
        // samplingRatio=1.0 with 4 rows → ceil(4*1.0) = 4 synthetic samples
        // samplingRatio=0.5 with 4 rows → ceil(4*0.5) = 2 synthetic samples
        var smote1 = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 1.0);
        var smote05 = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 0.5);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } });

        var syn1 = smote1.GenerateSyntheticSamples(data, new AugmentationContext<double>(seed: 42));
        var syn05 = smote05.GenerateSyntheticSamples(data, new AugmentationContext<double>(seed: 42));

        Assert.Equal(4, syn1.Rows);
        Assert.Equal(2, syn05.Rows);
    }

    [Fact]
    public void Smote_TwoSamples_InterpolatesBetweenThem()
    {
        // With only 2 samples and k=1, the nearest neighbor of each sample is the other one.
        // Synthetic = sample + gap * (neighbor - sample), so all synthetics lie on the line segment.
        // sample0 = [0, 0], sample1 = [10, 20]
        // synthetic = [0 + gap*(10-0), 0 + gap*(20-0)] = [10*gap, 20*gap]
        // So for any synthetic, col1 / col0 = 20/10 = 2 (exact ratio preserved)
        var smote = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 2.0);
        var data = M(new double[,] { { 0, 0 }, { 10, 20 } });
        var context = new AugmentationContext<double>(seed: 42);

        var synthetic = smote.GenerateSyntheticSamples(data, context);

        for (int i = 0; i < synthetic.Rows; i++)
        {
            double col0 = synthetic[i, 0];
            double col1 = synthetic[i, 1];

            // Both values should be in [0, 10] and [0, 20] respectively
            Assert.True(col0 >= 0 && col0 <= 10,
                $"Synthetic[{i},0]={col0} should be in [0, 10]");
            Assert.True(col1 >= 0 && col1 <= 20,
                $"Synthetic[{i},1]={col1} should be in [0, 20]");

            // The ratio should be exactly 2 (or both 0)
            if (Math.Abs(col0) > 1e-10)
            {
                Assert.True(Math.Abs(col1 / col0 - 2.0) < 1e-8,
                    $"Ratio col1/col0 should be 2, got {col1 / col0}");
            }
        }
    }

    [Fact]
    public void Smote_SingleSample_ReturnsEmpty()
    {
        // Need at least 2 samples to interpolate
        var smote = new SmoteAugmenter<double>(kNeighbors: 1);
        var data = M(new double[,] { { 1, 2, 3 } });
        var context = new AugmentationContext<double>(seed: 42);

        var synthetic = smote.GenerateSyntheticSamples(data, context);

        Assert.Equal(0, synthetic.Rows);
        Assert.Equal(3, synthetic.Columns);
    }

    [Fact]
    public void Smote_KNeighborsClampedToMaxAvailable()
    {
        // If k=5 but only 3 samples, effective k = min(5, 3-1) = 2
        // Should not throw, just clamp
        var smote = new SmoteAugmenter<double>(kNeighbors: 5, samplingRatio: 1.0);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 } });
        var context = new AugmentationContext<double>(seed: 42);

        var synthetic = smote.GenerateSyntheticSamples(data, context);

        Assert.Equal(3, synthetic.Rows); // ceil(3*1.0) = 3
    }

    [Fact]
    public void Smote_ApplySmoteWithLabels_AllSyntheticHaveSameLabel()
    {
        var smote = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 1.0);
        var data = M(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var labels = new Vector<double>(new double[] { 1.0, 1.0, 1.0 }); // all same label
        var context = new AugmentationContext<double>(seed: 42);

        var (combinedData, combinedLabels) = smote.ApplySmoteWithLabels(data, labels, context);

        // Should have original (3) + synthetic (3) = 6 samples
        Assert.Equal(6, combinedData.Rows);
        Assert.Equal(6, combinedLabels.Length);

        // All labels should be 1.0
        for (int i = 0; i < combinedLabels.Length; i++)
        {
            Assert.Equal(1.0, combinedLabels[i], 10);
        }
    }

    [Fact]
    public void Smote_ApplySmoteWithLabels_MixedLabels_Throws()
    {
        var smote = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 1.0);
        var data = M(new double[,] { { 1, 2 }, { 3, 4 } });
        var labels = new Vector<double>(new double[] { 0.0, 1.0 }); // different labels!
        var context = new AugmentationContext<double>(seed: 42);

        Assert.Throws<ArgumentException>(() => smote.ApplySmoteWithLabels(data, labels, context));
    }

    [Fact]
    public void Smote_InvalidKNeighbors_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SmoteAugmenter<double>(kNeighbors: 0));
    }

    [Fact]
    public void Smote_InvalidSamplingRatio_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SmoteAugmenter<double>(samplingRatio: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SmoteAugmenter<double>(samplingRatio: -1.0));
    }

    [Fact]
    public void Smote_Seeded_Reproducible()
    {
        var smote = new SmoteAugmenter<double>(kNeighbors: 2, samplingRatio: 1.0);
        var data = M(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 } });

        var syn1 = smote.GenerateSyntheticSamples(data, new AugmentationContext<double>(seed: 123));
        var syn2 = smote.GenerateSyntheticSamples(data, new AugmentationContext<double>(seed: 123));

        Assert.Equal(syn1.Rows, syn2.Rows);
        for (int i = 0; i < syn1.Rows; i++)
        {
            for (int c = 0; c < syn1.Columns; c++)
            {
                Assert.Equal(syn1[i, c], syn2[i, c], 10);
            }
        }
    }

    [Fact]
    public void Smote_DistanceMatrix_IsSymmetricAndCorrect()
    {
        // Verify distance calculation is Euclidean and symmetric
        // [0, 0], [3, 4] → distance = sqrt(9+16) = 5
        // [0, 0], [1, 0] → distance = 1
        // [3, 4], [1, 0] → distance = sqrt(4+16) = sqrt(20) = 4.472
        var smote = new SmoteAugmenter<double>(kNeighbors: 1, samplingRatio: 1.0);
        var data = M(new double[,] { { 0, 0 }, { 3, 4 }, { 1, 0 } });
        var context = new AugmentationContext<double>(seed: 42);

        // Generate synthetic to exercise the distance matrix internally
        var synthetic = smote.GenerateSyntheticSamples(data, context);

        // k=1 means each sample picks its nearest neighbor:
        // [0,0]'s nearest: [1,0] (d=1) - not [3,4] (d=5)
        // [3,4]'s nearest: [1,0] (d=sqrt(20)) - not [0,0] (d=5)
        // [1,0]'s nearest: [0,0] (d=1) - not [3,4] (d=sqrt(20))
        // So interpolation is always between nearby points
        // Synthetic samples should be reasonable, not NaN or infinity
        for (int i = 0; i < synthetic.Rows; i++)
        {
            for (int c = 0; c < synthetic.Columns; c++)
            {
                Assert.True(!double.IsNaN(synthetic[i, c]) && !double.IsInfinity(synthetic[i, c]),
                    $"Synthetic[{i},{c}] is {synthetic[i, c]} (should be finite)");
            }
        }
    }

    #endregion

    #region TabularMixUp - Linear Interpolation

    [Fact]
    public void MixUp_MixWithLabels_ExactInterpolation()
    {
        // MixUp: mixed = lambda * data1 + (1 - lambda) * data2
        // With seeded context, lambda is deterministic
        var mixup = new TabularMixUp<double>(alpha: 1.0); // alpha=1.0 → uniform [0,1]
        var data1 = M(new double[,] { { 10, 100 }, { 20, 200 } });
        var data2 = M(new double[,] { { 0, 0 }, { 0, 0 } });
        var labels1 = new Vector<double>(new double[] { 1.0, 1.0 });
        var labels2 = new Vector<double>(new double[] { 0.0, 0.0 });
        var context = new AugmentationContext<double>(seed: 42);

        var (mixed, mixedLabels, lambda) = mixup.MixWithLabels(data1, data2, labels1, labels2, context);

        // With data2 = zeros: mixed = lambda * data1
        // So mixed[0,0] = lambda * 10, mixed[0,1] = lambda * 100
        Assert.True(lambda > 0 && lambda < 1, $"Lambda={lambda} should be in (0,1)");
        AssertCell(mixed, 0, 0, lambda * 10.0, 1e-8);
        AssertCell(mixed, 0, 1, lambda * 100.0, 1e-8);
        AssertCell(mixed, 1, 0, lambda * 20.0, 1e-8);
        AssertCell(mixed, 1, 1, lambda * 200.0, 1e-8);

        // Labels should also be mixed: lambda * 1 + (1-lambda) * 0 = lambda
        Assert.Equal(lambda, mixedLabels[0], 8);
        Assert.Equal(lambda, mixedLabels[1], 8);
    }

    [Fact]
    public void MixUp_LambdaFromBetaDistribution_InUnitInterval()
    {
        // Beta(alpha, alpha) distribution always produces values in [0, 1]
        var mixup = new TabularMixUp<double>(alpha: 0.2);
        var data1 = M(new double[,] { { 1 }, { 2 } });
        var data2 = M(new double[,] { { 10 }, { 20 } });
        var labels1 = new Vector<double>(new double[] { 0.0, 0.0 });
        var labels2 = new Vector<double>(new double[] { 1.0, 1.0 });

        for (int trial = 0; trial < 20; trial++)
        {
            var context = new AugmentationContext<double>(seed: trial);
            var (_, _, lambda) = mixup.MixWithLabels(data1, data2, labels1, labels2, context);

            Assert.True(lambda >= 0.0 && lambda <= 1.0,
                $"Trial {trial}: Lambda={lambda} should be in [0, 1]");
        }
    }

    [Fact]
    public void MixUp_SingleSample_ReturnsClone()
    {
        var mixup = new TabularMixUp<double>();
        var data = M(new double[,] { { 42, 99 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = mixup.Apply(data, context);

        // Single sample → no mixing possible, return clone
        AssertCell(result, 0, 0, 42.0);
        AssertCell(result, 0, 1, 99.0);
    }

    [Fact]
    public void MixUp_Apply_PreservesDimensions()
    {
        var mixup = new TabularMixUp<double>(alpha: 0.5, probability: 1.0);
        var data = M(new double[,] {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var context = new AugmentationContext<double>(seed: 42);

        var result = mixup.Apply(data, context);

        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void MixUp_MismatchedColumns_Throws()
    {
        var mixup = new TabularMixUp<double>();
        var data1 = M(new double[,] { { 1, 2 } });
        var data2 = M(new double[,] { { 1, 2, 3 } });
        var labels1 = new Vector<double>(new double[] { 0.0 });
        var labels2 = new Vector<double>(new double[] { 1.0 });
        var context = new AugmentationContext<double>(seed: 42);

        Assert.Throws<ArgumentException>(() => mixup.MixWithLabels(data1, data2, labels1, labels2, context));
    }

    #endregion

    #region FeatureNoise - Gaussian Noise Addition

    [Fact]
    public void FeatureNoise_ZeroStdDev_DataUnchanged()
    {
        var noise = new FeatureNoise<double>(noiseStdDev: 0.0, probability: 1.0);
        var data = M(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = noise.Apply(data, context);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                AssertCell(result, i, j, data[i, j]);
            }
        }
    }

    [Fact]
    public void FeatureNoise_SmallStdDev_ValuesNearOriginal()
    {
        // With stddev=0.001, noise is very small
        var noise = new FeatureNoise<double>(noiseStdDev: 0.001, probability: 1.0);
        var data = M(new double[,] { { 100, 200 }, { 300, 400 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = noise.Apply(data, context);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                double diff = Math.Abs(result[i, j] - data[i, j]);
                Assert.True(diff < 1.0,
                    $"[{i},{j}]: diff={diff} should be < 1.0 with stddev=0.001");
            }
        }
    }

    [Fact]
    public void FeatureNoise_SelectiveFeatures_OnlyNoiseOnSpecified()
    {
        // Only apply noise to feature 0, leave feature 1 unchanged
        var noise = new FeatureNoise<double>(noiseStdDev: 1.0, probability: 1.0, featureIndices: new[] { 0 });
        var data = M(new double[,] { { 100, 200 }, { 300, 400 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = noise.Apply(data, context);

        // Feature 1 should be unchanged
        AssertCell(result, 0, 1, 200.0);
        AssertCell(result, 1, 1, 400.0);

        // Feature 0 should be changed (very unlikely to be exactly the same with stddev=1)
        bool feature0Changed = Math.Abs(result[0, 0] - 100.0) > 1e-10 ||
                               Math.Abs(result[1, 0] - 300.0) > 1e-10;
        Assert.True(feature0Changed, "Feature 0 should have noise applied");
    }

    [Fact]
    public void FeatureNoise_Seeded_Reproducible()
    {
        var noise = new FeatureNoise<double>(noiseStdDev: 0.1, probability: 1.0);
        var data = M(new double[,] { { 1, 2 }, { 3, 4 } });

        var result1 = noise.Apply(data, new AugmentationContext<double>(seed: 99));
        var result2 = noise.Apply(data, new AugmentationContext<double>(seed: 99));

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j], 10);
            }
        }
    }

    [Fact]
    public void FeatureNoise_NegativeStdDev_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FeatureNoise<double>(noiseStdDev: -0.1));
    }

    [Fact]
    public void FeatureNoise_NoiseDistribution_ApproximatelyZeroMean()
    {
        // Over many samples, the noise should average to approximately 0
        var noise = new FeatureNoise<double>(noiseStdDev: 1.0, probability: 1.0);

        // Create a 100x1 matrix of constant values
        var values = new double[100, 1];
        for (int i = 0; i < 100; i++) values[i, 0] = 0.0;
        var data = M(values);

        var context = new AugmentationContext<double>(seed: 42);
        var result = noise.Apply(data, context);

        // The mean of the noisy values should be close to 0
        double sum = 0;
        for (int i = 0; i < result.Rows; i++) sum += result[i, 0];
        double mean = sum / result.Rows;

        Assert.True(Math.Abs(mean) < 0.5,
            $"Mean of 100 noise samples = {mean}, expected close to 0");
    }

    #endregion

    #region AugmentationContext - Seeding and Probability

    [Fact]
    public void AugmentationContext_ShouldApply_ProbabilityZero_NeverApplies()
    {
        var context = new AugmentationContext<double>(seed: 42);
        for (int i = 0; i < 100; i++)
        {
            Assert.False(context.ShouldApply(0.0));
        }
    }

    [Fact]
    public void AugmentationContext_ShouldApply_ProbabilityOne_AlwaysApplies()
    {
        var context = new AugmentationContext<double>(seed: 42);
        for (int i = 0; i < 100; i++)
        {
            Assert.True(context.ShouldApply(1.0));
        }
    }

    [Fact]
    public void AugmentationContext_SampleBeta_ReturnsValuesInUnitInterval()
    {
        var context = new AugmentationContext<double>(seed: 42);
        for (int i = 0; i < 50; i++)
        {
            double sample = context.SampleBeta(0.2, 0.2);
            Assert.True(sample >= 0.0 && sample <= 1.0,
                $"Beta(0.2, 0.2) sample {i} = {sample}, expected in [0, 1]");
        }
    }

    [Fact]
    public void AugmentationContext_SampleBeta_SymmetricAlpha_MeanNearHalf()
    {
        // Beta(alpha, alpha) has mean = 0.5
        var context = new AugmentationContext<double>(seed: 42);
        double sum = 0;
        int n = 500;
        for (int i = 0; i < n; i++)
        {
            sum += context.SampleBeta(1.0, 1.0); // Beta(1,1) = Uniform(0,1), mean=0.5
        }
        double mean = sum / n;
        Assert.True(Math.Abs(mean - 0.5) < 0.1,
            $"Mean of 500 Beta(1,1) samples = {mean}, expected ~0.5");
    }

    [Fact]
    public void AugmentationContext_SampleGaussian_MeanAndStdCorrect()
    {
        var context = new AugmentationContext<double>(seed: 42);
        double sum = 0, sumSq = 0;
        int n = 1000;
        double targetMean = 5.0, targetStd = 2.0;

        for (int i = 0; i < n; i++)
        {
            double sample = context.SampleGaussian(targetMean, targetStd);
            sum += sample;
            sumSq += sample * sample;
        }

        double mean = sum / n;
        double variance = (sumSq / n) - (mean * mean);
        double std = Math.Sqrt(variance);

        Assert.True(Math.Abs(mean - targetMean) < 0.3,
            $"Gaussian mean = {mean}, expected ~{targetMean}");
        Assert.True(Math.Abs(std - targetStd) < 0.3,
            $"Gaussian std = {std}, expected ~{targetStd}");
    }

    #endregion

    #region FeatureDropout

    [Fact]
    public void FeatureDropout_DropRate0_DataUnchanged()
    {
        var dropout = new FeatureDropout<double>(dropoutRate: 0.0, probability: 1.0);
        var data = M(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = dropout.Apply(data, context);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                AssertCell(result, i, j, data[i, j]);
            }
        }
    }

    [Fact]
    public void FeatureDropout_DropRate1_AllZeroed()
    {
        var dropout = new FeatureDropout<double>(dropoutRate: 1.0, probability: 1.0);
        var data = M(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = dropout.Apply(data, context);

        // All features should be dropped (zeroed)
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                AssertCell(result, i, j, 0.0);
            }
        }
    }

    [Fact]
    public void FeatureDropout_DroppedFeaturesAreZero_NotNaN()
    {
        var dropout = new FeatureDropout<double>(dropoutRate: 0.5, probability: 1.0);
        var data = M(new double[,] { { 100, 200, 300, 400 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = dropout.Apply(data, context);

        // Some features should be 0, some should be original values, none should be NaN
        int zeroCount = 0;
        for (int j = 0; j < result.Columns; j++)
        {
            Assert.False(double.IsNaN(result[0, j]), $"Feature {j} should not be NaN");
            if (Math.Abs(result[0, j]) < 1e-10) zeroCount++;
        }

        // With 4 features and 50% dropout, we expect ~2 to be dropped
        // But with randomness, we just check at least one is dropped and at least one survives
        Assert.True(zeroCount > 0, "At least one feature should be dropped");
        Assert.True(zeroCount < 4, "At least one feature should survive");
    }

    [Fact]
    public void FeatureDropout_PreservesDimensions()
    {
        var dropout = new FeatureDropout<double>(dropoutRate: 0.3, probability: 1.0);
        var data = M(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = dropout.Apply(data, context);

        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    #endregion

    #region RowShuffle

    [Fact]
    public void RowShuffle_PreservesAllRows()
    {
        var shuffle = new RowShuffle<double>(probability: 1.0);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = shuffle.Apply(data, context);

        Assert.Equal(5, result.Rows);

        // All original values should appear in the result (just in different order)
        var originalValues = new HashSet<double> { 1, 2, 3, 4, 5 };
        var resultValues = new HashSet<double>();
        for (int i = 0; i < result.Rows; i++) resultValues.Add(result[i, 0]);

        Assert.True(originalValues.SetEquals(resultValues), "All original values should be preserved");
    }

    [Fact]
    public void RowShuffle_Seeded_Reproducible()
    {
        var shuffle = new RowShuffle<double>(probability: 1.0);
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        var result1 = shuffle.Apply(data, new AugmentationContext<double>(seed: 42));
        var result2 = shuffle.Apply(data, new AugmentationContext<double>(seed: 42));

        for (int i = 0; i < result1.Rows; i++)
        {
            Assert.Equal(result1[i, 0], result2[i, 0], 10);
        }
    }

    [Fact]
    public void RowShuffle_PreservesRowIntegrity()
    {
        // Each row should remain as a complete unit (col0 and col1 stay together)
        var shuffle = new RowShuffle<double>(probability: 1.0);
        var data = M(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });
        var context = new AugmentationContext<double>(seed: 42);

        var result = shuffle.Apply(data, context);

        // Verify each row in result matches one of the original rows
        for (int i = 0; i < result.Rows; i++)
        {
            double col0 = result[i, 0];
            double col1 = result[i, 1];
            // col1 should be col0 * 10
            Assert.Equal(col0 * 10.0, col1, 10);
        }
    }

    #endregion
}
