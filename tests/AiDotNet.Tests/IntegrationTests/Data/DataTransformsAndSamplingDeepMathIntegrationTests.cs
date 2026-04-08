using AiDotNet.Data.Sampling;
using AiDotNet.Data.Transforms;
using AiDotNet.Data.Transforms.Numeric;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

/// <summary>
/// Deep math integration tests for Data transforms and sampling.
/// Tests NormalizeTransform, StandardScaleTransform, MinMaxScaleTransform,
/// OneHotEncodeTransform, Compose, WeightedSampler balanced weights, and sampling distributions.
/// </summary>
public class DataTransformsAndSamplingDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    // ============================
    // NormalizeTransform Tests
    // ============================

    [Fact]
    public void Normalize_Formula_XMinusMeanOverStd()
    {
        // (x - mean) / std
        var mean = new[] { 2.0, 4.0 };
        var std = new[] { 1.0, 2.0 };
        var transform = new NormalizeTransform<double>(mean, std);

        var result = transform.Apply(new[] { 3.0, 8.0 });

        // (3 - 2) / 1 = 1.0, (8 - 4) / 2 = 2.0
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
    }

    [Fact]
    public void Normalize_MeanInput_ReturnsZero()
    {
        var mean = new[] { 5.0, 10.0 };
        var std = new[] { 2.0, 3.0 };
        var transform = new NormalizeTransform<double>(mean, std);

        var result = transform.Apply(new[] { 5.0, 10.0 });

        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
    }

    [Fact]
    public void Normalize_OneSigmaAbove_ReturnsOne()
    {
        var mean = new[] { 0.0 };
        var std = new[] { 3.0 };
        var transform = new NormalizeTransform<double>(mean, std);

        var result = transform.Apply(new[] { 3.0 });
        Assert.Equal(1.0, result[0], Tolerance);
    }

    [Fact]
    public void Normalize_ZeroStd_ReturnsDifference()
    {
        // When std is nearly zero, result = x - mean (std division skipped)
        var mean = new[] { 5.0 };
        var std = new[] { 0.0 };
        var transform = new NormalizeTransform<double>(mean, std);

        var result = transform.Apply(new[] { 7.0 });
        Assert.Equal(2.0, result[0], Tolerance); // 7 - 5 = 2, no division
    }

    [Fact]
    public void Normalize_GlobalMeanStd_AppliedToAllElements()
    {
        var transform = new NormalizeTransform<double>(10.0, 5.0, 3);

        var result = transform.Apply(new[] { 15.0, 20.0, 25.0 });

        // (15-10)/5=1, (20-10)/5=2, (25-10)/5=3
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void Normalize_MismatchedLengths_Throws()
    {
        var mean = new[] { 1.0, 2.0 };
        var std = new[] { 1.0 };
        Assert.Throws<ArgumentException>(() => new NormalizeTransform<double>(mean, std));
    }

    [Fact]
    public void Normalize_InputLengthMismatch_Throws()
    {
        var mean = new[] { 1.0, 2.0 };
        var std = new[] { 1.0, 1.0 };
        var transform = new NormalizeTransform<double>(mean, std);
        Assert.Throws<ArgumentException>(() => transform.Apply(new[] { 1.0 }));
    }

    // ============================
    // StandardScaleTransform Tests
    // ============================

    [Fact]
    public void StandardScale_ComputesMeanCorrectly()
    {
        // Data: [1, 3], [3, 7] => means: [2, 5]
        var data = new[] { new[] { 1.0, 3.0 }, new[] { 3.0, 7.0 } };
        var transform = new StandardScaleTransform<double>(data);

        // Apply to a sample at the mean: should get 0
        var result = transform.Apply(new[] { 2.0, 5.0 });
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
    }

    [Fact]
    public void StandardScale_ComputesStdCorrectly()
    {
        // Data: [0, 0], [2, 4] => means: [1, 2]
        // std feature 0: sqrt(((0-1)^2 + (2-1)^2)/2) = sqrt(2/2) = 1
        // std feature 1: sqrt(((0-2)^2 + (4-2)^2)/2) = sqrt(8/2) = 2
        var data = new[] { new[] { 0.0, 0.0 }, new[] { 2.0, 4.0 } };
        var transform = new StandardScaleTransform<double>(data);

        // Apply to mean + 1 std: should get 1.0
        var result = transform.Apply(new[] { 2.0, 4.0 });
        // (2 - 1) / 1 = 1.0, (4 - 2) / 2 = 1.0
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(1.0, result[1], Tolerance);
    }

    [Fact]
    public void StandardScale_FromPrecomputed_MatchesFormula()
    {
        var mean = new[] { 10.0, 20.0 };
        var std = new[] { 2.0, 5.0 };
        var transform = new StandardScaleTransform<double>(mean, std);

        var result = transform.Apply(new[] { 14.0, 30.0 });
        // (14-10)/2=2, (30-20)/5=2
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
    }

    [Fact]
    public void StandardScale_ConstantFeature_StdSetToOne()
    {
        // All values same => std would be 0, should be set to 1
        var data = new[] { new[] { 5.0 }, new[] { 5.0 }, new[] { 5.0 } };
        var transform = new StandardScaleTransform<double>(data);

        var result = transform.Apply(new[] { 7.0 });
        // (7 - 5) / 1 = 2 (std clamped to 1 when near zero)
        Assert.Equal(2.0, result[0], Tolerance);
    }

    [Fact]
    public void StandardScale_EmptyData_Throws()
    {
        Assert.Throws<ArgumentException>(() => new StandardScaleTransform<double>(Array.Empty<double[]>()));
    }

    // ============================
    // MinMaxScaleTransform Tests
    // ============================

    [Fact]
    public void MinMaxScale_DefaultRange_ScalesTo01()
    {
        // min=[0], max=[10], x=5 => (5-0)/(10-0) * (1-0) + 0 = 0.5
        var transform = new MinMaxScaleTransform<double>(new[] { 0.0 }, new[] { 10.0 });

        var result = transform.Apply(new[] { 5.0 });
        Assert.Equal(0.5, result[0], Tolerance);
    }

    [Fact]
    public void MinMaxScale_MinValue_MapsToTargetMin()
    {
        var transform = new MinMaxScaleTransform<double>(new[] { 2.0 }, new[] { 8.0 });

        var result = transform.Apply(new[] { 2.0 });
        Assert.Equal(0.0, result[0], Tolerance);
    }

    [Fact]
    public void MinMaxScale_MaxValue_MapsToTargetMax()
    {
        var transform = new MinMaxScaleTransform<double>(new[] { 2.0 }, new[] { 8.0 });

        var result = transform.Apply(new[] { 8.0 });
        Assert.Equal(1.0, result[0], Tolerance);
    }

    [Fact]
    public void MinMaxScale_CustomRange_ScalesCorrectly()
    {
        // Scale to [-1, 1]
        var transform = new MinMaxScaleTransform<double>(new[] { 0.0 }, new[] { 100.0 }, -1.0, 1.0);

        var result = transform.Apply(new[] { 50.0 });
        // (50-0)/100 * (1-(-1)) + (-1) = 0.5 * 2 - 1 = 0.0
        Assert.Equal(0.0, result[0], Tolerance);
    }

    [Fact]
    public void MinMaxScale_FromReferenceData_ComputesMinMax()
    {
        var data = new[]
        {
            new[] { 10.0, 100.0 },
            new[] { 20.0, 200.0 },
            new[] { 30.0, 300.0 }
        };
        var transform = new MinMaxScaleTransform<double>(data);

        // Feature 0: min=10, max=30. Midpoint=20 => (20-10)/(30-10) = 0.5
        // Feature 1: min=100, max=300. Midpoint=200 => (200-100)/(300-100) = 0.5
        var result = transform.Apply(new[] { 20.0, 200.0 });
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(0.5, result[1], Tolerance);
    }

    [Fact]
    public void MinMaxScale_ConstantFeature_MapsToTargetMin()
    {
        // All values the same => range = 0, should map to targetMin
        var transform = new MinMaxScaleTransform<double>(new[] { 5.0 }, new[] { 5.0 });

        var result = transform.Apply(new[] { 5.0 });
        Assert.Equal(0.0, result[0], Tolerance); // maps to targetMin = 0
    }

    [Fact]
    public void MinMaxScale_HandComputed_TwoFeatures()
    {
        // min=[0, -10], max=[100, 10]
        var transform = new MinMaxScaleTransform<double>(
            new[] { 0.0, -10.0 }, new[] { 100.0, 10.0 });

        var result = transform.Apply(new[] { 25.0, 0.0 });
        // Feature 0: (25-0)/(100-0) = 0.25
        // Feature 1: (0-(-10))/(10-(-10)) = 10/20 = 0.5
        Assert.Equal(0.25, result[0], Tolerance);
        Assert.Equal(0.5, result[1], Tolerance);
    }

    [Fact]
    public void MinMaxScale_InvalidRange_Throws()
    {
        // targetMin >= targetMax should throw
        Assert.Throws<ArgumentException>(() =>
            new MinMaxScaleTransform<double>(new[] { 0.0 }, new[] { 1.0 }, 1.0, 1.0));
        Assert.Throws<ArgumentException>(() =>
            new MinMaxScaleTransform<double>(new[] { 0.0 }, new[] { 1.0 }, 2.0, 1.0));
    }

    // ============================
    // OneHotEncodeTransform Tests
    // ============================

    [Fact]
    public void OneHot_Class0_FirstElementIsOne()
    {
        var transform = new OneHotEncodeTransform<double>(3);
        var result = transform.Apply(0);

        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
    }

    [Fact]
    public void OneHot_LastClass_LastElementIsOne()
    {
        var transform = new OneHotEncodeTransform<double>(4);
        var result = transform.Apply(3);

        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
        Assert.Equal(1.0, result[3], Tolerance);
    }

    [Fact]
    public void OneHot_SumIsAlwaysOne()
    {
        var transform = new OneHotEncodeTransform<double>(5);
        for (int i = 0; i < 5; i++)
        {
            var result = transform.Apply(i);
            var sum = result.Sum();
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    [Fact]
    public void OneHot_DifferentClasses_AreOrthogonal()
    {
        // Dot product of different one-hot vectors should be 0
        var transform = new OneHotEncodeTransform<double>(4);
        var v0 = transform.Apply(0);
        var v1 = transform.Apply(1);

        double dot = 0;
        for (int i = 0; i < v0.Length; i++) dot += v0[i] * v1[i];
        Assert.Equal(0.0, dot, Tolerance);
    }

    [Fact]
    public void OneHot_InvalidIndex_Throws()
    {
        var transform = new OneHotEncodeTransform<double>(3);
        Assert.Throws<ArgumentOutOfRangeException>(() => transform.Apply(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => transform.Apply(3));
    }

    [Fact]
    public void OneHot_InvalidNumClasses_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new OneHotEncodeTransform<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OneHotEncodeTransform<double>(-1));
    }

    // ============================
    // Compose Transform Tests
    // ============================

    [Fact]
    public void Compose_AppliesTransformsInOrder()
    {
        // First normalize, then apply identity (compose two transforms)
        var normalize = new NormalizeTransform<double>(new[] { 0.0 }, new[] { 2.0 });
        var identity = new IdentityTransform<double[]>();
        var composed = new Compose<double[]>(normalize, identity);

        var result = composed.Apply(new[] { 4.0 });
        // Normalize: (4 - 0) / 2 = 2.0, then identity: 2.0
        Assert.Equal(2.0, result[0], Tolerance);
    }

    // ============================
    // WeightedSampler Balanced Weights Tests
    // ============================

    [Fact]
    public void BalancedWeights_EqualClasses_EqualWeights()
    {
        // 4 samples, 2 classes, each with 2 samples
        var labels = new[] { 0, 0, 1, 1 };
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, 2);

        // weight = total / (numClasses * classCount) = 4 / (2 * 2) = 1.0
        Assert.Equal(4, weights.Length);
        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(1.0, weights[1], Tolerance);
        Assert.Equal(1.0, weights[2], Tolerance);
        Assert.Equal(1.0, weights[3], Tolerance);
    }

    [Fact]
    public void BalancedWeights_ImbalancedClasses_InverseFrequency()
    {
        // 10 samples: 9 class 0, 1 class 1
        var labels = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, 2);

        // Class 0: weight = 10 / (2 * 9) = 10/18 ≈ 0.5556
        // Class 1: weight = 10 / (2 * 1) = 10/2 = 5.0
        Assert.Equal(10.0 / 18.0, weights[0], Tolerance); // class 0
        Assert.Equal(5.0, weights[9], Tolerance); // class 1

        // Class 1 weight should be 9x class 0 weight (inverse of ratio)
        Assert.Equal(weights[9] / weights[0], 9.0, Tolerance);
    }

    [Fact]
    public void BalancedWeights_TotalWeightPerClass_IsEqual()
    {
        // After balancing, total weight per class should be equal
        var labels = new[] { 0, 0, 0, 1, 1, 2 };
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, 3);

        // Sum weights per class
        double class0Weight = weights[0] + weights[1] + weights[2]; // 3 samples
        double class1Weight = weights[3] + weights[4]; // 2 samples
        double class2Weight = weights[5]; // 1 sample

        // Total = 6 / (3 classes) = 2 per class
        // class0: 3 * (6/(3*3)) = 3 * 2/3 = 2
        // class1: 2 * (6/(3*2)) = 2 * 1 = 2
        // class2: 1 * (6/(3*1)) = 1 * 2 = 2
        Assert.Equal(2.0, class0Weight, Tolerance);
        Assert.Equal(2.0, class1Weight, Tolerance);
        Assert.Equal(2.0, class2Weight, Tolerance);
    }

    [Fact]
    public void BalancedWeights_EmptyClass_WeightIsZero()
    {
        // 3 classes but only 2 have samples
        var labels = new[] { 0, 0, 2, 2 };
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, 3);

        // Class 0: 4 / (3 * 2) = 2/3
        // Class 1: 0 samples, weight = 0
        // Class 2: 4 / (3 * 2) = 2/3
        Assert.Equal(4.0 / 6.0, weights[0], Tolerance); // class 0
        Assert.Equal(4.0 / 6.0, weights[2], Tolerance); // class 2
    }

    // ============================
    // WeightedSampler Sampling Tests
    // ============================

    [Fact]
    public void WeightedSampler_WithReplacement_SamplesCorrectCount()
    {
        var weights = new[] { 1.0, 1.0, 1.0, 1.0, 1.0 };
        var sampler = new WeightedSampler<double>(weights, numSamples: 10, replacement: true, seed: 42);

        var indices = sampler.GetIndices().ToList();
        Assert.Equal(10, indices.Count);
    }

    [Fact]
    public void WeightedSampler_WithoutReplacement_NoDuplicates()
    {
        var weights = new[] { 1.0, 1.0, 1.0, 1.0, 1.0 };
        var sampler = new WeightedSampler<double>(weights, replacement: false, seed: 42);

        var indices = sampler.GetIndices().ToList();
        Assert.Equal(5, indices.Count);
        Assert.Equal(indices.Distinct().Count(), indices.Count); // no duplicates
    }

    [Fact]
    public void WeightedSampler_ZeroWeight_NeverSampled()
    {
        // Index 2 has zero weight - should never appear
        var weights = new[] { 1.0, 1.0, 0.0, 1.0, 1.0 };
        var sampler = new WeightedSampler<double>(weights, numSamples: 1000, replacement: true, seed: 42);

        var indices = sampler.GetIndices().ToList();
        Assert.DoesNotContain(2, indices);
    }

    [Fact]
    public void WeightedSampler_HighWeight_SampledMoreOften()
    {
        // Index 0 has 100x the weight of others
        var weights = new[] { 100.0, 1.0, 1.0, 1.0, 1.0 };
        var sampler = new WeightedSampler<double>(weights, numSamples: 1000, replacement: true, seed: 42);

        var indices = sampler.GetIndices().ToList();
        var count0 = indices.Count(i => i == 0);

        // Index 0 should be sampled far more than any other
        // Expected: ~100/104 ≈ 96% of samples
        Assert.True(count0 > 800, $"Index 0 sampled {count0}/1000 times, expected >800 with 100x weight");
    }

    [Fact]
    public void WeightedSampler_AllIndicesInRange()
    {
        var weights = new[] { 1.0, 2.0, 3.0 };
        var sampler = new WeightedSampler<double>(weights, numSamples: 100, replacement: true, seed: 42);

        var indices = sampler.GetIndices().ToList();
        Assert.All(indices, idx => Assert.InRange(idx, 0, 2));
    }

    // ============================
    // Mathematical Properties Tests
    // ============================

    [Fact]
    public void Normalize_InvertibleWithDenormalize()
    {
        // If you normalize then multiply by std and add mean, you get the original
        var mean = new[] { 3.0, 7.0 };
        var std = new[] { 2.0, 5.0 };
        var transform = new NormalizeTransform<double>(mean, std);

        var original = new[] { 10.0, 22.0 };
        var normalized = transform.Apply(original);

        // Reverse: x = normalized * std + mean
        var recovered = new double[2];
        for (int i = 0; i < 2; i++)
        {
            recovered[i] = normalized[i] * std[i] + mean[i];
        }

        Assert.Equal(original[0], recovered[0], Tolerance);
        Assert.Equal(original[1], recovered[1], Tolerance);
    }

    [Fact]
    public void StandardScale_OutputMean_IsApproxZero()
    {
        // After z-score normalization on the training data, the output mean should be ~0
        var data = new[]
        {
            new[] { 10.0, 100.0 },
            new[] { 20.0, 200.0 },
            new[] { 30.0, 300.0 },
            new[] { 40.0, 400.0 }
        };
        var transform = new StandardScaleTransform<double>(data);

        double sum0 = 0, sum1 = 0;
        foreach (var row in data)
        {
            var result = transform.Apply(row);
            sum0 += result[0];
            sum1 += result[1];
        }

        Assert.Equal(0.0, sum0 / data.Length, 1e-8);
        Assert.Equal(0.0, sum1 / data.Length, 1e-8);
    }

    [Fact]
    public void MinMaxScale_OutputRange_WithinTargetBounds()
    {
        var data = new[]
        {
            new[] { -5.0, 100.0 },
            new[] { 15.0, 200.0 },
            new[] { 10.0, 150.0 }
        };
        var transform = new MinMaxScaleTransform<double>(data, -1.0, 1.0);

        foreach (var row in data)
        {
            var result = transform.Apply(row);
            for (int i = 0; i < result.Length; i++)
            {
                Assert.True(result[i] >= -1.0 - Tolerance, $"Scaled value {result[i]} below target min -1");
                Assert.True(result[i] <= 1.0 + Tolerance, $"Scaled value {result[i]} above target max 1");
            }
        }
    }
}
