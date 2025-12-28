using Xunit;
using AiDotNet.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for SamplingHelper to verify sampling operations.
/// </summary>
public class SamplingHelperIntegrationTests : IDisposable
{
    public SamplingHelperIntegrationTests()
    {
        // Clear any seed from previous tests
        SamplingHelper.ClearSeed();
    }

    public void Dispose()
    {
        // Clean up after each test
        SamplingHelper.ClearSeed();
    }

    #region SampleWithoutReplacement Tests - Basic Functionality

    [Fact]
    public void SampleWithoutReplacement_ValidInput_ReturnsCorrectSampleSize()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(100, 10);

        Assert.Equal(10, result.Length);
    }

    [Fact]
    public void SampleWithoutReplacement_ValidInput_ReturnsUniqueIndices()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(50, 20);

        var distinctCount = result.Distinct().Count();
        Assert.Equal(20, distinctCount);
    }

    [Fact]
    public void SampleWithoutReplacement_ValidInput_IndicesInValidRange()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(100, 30);

        Assert.All(result, index => Assert.InRange(index, 0, 99));
    }

    [Fact]
    public void SampleWithoutReplacement_SampleSizeEqualsPopulation_ReturnsAllIndices()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(10, 10);

        Assert.Equal(10, result.Length);
        Assert.Equal(10, result.Distinct().Count());

        // Should contain all indices 0-9
        var sortedResult = result.OrderBy(x => x).ToArray();
        for (int i = 0; i < 10; i++)
        {
            Assert.Contains(i, result);
        }
    }

    [Fact]
    public void SampleWithoutReplacement_SampleSizeOne_ReturnsSingleIndex()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(100, 1);

        Assert.Single(result);
        Assert.InRange(result[0], 0, 99);
    }

    [Fact]
    public void SampleWithoutReplacement_SampleSizeZero_ReturnsEmptyArray()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(100, 0);

        Assert.Empty(result);
    }

    #endregion

    #region SampleWithoutReplacement Tests - Error Cases

    [Fact]
    public void SampleWithoutReplacement_SampleSizeGreaterThanPopulation_ThrowsArgumentException()
    {
        SamplingHelper.SetSeed(42);

        Assert.Throws<ArgumentException>(() =>
            SamplingHelper.SampleWithoutReplacement(10, 20));
    }

    [Fact]
    public void SampleWithoutReplacement_PopulationSizeZero_SampleGreaterThanZero_ThrowsArgumentException()
    {
        SamplingHelper.SetSeed(42);

        Assert.Throws<ArgumentException>(() =>
            SamplingHelper.SampleWithoutReplacement(0, 1));
    }

    #endregion

    #region SampleWithReplacement Tests - Basic Functionality

    [Fact]
    public void SampleWithReplacement_ValidInput_ReturnsCorrectSampleSize()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(100, 10);

        Assert.Equal(10, result.Length);
    }

    [Fact]
    public void SampleWithReplacement_ValidInput_IndicesInValidRange()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(100, 50);

        Assert.All(result, index => Assert.InRange(index, 0, 99));
    }

    [Fact]
    public void SampleWithReplacement_SampleSizeGreaterThanPopulation_Allowed()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(10, 100);

        Assert.Equal(100, result.Length);
        Assert.All(result, index => Assert.InRange(index, 0, 9));
    }

    [Fact]
    public void SampleWithReplacement_MayHaveDuplicates()
    {
        SamplingHelper.SetSeed(42);
        // Large sample from small population should have duplicates
        var result = SamplingHelper.SampleWithReplacement(5, 100);

        var uniqueCount = result.Distinct().Count();
        Assert.True(uniqueCount < result.Length, "Expected duplicates in sample with replacement");
    }

    [Fact]
    public void SampleWithReplacement_SampleSizeZero_ReturnsEmptyArray()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(100, 0);

        Assert.Empty(result);
    }

    [Fact]
    public void SampleWithReplacement_PopulationSizeOne_AllIndicesAreZero()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(1, 10);

        Assert.All(result, index => Assert.Equal(0, index));
    }

    #endregion

    #region CreateBootstrapSamples Tests - Basic Functionality

    [Fact]
    public void CreateBootstrapSamples_ValidInput_ReturnsCorrectNumberOfSamples()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 5);

        Assert.Equal(5, samples.Count);
    }

    [Fact]
    public void CreateBootstrapSamples_DefaultSampleSize_EqualToDataLength()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3, 4, 5 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 3);

        Assert.All(samples, sample => Assert.Equal(5, sample.Length));
    }

    [Fact]
    public void CreateBootstrapSamples_CustomSampleSize_UsesProvidedSize()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 4, sampleSize: 3);

        Assert.All(samples, sample => Assert.Equal(3, sample.Length));
    }

    [Fact]
    public void CreateBootstrapSamples_SampleSizeLargerThanData_Allowed()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 2, sampleSize: 10);

        Assert.All(samples, sample => Assert.Equal(10, sample.Length));
    }

    [Fact]
    public void CreateBootstrapSamples_ContainsValuesFromOriginalData()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 10, 20, 30, 40, 50 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 3);

        foreach (var sample in samples)
        {
            Assert.All(sample, value => Assert.Contains(value, data));
        }
    }

    [Fact]
    public void CreateBootstrapSamples_MayHaveDuplicateValues()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3, 4, 5 };

        // Create multiple samples to increase chance of finding duplicates
        var samples = SamplingHelper.CreateBootstrapSamples(data, 20);

        // At least one sample should have duplicates (bootstrap is with replacement)
        var hasDuplicates = samples.Any(sample => sample.Distinct().Count() < sample.Length);
        Assert.True(hasDuplicates, "Expected at least one bootstrap sample to have duplicate values");
    }

    [Fact]
    public void CreateBootstrapSamples_StringData_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { "apple", "banana", "cherry", "date" };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 3);

        Assert.Equal(3, samples.Count);
        foreach (var sample in samples)
        {
            Assert.Equal(4, sample.Length);
            Assert.All(sample, value => Assert.Contains(value, data));
        }
    }

    [Fact]
    public void CreateBootstrapSamples_DoubleData_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1.5, 2.5, 3.5, 4.5, 5.5 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 5);

        Assert.Equal(5, samples.Count);
        foreach (var sample in samples)
        {
            Assert.All(sample, value => Assert.Contains(value, data));
        }
    }

    [Fact]
    public void CreateBootstrapSamples_NumberOfSamplesZero_ReturnsEmptyList()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 0);

        Assert.Empty(samples);
    }

    #endregion

    #region SetSeed and ClearSeed Tests

    [Fact]
    public void SetSeed_SameSeed_ProducesSameResults()
    {
        SamplingHelper.SetSeed(42);
        var result1 = SamplingHelper.SampleWithoutReplacement(100, 10);

        SamplingHelper.SetSeed(42);
        var result2 = SamplingHelper.SampleWithoutReplacement(100, 10);

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void SetSeed_DifferentSeeds_ProduceDifferentResults()
    {
        SamplingHelper.SetSeed(42);
        var result1 = SamplingHelper.SampleWithoutReplacement(100, 10);

        SamplingHelper.SetSeed(123);
        var result2 = SamplingHelper.SampleWithoutReplacement(100, 10);

        Assert.NotEqual(result1, result2);
    }

    [Fact]
    public void SetSeed_SameSeed_SampleWithReplacement_ProducesSameResults()
    {
        SamplingHelper.SetSeed(42);
        var result1 = SamplingHelper.SampleWithReplacement(100, 50);

        SamplingHelper.SetSeed(42);
        var result2 = SamplingHelper.SampleWithReplacement(100, 50);

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void SetSeed_SameSeed_BootstrapSamples_ProducesSameResults()
    {
        var data = new[] { 1, 2, 3, 4, 5 };

        SamplingHelper.SetSeed(42);
        var samples1 = SamplingHelper.CreateBootstrapSamples(data, 3);

        SamplingHelper.SetSeed(42);
        var samples2 = SamplingHelper.CreateBootstrapSamples(data, 3);

        Assert.Equal(samples1.Count, samples2.Count);
        for (int i = 0; i < samples1.Count; i++)
        {
            Assert.Equal(samples1[i], samples2[i]);
        }
    }

    [Fact]
    public void ClearSeed_AfterSetSeed_RestoresRandomBehavior()
    {
        SamplingHelper.SetSeed(42);
        var result1 = SamplingHelper.SampleWithReplacement(1000, 100);

        SamplingHelper.ClearSeed();

        // After clearing, sampling should produce different results
        // Note: There's a tiny chance they could match by accident, but it's astronomically unlikely
        var result2 = SamplingHelper.SampleWithReplacement(1000, 100);
        var result3 = SamplingHelper.SampleWithReplacement(1000, 100);

        // At least one pair should be different (with high probability)
        Assert.True(
            !result1.SequenceEqual(result2) || !result2.SequenceEqual(result3),
            "Expected different results after clearing seed");
    }

    [Fact]
    public void SetSeed_SequentialCalls_ProduceDeterministicSequence()
    {
        SamplingHelper.SetSeed(42);
        var first = SamplingHelper.SampleWithReplacement(100, 5);
        var second = SamplingHelper.SampleWithReplacement(100, 5);
        var third = SamplingHelper.SampleWithReplacement(100, 5);

        // Reset and verify sequence is the same
        SamplingHelper.SetSeed(42);
        var first2 = SamplingHelper.SampleWithReplacement(100, 5);
        var second2 = SamplingHelper.SampleWithReplacement(100, 5);
        var third2 = SamplingHelper.SampleWithReplacement(100, 5);

        Assert.Equal(first, first2);
        Assert.Equal(second, second2);
        Assert.Equal(third, third2);
    }

    #endregion

    #region Large Dataset Tests

    [Fact]
    public void SampleWithoutReplacement_LargePopulation_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(10000, 1000);

        Assert.Equal(1000, result.Length);
        Assert.Equal(1000, result.Distinct().Count());
        Assert.All(result, index => Assert.InRange(index, 0, 9999));
    }

    [Fact]
    public void SampleWithReplacement_LargeSample_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(1000, 10000);

        Assert.Equal(10000, result.Length);
        Assert.All(result, index => Assert.InRange(index, 0, 999));
    }

    [Fact]
    public void CreateBootstrapSamples_ManyBootstraps_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var data = new int[100];
        for (int i = 0; i < 100; i++) data[i] = i;

        var samples = SamplingHelper.CreateBootstrapSamples(data, 1000, sampleSize: 50);

        Assert.Equal(1000, samples.Count);
        Assert.All(samples, sample =>
        {
            Assert.Equal(50, sample.Length);
            Assert.All(sample, value => Assert.InRange(value, 0, 99));
        });
    }

    #endregion

    #region Statistical Properties Tests

    [Fact]
    public void SampleWithoutReplacement_RepeatedCalls_DifferentResults()
    {
        SamplingHelper.ClearSeed();

        var results = new List<int[]>();
        for (int i = 0; i < 10; i++)
        {
            results.Add(SamplingHelper.SampleWithoutReplacement(100, 10));
        }

        // Not all results should be the same
        var allSame = results.All(r => r.SequenceEqual(results[0]));
        Assert.False(allSame, "Repeated calls without seed should produce different results");
    }

    [Fact]
    public void SampleWithReplacement_Distribution_CoversMostIndices()
    {
        SamplingHelper.SetSeed(42);

        // With large enough sample from small population, should hit most indices
        var result = SamplingHelper.SampleWithReplacement(10, 1000);

        var uniqueIndices = result.Distinct().Count();
        // With 1000 samples from 10 options, we should hit all 10 with very high probability
        Assert.Equal(10, uniqueIndices);
    }

    [Fact]
    public void CreateBootstrapSamples_AverageVariation_IsReasonable()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 100);

        // Calculate average unique values in each bootstrap sample
        var avgUniqueRatio = samples.Average(s => (double)s.Distinct().Count() / s.Length);

        // With replacement, we expect around 63.2% unique values on average
        // (1 - 1/e ≈ 0.632 for large n)
        // Allow some variance but check it's reasonable
        Assert.True(avgUniqueRatio > 0.5, $"Expected unique ratio > 0.5, got {avgUniqueRatio}");
        Assert.True(avgUniqueRatio < 0.8, $"Expected unique ratio < 0.8, got {avgUniqueRatio}");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SampleWithoutReplacement_PopulationSizeZero_SampleSizeZero_ReturnsEmptyArray()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithoutReplacement(0, 0);

        Assert.Empty(result);
    }

    [Fact]
    public void SampleWithReplacement_PopulationSizeZero_SampleSizeZero_ReturnsEmptyArray()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(0, 0);

        Assert.Empty(result);
    }

    [Fact]
    public void CreateBootstrapSamples_SingleElementData_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var data = new[] { 42 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 5, sampleSize: 10);

        Assert.Equal(5, samples.Count);
        foreach (var sample in samples)
        {
            Assert.Equal(10, sample.Length);
            Assert.All(sample, value => Assert.Equal(42, value));
        }
    }

    [Fact]
    public void CreateBootstrapSamples_EmptyData_ReturnsEmptySamples()
    {
        SamplingHelper.SetSeed(42);
        var data = Array.Empty<int>();

        var samples = SamplingHelper.CreateBootstrapSamples(data, 3);

        Assert.Equal(3, samples.Count);
        Assert.All(samples, sample => Assert.Empty(sample));
    }

    [Fact]
    public void SampleWithReplacement_LargePopulationSmallSample_ValidIndices()
    {
        SamplingHelper.SetSeed(42);
        var result = SamplingHelper.SampleWithReplacement(int.MaxValue / 2, 10);

        Assert.Equal(10, result.Length);
        Assert.All(result, index => Assert.True(index >= 0));
    }

    #endregion

    #region Complex Object Tests

    private class TestPerson
    {
        public string Name { get; init; } = string.Empty;
        public int Age { get; init; }
    }

    [Fact]
    public void CreateBootstrapSamples_ComplexObjects_WorksCorrectly()
    {
        SamplingHelper.SetSeed(42);
        var data = new[]
        {
            new TestPerson { Name = "Alice", Age = 25 },
            new TestPerson { Name = "Bob", Age = 30 },
            new TestPerson { Name = "Charlie", Age = 35 }
        };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 5);

        Assert.Equal(5, samples.Count);
        foreach (var sample in samples)
        {
            Assert.Equal(3, sample.Length);
            foreach (var person in sample)
            {
                Assert.Contains(person, data);
            }
        }
    }

    [Fact]
    public void CreateBootstrapSamples_PreservesObjectReferences()
    {
        SamplingHelper.SetSeed(42);
        var person1 = new TestPerson { Name = "Alice", Age = 25 };
        var person2 = new TestPerson { Name = "Bob", Age = 30 };
        var data = new[] { person1, person2 };

        var samples = SamplingHelper.CreateBootstrapSamples(data, 3);

        foreach (var sample in samples)
        {
            foreach (var person in sample)
            {
                Assert.True(ReferenceEquals(person, person1) || ReferenceEquals(person, person2));
            }
        }
    }

    #endregion
}
