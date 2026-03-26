using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for active learning strategies implementing IActiveLearningStrategy.
/// Tests mathematical invariants: index validity, uniqueness, count matching,
/// score finiteness, and input immutability.
/// </summary>
public abstract class ActiveLearningTestBase
{
    /// <summary>Factory method — subclasses return their concrete strategy instance.</summary>
    protected abstract IActiveLearningStrategy<double> CreateStrategy();

    /// <summary>Creates a mock model for testing. Override if the strategy needs a specific model type.</summary>
    protected abstract IFullModel<double, Tensor<double>, Tensor<double>> CreateMockModel();

    /// <summary>Number of samples in the unlabeled pool.</summary>
    protected virtual int PoolSize => 20;

    /// <summary>Feature dimension of each sample.</summary>
    protected virtual int FeatureDim => 4;

    /// <summary>Number of samples to select per batch.</summary>
    protected virtual int BatchSize => 5;

    /// <summary>Creates a synthetic unlabeled pool tensor [poolSize, featureDim].</summary>
    protected virtual Tensor<double> CreateUnlabeledPool()
    {
        var rng = new Random(42);
        var data = new double[PoolSize * FeatureDim];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble() * 2.0 - 1.0;
        }

        return new Tensor<double>(data, new[] { PoolSize, FeatureDim });
    }

    // =========================================================================
    // INVARIANT 1: Selected count matches requested batch size
    // =========================================================================

    [Fact]
    public void SelectSamples_ReturnsRequestedCount()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var selected = strategy.SelectSamples(model, pool, BatchSize);

        Assert.Equal(BatchSize, selected.Length);
    }

    // =========================================================================
    // INVARIANT 2: All selected indices are unique (no duplicates)
    // =========================================================================

    [Fact]
    public void SelectSamples_IndicesAreUnique()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var selected = strategy.SelectSamples(model, pool, BatchSize);

        var uniqueIndices = new HashSet<int>(selected);
        Assert.Equal(selected.Length, uniqueIndices.Count);
    }

    // =========================================================================
    // INVARIANT 3: All selected indices are within valid range [0, poolSize)
    // =========================================================================

    [Fact]
    public void SelectSamples_IndicesInRange()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var selected = strategy.SelectSamples(model, pool, BatchSize);

        foreach (var idx in selected)
        {
            Assert.True(idx >= 0 && idx < PoolSize,
                $"Selected index {idx} is out of range [0, {PoolSize}). " +
                "Active learning must select from the unlabeled pool.");
        }
    }

    // =========================================================================
    // INVARIANT 4: Informativeness scores are finite
    // =========================================================================

    [Fact]
    public void ComputeScores_AreFinite()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var scores = strategy.ComputeInformativenessScores(model, pool);

        Assert.Equal(PoolSize, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.False(double.IsNaN(scores[i]),
                $"Informativeness score at index {i} is NaN.");
            Assert.False(double.IsInfinity(scores[i]),
                $"Informativeness score at index {i} is Infinity.");
        }
    }

    // =========================================================================
    // INVARIANT 5: Requesting more than pool size is handled gracefully
    // =========================================================================

    [Fact]
    public void SelectSamples_RequestMoreThanPool_HandlesGracefully()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        int[] selected;
        try
        {
            selected = strategy.SelectSamples(model, pool, PoolSize + 10);
        }
        catch (ArgumentException)
        {
            // Throwing is acceptable behavior for invalid request
            return;
        }

        // If it doesn't throw, it should return at most poolSize samples
        Assert.True(selected.Length <= PoolSize,
            $"Selected {selected.Length} samples from pool of {PoolSize}. " +
            "Cannot select more samples than available in pool.");
    }

    // =========================================================================
    // INVARIANT 6: Does not mutate input pool
    // =========================================================================

    [Fact]
    public void SelectSamples_DoesNotMutatePool()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        // Clone pool data
        var original = new double[pool.Length];
        for (int i = 0; i < pool.Length; i++)
            original[i] = pool[i];

        strategy.SelectSamples(model, pool, BatchSize);

        for (int i = 0; i < pool.Length; i++)
        {
            Assert.True(original[i] == pool[i],
                $"Unlabeled pool was mutated at flat index {i}. " +
                "Active learning strategies must not modify input data.");
        }
    }

    // =========================================================================
    // INVARIANT 7: Batch size of 1 returns exactly 1 sample
    // =========================================================================

    [Fact]
    public void SelectSamples_BatchSizeOne_ReturnsSingleSample()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var selected = strategy.SelectSamples(model, pool, 1);

        Assert.Single(selected);
        Assert.True(selected[0] >= 0 && selected[0] < PoolSize);
    }
}
