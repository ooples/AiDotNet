using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for active learning strategies implementing IActiveLearningStrategy.
/// Tests deep mathematical invariants: selection validity, informativeness ordering,
/// diversity properties, score monotonicity, and information-theoretic consistency.
/// </summary>
public abstract class ActiveLearningTestBase
{
    /// <summary>Factory method — subclasses return their concrete strategy instance.</summary>
    protected abstract IActiveLearningStrategy<double> CreateStrategy();

    /// <summary>Creates a mock model for testing.</summary>
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
            data[i] = rng.NextDouble() * 2.0 - 1.0;
        return new Tensor<double>(data, new[] { PoolSize, FeatureDim });
    }

    // =========================================================================
    // SELECTION VALIDITY INVARIANTS
    // =========================================================================

    // INVARIANT 1: Selected count matches requested batch size
    [Fact]
    public void SelectSamples_ReturnsRequestedCount()
    {
        var strategy = CreateStrategy();
        var selected = strategy.SelectSamples(CreateMockModel(), CreateUnlabeledPool(), BatchSize);
        Assert.Equal(BatchSize, selected.Length);
    }

    // INVARIANT 2: All selected indices are unique
    [Fact]
    public void SelectSamples_IndicesAreUnique()
    {
        var strategy = CreateStrategy();
        var selected = strategy.SelectSamples(CreateMockModel(), CreateUnlabeledPool(), BatchSize);
        Assert.Equal(selected.Length, new HashSet<int>(selected).Count);
    }

    // INVARIANT 3: All selected indices are within valid range
    [Fact]
    public void SelectSamples_IndicesInRange()
    {
        var strategy = CreateStrategy();
        var selected = strategy.SelectSamples(CreateMockModel(), CreateUnlabeledPool(), BatchSize);
        foreach (var idx in selected)
        {
            Assert.True(idx >= 0 && idx < PoolSize,
                $"Selected index {idx} out of range [0, {PoolSize}).");
        }
    }

    // =========================================================================
    // INFORMATIVENESS ORDERING INVARIANTS
    // =========================================================================

    // INVARIANT 4: Selected samples have higher-than-average informativeness scores
    // The whole point of active learning: selected samples should be MORE informative.
    [Fact]
    public void SelectSamples_SelectedHaveHigherThanAverageScores()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var scores = strategy.ComputeInformativenessScores(model, pool);
        var selected = strategy.SelectSamples(model, pool, BatchSize);

        // Compute average score of ALL samples
        double totalScore = 0;
        for (int i = 0; i < scores.Length; i++)
            totalScore += scores[i];
        double avgScore = totalScore / scores.Length;

        // Compute average score of SELECTED samples
        double selectedTotal = 0;
        foreach (var idx in selected)
            selectedTotal += scores[idx];
        double selectedAvg = selectedTotal / selected.Length;

        // Selected samples should have scores >= average (they're the "most informative")
        Assert.True(selectedAvg >= avgScore - 1e-10,
            $"Selected samples' avg score ({selectedAvg:F6}) should be >= " +
            $"pool average ({avgScore:F6}). Active learning selects the MOST informative samples.");
    }

    // INVARIANT 5: Scores are non-negative (informativeness is a non-negative measure)
    [Fact]
    public void ComputeScores_AreNonNegative()
    {
        var strategy = CreateStrategy();
        var scores = strategy.ComputeInformativenessScores(CreateMockModel(), CreateUnlabeledPool());

        for (int i = 0; i < scores.Length; i++)
        {
            Assert.True(scores[i] >= -1e-10,
                $"Informativeness score at index {i} = {scores[i]:E4} is negative. " +
                "Uncertainty/informativeness measures should be non-negative.");
        }
    }

    // INVARIANT 6: Scores are finite
    [Fact]
    public void ComputeScores_AreFinite()
    {
        var strategy = CreateStrategy();
        var scores = strategy.ComputeInformativenessScores(CreateMockModel(), CreateUnlabeledPool());

        Assert.Equal(PoolSize, scores.Length);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.False(double.IsNaN(scores[i]), $"Score[{i}] is NaN.");
            Assert.False(double.IsInfinity(scores[i]), $"Score[{i}] is Infinity.");
        }
    }

    // INVARIANT 7: Score count matches pool size
    [Fact]
    public void ComputeScores_CountMatchesPoolSize()
    {
        var strategy = CreateStrategy();
        var scores = strategy.ComputeInformativenessScores(CreateMockModel(), CreateUnlabeledPool());
        Assert.Equal(PoolSize, scores.Length);
    }

    // =========================================================================
    // DIVERSITY & COVERAGE INVARIANTS
    // =========================================================================

    // INVARIANT 8: Selected samples are spread across the input space
    // Pairwise distances between selected samples should be non-trivial (not all identical).
    [Fact]
    public void SelectSamples_AreSpreadAcrossInputSpace()
    {
        var strategy = CreateStrategy();
        var pool = CreateUnlabeledPool();
        var selected = strategy.SelectSamples(CreateMockModel(), pool, BatchSize);

        // Compute pairwise L2 distances between selected samples
        double minDist = double.MaxValue;
        for (int a = 0; a < selected.Length; a++)
        {
            for (int b = a + 1; b < selected.Length; b++)
            {
                double dist = 0;
                for (int d = 0; d < FeatureDim; d++)
                {
                    double diff = pool[selected[a] * FeatureDim + d] - pool[selected[b] * FeatureDim + d];
                    dist += diff * diff;
                }

                dist = Math.Sqrt(dist);
                minDist = Math.Min(minDist, dist);
            }
        }

        // Selected samples should not all be identical points
        Assert.True(minDist > 1e-10,
            $"Minimum pairwise distance between selected samples is {minDist:E4}. " +
            "Selected samples should represent diverse regions of the input space.");
    }

    // INVARIANT 9: Selecting k<n from pool is a SUBSET of selecting k+1<n
    // If you select 3 samples, the top-1 sample should also appear in the top-5.
    [Fact]
    public void SelectSamples_TopKContainsTopOne()
    {
        var strategy = CreateStrategy();
        var model = CreateMockModel();
        var pool = CreateUnlabeledPool();

        var top1 = strategy.SelectSamples(model, pool, 1);
        var top5 = strategy.SelectSamples(model, pool, Math.Min(5, PoolSize));

        // The single most informative sample should appear in the top-5
        bool top1InTop5 = Array.IndexOf(top5, top1[0]) >= 0;
        Assert.True(top1InTop5,
            $"Most informative sample (index {top1[0]}) should appear in top-5 selection. " +
            "Active learning selection should be consistent across batch sizes.");
    }

    // =========================================================================
    // EDGE CASE INVARIANTS
    // =========================================================================

    // INVARIANT 10: Requesting more than pool size is handled gracefully
    [Fact]
    public void SelectSamples_RequestMoreThanPool_HandlesGracefully()
    {
        var strategy = CreateStrategy();
        int[] selected;
        try
        {
            selected = strategy.SelectSamples(CreateMockModel(), CreateUnlabeledPool(), PoolSize + 10);
        }
        catch (ArgumentException)
        {
            return; // Throwing is acceptable
        }

        Assert.True(selected.Length <= PoolSize,
            $"Selected {selected.Length} from pool of {PoolSize}.");
    }

    // INVARIANT 11: Batch size of 1 returns exactly 1
    [Fact]
    public void SelectSamples_BatchSizeOne_ReturnsSingle()
    {
        var strategy = CreateStrategy();
        var selected = strategy.SelectSamples(CreateMockModel(), CreateUnlabeledPool(), 1);
        Assert.Single(selected);
        Assert.True(selected[0] >= 0 && selected[0] < PoolSize);
    }

    // INVARIANT 12: Does not mutate input pool
    [Fact]
    public void SelectSamples_DoesNotMutatePool()
    {
        var strategy = CreateStrategy();
        var pool = CreateUnlabeledPool();

        var original = new double[pool.Length];
        for (int i = 0; i < pool.Length; i++)
            original[i] = pool[i];

        strategy.SelectSamples(CreateMockModel(), pool, BatchSize);

        for (int i = 0; i < pool.Length; i++)
        {
            Assert.True(original[i] == pool[i],
                $"Pool mutated at index {i}.");
        }
    }
}
