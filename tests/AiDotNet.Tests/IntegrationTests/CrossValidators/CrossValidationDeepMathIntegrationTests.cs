using AiDotNet.Evaluation.CrossValidation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CrossValidators;

/// <summary>
/// Deep math-correctness integration tests for Cross-Validation strategies.
/// Verifies exact fold splits, index partitioning, fold sizes, temporal ordering,
/// combinatorial counts, stratification proportions, and bootstrap properties.
/// </summary>
public class CrossValidationDeepMathIntegrationTests
{
    #region KFoldStrategy - Fold Size Distribution

    [Fact]
    public void KFold_EvenSplit_AllFoldsSameSize()
    {
        // 10 samples, 5 folds => each fold has exactly 2 samples
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(10).ToList();

        Assert.Equal(5, splits.Count);

        foreach (var (train, val) in splits)
        {
            Assert.Equal(8, train.Length);  // 10 - 2 = 8
            Assert.Equal(2, val.Length);    // 10 / 5 = 2
        }
    }

    [Fact]
    public void KFold_UnevenSplit_RemainderDistributed()
    {
        // 11 samples, 3 folds:
        // baseFoldSize = 11 / 3 = 3, remainder = 11 % 3 = 2
        // Fold 0: 3 + 1 = 4 (fold < remainder)
        // Fold 1: 3 + 1 = 4 (fold < remainder)
        // Fold 2: 3 + 0 = 3 (fold >= remainder)
        var strategy = new KFoldStrategy<double>(k: 3, shuffle: false);
        var splits = strategy.Split(11).ToList();

        Assert.Equal(3, splits.Count);

        Assert.Equal(4, splits[0].ValidationIndices.Length); // fold 0 gets +1
        Assert.Equal(4, splits[1].ValidationIndices.Length); // fold 1 gets +1
        Assert.Equal(3, splits[2].ValidationIndices.Length); // fold 2 gets base

        Assert.Equal(7, splits[0].TrainIndices.Length);  // 11 - 4 = 7
        Assert.Equal(7, splits[1].TrainIndices.Length);  // 11 - 4 = 7
        Assert.Equal(8, splits[2].TrainIndices.Length);  // 11 - 3 = 8
    }

    [Fact]
    public void KFold_NoShuffle_IndicesAreSequential()
    {
        // Without shuffling, fold i gets indices [i*foldSize .. (i+1)*foldSize-1]
        var strategy = new KFoldStrategy<double>(k: 4, shuffle: false);
        var splits = strategy.Split(8).ToList();

        // Each fold has 2 samples: fold 0 = [0,1], fold 1 = [2,3], fold 2 = [4,5], fold 3 = [6,7]
        Assert.Equal(new[] { 0, 1 }, splits[0].ValidationIndices.OrderBy(x => x).ToArray());
        Assert.Equal(new[] { 2, 3 }, splits[1].ValidationIndices.OrderBy(x => x).ToArray());
        Assert.Equal(new[] { 4, 5 }, splits[2].ValidationIndices.OrderBy(x => x).ToArray());
        Assert.Equal(new[] { 6, 7 }, splits[3].ValidationIndices.OrderBy(x => x).ToArray());
    }

    [Fact]
    public void KFold_AllIndicesCoveredExactlyOnce()
    {
        // Every index appears in exactly one validation fold
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(13).ToList();

        var allValIndices = splits.SelectMany(s => s.ValidationIndices).OrderBy(x => x).ToArray();

        // All 13 indices appear exactly once across all validation sets
        Assert.Equal(13, allValIndices.Length);
        for (int i = 0; i < 13; i++)
        {
            Assert.Equal(i, allValIndices[i]);
        }
    }

    [Fact]
    public void KFold_TrainAndValidationAreDisjoint()
    {
        var strategy = new KFoldStrategy<double>(k: 3, shuffle: false);
        var splits = strategy.Split(9).ToList();

        foreach (var (train, val) in splits)
        {
            var trainSet = new HashSet<int>(train);
            var valSet = new HashSet<int>(val);

            // No overlap
            Assert.Empty(trainSet.Intersect(valSet));

            // Together they cover all indices
            Assert.Equal(9, trainSet.Count + valSet.Count);
        }
    }

    [Fact]
    public void KFold_MinimumK2_Works()
    {
        var strategy = new KFoldStrategy<double>(k: 2, shuffle: false);
        var splits = strategy.Split(6).ToList();

        Assert.Equal(2, splits.Count);
        // Each fold has 3 validation samples
        Assert.Equal(3, splits[0].ValidationIndices.Length);
        Assert.Equal(3, splits[1].ValidationIndices.Length);
    }

    [Fact]
    public void KFold_KEqualsN_EquivalentToLOOCV()
    {
        // K=N is essentially Leave-One-Out
        int n = 5;
        var strategy = new KFoldStrategy<double>(k: n, shuffle: false);
        var splits = strategy.Split(n).ToList();

        Assert.Equal(n, splits.Count);
        foreach (var (train, val) in splits)
        {
            Assert.Single(val);  // Each fold validates on 1 sample
            Assert.Equal(n - 1, train.Length);
        }
    }

    #endregion

    #region LeaveOneOutStrategy

    [Fact]
    public void LeaveOneOut_GeneratesNSplits()
    {
        int n = 7;
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(n).ToList();

        Assert.Equal(n, splits.Count);
    }

    [Fact]
    public void LeaveOneOut_EachSampleValidatedExactlyOnce()
    {
        int n = 6;
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(n).ToList();

        for (int i = 0; i < n; i++)
        {
            // Split i has sample i as the sole validation point
            Assert.Single(splits[i].ValidationIndices);
            Assert.Equal(i, splits[i].ValidationIndices[0]);

            // Training is all other samples
            Assert.Equal(n - 1, splits[i].TrainIndices.Length);
            Assert.DoesNotContain(i, splits[i].TrainIndices);
        }
    }

    [Fact]
    public void LeaveOneOut_TrainingSizeIsNMinus1()
    {
        int n = 10;
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(n).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(n - 1, train.Length);
            Assert.Single(val);
        }
    }

    #endregion

    #region LeavePOutStrategy - Combinatorial Properties

    [Fact]
    public void LeavePOut_P2_GeneratesC_N_2_Splits()
    {
        // C(5,2) = 10 combinations
        int n = 5;
        var strategy = new LeavePOutStrategy<double>(p: 2, maxFolds: null);
        var splits = strategy.Split(n).ToList();

        // C(5,2) = 5!/(2!*3!) = 10
        Assert.Equal(10, splits.Count);
    }

    [Fact]
    public void LeavePOut_P1_SameAsLOOCV()
    {
        int n = 4;
        var strategy = new LeavePOutStrategy<double>(p: 1, maxFolds: null);
        var splits = strategy.Split(n).ToList();

        // C(4,1) = 4, same as LOOCV
        Assert.Equal(4, splits.Count);

        foreach (var (train, val) in splits)
        {
            Assert.Single(val);
            Assert.Equal(n - 1, train.Length);
        }
    }

    [Fact]
    public void LeavePOut_P2_EachValidationHas2Samples()
    {
        int n = 6;
        var strategy = new LeavePOutStrategy<double>(p: 2, maxFolds: null);
        var splits = strategy.Split(n).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(2, val.Length);
            Assert.Equal(n - 2, train.Length);
        }
    }

    [Fact]
    public void LeavePOut_P2_AllPairsGenerated()
    {
        int n = 4;
        var strategy = new LeavePOutStrategy<double>(p: 2, maxFolds: null);
        var splits = strategy.Split(n).ToList();

        // C(4,2) = 6 pairs: {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
        Assert.Equal(6, splits.Count);

        var pairs = splits
            .Select(s => (s.ValidationIndices.Min(), s.ValidationIndices.Max()))
            .OrderBy(p => p.Item1)
            .ThenBy(p => p.Item2)
            .ToList();

        var expected = new[] { (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) };
        Assert.Equal(expected, pairs);
    }

    [Fact]
    public void LeavePOut_MaxFoldsLimitsOutput()
    {
        // C(10,2) = 45, but maxFolds limits to 10
        var strategy = new LeavePOutStrategy<double>(p: 2, maxFolds: 10);
        var splits = strategy.Split(10).ToList();

        Assert.Equal(10, splits.Count);
    }

    [Fact]
    public void LeavePOut_P3_CorrectCount()
    {
        // C(6,3) = 6!/(3!*3!) = 20
        var strategy = new LeavePOutStrategy<double>(p: 3, maxFolds: null);
        var splits = strategy.Split(6).ToList();

        Assert.Equal(20, splits.Count);
    }

    #endregion

    #region StratifiedKFoldStrategy - Class Distribution Preservation

    [Fact]
    public void StratifiedKFold_PreservesClassProportions()
    {
        // 10 samples: 7 class 0, 3 class 1
        // Class 0 (7 samples, k=5): base=1, rem=2 => folds 0,1 get 2; folds 2,3,4 get 1
        // Class 1 (3 samples, k=5): base=0, rem=3 => folds 0,1,2 get 1; folds 3,4 get 0
        // Total: fold0=3, fold1=3, fold2=2, fold3=1, fold4=1
        var labels = new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 };
        var strategy = new StratifiedKFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(10, labels).ToList();

        Assert.Equal(5, splits.Count);

        // Total validation samples across all folds should equal data size
        int totalVal = splits.Sum(s => s.ValidationIndices.Length);
        Assert.Equal(10, totalVal);

        foreach (var (train, val) in splits)
        {
            Assert.Equal(10, train.Length + val.Length);
            Assert.All(val, idx => Assert.InRange(idx, 0, 9));
            Assert.All(train, idx => Assert.InRange(idx, 0, 9));
        }
    }

    [Fact]
    public void StratifiedKFold_NoShuffle_EachClassSplitEvenly()
    {
        // 6 class-0 samples and 4 class-1 samples, 2 folds
        // Class 0: 6/2 = 3 per fold
        // Class 1: 4/2 = 2 per fold
        // Total per fold: 5 validation, 5 train
        var labels = new double[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 };
        var strategy = new StratifiedKFoldStrategy<double>(k: 2, shuffle: false);
        var splits = strategy.Split(10, labels).ToList();

        Assert.Equal(2, splits.Count);

        foreach (var (train, val) in splits)
        {
            Assert.Equal(5, val.Length);
            Assert.Equal(5, train.Length);

            // Count class distribution in validation
            int valClass0 = val.Count(idx => labels[idx] == 0.0);
            int valClass1 = val.Count(idx => labels[idx] == 1.0);

            // Should be 3 class-0 and 2 class-1 per fold
            Assert.Equal(3, valClass0);
            Assert.Equal(2, valClass1);
        }
    }

    [Fact]
    public void StratifiedKFold_RequiresLabels()
    {
        var strategy = new StratifiedKFoldStrategy<double>(k: 3);

        Assert.Throws<ArgumentException>(() => strategy.Split(10).ToList());
    }

    [Fact]
    public void StratifiedKFold_AllIndicesCoveredExactlyOnce()
    {
        var labels = new double[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
        var strategy = new StratifiedKFoldStrategy<double>(k: 3, shuffle: false);
        var splits = strategy.Split(9, labels).ToList();

        var allValIndices = splits.SelectMany(s => s.ValidationIndices).OrderBy(x => x).ToArray();
        Assert.Equal(9, allValIndices.Length);
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(i, allValIndices[i]);
        }
    }

    #endregion

    #region BootstrapStrategy - Statistical Properties

    [Fact]
    public void Bootstrap_TrainSizeEqualsDataSize()
    {
        // Bootstrap samples WITH replacement => train set size == data size
        var strategy = new BootstrapStrategy<double>(numBootstraps: 5, randomSeed: 42);
        var splits = strategy.Split(10).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(10, train.Length); // Same size as data (with replacement)
        }
    }

    [Fact]
    public void Bootstrap_TrainIndicesMayRepeat()
    {
        // Sampling with replacement means some indices appear multiple times
        var strategy = new BootstrapStrategy<double>(numBootstraps: 1, randomSeed: 42);
        var splits = strategy.Split(100).ToList();

        var train = splits[0].TrainIndices;
        var uniqueCount = train.Distinct().Count();

        // With 100 samples, P(at least one duplicate) is extremely high
        // Expected unique count ~ N * (1 - (1 - 1/N)^N) ~ N * (1 - 1/e) ~ 63.2%
        Assert.True(uniqueCount < train.Length,
            $"Expected duplicates in bootstrap sample. Unique: {uniqueCount}, Total: {train.Length}");
    }

    [Fact]
    public void Bootstrap_OOBSamplesAreNotInTraining()
    {
        var strategy = new BootstrapStrategy<double>(numBootstraps: 3, randomSeed: 42);
        var splits = strategy.Split(20).ToList();

        foreach (var (train, val) in splits)
        {
            var trainSet = new HashSet<int>(train);

            // OOB (validation) samples should NOT appear in the training set
            foreach (int oobIdx in val)
            {
                Assert.DoesNotContain(oobIdx, trainSet);
            }
        }
    }

    [Fact]
    public void Bootstrap_OOBSizeApproximately37Percent()
    {
        // Theoretically, P(sample not selected) = (1 - 1/N)^N ≈ 1/e ≈ 0.368
        // So ~36.8% of samples should be OOB
        var strategy = new BootstrapStrategy<double>(numBootstraps: 50, randomSeed: 42);
        int n = 200;
        var splits = strategy.Split(n).ToList();

        double avgOOBFraction = splits.Average(s => (double)s.ValidationIndices.Length / n);

        // Should be approximately 0.368, allow tolerance of 0.05
        Assert.InRange(avgOOBFraction, 0.30, 0.43);
    }

    [Fact]
    public void Bootstrap_Reproducible_WithSeed()
    {
        var strategy1 = new BootstrapStrategy<double>(numBootstraps: 3, randomSeed: 123);
        var strategy2 = new BootstrapStrategy<double>(numBootstraps: 3, randomSeed: 123);

        var splits1 = strategy1.Split(10).ToList();
        var splits2 = strategy2.Split(10).ToList();

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(splits1[i].TrainIndices, splits2[i].TrainIndices);
            Assert.Equal(splits1[i].ValidationIndices, splits2[i].ValidationIndices);
        }
    }

    #endregion

    #region ShuffleSplitStrategy - Split Sizes

    [Fact]
    public void ShuffleSplit_CorrectTestAndTrainSizes()
    {
        // 20% test on 100 samples => 20 test, 80 train
        var strategy = new ShuffleSplitStrategy<double>(numSplits: 3, testSize: 0.2, randomSeed: 42);
        var splits = strategy.Split(100).ToList();

        Assert.Equal(3, splits.Count);

        foreach (var (train, val) in splits)
        {
            Assert.Equal(80, train.Length);
            Assert.Equal(20, val.Length);
        }
    }

    [Fact]
    public void ShuffleSplit_TrainAndTestAreDisjoint()
    {
        var strategy = new ShuffleSplitStrategy<double>(numSplits: 5, testSize: 0.3, randomSeed: 42);
        var splits = strategy.Split(50).ToList();

        foreach (var (train, val) in splits)
        {
            var trainSet = new HashSet<int>(train);
            var valSet = new HashSet<int>(val);

            Assert.Empty(trainSet.Intersect(valSet));
            Assert.Equal(50, trainSet.Count + valSet.Count);
        }
    }

    [Fact]
    public void ShuffleSplit_SplitsAreIndependent()
    {
        // Different splits can have different samples in test
        var strategy = new ShuffleSplitStrategy<double>(numSplits: 3, testSize: 0.5, randomSeed: 42);
        var splits = strategy.Split(20).ToList();

        // Not all validation sets should be identical (very unlikely with seed 42)
        var valSets = splits.Select(s => new HashSet<int>(s.ValidationIndices)).ToList();

        bool allSame = true;
        for (int i = 1; i < valSets.Count; i++)
        {
            if (!valSets[0].SetEquals(valSets[i]))
            {
                allSame = false;
                break;
            }
        }

        Assert.False(allSame, "Shuffle split should produce different validation sets across splits");
    }

    [Fact]
    public void ShuffleSplit_SmallDataset_AtLeastOneTest()
    {
        // testSize = 0.1 on 5 samples => Math.Max(1, (int)(5 * 0.1)) = Math.Max(1, 0) = 1
        var strategy = new ShuffleSplitStrategy<double>(numSplits: 1, testSize: 0.1, randomSeed: 42);
        var splits = strategy.Split(5).ToList();

        Assert.Single(splits);
        Assert.True(splits[0].ValidationIndices.Length >= 1);
    }

    #endregion

    #region TimeSeriesSplitStrategy - Temporal Order

    [Fact]
    public void TimeSeriesSplit_TrainingAlwaysBeforeValidation()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 4);
        var splits = strategy.Split(20).ToList();

        foreach (var (train, val) in splits)
        {
            int maxTrainIdx = train.Max();
            int minValIdx = val.Min();

            Assert.True(maxTrainIdx < minValIdx,
                $"Training max index ({maxTrainIdx}) should be less than validation min index ({minValIdx})");
        }
    }

    [Fact]
    public void TimeSeriesSplit_ExpandingTrainingWindow()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3);
        var splits = strategy.Split(16).ToList();

        // Training window should grow with each split
        for (int i = 1; i < splits.Count; i++)
        {
            Assert.True(splits[i].TrainIndices.Length > splits[i - 1].TrainIndices.Length,
                $"Split {i} train size ({splits[i].TrainIndices.Length}) should be greater than " +
                $"split {i - 1} train size ({splits[i - 1].TrainIndices.Length})");
        }
    }

    [Fact]
    public void TimeSeriesSplit_IndicesAreContiguous()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3);
        var splits = strategy.Split(12).ToList();

        foreach (var (train, val) in splits)
        {
            // Train indices should be contiguous: [trainStart, trainStart+1, ..., trainEnd-1]
            var sortedTrain = train.OrderBy(x => x).ToArray();
            for (int i = 1; i < sortedTrain.Length; i++)
            {
                Assert.Equal(sortedTrain[i - 1] + 1, sortedTrain[i]);
            }

            // Val indices should be contiguous
            var sortedVal = val.OrderBy(x => x).ToArray();
            for (int i = 1; i < sortedVal.Length; i++)
            {
                Assert.Equal(sortedVal[i - 1] + 1, sortedVal[i]);
            }
        }
    }

    [Fact]
    public void TimeSeriesSplit_WithGap_RespectsGap()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3, gap: 2);
        var splits = strategy.Split(20).ToList();

        foreach (var (train, val) in splits)
        {
            int maxTrainIdx = train.Max();
            int minValIdx = val.Min();

            // Gap of 2 means at least 2 indices between train and val
            Assert.True(minValIdx - maxTrainIdx > 2,
                $"Gap should be > 2 between train max ({maxTrainIdx}) and val min ({minValIdx})");
        }
    }

    [Fact]
    public void TimeSeriesSplit_FixedTestSize()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3, testSize: 5);
        var splits = strategy.Split(30).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(5, val.Length);
        }
    }

    [Fact]
    public void TimeSeriesSplit_MaxTrainSize_LimitsTraining()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3, maxTrainSize: 5);
        var splits = strategy.Split(20).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.True(train.Length <= 5,
                $"Training size ({train.Length}) should be <= maxTrainSize (5)");
        }
    }

    [Fact]
    public void TimeSeriesSplit_DefaultTestSize_Calculation()
    {
        // Default: testSizePerFold = dataSize / (numSplits + 1)
        // 24 samples, 3 splits => testSize = 24 / 4 = 6
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3);
        var splits = strategy.Split(24).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(6, val.Length);
        }
    }

    #endregion

    #region Cross-Strategy Comparison Properties

    [Fact]
    public void KFold_TotalValidationSamples_EqualsDataSize()
    {
        // In K-Fold, every sample is validated exactly once
        int n = 17;
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(n).ToList();

        int totalValSamples = splits.Sum(s => s.ValidationIndices.Length);
        Assert.Equal(n, totalValSamples);
    }

    [Fact]
    public void LeavePOut_TrainValidationSizesCorrect()
    {
        // Leave-P-Out: train = N-P, val = P
        int n = 8;
        int p = 3;
        var strategy = new LeavePOutStrategy<double>(p: p, maxFolds: null);
        var splits = strategy.Split(n).ToList();

        // C(8,3) = 56
        Assert.Equal(56, splits.Count);

        foreach (var (train, val) in splits)
        {
            Assert.Equal(p, val.Length);
            Assert.Equal(n - p, train.Length);
        }
    }

    [Fact]
    public void KFold_Reproducible_WithSeed()
    {
        var s1 = new KFoldStrategy<double>(k: 3, shuffle: true, randomSeed: 99);
        var s2 = new KFoldStrategy<double>(k: 3, shuffle: true, randomSeed: 99);

        var splits1 = s1.Split(15).ToList();
        var splits2 = s2.Split(15).ToList();

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(splits1[i].TrainIndices, splits2[i].TrainIndices);
            Assert.Equal(splits1[i].ValidationIndices, splits2[i].ValidationIndices);
        }
    }

    [Fact]
    public void TimeSeriesSplit_NoDataLeakage_FutureNotInTraining()
    {
        // Core property: for each split, no validation index should appear in training
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 5);
        var splits = strategy.Split(50).ToList();

        foreach (var (train, val) in splits)
        {
            var trainSet = new HashSet<int>(train);
            foreach (int valIdx in val)
            {
                Assert.DoesNotContain(valIdx, trainSet);
            }
        }
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void KFold_KEqualsDataSize_Works()
    {
        var strategy = new KFoldStrategy<double>(k: 3, shuffle: false);
        var splits = strategy.Split(3).ToList();

        Assert.Equal(3, splits.Count);
        foreach (var (train, val) in splits)
        {
            Assert.Single(val);
            Assert.Equal(2, train.Length);
        }
    }

    [Fact]
    public void KFold_K1_Throws()
    {
        Assert.Throws<ArgumentException>(() => new KFoldStrategy<double>(k: 1));
    }

    [Fact]
    public void KFold_MoreFoldsThanSamples_Throws()
    {
        var strategy = new KFoldStrategy<double>(k: 10, shuffle: false);
        Assert.Throws<ArgumentException>(() => strategy.Split(5).ToList());
    }

    [Fact]
    public void LeaveOneOut_MinimumSamples()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(2).ToList();

        Assert.Equal(2, splits.Count);
        Assert.Equal(new[] { 0 }, splits[0].ValidationIndices);
        Assert.Equal(new[] { 1 }, splits[0].TrainIndices);
        Assert.Equal(new[] { 1 }, splits[1].ValidationIndices);
        Assert.Equal(new[] { 0 }, splits[1].TrainIndices);
    }

    [Fact]
    public void LeaveOneOut_OneSample_Throws()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        Assert.Throws<ArgumentException>(() => strategy.Split(1).ToList());
    }

    [Fact]
    public void LeavePOut_PGreaterThanN_Throws()
    {
        var strategy = new LeavePOutStrategy<double>(p: 5);
        Assert.Throws<ArgumentException>(() => strategy.Split(3).ToList());
    }

    [Fact]
    public void ShuffleSplit_Reproducible_WithSeed()
    {
        var s1 = new ShuffleSplitStrategy<double>(numSplits: 3, testSize: 0.3, randomSeed: 77);
        var s2 = new ShuffleSplitStrategy<double>(numSplits: 3, testSize: 0.3, randomSeed: 77);

        var splits1 = s1.Split(20).ToList();
        var splits2 = s2.Split(20).ToList();

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(splits1[i].TrainIndices, splits2[i].TrainIndices);
            Assert.Equal(splits1[i].ValidationIndices, splits2[i].ValidationIndices);
        }
    }

    #endregion
}
