using AiDotNet.Evaluation.CrossValidation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Integration tests for cross-validation strategies:
/// KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit,
/// RepeatedKFold, Bootstrap, MonteCarlo, TimeSeriesSplit,
/// SlidingWindow, BlockedKFold, PurgedKFold, GroupKFold,
/// StratifiedGroupKFold, NestedCV, and CVFold struct.
/// </summary>
public class CrossValidationIntegrationTests
{
    private const int DataSize = 100;

    #region KFoldStrategy

    [Fact]
    public void KFold_DefaultK5_Produces5Splits()
    {
        var strategy = new KFoldStrategy<double>();
        Assert.Equal(5, strategy.NumSplits);
        Assert.Equal("5-Fold", strategy.Name);

        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void KFold_EachFoldCoversAllData()
    {
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            // Train + val = all data
            var allIndices = train.Concat(val).OrderBy(x => x).ToArray();
            Assert.Equal(DataSize, allIndices.Length);
            Assert.Equal(Enumerable.Range(0, DataSize).ToArray(), allIndices);
        }
    }

    [Fact]
    public void KFold_ValidationSizesRoughlyEqual()
    {
        var strategy = new KFoldStrategy<double>(k: 5);
        var splits = strategy.Split(DataSize).ToList();

        var sizes = splits.Select(s => s.ValidationIndices.Length).ToList();
        // Each validation fold should be ~20 for 100 samples / 5 folds
        foreach (var size in sizes)
        {
            Assert.True(size >= 19 && size <= 21, $"Validation size {size} should be ~20");
        }
    }

    [Fact]
    public void KFold_NoOverlapBetweenTrainAndVal()
    {
        var strategy = new KFoldStrategy<double>(k: 5, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            var overlap = train.Intersect(val).Count();
            Assert.Equal(0, overlap);
        }
    }

    [Fact]
    public void KFold_K10_Produces10Splits()
    {
        var strategy = new KFoldStrategy<double>(k: 10);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(10, splits.Count);
    }

    [Fact]
    public void KFold_InvalidK_Throws()
    {
        Assert.Throws<ArgumentException>(() => new KFoldStrategy<double>(k: 1));
        Assert.Throws<ArgumentException>(() => new KFoldStrategy<double>(k: 0));
    }

    [Fact]
    public void KFold_DataSizeLessThanK_Throws()
    {
        var strategy = new KFoldStrategy<double>(k: 5);
        Assert.Throws<ArgumentException>(() => strategy.Split(3).ToList());
    }

    [Fact]
    public void KFold_SeededReproducible()
    {
        var s1 = new KFoldStrategy<double>(k: 5, shuffle: true, randomSeed: 42);
        var s2 = new KFoldStrategy<double>(k: 5, shuffle: true, randomSeed: 42);

        var splits1 = s1.Split(DataSize).ToList();
        var splits2 = s2.Split(DataSize).ToList();

        for (int i = 0; i < splits1.Count; i++)
        {
            Assert.Equal(splits1[i].TrainIndices, splits2[i].TrainIndices);
            Assert.Equal(splits1[i].ValidationIndices, splits2[i].ValidationIndices);
        }
    }

    [Fact]
    public void KFold_Description_ContainsInfo()
    {
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: true);
        Assert.Contains("5-fold", strategy.Description);
        Assert.Contains("shuffling", strategy.Description);
    }

    #endregion

    #region StratifiedKFoldStrategy

    [Fact]
    public void StratifiedKFold_DefaultK5_Produces5Splits()
    {
        var strategy = new StratifiedKFoldStrategy<double>(k: 5);
        Assert.Equal(5, strategy.NumSplits);
    }

    [Fact]
    public void StratifiedKFold_InvalidK_Throws()
    {
        Assert.Throws<ArgumentException>(() => new StratifiedKFoldStrategy<double>(k: 1));
    }

    [Fact]
    public void StratifiedKFold_WithLabels_NoOverlap()
    {
        var strategy = new StratifiedKFoldStrategy<double>(k: 5, randomSeed: 42);
        var labels = new double[DataSize];
        for (int i = 0; i < DataSize; i++) labels[i] = i % 3; // 3 classes

        var splits = strategy.Split(DataSize, labels).ToList();
        Assert.Equal(5, splits.Count);

        foreach (var (train, val) in splits)
        {
            var overlap = train.Intersect(val).Count();
            Assert.Equal(0, overlap);
        }
    }

    #endregion

    #region LeaveOneOutStrategy

    [Fact]
    public void LeaveOneOut_ProducesNSplits()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(10).ToList();

        Assert.Equal(10, splits.Count);
        Assert.Equal("LeaveOneOut", strategy.Name);
    }

    [Fact]
    public void LeaveOneOut_ValidationIsOneElement()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(10).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Single(val);
            Assert.Equal(9, train.Length);
        }
    }

    [Fact]
    public void LeaveOneOut_EachSampleValidatedOnce()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        var splits = strategy.Split(10).ToList();

        var validatedIndices = splits.Select(s => s.ValidationIndices[0]).OrderBy(x => x).ToArray();
        Assert.Equal(Enumerable.Range(0, 10).ToArray(), validatedIndices);
    }

    [Fact]
    public void LeaveOneOut_TooFewSamples_Throws()
    {
        var strategy = new LeaveOneOutStrategy<double>();
        Assert.Throws<ArgumentException>(() => strategy.Split(1).ToList());
    }

    #endregion

    #region LeavePOutStrategy

    [Fact]
    public void LeavePOut_Default_P2()
    {
        var strategy = new LeavePOutStrategy<double>(p: 2);
        Assert.Equal("Leave-2-Out", strategy.Name);
    }

    [Fact]
    public void LeavePOut_ValidationSizeIsP()
    {
        var strategy = new LeavePOutStrategy<double>(p: 2, maxFolds: 10);
        var splits = strategy.Split(10).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(2, val.Length);
            Assert.Equal(8, train.Length);
        }
    }

    [Fact]
    public void LeavePOut_InvalidP_Throws()
    {
        Assert.Throws<ArgumentException>(() => new LeavePOutStrategy<double>(p: 0));
    }

    #endregion

    #region ShuffleSplitStrategy

    [Fact]
    public void ShuffleSplit_DefaultProduces10Splits()
    {
        var strategy = new ShuffleSplitStrategy<double>();
        Assert.Equal(10, strategy.NumSplits);
    }

    [Fact]
    public void ShuffleSplit_TestSizeApproximatelyCorrect()
    {
        var strategy = new ShuffleSplitStrategy<double>(numSplits: 5, testSize: 0.2, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            // 20% of 100 = 20
            Assert.True(val.Length >= 18 && val.Length <= 22, $"Validation size {val.Length} should be ~20");
            Assert.True(train.Length >= 78 && train.Length <= 82, $"Train size {train.Length} should be ~80");
        }
    }

    [Fact]
    public void ShuffleSplit_InvalidTestSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ShuffleSplitStrategy<double>(testSize: 0.0));
        Assert.Throws<ArgumentException>(() => new ShuffleSplitStrategy<double>(testSize: 1.0));
    }

    [Fact]
    public void ShuffleSplit_InvalidNumSplits_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ShuffleSplitStrategy<double>(numSplits: 0));
    }

    #endregion

    #region RepeatedKFoldStrategy

    [Fact]
    public void RepeatedKFold_ProducesKTimesRSplits()
    {
        var strategy = new RepeatedKFoldStrategy<double>(k: 5, repetitions: 3, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(15, splits.Count); // 5 * 3
    }

    [Fact]
    public void RepeatedKFold_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new RepeatedKFoldStrategy<double>(k: 1));
        Assert.Throws<ArgumentException>(() => new RepeatedKFoldStrategy<double>(k: 5, repetitions: 0));
    }

    [Fact]
    public void RepeatedKFold_Properties()
    {
        var strategy = new RepeatedKFoldStrategy<double>(k: 5, repetitions: 10);
        Assert.Equal(50, strategy.NumSplits);
        Assert.Contains("5-Fold", strategy.Name);
    }

    #endregion

    #region BootstrapStrategy

    [Fact]
    public void Bootstrap_DefaultProduces100Splits()
    {
        var strategy = new BootstrapStrategy<double>();
        Assert.Equal(100, strategy.NumSplits);
    }

    [Fact]
    public void Bootstrap_ProducesCorrectSplitCount()
    {
        var strategy = new BootstrapStrategy<double>(numBootstraps: 10, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(10, splits.Count);
    }

    [Fact]
    public void Bootstrap_TrainSizeEqualsDataSize()
    {
        var strategy = new BootstrapStrategy<double>(numBootstraps: 5, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            // Bootstrap train set has same size as data (with replacement)
            Assert.Equal(DataSize, train.Length);
            // Validation is out-of-bag samples
            Assert.True(val.Length > 0, "OOB samples should exist");
        }
    }

    [Fact]
    public void Bootstrap_InvalidNumBootstraps_Throws()
    {
        Assert.Throws<ArgumentException>(() => new BootstrapStrategy<double>(numBootstraps: 0));
    }

    #endregion

    #region MonteCarloStrategy

    [Fact]
    public void MonteCarlo_ProducesCorrectSplitCount()
    {
        var strategy = new MonteCarloStrategy<double>(nIterations: 20, testSize: 0.2, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(20, splits.Count);
    }

    [Fact]
    public void MonteCarlo_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MonteCarloStrategy<double>(nIterations: 0));
        Assert.Throws<ArgumentException>(() => new MonteCarloStrategy<double>(testSize: 0.0));
        Assert.Throws<ArgumentException>(() => new MonteCarloStrategy<double>(testSize: 1.0));
    }

    [Fact]
    public void MonteCarlo_Properties()
    {
        var strategy = new MonteCarloStrategy<double>(nIterations: 50);
        Assert.Equal(50, strategy.NumSplits);
        Assert.Contains("Monte Carlo", strategy.Name);
    }

    #endregion

    #region TimeSeriesSplitStrategy

    [Fact]
    public void TimeSeriesSplit_ProducesCorrectSplits()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 5);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void TimeSeriesSplit_TrainingSizeGrows()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 5);
        var splits = strategy.Split(DataSize).ToList();

        for (int i = 0; i < splits.Count - 1; i++)
        {
            Assert.True(splits[i + 1].TrainIndices.Length >= splits[i].TrainIndices.Length,
                "Training size should grow or stay the same across splits");
        }
    }

    [Fact]
    public void TimeSeriesSplit_TrainBeforeValidation()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            if (train.Length > 0 && val.Length > 0)
            {
                Assert.True(train.Max() < val.Min(),
                    "All train indices should be before validation indices");
            }
        }
    }

    [Fact]
    public void TimeSeriesSplit_InvalidNumSplits_Throws()
    {
        Assert.Throws<ArgumentException>(() => new TimeSeriesSplitStrategy<double>(numSplits: 1));
    }

    [Fact]
    public void TimeSeriesSplit_WithGap_GreaterThanZero()
    {
        var strategy = new TimeSeriesSplitStrategy<double>(numSplits: 3, gap: 5);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            if (train.Length > 0 && val.Length > 0)
            {
                int gapBetween = val.Min() - train.Max();
                Assert.True(gapBetween >= 5, $"Gap should be at least 5, was {gapBetween}");
            }
        }
    }

    [Fact]
    public void TimeSeriesSplit_NegativeGap_Throws()
    {
        Assert.Throws<ArgumentException>(() => new TimeSeriesSplitStrategy<double>(gap: -1));
    }

    #endregion

    #region SlidingWindowStrategy

    [Fact]
    public void SlidingWindow_ProducesSplits()
    {
        var strategy = new SlidingWindowStrategy<double>(windowSize: 20, testSize: 5);
        var splits = strategy.Split(DataSize).ToList();
        Assert.True(splits.Count > 0);
    }

    [Fact]
    public void SlidingWindow_TrainSizeEqualsWindow()
    {
        var strategy = new SlidingWindowStrategy<double>(windowSize: 20, testSize: 5);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            Assert.Equal(20, train.Length);
            Assert.Equal(5, val.Length);
        }
    }

    [Fact]
    public void SlidingWindow_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new SlidingWindowStrategy<double>(windowSize: 0, testSize: 5));
        Assert.Throws<ArgumentException>(() => new SlidingWindowStrategy<double>(windowSize: 20, testSize: 0));
    }

    #endregion

    #region BlockedKFoldStrategy

    [Fact]
    public void BlockedKFold_ProducesCorrectSplits()
    {
        var strategy = new BlockedKFoldStrategy<double>(nFolds: 5);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void BlockedKFold_NoOverlapBetweenTrainAndVal()
    {
        var strategy = new BlockedKFoldStrategy<double>(nFolds: 5, gapSize: 2);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            var overlap = train.Intersect(val).Count();
            Assert.Equal(0, overlap);
        }
    }

    [Fact]
    public void BlockedKFold_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new BlockedKFoldStrategy<double>(nFolds: 1));
        Assert.Throws<ArgumentException>(() => new BlockedKFoldStrategy<double>(gapSize: -1));
    }

    #endregion

    #region PurgedKFoldStrategy

    [Fact]
    public void PurgedKFold_ProducesCorrectSplits()
    {
        var strategy = new PurgedKFoldStrategy<double>(k: 5);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void PurgedKFold_NoOverlap()
    {
        var strategy = new PurgedKFoldStrategy<double>(k: 5, purgeGap: 3);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            var overlap = train.Intersect(val).Count();
            Assert.Equal(0, overlap);
        }
    }

    [Fact]
    public void PurgedKFold_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new PurgedKFoldStrategy<double>(k: 1));
        Assert.Throws<ArgumentException>(() => new PurgedKFoldStrategy<double>(purgeGap: -1));
    }

    #endregion

    #region GroupKFoldStrategy

    [Fact]
    public void GroupKFold_ProducesSplits()
    {
        var groups = new int[DataSize];
        for (int i = 0; i < DataSize; i++) groups[i] = i % 10; // 10 groups

        var strategy = new GroupKFoldStrategy<double>(groups, k: 5);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void GroupKFold_SameGroupNotInTrainAndVal()
    {
        var groups = new int[DataSize];
        for (int i = 0; i < DataSize; i++) groups[i] = i % 10;

        var strategy = new GroupKFoldStrategy<double>(groups, k: 5);
        var splits = strategy.Split(DataSize).ToList();

        foreach (var (train, val) in splits)
        {
            var trainGroups = train.Select(i => groups[i]).Distinct().ToHashSet();
            var valGroups = val.Select(i => groups[i]).Distinct().ToHashSet();
            var overlap = trainGroups.Intersect(valGroups).Count();
            Assert.Equal(0, overlap);
        }
    }

    [Fact]
    public void GroupKFold_NullGroups_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new GroupKFoldStrategy<double>(null));
    }

    #endregion

    #region StratifiedGroupKFoldStrategy

    [Fact]
    public void StratifiedGroupKFold_ProducesSplits()
    {
        var groups = new int[DataSize];
        for (int i = 0; i < DataSize; i++) groups[i] = i % 10;

        var strategy = new StratifiedGroupKFoldStrategy<double>(nFolds: 5, groups: groups, randomSeed: 42);
        var labels = new double[DataSize];
        for (int i = 0; i < DataSize; i++) labels[i] = i % 3;

        var splits = strategy.Split(DataSize, labels).ToList();
        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void StratifiedGroupKFold_InvalidFolds_Throws()
    {
        Assert.Throws<ArgumentException>(() => new StratifiedGroupKFoldStrategy<double>(nFolds: 1));
    }

    #endregion

    #region NestedCVStrategy

    [Fact]
    public void NestedCV_ProducesSplits()
    {
        var strategy = new NestedCVStrategy<double>(outerFolds: 3, innerFolds: 2, randomSeed: 42);
        var splits = strategy.Split(DataSize).ToList();
        Assert.Equal(3, splits.Count);
    }

    [Fact]
    public void NestedCV_InvalidParams_Throws()
    {
        Assert.Throws<ArgumentException>(() => new NestedCVStrategy<double>(outerFolds: 1));
        Assert.Throws<ArgumentException>(() => new NestedCVStrategy<double>(innerFolds: 1));
    }

    [Fact]
    public void NestedCV_Properties()
    {
        var strategy = new NestedCVStrategy<double>(outerFolds: 5, innerFolds: 3);
        Assert.Equal(5, strategy.NumSplits);
        Assert.Contains("Nested", strategy.Name);
    }

    #endregion

    #region CVFold Struct

    [Fact]
    public void CVFold_Properties_SetCorrectly()
    {
        var fold = new CVFold<double>
        {
            FoldIndex = 2,
            TrainIndices = new[] { 0, 1, 2 },
            ValidationIndices = new[] { 3, 4 }
        };

        Assert.Equal(2, fold.FoldIndex);
        Assert.Equal(3, fold.TrainSize);
        Assert.Equal(2, fold.ValidationSize);
    }

    [Fact]
    public void CVFold_DefaultValues()
    {
        var fold = new CVFold<double>();
        Assert.Equal(0, fold.FoldIndex);
        Assert.Equal(0, fold.TrainSize);
        Assert.Equal(0, fold.ValidationSize);
    }

    #endregion

    #region Cross-Strategy Tests

    [Fact]
    public void AllStrategies_Float_Work()
    {
        // Verify all strategies work with float type
        var kfold = new KFoldStrategy<float>(k: 3);
        var splits = kfold.Split(30).ToList();
        Assert.Equal(3, splits.Count);

        var loo = new LeaveOneOutStrategy<float>();
        var looSplits = loo.Split(5).ToList();
        Assert.Equal(5, looSplits.Count);
    }

    [Fact]
    public void AllStrategies_NoOverlap()
    {
        // Test that train and validation never overlap for basic strategies
        var strategies = new ICrossValidationStrategy<double>[]
        {
            new KFoldStrategy<double>(k: 5, randomSeed: 42),
            new LeaveOneOutStrategy<double>(),
            new LeavePOutStrategy<double>(p: 2, maxFolds: 20),
            new BlockedKFoldStrategy<double>(nFolds: 5),
            new PurgedKFoldStrategy<double>(k: 5),
        };

        int testSize = 20; // Small size for combinatorial strategies

        foreach (var strategy in strategies)
        {
            var splits = strategy.Split(testSize).ToList();
            foreach (var (train, val) in splits)
            {
                var overlap = train.Intersect(val).Count();
                Assert.Equal(0, overlap);
            }
        }
    }

    #endregion
}
