using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.ActiveLearning;

/// <summary>
/// Comprehensive tests for all Active Learning strategies.
/// </summary>
public class ActiveLearningStrategyTests
{
    #region UncertaintySampling Tests

    [Fact]
    public void UncertaintySampling_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new UncertaintySampling<double>();

        // Assert - default measure is Entropy
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Entropy", strategy.Name);
        Assert.False(strategy.UseBatchDiversity);
    }

    [Fact]
    public void UncertaintySampling_Constructor_CustomMeasure_ReflectedInName()
    {
        // Arrange & Act
        var strategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.Entropy);

        // Assert
        Assert.Contains("Entropy", strategy.Name);
    }

    [Fact]
    public void UncertaintySampling_SelectSamples_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(null!, pool, 3));
    }

    [Fact]
    public void UncertaintySampling_SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(model, null!, 3));
    }

    [Fact]
    public void UncertaintySampling_SelectSamples_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);
        var batchSize = 5;

        // Act
        var selected = strategy.SelectSamples(model, pool, batchSize);

        // Assert
        Assert.Equal(batchSize, selected.Length);
        Assert.All(selected, idx => Assert.InRange(idx, 0, 19));
        Assert.Equal(selected.Distinct().Count(), selected.Length); // All unique
    }

    [Fact]
    public void UncertaintySampling_SelectSamples_BatchSizeLargerThanPool_ReturnsPoolSize()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(5, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 10);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void UncertaintySampling_ComputeInformativenessScores_ReturnsVectorWithCorrectLength()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(15, scores.Length);
    }

    [Fact]
    public void UncertaintySampling_GetSelectionStatistics_ReturnsExpectedKeys()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act
        _ = strategy.ComputeInformativenessScores(model, pool);
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.Contains("MinScore", stats.Keys);
        Assert.Contains("MaxScore", stats.Keys);
        Assert.Contains("MeanScore", stats.Keys);
    }

    [Fact]
    public void UncertaintySampling_UseBatchDiversity_CanBeSetAndGet()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();

        // Act
        strategy.UseBatchDiversity = true;

        // Assert
        Assert.True(strategy.UseBatchDiversity);
    }

    #endregion

    #region MarginSampling Tests

    [Fact]
    public void MarginSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new MarginSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Margin", strategy.Name);
    }

    [Fact]
    public void MarginSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new MarginSampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void MarginSampling_ComputeInformativenessScores_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new MarginSampling<double>();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.ComputeInformativenessScores(null!, pool));
    }

    #endregion

    #region EntropySampling Tests

    [Fact]
    public void EntropySampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new EntropySampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Entropy", strategy.Name);
    }

    [Fact]
    public void EntropySampling_SelectSamples_ValidInput_ReturnsUniqueIndices()
    {
        // Arrange
        var strategy = new EntropySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Distinct().Count(), selected.Length);
    }

    #endregion

    #region LeastConfidenceSampling Tests

    [Fact]
    public void LeastConfidenceSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new LeastConfidenceSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("LeastConfidence", strategy.Name);
    }

    [Fact]
    public void LeastConfidenceSampling_SelectSamples_SingleSample_ReturnsOneIndex()
    {
        // Arrange
        var strategy = new LeastConfidenceSampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 1);

        // Assert
        Assert.Single(selected);
    }

    #endregion

    #region RandomSampling Tests

    [Fact]
    public void RandomSampling_Constructor_DefaultSeed_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new RandomSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Random", strategy.Name);
    }

    [Fact]
    public void RandomSampling_Constructor_WithSeed_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new RandomSampling<double>(seed: 42);

        // Assert
        Assert.NotNull(strategy);
    }

    [Fact]
    public void RandomSampling_SelectSamples_DifferentCallsSameSeed_ReturnsSameSelection()
    {
        // Arrange
        var strategy1 = new RandomSampling<double>(seed: 42);
        var strategy2 = new RandomSampling<double>(seed: 42);
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected1 = strategy1.SelectSamples(model, pool, 5);
        var selected2 = strategy2.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(selected1, selected2);
    }

    [Fact]
    public void RandomSampling_SelectSamples_ReturnsUniqueIndices()
    {
        // Arrange
        var strategy = new RandomSampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 10);

        // Assert
        Assert.Equal(10, selected.Length);
        Assert.Equal(selected.Distinct().Count(), selected.Length);
    }

    #endregion

    #region BALD Tests

    [Fact]
    public void BALD_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new BALD<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("BALD", strategy.Name);
    }

    [Fact]
    public void BALD_Constructor_CustomParameters_ReflectedInInstance()
    {
        // Arrange & Act
        var strategy = new BALD<double>(numMcSamples: 20, dropoutRate: 0.3);

        // Assert
        Assert.Contains("MC20", strategy.Name);
    }

    [Fact]
    public void BALD_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new BALD<double>(numMcSamples: 5);
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    [Fact]
    public void BALD_SelectSamples_WithDiversity_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new BALD<double>(numMcSamples: 5);
        strategy.UseBatchDiversity = true;
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    #endregion

    #region BatchBALD Tests

    [Fact]
    public void BatchBALD_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new BatchBALD<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("BatchBALD", strategy.Name);
    }

    [Fact]
    public void BatchBALD_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new BatchBALD<double>(numMcSamples: 3);
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    #endregion

    #region QueryByCommittee Tests

    [Fact]
    public void QueryByCommittee_Constructor_ValidCommittee_CreatesInstance()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel()
        };

        // Act
        var strategy = new QueryByCommittee<double>(committee);

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("QueryByCommittee", strategy.Name);
        Assert.Equal(2, strategy.Committee.Count);
    }

    [Fact]
    public void QueryByCommittee_Constructor_NullCommittee_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryByCommittee<double>(null!));
    }

    [Fact]
    public void QueryByCommittee_Constructor_SingleMember_ThrowsArgumentException()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel()
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new QueryByCommittee<double>(committee));
    }

    [Fact]
    public void QueryByCommittee_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel()
        };
        var strategy = new QueryByCommittee<double>(committee);
        var model = ActiveLearningTestHelper.CreateMockModel(); // Not used but required by interface
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    [Fact]
    public void QueryByCommittee_GetSelectionStatistics_IncludesCommitteeSize()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel()
        };
        var strategy = new QueryByCommittee<double>(committee);
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act
        _ = strategy.ComputeInformativenessScores(model, pool);
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.Contains("CommitteeSize", stats.Keys);
        Assert.Equal(3.0, stats["CommitteeSize"]);
    }

    [Fact]
    public void QueryByCommittee_Constructor_DifferentMeasures_ReflectedInName()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel(),
            ActiveLearningTestHelper.CreateMockModel()
        };

        // Act
        var strategy = new QueryByCommittee<double>(committee, QueryByCommittee<double>.DisagreementMeasure.KLDivergence);

        // Assert
        Assert.Contains("KLDivergence", strategy.Name);
    }

    #endregion

    #region DiversitySampling Tests

    [Fact]
    public void DiversitySampling_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new DiversitySampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("DiversitySampling", strategy.Name);
        Assert.True(strategy.UseBatchDiversity);
    }

    [Fact]
    public void DiversitySampling_Constructor_DifferentMethods_ReflectedInName()
    {
        // Arrange & Act
        var strategy1 = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.FarthestFirst);
        var strategy2 = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.DensityPeaks);

        // Assert
        Assert.Contains("FarthestFirst", strategy1.Name);
        Assert.Contains("DensityPeaks", strategy2.Name);
    }

    [Fact]
    public void DiversitySampling_SelectSamples_ValidInput_ReturnsUniqueIndices()
    {
        // Arrange
        var strategy = new DiversitySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
        Assert.Equal(selected.Distinct().Count(), selected.Length);
    }

    [Fact]
    public void DiversitySampling_GetSelectionStatistics_IncludesCoverageRadius()
    {
        // Arrange
        var strategy = new DiversitySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        _ = strategy.SelectSamples(model, pool, 3);
        var stats = strategy.GetSelectionStatistics();

        // Assert
        Assert.Contains("CoverageRadius", stats.Keys);
    }

    [Fact]
    public void DiversitySampling_CoverageRadius_UpdatedAfterSelection()
    {
        // Arrange
        var strategy = new DiversitySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var initialRadius = strategy.CoverageRadius;
        _ = strategy.SelectSamples(model, pool, 3);
        var finalRadius = strategy.CoverageRadius;

        // Assert
        Assert.NotEqual(initialRadius, finalRadius);
    }

    #endregion

    #region HybridSampling Tests

    [Fact]
    public void HybridSampling_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategies = new List<(IActiveLearningStrategy<double> Strategy, double Weight)>
        {
            (new UncertaintySampling<double>(), 0.5),
            (new RandomSampling<double>(), 0.5)
        };
        var strategy = new HybridSampling<double>(strategies);

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Hybrid", strategy.Name);
    }

    [Fact]
    public void HybridSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategies = new List<(IActiveLearningStrategy<double> Strategy, double Weight)>
        {
            (new UncertaintySampling<double>(), 0.5),
            (new RandomSampling<double>(), 0.5)
        };
        var strategy = new HybridSampling<double>(strategies);
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    #endregion

    #region ExpectedModelChange Tests

    [Fact]
    public void ExpectedModelChange_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new ExpectedModelChange<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("ExpectedModelChange", strategy.Name);
    }

    [Fact]
    public void ExpectedModelChange_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new ExpectedModelChange<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    #endregion

    #region VariationRatios Tests

    [Fact]
    public void VariationRatios_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new VariationRatios<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Variation", strategy.Name);
    }

    [Fact]
    public void VariationRatios_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new VariationRatios<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 4);

        // Assert
        Assert.Equal(4, selected.Length);
    }

    #endregion

    #region CoreSetSelection Tests

    [Fact]
    public void CoreSetSelection_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new CoreSetSelection<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("CoreSet", strategy.Name);
    }

    [Fact]
    public void CoreSetSelection_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new CoreSetSelection<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(20, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 5);

        // Assert
        Assert.Equal(5, selected.Length);
    }

    #endregion

    #region DensityWeightedSampling Tests

    [Fact]
    public void DensityWeightedSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new DensityWeightedSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("DensityWeighted", strategy.Name);
    }

    [Fact]
    public void DensityWeightedSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new DensityWeightedSampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 3);

        // Assert
        Assert.Equal(3, selected.Length);
    }

    #endregion

    #region InformationDensity Tests

    [Fact]
    public void InformationDensity_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new InformationDensity<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("InformationDensity", strategy.Name);
    }

    [Fact]
    public void InformationDensity_SelectSamples_ValidInput_ReturnsCorrectCount()
    {
        // Arrange
        var strategy = new InformationDensity<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act
        var selected = strategy.SelectSamples(model, pool, 4);

        // Assert
        Assert.Equal(4, selected.Length);
    }

    #endregion
}
