using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.ActiveLearning;

/// <summary>
/// Comprehensive tests for all Active Learning strategies.
/// </summary>
public class ActiveLearningStrategyTests
{
    #region UncertaintySampling Tests

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new UncertaintySampling<double>();

        // Assert - default measure is Entropy
        Assert.NotNull(strategy);
        Assert.Equal("UncertaintySampling-Entropy", strategy.Name);
        Assert.False(strategy.UseBatchDiversity);
    }

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_Constructor_CustomMeasure_ReflectedInName()
    {
        // Arrange & Act
        var strategy = new UncertaintySampling<double>(UncertaintySampling<double>.UncertaintyMeasure.Entropy);

        // Assert
        Assert.Contains("Entropy", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_SelectSamples_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(global::AiDotNet.Tests.TestNull.Of<AiDotNet.Interfaces.IFullModel<double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>>(), pool, 3));
    }

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_SelectSamples_NullPool_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new UncertaintySampling<double>();
        var model = ActiveLearningTestHelper.CreateMockModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.SelectSamples(model, global::AiDotNet.Tests.TestNull.Of<AiDotNet.Tensors.LinearAlgebra.Tensor<double>>(), 3));
    }

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_SelectSamples_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_SelectSamples_BatchSizeLargerThanPool_ReturnsPoolSize()
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

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_ComputeInformativenessScores_ReturnsVectorWithCorrectLength()
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

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_GetSelectionStatistics_ReturnsExpectedKeys()
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

    [Fact(Timeout = 60000)]
    public async Task UncertaintySampling_UseBatchDiversity_CanBeSetAndGet()
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

    [Fact(Timeout = 60000)]
    public async Task MarginSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new MarginSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Margin", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task MarginSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task MarginSampling_ComputeInformativenessScores_NullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new MarginSampling<double>();
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(10, 5);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.ComputeInformativenessScores(global::AiDotNet.Tests.TestNull.Of<AiDotNet.Interfaces.IFullModel<double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>>(), pool));
    }

    #endregion

    #region EntropySampling Tests

    [Fact(Timeout = 60000)]
    public async Task EntropySampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new EntropySampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Entropy", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task EntropySampling_SelectSamples_ValidInput_ReturnsUniqueIndices()
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

    [Fact(Timeout = 60000)]
    public async Task EntropySampling_BatchDiversity_NaNPredictions_TerminatesWithoutSelectingNaNScores()
    {
        // Arrange - NaN parameters make every logit, probability and entropy score NaN. The greedy
        // diversity loop used to retry forever when no remaining candidate beat MinValue.
        var strategy = new EntropySampling<double> { UseBatchDiversity = true };
        var model = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var nanParameters = new Vector<double>(10);
        for (int i = 0; i < nanParameters.Length; i++)
        {
            nanParameters[i] = double.NaN;
        }

        model.SetParameters(nanParameters);
        var pool = ActiveLearningTestHelper.CreateUnlabeledPool(15, 5);

        // Act - run off the test thread so the timeout can fire if selection hangs
        var selected = await Task.Run(() => strategy.SelectSamples(model, pool, 5));

        // Assert - no candidate has a comparable score, so the greedy selection stops with none
        Assert.Empty(selected);
    }

    #endregion

    #region LeastConfidenceSampling Tests

    [Fact(Timeout = 60000)]
    public async Task LeastConfidenceSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new LeastConfidenceSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("LeastConfidence", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task LeastConfidenceSampling_SelectSamples_SingleSample_ReturnsOneIndex()
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

    [Fact(Timeout = 60000)]
    public async Task RandomSampling_Constructor_DefaultSeed_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new RandomSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Random", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task RandomSampling_Constructor_WithSeed_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new RandomSampling<double>(seed: 42);

        // Assert
        Assert.NotNull(strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task RandomSampling_SelectSamples_DifferentCallsSameSeed_ReturnsSameSelection()
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

    [Fact(Timeout = 60000)]
    public async Task RandomSampling_SelectSamples_ReturnsUniqueIndices()
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

    [Fact(Timeout = 60000)]
    public async Task BALD_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new BALD<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("BALD", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task BALD_Constructor_CustomParameters_ReflectedInInstance()
    {
        // Arrange & Act
        var strategy = new BALD<double>(numMcSamples: 20, dropoutRate: 0.3);

        // Assert
        Assert.Contains("MC20", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task BALD_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task BALD_SelectSamples_WithDiversity_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task BALD_Half_OneHotProbabilities_GiveFiniteZeroScore()
    {
        // Arrange - one Monte Carlo pass with probabilities [1, 0]. The entropy terms use
        // (p + eps) * log(p + eps); 1e-10 rounds to 0 in Half, which made the term for p = 0 equal
        // 0 * log(0) = NaN. With a single pass the mutual information H(mean p) - mean H(p) is exactly 0.
        var probabilities = new Tensor<Half>(new[] { 1, 2 });
        probabilities[0] = (Half)1f;
        probabilities[1] = (Half)0f;

        // Act
        var scores = ActiveLearningEngineOps.BaldScores(new List<Tensor<Half>> { probabilities }, 1);

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        double score = ops.ToDouble(scores[0]);
        Assert.False(double.IsNaN(score));
        Assert.Equal(0.0, score, 6);
    }

    #endregion

    #region BatchBALD Tests

    [Fact(Timeout = 60000)]
    public async Task BatchBALD_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new BatchBALD<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("BatchBALD", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task BatchBALD_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_Constructor_ValidCommittee_CreatesInstance()
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

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_Constructor_NullCommittee_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new QueryByCommittee<double>(global::AiDotNet.Tests.TestNull.Of<System.Collections.Generic.IEnumerable<AiDotNet.Interfaces.IFullModel<double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>>>()));
    }

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_Constructor_SingleMember_ThrowsArgumentException()
    {
        // Arrange
        var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
        {
            ActiveLearningTestHelper.CreateMockModel()
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new QueryByCommittee<double>(committee));
    }

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_GetSelectionStatistics_IncludesCommitteeSize()
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

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_Constructor_DifferentMeasures_ReflectedInName()
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

    [Fact(Timeout = 60000)]
    public async Task DiversitySampling_Constructor_DefaultParameters_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new DiversitySampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("DiversitySampling", strategy.Name);
        Assert.True(strategy.UseBatchDiversity);
    }

    [Fact(Timeout = 60000)]
    public async Task DiversitySampling_Constructor_DifferentMethods_ReflectedInName()
    {
        // Arrange & Act
        var strategy1 = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.FarthestFirst);
        var strategy2 = new DiversitySampling<double>(DiversitySampling<double>.DiversityMethod.DensityPeaks);

        // Assert
        Assert.Contains("FarthestFirst", strategy1.Name);
        Assert.Contains("DensityPeaks", strategy2.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task DiversitySampling_SelectSamples_ValidInput_ReturnsUniqueIndices()
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

    [Fact(Timeout = 60000)]
    public async Task DiversitySampling_GetSelectionStatistics_IncludesCoverageRadius()
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

    [Fact(Timeout = 60000)]
    public async Task DiversitySampling_CoverageRadius_UpdatedAfterSelection()
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

    [Fact(Timeout = 60000)]
    public async Task HybridSampling_Constructor_DefaultParameters_CreatesInstance()
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

    [Fact(Timeout = 60000)]
    public async Task HybridSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task ExpectedModelChange_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new ExpectedModelChange<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("ExpectedModelChange", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task ExpectedModelChange_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task VariationRatios_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new VariationRatios<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("Variation", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationRatios_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task CoreSetSelection_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new CoreSetSelection<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("CoreSet", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task CoreSetSelection_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task DensityWeightedSampling_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new DensityWeightedSampling<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("DensityWeighted", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task DensityWeightedSampling_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    [Fact(Timeout = 60000)]
    public async Task InformationDensity_Constructor_CreatesInstance()
    {
        // Arrange & Act
        var strategy = new InformationDensity<double>();

        // Assert
        Assert.NotNull(strategy);
        Assert.Contains("InformationDensity", strategy.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task InformationDensity_SelectSamples_ValidInput_ReturnsCorrectCount()
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

    #region Half Overflow Regressions

    private static Tensor<Half> HalfRows(int rows, int columns, params float[] values)
    {
        var data = new Vector<Half>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            data[i] = (Half)values[i];
        }

        return new Tensor<Half>(new[] { rows, columns }, data);
    }

    [Fact(Timeout = 60000)]
    public async Task ActiveLearningEngineOps_Half_DistancesAndNorms_DoNotOverflow()
    {
        // Arrange - |(300, 400)| = 500, but 300^2 = 90000 is past Half's maximum 65504, so the
        // sum-of-squares distances and norms were infinite.
        var rows = HalfRows(2, 2, 300f, 400f, 0f, 0f);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        var norms = ActiveLearningEngineOps.RowNorms(rows);
        var block = ActiveLearningEngineOps.DistanceBlock(rows, 0, 1);
        var toRow = ActiveLearningEngineOps.DistancesToRow(rows, 1);

        // Assert
        Assert.Equal(500.0, ops.ToDouble(norms[0]), 6);
        Assert.Equal(0.0, ops.ToDouble(norms[1]), 6);
        Assert.Equal(500.0, ops.ToDouble(block[0, 1]), 6);
        Assert.Equal(500.0, ops.ToDouble(toRow[0]), 6);
    }

    [Fact(Timeout = 60000)]
    public async Task ActiveLearningEngineOps_Half_CosineOfLargeRows_IsFinite()
    {
        // Arrange - the dot product 300 * 600 and the norm product overflow Half; their ratio was
        // inf / inf = NaN. Same direction gives 1, orthogonal rows give 0.
        var rows = HalfRows(3, 2, 300f, 0f, 600f, 0f, 0f, 500f);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        var cosine = ActiveLearningEngineOps.CosineBlock(rows, ActiveLearningEngineOps.RowNorms(rows), 0, 1);

        // Assert
        Assert.Equal(1.0, ops.ToDouble(cosine[0, 0]), 6);
        Assert.Equal(1.0, ops.ToDouble(cosine[0, 1]), 6);
        Assert.Equal(0.0, ops.ToDouble(cosine[0, 2]), 6);
    }

    [Fact(Timeout = 60000)]
    public async Task CoreSetSelection_Half_LargeValues_ScoresDistanceToCenter()
    {
        // Arrange - the center (40000 + 39040) / 2 = 39520 was a sum of 79040 (infinite in Half)
        // divided by 2, and every squared distance 480^2 overflowed as well.
        var strategy = new CoreSetSelection<Half>();
        var model = new MockFullModel<Half>(2);
        var pool = HalfRows(2, 1, 40000f, 39040f);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert
        Assert.Equal(480.0, ops.ToDouble(scores[0]), 6);
        Assert.Equal(480.0, ops.ToDouble(scores[1]), 6);
    }

    [Fact(Timeout = 60000)]
    public async Task InformationDensity_Half_RbfWithSmallGamma_KeepsNonzeroDensity()
    {
        // Arrange - exp(-gamma * d^2) with gamma = 1e-5 and d = 300 is exp(-0.9), about 0.41, but
        // d^2 = 90000 overflowed Half, which made the similarity exp(-inf) = 0 and every score 0.
        var strategy = new InformationDensity<Half>(beta: 1.0,
            similarityMeasure: InformationDensity<Half>.SimilarityMeasure.RBF, rbfGamma: 1e-5);
        var model = new MockFullModel<Half>(2);
        var pool = HalfRows(2, 1, 0f, 300f);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        var scores = strategy.ComputeInformativenessScores(model, pool);

        // Assert - uncertainty (entropy of non-degenerate logits) times a density of about 0.41
        Assert.True(ops.ToDouble(scores[0]) > 0.0, $"score 0 = {ops.ToDouble(scores[0])}");
        Assert.True(ops.ToDouble(scores[1]) > 0.0, $"score 1 = {ops.ToDouble(scores[1])}");
    }

    [Fact(Timeout = 60000)]
    public async Task RankedBatchStrategy_Half_LargeValues_NormalizedDistanceIsFinite()
    {
        // Arrange - |300 - 290| / (2 * 300) = 1/60. The bound (2 * 300)^2 = 360000 overflowed Half,
        // so the diversity was 10 / inf = 0 and every candidate counted as a duplicate.
        var strategy = new AiDotNet.ActiveLearning.Batch.RankedBatchStrategy<Half, Vector<Half>, Vector<Half>>();
        var a = new Vector<Half>(new[] { (Half)300f });
        var b = new Vector<Half>(new[] { (Half)290f });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        double diversity = ops.ToDouble(strategy.ComputeDiversity(a, b));

        // Assert
        Assert.Equal(1.0 / 60.0, diversity, 3);
    }

    [Fact(Timeout = 60000)]
    public async Task PredictionStabilityCriterion_Half_LargePredictions_CorrelationIsOne()
    {
        // Arrange - identical predictions correlate perfectly (stability 1). The centered sums of
        // squares (300^2 + 300^2) overflowed Half, so the correlation was inf / inf = NaN.
        var criterion = new AiDotNet.ActiveLearning.StoppingCriteria.PredictionStabilityCriterion<Half>(
            0.9, 1, AiDotNet.ActiveLearning.StoppingCriteria.StabilityMeasure.Correlation);
        var predictions = new Vector<Half>(new[] { (Half)0f, (Half)300f, (Half)600f });
        var context = new AiDotNet.ActiveLearning.Interfaces.ActiveLearningContext<Half>
        {
            PreviousPredictions = predictions,
            CurrentPredictions = predictions
        };
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        criterion.ShouldStop(context);

        // Assert
        Assert.Equal(1.0, ops.ToDouble(criterion.CurrentStability), 2);
    }

    [Fact(Timeout = 60000)]
    public async Task QueryByCommittee_Half_MoreThan2048Votes_VoteFractionsStayExact()
    {
        // Arrange - 2250 of 3000 members vote for class 0. Summing one-hot votes in Half stops at
        // 2048, which gave 2048 / 3000 instead of 0.75.
        var predictions = new List<Tensor<Half>>();
        for (int m = 0; m < 3000; m++)
        {
            predictions.Add(m < 2250 ? HalfRows(1, 2, 1f, 0f) : HalfRows(1, 2, 0f, 1f));
        }

        var votedClasses = new int[3000];
        for (int m = 2250; m < 3000; m++)
        {
            votedClasses[m] = 1;
        }

        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Act
        var fractions = QueryByCommittee<Half>.VoteFractions(predictions, 1, 2, 3000);
        var shares = AiDotNet.ActiveLearning.Strategies.Committee.QueryByCommitteeStrategy<Half, Vector<Half>, Vector<Half>>
            .VoteShares(votedClasses, 2, 3000);

        // Assert
        Assert.Equal(0.75, ops.ToDouble(fractions[0, 0]), 6);
        Assert.Equal(0.25, ops.ToDouble(fractions[0, 1]), 6);
        Assert.Equal(0.75, ops.ToDouble(shares[0]), 6);
        Assert.Equal(0.25, ops.ToDouble(shares[1]), 6);
    }

    #endregion
}

