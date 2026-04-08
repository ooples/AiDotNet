using AiDotNet.Enums;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Strategies;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.KnowledgeDistillation;

/// <summary>
/// Comprehensive integration tests for the KnowledgeDistillation module.
/// Tests cover distillation loss functions, strategies, and configuration classes.
/// </summary>
public class KnowledgeDistillationIntegrationTests
{
    private const int BatchSize = 4;
    private const int NumClasses = 10;
    private const double DefaultTemperature = 3.0;
    private const double DefaultAlpha = 0.3;

    #region Helper Methods

    private static Matrix<double> CreateLogits(int batchSize, int numClasses, double baseValue = 0.0)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var logits = new Matrix<double>(batchSize, numClasses);
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                logits[r, c] = baseValue + random.NextDouble() * 2 - 1;
            }
        }
        return logits;
    }

    private static Matrix<double> CreateOneHotLabels(int batchSize, int numClasses)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var labels = new Matrix<double>(batchSize, numClasses);
        for (int r = 0; r < batchSize; r++)
        {
            int classIdx = random.Next(numClasses);
            labels[r, classIdx] = 1.0;
        }
        return labels;
    }

    private static void AssertMatrixFinite(Matrix<double> matrix, string context = "Matrix")
    {
        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                Assert.False(double.IsNaN(matrix[r, c]), $"{context}[{r},{c}] is NaN");
                Assert.False(double.IsInfinity(matrix[r, c]), $"{context}[{r},{c}] is Infinity");
            }
        }
    }

    #endregion

    #region DistillationLoss Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_Constructor_DefaultValues()
    {
        var loss = new DistillationLoss<double>();

        Assert.Equal(3.0, loss.Temperature);
        Assert.Equal(0.3, loss.Alpha);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_Constructor_CustomValues()
    {
        var loss = new DistillationLoss<double>(temperature: 5.0, alpha: 0.5);

        Assert.Equal(5.0, loss.Temperature);
        Assert.Equal(0.5, loss.Alpha);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_SoftLossOnly()
    {
        var loss = new DistillationLoss<double>(DefaultTemperature, DefaultAlpha);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses, baseValue: 1.0);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits, trueLabelsBatch: null);

        Assert.False(double.IsNaN(lossValue));
        Assert.False(double.IsInfinity(lossValue));
        // KL divergence mathematically is non-negative, but small numerical precision
        // differences may result in very small negative values (near zero)
        Assert.True(lossValue >= -1e-8, $"Loss should be non-negative (or near zero due to precision). Got: {lossValue}");
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_WithTrueLabels()
    {
        var loss = new DistillationLoss<double>(DefaultTemperature, DefaultAlpha);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses, baseValue: 1.0);
        var trueLabels = CreateOneHotLabels(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits, trueLabels);

        Assert.False(double.IsNaN(lossValue));
        Assert.False(double.IsInfinity(lossValue));
        Assert.True(lossValue >= 0, "Loss should be non-negative");
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_IdenticalOutputs_ZeroSoftLoss()
    {
        var loss = new DistillationLoss<double>(DefaultTemperature, alpha: 0.0);
        var logits = CreateLogits(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(logits, logits, trueLabelsBatch: null);

        Assert.True(lossValue < 0.001, "Loss should be near zero for identical outputs");
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeGradient_ReturnsCorrectShape()
    {
        var loss = new DistillationLoss<double>(DefaultTemperature, DefaultAlpha);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var gradient = loss.ComputeGradient(studentLogits, teacherLogits);

        Assert.Equal(BatchSize, gradient.Rows);
        Assert.Equal(NumClasses, gradient.Columns);
        AssertMatrixFinite(gradient, "Gradient");
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeGradient_WithTrueLabels()
    {
        var loss = new DistillationLoss<double>(DefaultTemperature, DefaultAlpha);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);
        var trueLabels = CreateOneHotLabels(BatchSize, NumClasses);

        var gradient = loss.ComputeGradient(studentLogits, teacherLogits, trueLabels);

        Assert.Equal(BatchSize, gradient.Rows);
        Assert.Equal(NumClasses, gradient.Columns);
        AssertMatrixFinite(gradient, "Gradient with labels");
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_NullStudent_Throws()
    {
        var loss = new DistillationLoss<double>();
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        Assert.Throws<ArgumentNullException>(() => loss.ComputeLoss(null!, teacherLogits));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_NullTeacher_Throws()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);

        Assert.Throws<ArgumentNullException>(() => loss.ComputeLoss(studentLogits, null!));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_DimensionMismatch_Throws()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses + 5);

        Assert.Throws<ArgumentException>(() => loss.ComputeLoss(studentLogits, teacherLogits));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ComputeLoss_BatchSizeMismatch_Throws()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize + 2, NumClasses);

        Assert.Throws<ArgumentException>(() => loss.ComputeLoss(studentLogits, teacherLogits));
    }

    #endregion

    #region DistillationStrategyBase Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategy_InvalidTemperature_Throws()
    {
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: 0));
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(temperature: -1));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategy_InvalidAlpha_Throws()
    {
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(alpha: -0.1));
        Assert.Throws<ArgumentException>(() => new DistillationLoss<double>(alpha: 1.5));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategy_AlphaBoundaryValues_Valid()
    {
        var lossAlpha0 = new DistillationLoss<double>(alpha: 0.0);
        var lossAlpha1 = new DistillationLoss<double>(alpha: 1.0);

        Assert.Equal(0.0, lossAlpha0.Alpha);
        Assert.Equal(1.0, lossAlpha1.Alpha);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategy_TemperaturePropertySet()
    {
        var loss = new DistillationLoss<double>();
        loss.Temperature = 5.0;

        Assert.Equal(5.0, loss.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategy_AlphaPropertySet()
    {
        var loss = new DistillationLoss<double>();
        loss.Alpha = 0.7;

        Assert.Equal(0.7, loss.Alpha);
    }

    #endregion

    #region FeatureDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_Constructor_ValidLayerPairs()
    {
        var layerPairs = new[] { "layer1:layer1", "layer2:layer2" };
        var strategy = new FeatureDistillationStrategy<double>(layerPairs, featureWeight: 0.5);

        // No exception means success
        Assert.NotNull(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_Constructor_NullLayerPairs_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new FeatureDistillationStrategy<double>(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_Constructor_EmptyLayerPairs_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FeatureDistillationStrategy<double>(Array.Empty<string>()));
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_Constructor_InvalidLayerPairFormat_Throws()
    {
        // Missing colon separator
        Assert.Throws<ArgumentException>(() => new FeatureDistillationStrategy<double>(new[] { "invalid" }));
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_Constructor_InvalidFeatureWeight_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FeatureDistillationStrategy<double>(new[] { "a:b" }, featureWeight: -0.1));
        Assert.Throws<ArgumentException>(() => new FeatureDistillationStrategy<double>(new[] { "a:b" }, featureWeight: 1.5));
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_ComputeFeatureLoss_ValidOutput()
    {
        var layerPairs = new[] { "layer1:layer1" };
        var strategy = new FeatureDistillationStrategy<double>(layerPairs, featureWeight: 0.5);

        var teacherFeatures = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var studentFeatures = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1 });

        Func<string, Vector<double>> teacherExtractor = _ => teacherFeatures;
        Func<string, Vector<double>> studentExtractor = _ => studentFeatures;
        var input = new Vector<double>(new double[] { 1.0 });

        var lossValue = strategy.ComputeFeatureLoss(teacherExtractor, studentExtractor, input);

        Assert.False(double.IsNaN(lossValue));
        Assert.False(double.IsInfinity(lossValue));
        Assert.True(lossValue >= 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureDistillation_ComputeFeatureGradient_ValidOutput()
    {
        var layerPairs = new[] { "layer1:layer1" };
        var strategy = new FeatureDistillationStrategy<double>(layerPairs, featureWeight: 0.5);

        var teacherFeatures = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var studentFeatures = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1 });

        var gradient = strategy.ComputeFeatureGradient(studentFeatures, teacherFeatures);

        Assert.Equal(4, gradient.Length);
        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.False(double.IsNaN(gradient[i]));
            Assert.False(double.IsInfinity(gradient[i]));
        }
    }

    #endregion

    #region AttentionDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_Constructor_ValidLayers()
    {
        var layers = new[] { "layer.0.attention", "layer.1.attention" };
        var strategy = new AttentionDistillationStrategy<double>(layers, attentionWeight: 0.3);

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_Constructor_NullLayers_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new AttentionDistillationStrategy<double>(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_Constructor_EmptyLayers_Throws()
    {
        Assert.Throws<ArgumentException>(() => new AttentionDistillationStrategy<double>(Array.Empty<string>()));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_Constructor_InvalidWeight_Throws()
    {
        Assert.Throws<ArgumentException>(() => new AttentionDistillationStrategy<double>(new[] { "layer" }, attentionWeight: -0.1));
        Assert.Throws<ArgumentException>(() => new AttentionDistillationStrategy<double>(new[] { "layer" }, attentionWeight: 1.5));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_ComputeLoss_ValidOutput()
    {
        var layers = new[] { "layer.0.attention" };
        var strategy = new AttentionDistillationStrategy<double>(layers);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
        Assert.True(lossValue >= 0);
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionDistillation_ComputeAttentionLoss_ValidOutput()
    {
        var layers = new[] { "layer1" };
        var strategy = new AttentionDistillationStrategy<double>(layers, attentionWeight: 0.3);

        var attention = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 });
        Func<string, Vector<double>> extractor = _ => attention;

        var lossValue = strategy.ComputeAttentionLoss(extractor, extractor);

        // Identical attention should give near-zero loss
        Assert.True(lossValue < 0.01, "Identical attention should produce near-zero loss");
    }

    #endregion

    #region ContrastiveDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task ContrastiveDistillation_Constructor_DefaultValues()
    {
        var strategy = new ContrastiveDistillationStrategy<double>();

        // ContrastiveDistillationStrategy uses 0.07 as default temperature (typical for contrastive learning)
        Assert.Equal(0.07, strategy.Temperature, precision: 4);
    }

    [Fact(Timeout = 120000)]
    public async Task ContrastiveDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new ContrastiveDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region ProbabilisticDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task ProbabilisticDistillation_Constructor_DefaultValues()
    {
        var strategy = new ProbabilisticDistillationStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task ProbabilisticDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new ProbabilisticDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region HybridDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_Constructor_ValidStrategies()
    {
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (new DistillationLoss<double>(), 0.5),
            (new ContrastiveDistillationStrategy<double>(), 0.5)
        };
        var strategy = new HybridDistillationStrategy<double>(strategies);

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_Constructor_NullStrategies_Throws()
    {
        Assert.Throws<ArgumentException>(() => new HybridDistillationStrategy<double>(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_Constructor_EmptyStrategies_Throws()
    {
        Assert.Throws<ArgumentException>(() => new HybridDistillationStrategy<double>(
            Array.Empty<(IDistillationStrategy<double>, double)>()));
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_Constructor_WeightsDontSumToOne_Throws()
    {
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (new DistillationLoss<double>(), 0.3),
            (new ContrastiveDistillationStrategy<double>(), 0.3)
        };

        Assert.Throws<ArgumentException>(() => new HybridDistillationStrategy<double>(strategies));
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_Constructor_NegativeWeight_Throws()
    {
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (new DistillationLoss<double>(), -0.5),
            (new ContrastiveDistillationStrategy<double>(), 1.5)
        };

        Assert.Throws<ArgumentException>(() => new HybridDistillationStrategy<double>(strategies));
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_ComputeLoss_ValidOutput()
    {
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (new DistillationLoss<double>(), 0.6),
            (new ContrastiveDistillationStrategy<double>(), 0.4)
        };
        var strategy = new HybridDistillationStrategy<double>(strategies);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task HybridDistillation_GetStrategies_ReturnsConfiguredStrategies()
    {
        var loss = new DistillationLoss<double>();
        var contrastive = new ContrastiveDistillationStrategy<double>();
        var strategies = new (IDistillationStrategy<double>, double)[]
        {
            (loss, 0.5),
            (contrastive, 0.5)
        };
        var strategy = new HybridDistillationStrategy<double>(strategies);

        var result = strategy.GetStrategies();

        Assert.Equal(2, result.Length);
    }

    #endregion

    #region CurriculumDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task EasyToHardCurriculum_Constructor_DefaultValues()
    {
        var strategy = new EasyToHardCurriculumStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task EasyToHardCurriculum_ComputeLoss_ValidOutput()
    {
        var strategy = new EasyToHardCurriculumStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task HardToEasyCurriculum_Constructor_DefaultValues()
    {
        var strategy = new HardToEasyCurriculumStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task HardToEasyCurriculum_ComputeLoss_ValidOutput()
    {
        var strategy = new HardToEasyCurriculumStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region AdaptiveDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task AccuracyBasedAdaptive_Constructor_DefaultValues()
    {
        var strategy = new AccuracyBasedAdaptiveStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task AccuracyBasedAdaptive_ComputeLoss_ValidOutput()
    {
        var strategy = new AccuracyBasedAdaptiveStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task ConfidenceBasedAdaptive_Constructor_DefaultValues()
    {
        var strategy = new ConfidenceBasedAdaptiveStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task EntropyBasedAdaptive_Constructor_DefaultValues()
    {
        var strategy = new EntropyBasedAdaptiveStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    #endregion

    #region VariationalDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task VariationalDistillation_Constructor_DefaultValues()
    {
        var strategy = new VariationalDistillationStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task VariationalDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new VariationalDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region NeuronSelectivityDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task NeuronSelectivityDistillation_Constructor_DefaultValues()
    {
        var strategy = new NeuronSelectivityDistillationStrategy<double>();

        Assert.Equal(3.0, strategy.Temperature);
    }

    [Fact(Timeout = 120000)]
    public async Task NeuronSelectivityDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new NeuronSelectivityDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region RelationalDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task RelationalDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new RelationalDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region SimilarityPreservingStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task SimilarityPreserving_ComputeLoss_ValidOutput()
    {
        var strategy = new SimilarityPreservingStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region FlowBasedDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task FlowBasedDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new FlowBasedDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region FactorTransferDistillationStrategy Tests

    [Fact(Timeout = 120000)]
    public async Task FactorTransferDistillation_ComputeLoss_ValidOutput()
    {
        var strategy = new FactorTransferDistillationStrategy<double>();
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = strategy.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region DistillationStrategyFactory Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_ResponseBased()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ResponseBased, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<DistillationLoss<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_ContrastiveBased()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ContrastiveBased, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<ContrastiveDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_RelationBased()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.RelationBased, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<RelationalDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_SimilarityPreserving()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.SimilarityPreserving, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<SimilarityPreservingStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_FlowBased()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.FlowBased, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<FlowBasedDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_ProbabilisticTransfer()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.ProbabilisticTransfer, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<ProbabilisticDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_VariationalInformation()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.VariationalInformation, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<VariationalDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_FactorTransfer()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.FactorTransfer, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<FactorTransferDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_NeuronSelectivity()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.NeuronSelectivity, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<NeuronSelectivityDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_CreateStrategy_Hybrid()
    {
        var strategy = DistillationStrategyFactory<double>.CreateStrategy(
            DistillationStrategyType.Hybrid, DefaultTemperature, DefaultAlpha);

        Assert.NotNull(strategy);
        Assert.IsType<HybridDistillationStrategy<double>>(strategy);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationStrategyFactory_Configure_FluentBuilder()
    {
        var strategy = DistillationStrategyFactory<double>
            .Configure(DistillationStrategyType.ResponseBased)
            .WithTemperature(5.0)
            .WithAlpha(0.5)
            .Build();

        Assert.NotNull(strategy);
        Assert.IsType<DistillationLoss<double>>(strategy);
    }

    #endregion

    #region DistillationForwardResult Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationForwardResult_Constructor_SetsProperties()
    {
        var output = CreateLogits(BatchSize, NumClasses);

        var result = new DistillationForwardResult<double>(output);

        Assert.Same(output, result.FinalOutput);
        Assert.False(result.HasIntermediateActivations);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationForwardResult_Constructor_WithIntermediateActivations()
    {
        var output = CreateLogits(BatchSize, NumClasses);
        var activations = new IntermediateActivations<double>();
        activations.Add("layer1", CreateLogits(BatchSize, NumClasses));

        var result = new DistillationForwardResult<double>(output, activations);

        Assert.Same(output, result.FinalOutput);
        Assert.True(result.HasIntermediateActivations);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationForwardResult_NullOutput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new DistillationForwardResult<double>(null!));
    }

    #endregion

    #region DistillationCheckpointConfig Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationCheckpointConfig_DefaultValues()
    {
        var config = new DistillationCheckpointConfig();

        Assert.Equal("./checkpoints", config.CheckpointDirectory);
        Assert.Equal(10, config.SaveEveryEpochs);
        Assert.Equal(0, config.SaveEveryBatches);
        Assert.Equal(3, config.KeepBestN);
        Assert.False(config.SaveTeacher);
        Assert.True(config.SaveStudent);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationCheckpointConfig_CustomValues()
    {
        var config = new DistillationCheckpointConfig
        {
            CheckpointDirectory = "/custom/path",
            SaveEveryEpochs = 5,
            SaveEveryBatches = 100,
            KeepBestN = 10,
            SaveTeacher = true,
            SaveStudent = false
        };

        Assert.Equal("/custom/path", config.CheckpointDirectory);
        Assert.Equal(5, config.SaveEveryEpochs);
        Assert.Equal(100, config.SaveEveryBatches);
        Assert.Equal(10, config.KeepBestN);
        Assert.True(config.SaveTeacher);
        Assert.False(config.SaveStudent);
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationCheckpointConfig_BestMetricSettings()
    {
        var config = new DistillationCheckpointConfig
        {
            BestMetric = "accuracy",
            LowerIsBetter = false
        };

        Assert.Equal("accuracy", config.BestMetric);
        Assert.False(config.LowerIsBetter);
    }

    #endregion

    #region IntermediateActivations Tests

    [Fact(Timeout = 120000)]
    public async Task IntermediateActivations_AddAndGet()
    {
        var activations = new IntermediateActivations<double>();
        var matrix = CreateLogits(BatchSize, NumClasses);

        activations.Add("layer1", matrix);
        var retrieved = activations.Get("layer1");

        // Get() returns a defensive clone, so we compare values, not references
        Assert.NotNull(retrieved);
        Assert.Equal(matrix.Rows, retrieved.Rows);
        Assert.Equal(matrix.Columns, retrieved.Columns);
        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                Assert.Equal(matrix[r, c], retrieved[r, c]);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task IntermediateActivations_Get_NonExistent_ReturnsNull()
    {
        var activations = new IntermediateActivations<double>();

        var result = activations.Get("nonexistent");

        Assert.Null(result);
    }

    [Fact(Timeout = 120000)]
    public async Task IntermediateActivations_LayerCount()
    {
        var activations = new IntermediateActivations<double>();
        activations.Add("layer1", CreateLogits(1, 1));
        activations.Add("layer2", CreateLogits(1, 1));

        Assert.Equal(2, activations.LayerCount);
    }

    #endregion

    #region Edge Cases Tests

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_SingleSample_Works()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(1, NumClasses);
        var teacherLogits = CreateLogits(1, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_LargeBatch_Works()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(128, NumClasses);
        var teacherLogits = CreateLogits(128, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_HighTemperature_Works()
    {
        var loss = new DistillationLoss<double>(temperature: 20.0);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_LowTemperature_Works()
    {
        var loss = new DistillationLoss<double>(temperature: 0.5);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_PureHardLoss_Works()
    {
        var loss = new DistillationLoss<double>(alpha: 1.0);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);
        var trueLabels = CreateOneHotLabels(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits, trueLabels);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_PureSoftLoss_Works()
    {
        var loss = new DistillationLoss<double>(alpha: 0.0);
        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);
        var trueLabels = CreateOneHotLabels(BatchSize, NumClasses);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits, trueLabels);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_BinaryClassification_Works()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(BatchSize, 2);
        var teacherLogits = CreateLogits(BatchSize, 2);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    [Fact(Timeout = 120000)]
    public async Task DistillationLoss_ManyClasses_Works()
    {
        var loss = new DistillationLoss<double>();
        var studentLogits = CreateLogits(BatchSize, 1000);
        var teacherLogits = CreateLogits(BatchSize, 1000);

        var lossValue = loss.ComputeLoss(studentLogits, teacherLogits);

        Assert.False(double.IsNaN(lossValue));
    }

    #endregion

    #region Cross-Strategy Consistency Tests

    [Fact(Timeout = 120000)]
    public async Task AllIDistillationStrategies_SupportSameInterface()
    {
        // Strategies that implement IDistillationStrategy<T>
        var strategies = new IDistillationStrategy<double>[]
        {
            new DistillationLoss<double>(),
            new AttentionDistillationStrategy<double>(new[] { "layer1" }),
            new ContrastiveDistillationStrategy<double>(),
            new ProbabilisticDistillationStrategy<double>(),
            new HybridDistillationStrategy<double>(new[]
            {
                ((IDistillationStrategy<double>)new DistillationLoss<double>(), 0.5),
                ((IDistillationStrategy<double>)new ContrastiveDistillationStrategy<double>(), 0.5)
            }),
            new VariationalDistillationStrategy<double>(),
            new NeuronSelectivityDistillationStrategy<double>(),
            new EasyToHardCurriculumStrategy<double>(),
            new HardToEasyCurriculumStrategy<double>(),
            new AccuracyBasedAdaptiveStrategy<double>(),
            new ConfidenceBasedAdaptiveStrategy<double>(),
            new EntropyBasedAdaptiveStrategy<double>(),
            new RelationalDistillationStrategy<double>(),
            new SimilarityPreservingStrategy<double>(),
            new FlowBasedDistillationStrategy<double>(),
            new FactorTransferDistillationStrategy<double>()
        };

        var studentLogits = CreateLogits(BatchSize, NumClasses);
        var teacherLogits = CreateLogits(BatchSize, NumClasses);

        foreach (var strategy in strategies)
        {
            var loss = strategy.ComputeLoss(studentLogits, teacherLogits);
            Assert.False(double.IsNaN(loss), $"Strategy {strategy.GetType().Name} produced NaN loss");
            Assert.False(double.IsInfinity(loss), $"Strategy {strategy.GetType().Name} produced Infinity loss");
        }
    }

    #endregion
}
