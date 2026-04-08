using AiDotNet.FederatedLearning.DriftDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Tests for federated concept drift detection and adaptation (#852).
/// </summary>
public class FederatedDriftDetectionTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    private static Dictionary<int, Tensor<double>> CreateClientModels(int clientCount, int modelSize, double divergence = 0.1)
    {
        var models = new Dictionary<int, Tensor<double>>();
        var rng = new Random(42);

        for (int c = 0; c < clientCount; c++)
        {
            var values = new double[modelSize];
            for (int i = 0; i < modelSize; i++)
            {
                values[i] = 0.5 + c * divergence * (rng.NextDouble() - 0.5);
            }

            models[c] = CreateTensor(values);
        }

        return models;
    }

    private static Tensor<double> CreateGlobalModel(int size)
    {
        var values = new double[size];
        for (int i = 0; i < size; i++)
        {
            values[i] = 0.5 + i * 0.01;
        }

        return CreateTensor(values);
    }

    private static Dictionary<int, double> CreateStableMetrics(int clientCount, double baseLoss = 0.5)
    {
        var metrics = new Dictionary<int, double>();
        for (int c = 0; c < clientCount; c++)
        {
            metrics[c] = baseLoss + c * 0.01;
        }

        return metrics;
    }

    private static Dictionary<int, double> CreateDriftingMetrics(int clientCount, double baseLoss, int driftingClient, double driftMagnitude)
    {
        var metrics = new Dictionary<int, double>();
        for (int c = 0; c < clientCount; c++)
        {
            metrics[c] = c == driftingClient ? baseLoss + driftMagnitude : baseLoss + c * 0.01;
        }

        return metrics;
    }

    // ========== StatisticalDriftDetector Tests ==========

    [Fact]
    public void Statistical_PageHinkley_DetectsNoDriftOnStableData()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.PageHinkley,
            SensitivityThreshold = 0.01,
            LookbackWindowRounds = 10
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);

        // Feed several rounds of stable metrics
        DriftReport lastReport = null;
        for (int round = 0; round < 8; round++)
        {
            var stableMetrics = CreateStableMetrics(3, 0.5 - round * 0.01);
            lastReport = detector.DetectDrift(round, clientModels, globalModel, stableMetrics);
        }

        Assert.NotNull(lastReport);
        Assert.False(lastReport.GlobalDriftDetected, "Stable data should not trigger global drift");
    }

    [Fact]
    public void Statistical_ADWIN_ReturnsReport()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.ADWIN,
            SensitivityThreshold = 0.01,
            LookbackWindowRounds = 20
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);

        // Feed enough rounds to get a meaningful report
        DriftReport report = null;
        for (int round = 0; round < 10; round++)
        {
            var metrics = CreateStableMetrics(3, 0.5);
            report = detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        Assert.NotNull(report);
        Assert.Equal(9, report.Round);
        Assert.False(string.IsNullOrEmpty(report.Summary));
        Assert.Equal(3, report.ClientResults.Count);
    }

    [Fact]
    public void Statistical_DDM_ReturnsValidReport()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.DDM,
            SensitivityThreshold = 0.01,
            LookbackWindowRounds = 20
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);

        DriftReport report = null;
        for (int round = 0; round < 10; round++)
        {
            var metrics = CreateStableMetrics(3, 0.5);
            report = detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        Assert.NotNull(report);
        Assert.InRange(report.DriftingClientFraction, 0.0, 1.0);
        Assert.InRange(report.AverageDriftScore, 0.0, 1.0);
    }

    [Fact]
    public void Statistical_GetAdaptiveWeights_PreservesNormalization()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.PageHinkley,
            AdaptAggregationWeights = true,
            MinDriftWeight = 0.1
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);

        // Feed data to build history
        DriftReport report = null;
        for (int round = 0; round < 5; round++)
        {
            var metrics = CreateStableMetrics(3, 0.5);
            report = detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        var originalWeights = new Dictionary<int, double>
        {
            { 0, 0.33 }, { 1, 0.34 }, { 2, 0.33 }
        };

        var adjusted = detector.GetAdaptiveWeights(originalWeights, report);

        Assert.Equal(3, adjusted.Count);

        // Weights should sum to approximately 1.0
        double sum = 0;
        foreach (var w in adjusted.Values) sum += w;
        Assert.InRange(sum, 0.99, 1.01);
    }

    [Fact]
    public void Statistical_GetAdaptiveWeights_WithDisabledAdaptation_ReturnsOriginal()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.ADWIN,
            AdaptAggregationWeights = false
        };
        var detector = new StatisticalDriftDetector<double>(options);

        var originalWeights = new Dictionary<int, double>
        {
            { 0, 0.5 }, { 1, 0.3 }, { 2, 0.2 }
        };

        var report = new DriftReport { Round = 1 };
        var adjusted = detector.GetAdaptiveWeights(originalWeights, report);

        Assert.Equal(originalWeights[0], adjusted[0], 12);
        Assert.Equal(originalWeights[1], adjusted[1], 12);
        Assert.Equal(originalWeights[2], adjusted[2], 12);
    }

    [Fact]
    public void Statistical_Reset_ClearsState()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.PageHinkley
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);

        // Feed some data
        for (int round = 0; round < 5; round++)
        {
            var metrics = CreateStableMetrics(3, 0.5);
            detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        detector.Reset();

        // After reset, first report should show no drift
        var report = detector.DetectDrift(0, clientModels, globalModel, CreateStableMetrics(3, 0.5));
        Assert.False(report.GlobalDriftDetected);
    }

    [Fact]
    public void Statistical_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new StatisticalDriftDetector<double>(null));
    }

    [Fact]
    public void Statistical_NullClientMetrics_Throws()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.ADWIN };
        var detector = new StatisticalDriftDetector<double>(options);

        Assert.Throws<ArgumentNullException>(() =>
            detector.DetectDrift(0, CreateClientModels(3, 10), CreateGlobalModel(10), null));
    }

    [Fact]
    public void Statistical_MethodName_MatchesOption()
    {
        var options = new FederatedDriftOptions { Method = FederatedDriftMethod.DDM };
        var detector = new StatisticalDriftDetector<double>(options);

        Assert.Equal("DDM", detector.MethodName);
    }

    // ========== ModelDriftDetector Tests ==========

    [Fact]
    public void Model_GradientDivergence_DetectsOnStableModels()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.GradientDivergence,
            SensitivityThreshold = 0.01,
            LookbackWindowRounds = 10
        };
        var detector = new ModelDriftDetector<double>(options);
        var globalModel = CreateGlobalModel(10);

        // Feed multiple rounds with similar models
        DriftReport report = null;
        for (int round = 0; round < 5; round++)
        {
            var clientModels = CreateClientModels(3, 10, 0.01);
            var metrics = CreateStableMetrics(3, 0.5);
            report = detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        Assert.NotNull(report);
        Assert.Equal(3, report.ClientResults.Count);
    }

    [Fact]
    public void Model_WeightDivergence_ReturnsValidReport()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.WeightDivergence,
            SensitivityThreshold = 0.01,
            LookbackWindowRounds = 10
        };
        var detector = new ModelDriftDetector<double>(options);
        var globalModel = CreateGlobalModel(10);

        DriftReport report = null;
        for (int round = 0; round < 5; round++)
        {
            var clientModels = CreateClientModels(3, 10, 0.1);
            var metrics = CreateStableMetrics(3, 0.5);
            report = detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        Assert.NotNull(report);
        Assert.InRange(report.DriftingClientFraction, 0.0, 1.0);
        Assert.InRange(report.AverageDriftScore, 0.0, 1.0);
    }

    [Fact]
    public void Model_NullClientModels_Throws()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.GradientDivergence };
        var detector = new ModelDriftDetector<double>(options);

        Assert.Throws<ArgumentNullException>(() =>
            detector.DetectDrift(0, null, CreateGlobalModel(10), CreateStableMetrics(3)));
    }

    [Fact]
    public void Model_NullGlobalModel_Throws()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.GradientDivergence };
        var detector = new ModelDriftDetector<double>(options);

        Assert.Throws<ArgumentNullException>(() =>
            detector.DetectDrift(0, CreateClientModels(3, 10), null, CreateStableMetrics(3)));
    }

    [Fact]
    public void Model_Reset_ClearsHistory()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.GradientDivergence };
        var detector = new ModelDriftDetector<double>(options);
        var globalModel = CreateGlobalModel(10);

        for (int round = 0; round < 5; round++)
        {
            var clientModels = CreateClientModels(3, 10, 0.1);
            var metrics = CreateStableMetrics(3, 0.5);
            detector.DetectDrift(round, clientModels, globalModel, metrics);
        }

        detector.Reset();

        // After reset, first detection should have no history
        var report = detector.DetectDrift(0, CreateClientModels(3, 10), globalModel, CreateStableMetrics(3));
        Assert.NotNull(report);
        Assert.Equal(0, report.Round);
    }

    [Fact]
    public void Model_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ModelDriftDetector<double>(null));
    }

    [Fact]
    public void Model_MethodName_MatchesOption()
    {
        var options = new FederatedDriftOptions { Method = FederatedDriftMethod.WeightDivergence };
        var detector = new ModelDriftDetector<double>(options);

        Assert.Equal("WeightDivergence", detector.MethodName);
    }

    // ========== DriftAdaptiveAggregator Tests ==========

    [Fact]
    public void DriftAdaptive_SkipsWhenDisabled()
    {
        var options = new FederatedDriftOptions { Enabled = false };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);
        var metrics = CreateStableMetrics(3);
        var weights = new Dictionary<int, double>
        {
            { 0, 0.33 }, { 1, 0.34 }, { 2, 0.33 }
        };

        var (report, adjustedWeights) = aggregator.ProcessRound(1, clientModels, globalModel, metrics, weights);

        Assert.NotNull(report);
        Assert.False(report.GlobalDriftDetected);
        Assert.Equal(3, adjustedWeights.Count);
    }

    [Fact]
    public void DriftAdaptive_ProcessRound_ReturnsReportAndWeights()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.PageHinkley,
            SensitivityThreshold = 0.01,
            DetectionFrequency = 1
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);
        var weights = new Dictionary<int, double>
        {
            { 0, 0.33 }, { 1, 0.34 }, { 2, 0.33 }
        };

        // Feed several rounds
        for (int round = 0; round < 5; round++)
        {
            var metrics = CreateStableMetrics(3, 0.5);
            var (report, adjustedWeights) = aggregator.ProcessRound(
                round, clientModels, globalModel, metrics, weights);

            Assert.NotNull(report);
            Assert.NotNull(adjustedWeights);
        }
    }

    [Fact]
    public void DriftAdaptive_SkipsNonDetectionRounds()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.ADWIN,
            DetectionFrequency = 3
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);
        var metrics = CreateStableMetrics(3);
        var weights = new Dictionary<int, double>
        {
            { 0, 0.33 }, { 1, 0.34 }, { 2, 0.33 }
        };

        // Round 1 should be skipped (1 % 3 != 0)
        var (report1, _) = aggregator.ProcessRound(1, clientModels, globalModel, metrics, weights);
        Assert.Contains("skipped", report1.Summary);

        // Round 3 should run (3 % 3 == 0)
        var (report3, _) = aggregator.ProcessRound(3, clientModels, globalModel, metrics, weights);
        Assert.DoesNotContain("skipped", report3.Summary);
    }

    [Fact]
    public void DriftAdaptive_GetDriftingClients_EmptyWhenNoDrift()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.PageHinkley };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var driftingClients = aggregator.GetDriftingClients();
        Assert.Empty(driftingClients);
    }

    [Fact]
    public void DriftAdaptive_GetClientsNeedingRetraining_EmptyByDefault()
    {
        var options = new FederatedDriftOptions { Enabled = true, Method = FederatedDriftMethod.ADWIN };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var retrainClients = aggregator.GetClientsNeedingRetraining();
        Assert.Empty(retrainClients);
    }

    [Fact]
    public void DriftAdaptive_Reset_ClearsLatestReport()
    {
        var options = new FederatedDriftOptions
        {
            Enabled = true,
            Method = FederatedDriftMethod.PageHinkley,
            DetectionFrequency = 1
        };
        var detector = new StatisticalDriftDetector<double>(options);
        var aggregator = new DriftAdaptiveAggregator<double>(options, detector);

        var clientModels = CreateClientModels(3, 10);
        var globalModel = CreateGlobalModel(10);
        var metrics = CreateStableMetrics(3);
        var weights = new Dictionary<int, double> { { 0, 0.5 }, { 1, 0.3 }, { 2, 0.2 } };

        aggregator.ProcessRound(0, clientModels, globalModel, metrics, weights);
        Assert.NotNull(aggregator.LatestReport);

        aggregator.Reset();
        Assert.Null(aggregator.LatestReport);
    }

    [Fact]
    public void DriftAdaptive_NullOptions_Throws()
    {
        var detector = new StatisticalDriftDetector<double>(new FederatedDriftOptions());
        Assert.Throws<ArgumentNullException>(() => new DriftAdaptiveAggregator<double>(null, detector));
    }

    [Fact]
    public void DriftAdaptive_NullDetector_Throws()
    {
        var options = new FederatedDriftOptions();
        Assert.Throws<ArgumentNullException>(() => new DriftAdaptiveAggregator<double>(options, null));
    }

    // ========== DriftReport and Related Types Tests ==========

    [Fact]
    public void DriftReport_DefaultValues()
    {
        var report = new DriftReport();

        Assert.Equal(0, report.Round);
        Assert.False(report.GlobalDriftDetected);
        Assert.NotNull(report.ClientResults);
        Assert.Empty(report.ClientResults);
        Assert.Equal(0.0, report.DriftingClientFraction);
        Assert.Equal(0.0, report.AverageDriftScore);
        Assert.Equal(string.Empty, report.Summary);
    }

    [Fact]
    public void ClientDriftResult_DefaultValues()
    {
        var result = new ClientDriftResult();

        Assert.Equal(0, result.ClientId);
        Assert.Equal(0.0, result.DriftScore);
        Assert.Equal(DriftType.None, result.DriftType);
        Assert.Equal(DriftAction.None, result.RecommendedAction);
        Assert.Equal(1.0, result.SuggestedWeightMultiplier);
    }

    [Fact]
    public void DriftType_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(DriftType), DriftType.None));
        Assert.True(Enum.IsDefined(typeof(DriftType), DriftType.Warning));
        Assert.True(Enum.IsDefined(typeof(DriftType), DriftType.Sudden));
        Assert.True(Enum.IsDefined(typeof(DriftType), DriftType.Gradual));
        Assert.True(Enum.IsDefined(typeof(DriftType), DriftType.Recurring));
    }

    [Fact]
    public void DriftAction_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(DriftAction), DriftAction.None));
        Assert.True(Enum.IsDefined(typeof(DriftAction), DriftAction.Monitor));
        Assert.True(Enum.IsDefined(typeof(DriftAction), DriftAction.ReduceWeight));
        Assert.True(Enum.IsDefined(typeof(DriftAction), DriftAction.SelectiveRetrain));
        Assert.True(Enum.IsDefined(typeof(DriftAction), DriftAction.TemporaryExclude));
    }

    [Fact]
    public void FederatedDriftMethod_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(FederatedDriftMethod), FederatedDriftMethod.PageHinkley));
        Assert.True(Enum.IsDefined(typeof(FederatedDriftMethod), FederatedDriftMethod.ADWIN));
        Assert.True(Enum.IsDefined(typeof(FederatedDriftMethod), FederatedDriftMethod.DDM));
        Assert.True(Enum.IsDefined(typeof(FederatedDriftMethod), FederatedDriftMethod.GradientDivergence));
        Assert.True(Enum.IsDefined(typeof(FederatedDriftMethod), FederatedDriftMethod.WeightDivergence));
    }

    // ========== Options Tests ==========

    [Fact]
    public void FederatedDriftOptions_DefaultValues()
    {
        var options = new FederatedDriftOptions();

        Assert.False(options.Enabled);
        Assert.Equal(FederatedDriftMethod.ADWIN, options.Method);
        Assert.Equal(0.01, options.SensitivityThreshold);
        Assert.Equal(20, options.LookbackWindowRounds);
        Assert.True(options.AdaptAggregationWeights);
        Assert.Equal(0.1, options.MinDriftWeight);
        Assert.Equal(1, options.DetectionFrequency);
        Assert.False(options.TriggerSelectiveRetraining);
        Assert.Equal(0.3, options.GlobalDriftThreshold);
    }

    // ========== Integration with FederatedLearningOptions ==========

    [Fact]
    public void FederatedLearningOptions_CanSetDriftOptions()
    {
        var flOptions = new FederatedLearningOptions
        {
            DriftDetection = new FederatedDriftOptions
            {
                Enabled = true,
                Method = FederatedDriftMethod.GradientDivergence,
                SensitivityThreshold = 0.05
            }
        };

        Assert.NotNull(flOptions.DriftDetection);
        Assert.True(flOptions.DriftDetection.Enabled);
        Assert.Equal(FederatedDriftMethod.GradientDivergence, flOptions.DriftDetection.Method);
    }
}
