using Xunit;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.CurriculumLearning.Schedulers;
using AiDotNet.CurriculumLearning.Interfaces;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Integration tests for the Configuration module.
/// Tests all configuration options classes for default values, property setting, and validation.
/// </summary>
public class ConfigurationIntegrationTests
{
    #region AutoML Options Tests

    [Fact]
    public void AutoMLBudgetOptions_DefaultValues_AreCorrect()
    {
        var options = new AutoMLBudgetOptions();

        Assert.Equal(AutoMLBudgetPreset.Standard, options.Preset);
        Assert.Null(options.TimeLimitOverride);
        Assert.Null(options.TrialLimitOverride);
    }

    [Fact]
    public void AutoMLBudgetOptions_CanSetAllProperties()
    {
        var options = new AutoMLBudgetOptions
        {
            Preset = AutoMLBudgetPreset.Fast,
            TimeLimitOverride = TimeSpan.FromMinutes(30),
            TrialLimitOverride = 50
        };

        Assert.Equal(AutoMLBudgetPreset.Fast, options.Preset);
        Assert.Equal(TimeSpan.FromMinutes(30), options.TimeLimitOverride);
        Assert.Equal(50, options.TrialLimitOverride);
    }

    [Fact]
    public void AutoMLBudgetOptions_SupportsAllPresets()
    {
        // Verify all presets are valid
        foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
        {
            var options = new AutoMLBudgetOptions { Preset = preset };
            Assert.Equal(preset, options.Preset);
        }
    }

    #endregion

    #region RL Training Options Tests

    [Fact]
    public void RLTrainingOptions_DefaultValues_AreCorrect()
    {
        var options = new RLTrainingOptions<double>();

        Assert.Equal(1000, options.Episodes);
        Assert.Equal(500, options.MaxStepsPerEpisode);
        Assert.Equal(1000, options.WarmupSteps);
        Assert.Equal(1, options.TrainFrequency);
        Assert.Equal(1, options.GradientSteps);
        Assert.Equal(64, options.BatchSize);
        Assert.Equal(10, options.LogFrequency);
        Assert.Null(options.Environment);
        Assert.Null(options.ExplorationStrategy);
        Assert.Null(options.ReplayBuffer);
        Assert.Null(options.Seed);
        Assert.False(options.NormalizeObservations);
        Assert.False(options.NormalizeRewards);
        Assert.False(options.UsePrioritizedReplay);
    }

    [Fact]
    public void RLTrainingOptions_CanSetAllProperties()
    {
        var options = new RLTrainingOptions<double>
        {
            Episodes = 2000,
            MaxStepsPerEpisode = 1000,
            WarmupSteps = 500,
            TrainFrequency = 4,
            GradientSteps = 2,
            BatchSize = 128,
            LogFrequency = 5,
            Seed = 42,
            NormalizeObservations = true,
            NormalizeRewards = true,
            UsePrioritizedReplay = true
        };

        Assert.Equal(2000, options.Episodes);
        Assert.Equal(1000, options.MaxStepsPerEpisode);
        Assert.Equal(500, options.WarmupSteps);
        Assert.Equal(4, options.TrainFrequency);
        Assert.Equal(2, options.GradientSteps);
        Assert.Equal(128, options.BatchSize);
        Assert.Equal(5, options.LogFrequency);
        Assert.Equal(42, options.Seed);
        Assert.True(options.NormalizeObservations);
        Assert.True(options.NormalizeRewards);
        Assert.True(options.UsePrioritizedReplay);
    }

    [Fact]
    public void RLTrainingOptions_SupportsCallbacks()
    {
        bool episodeCallbackCalled = false;
        bool stepCallbackCalled = false;
        bool trainingStartCalled = false;
        bool trainingCompleteCalled = false;

        var options = new RLTrainingOptions<double>
        {
            OnEpisodeComplete = _ => episodeCallbackCalled = true,
            OnStepComplete = _ => stepCallbackCalled = true,
            OnTrainingStart = () => trainingStartCalled = true,
            OnTrainingComplete = _ => trainingCompleteCalled = true
        };

        // Verify callbacks are set
        Assert.NotNull(options.OnEpisodeComplete);
        Assert.NotNull(options.OnStepComplete);
        Assert.NotNull(options.OnTrainingStart);
        Assert.NotNull(options.OnTrainingComplete);

        // Invoke callbacks
        options.OnEpisodeComplete(new RLEpisodeMetrics<double>());
        options.OnStepComplete(new RLStepMetrics<double>());
        options.OnTrainingStart();
        options.OnTrainingComplete(new RLTrainingSummary<double>());

        Assert.True(episodeCallbackCalled);
        Assert.True(stepCallbackCalled);
        Assert.True(trainingStartCalled);
        Assert.True(trainingCompleteCalled);
    }

    #endregion

    #region RLCheckpointConfig Tests

    [Fact]
    public void RLCheckpointConfig_DefaultValues_AreCorrect()
    {
        var config = new RLCheckpointConfig();

        Assert.Equal("./checkpoints", config.CheckpointDirectory);
        Assert.Equal(100, config.SaveEveryEpisodes);
        Assert.Equal(3, config.KeepBestN);
        Assert.True(config.SaveOnBestReward);
    }

    [Fact]
    public void RLCheckpointConfig_CanSetAllProperties()
    {
        var config = new RLCheckpointConfig
        {
            CheckpointDirectory = "/custom/path",
            SaveEveryEpisodes = 50,
            KeepBestN = 5,
            SaveOnBestReward = false
        };

        Assert.Equal("/custom/path", config.CheckpointDirectory);
        Assert.Equal(50, config.SaveEveryEpisodes);
        Assert.Equal(5, config.KeepBestN);
        Assert.False(config.SaveOnBestReward);
    }

    #endregion

    #region RLEarlyStoppingConfig Tests

    [Fact]
    public void RLEarlyStoppingConfig_DefaultValues_AreCorrect()
    {
        var config = new RLEarlyStoppingConfig<double>();

        Assert.Equal(100, config.PatienceEpisodes);
        Assert.Equal(0.01, config.MinImprovement);
        // For unconstrained generic T?, when T is a value type like double,
        // T? is just T (not Nullable<T>), so default is 0
        Assert.Equal(0.0, config.RewardThreshold);
    }

    [Fact]
    public void RLEarlyStoppingConfig_CanSetAllProperties()
    {
        var config = new RLEarlyStoppingConfig<double>
        {
            PatienceEpisodes = 50,
            MinImprovement = 0.001,
            RewardThreshold = 100.0
        };

        Assert.Equal(50, config.PatienceEpisodes);
        Assert.Equal(0.001, config.MinImprovement);
        Assert.Equal(100.0, config.RewardThreshold);
    }

    [Fact]
    public void RLEarlyStoppingConfig_WorksWithFloat()
    {
        var config = new RLEarlyStoppingConfig<float>();

        Assert.Equal(100, config.PatienceEpisodes);
        Assert.Equal(0.01f, config.MinImprovement, 0.0001f);
    }

    #endregion

    #region ExplorationScheduleConfig Tests

    [Fact]
    public void ExplorationScheduleConfig_DefaultValues_AreCorrect()
    {
        var config = new ExplorationScheduleConfig<double>();

        Assert.Equal(1.0, config.InitialEpsilon);
        Assert.Equal(0.01, config.FinalEpsilon);
        Assert.Equal(100000, config.DecaySteps);
        Assert.Equal(ExplorationDecayType.Linear, config.DecayType);
    }

    [Fact]
    public void ExplorationScheduleConfig_CanSetAllProperties()
    {
        var config = new ExplorationScheduleConfig<double>
        {
            InitialEpsilon = 0.9,
            FinalEpsilon = 0.05,
            DecaySteps = 50000,
            DecayType = ExplorationDecayType.Exponential
        };

        Assert.Equal(0.9, config.InitialEpsilon);
        Assert.Equal(0.05, config.FinalEpsilon);
        Assert.Equal(50000, config.DecaySteps);
        Assert.Equal(ExplorationDecayType.Exponential, config.DecayType);
    }

    [Fact]
    public void ExplorationScheduleConfig_SupportsAllDecayTypes()
    {
        foreach (ExplorationDecayType decayType in Enum.GetValues(typeof(ExplorationDecayType)))
        {
            var config = new ExplorationScheduleConfig<double> { DecayType = decayType };
            Assert.Equal(decayType, config.DecayType);
        }
    }

    #endregion

    #region InferenceOptimizationConfig Tests

    [Fact]
    public void InferenceOptimizationConfig_Default_HasCorrectValues()
    {
        var config = InferenceOptimizationConfig.Default;

        Assert.True(config.EnableKVCache);
        Assert.True(config.EnableBatching);
        Assert.False(config.EnableSpeculativeDecoding);
    }

    [Fact]
    public void InferenceOptimizationConfig_HighPerformance_HasCorrectValues()
    {
        var config = InferenceOptimizationConfig.HighPerformance;

        Assert.True(config.EnableKVCache);
        Assert.Equal(2048, config.KVCacheMaxSizeMB);
        Assert.True(config.EnableBatching);
        Assert.Equal(64, config.MaxBatchSize);
        Assert.True(config.EnableSpeculativeDecoding);
        Assert.Equal(5, config.SpeculationDepth);
    }

    [Fact]
    public void InferenceOptimizationConfig_DefaultValues_AreCorrect()
    {
        var config = new InferenceOptimizationConfig();

        // KV Cache defaults
        Assert.True(config.EnableKVCache);
        Assert.Equal(1024, config.KVCacheMaxSizeMB);
        Assert.Equal(AiDotNet.Configuration.CacheEvictionPolicy.LRU, config.KVCacheEvictionPolicy);
        Assert.False(config.UseSlidingWindowKVCache);
        Assert.Equal(1024, config.KVCacheWindowSize);
        Assert.Equal(KVCachePrecisionMode.Auto, config.KVCachePrecision);
        Assert.Equal(KVCacheQuantizationMode.None, config.KVCacheQuantization);
        Assert.True(config.EnablePagedKVCache);
        Assert.Equal(16, config.PagedKVCacheBlockSize);

        // Attention defaults
        Assert.True(config.EnableFlashAttention);
        Assert.Equal(AttentionMaskingMode.Auto, config.AttentionMasking);

        // Batching defaults
        Assert.True(config.EnableBatching);
        Assert.Equal(32, config.MaxBatchSize);
        Assert.Equal(1, config.MinBatchSize);
        Assert.Equal(10, config.BatchTimeoutMs);
        Assert.True(config.AdaptiveBatchSize);

        // Speculative decoding defaults
        Assert.False(config.EnableSpeculativeDecoding);
        Assert.Equal(DraftModelType.NGram, config.DraftModelType);
        Assert.Equal(4, config.SpeculationDepth);
        Assert.False(config.UseTreeSpeculation);
        Assert.Equal(SpeculationPolicy.Auto, config.SpeculationPolicy);
        Assert.Equal(SpeculativeMethod.Auto, config.SpeculativeMethod);

        // Quantization defaults
        Assert.False(config.EnableWeightOnlyQuantization);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_AcceptsValidConfig()
    {
        var config = InferenceOptimizationConfig.Default;

        // Should not throw
        config.Validate();
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsInvalidKVCacheMaxSize()
    {
        var config = new InferenceOptimizationConfig { KVCacheMaxSizeMB = 0 };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("KVCacheMaxSizeMB must be positive", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsInvalidMaxBatchSize()
    {
        var config = new InferenceOptimizationConfig { MaxBatchSize = 0 };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("MaxBatchSize must be positive", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsInvalidMinBatchSize()
    {
        var config = new InferenceOptimizationConfig { MinBatchSize = 0 };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("MinBatchSize must be positive", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsMinGreaterThanMax()
    {
        var config = new InferenceOptimizationConfig
        {
            MinBatchSize = 64,
            MaxBatchSize = 32
        };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("MinBatchSize", ex.Message);
        Assert.Contains("cannot exceed MaxBatchSize", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsNegativeBatchTimeout()
    {
        var config = new InferenceOptimizationConfig { BatchTimeoutMs = -1 };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("BatchTimeoutMs must be non-negative", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsNegativeSpeculationDepth()
    {
        var config = new InferenceOptimizationConfig { SpeculationDepth = -1 };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("SpeculationDepth must be non-negative", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsInvalidSlidingWindowSize()
    {
        var config = new InferenceOptimizationConfig
        {
            UseSlidingWindowKVCache = true,
            KVCacheWindowSize = 0
        };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("KVCacheWindowSize must be positive", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_Validate_RejectsInvalidPagedBlockSize()
    {
        var config = new InferenceOptimizationConfig
        {
            EnablePagedKVCache = true,
            PagedKVCacheBlockSize = 0
        };

        var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
        Assert.Contains("PagedKVCacheBlockSize must be positive", ex.Message);
    }

    [Fact]
    public void InferenceOptimizationConfig_SupportsAllEnumValues()
    {
        var config = new InferenceOptimizationConfig();

        // Test all CacheEvictionPolicy values
        foreach (AiDotNet.Configuration.CacheEvictionPolicy policy in Enum.GetValues(typeof(AiDotNet.Configuration.CacheEvictionPolicy)))
        {
            config.KVCacheEvictionPolicy = policy;
            Assert.Equal(policy, config.KVCacheEvictionPolicy);
        }

        // Test all DraftModelType values
        foreach (DraftModelType type in Enum.GetValues(typeof(DraftModelType)))
        {
            config.DraftModelType = type;
            Assert.Equal(type, config.DraftModelType);
        }

        // Test all AttentionMaskingMode values
        foreach (AttentionMaskingMode mode in Enum.GetValues(typeof(AttentionMaskingMode)))
        {
            config.AttentionMasking = mode;
            Assert.Equal(mode, config.AttentionMasking);
        }

        // Test all SpeculationPolicy values
        foreach (SpeculationPolicy policy in Enum.GetValues(typeof(SpeculationPolicy)))
        {
            config.SpeculationPolicy = policy;
            Assert.Equal(policy, config.SpeculationPolicy);
        }

        // Test all SpeculativeMethod values
        foreach (SpeculativeMethod method in Enum.GetValues(typeof(SpeculativeMethod)))
        {
            config.SpeculativeMethod = method;
            Assert.Equal(method, config.SpeculativeMethod);
        }
    }

    #endregion

    #region ResNetConfiguration Tests

    [Fact]
    public void ResNetConfiguration_DefaultParameters_AreCorrect()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 1000);

        Assert.Equal(ResNetVariant.ResNet50, config.Variant);
        Assert.Equal(1000, config.NumClasses);
        Assert.Equal(224, config.InputHeight);
        Assert.Equal(224, config.InputWidth);
        Assert.Equal(3, config.InputChannels);
        Assert.True(config.IncludeClassifier);
        Assert.True(config.ZeroInitResidual);
        Assert.False(config.UseAutodiff);
    }

    [Fact]
    public void ResNetConfiguration_InputShape_ComputedCorrectly()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10,
            inputHeight: 32, inputWidth: 32, inputChannels: 3);

        var shape = config.InputShape;
        Assert.Equal(3, shape.Length);
        Assert.Equal(3, shape[0]); // channels
        Assert.Equal(32, shape[1]); // height
        Assert.Equal(32, shape[2]); // width

        Assert.Equal(3 * 32 * 32, config.TotalInputSize);
    }

    [Fact]
    public void ResNetConfiguration_UsesBottleneck_CorrectForVariants()
    {
        // BasicBlock variants (no bottleneck)
        Assert.False(new ResNetConfiguration(ResNetVariant.ResNet18, 10).UsesBottleneck);
        Assert.False(new ResNetConfiguration(ResNetVariant.ResNet34, 10).UsesBottleneck);

        // BottleneckBlock variants
        Assert.True(new ResNetConfiguration(ResNetVariant.ResNet50, 10).UsesBottleneck);
        Assert.True(new ResNetConfiguration(ResNetVariant.ResNet101, 10).UsesBottleneck);
        Assert.True(new ResNetConfiguration(ResNetVariant.ResNet152, 10).UsesBottleneck);
    }

    [Fact]
    public void ResNetConfiguration_BlockCounts_CorrectForAllVariants()
    {
        // ResNet18: [2, 2, 2, 2]
        var config18 = new ResNetConfiguration(ResNetVariant.ResNet18, 10);
        Assert.Equal(new[] { 2, 2, 2, 2 }, config18.BlockCounts);

        // ResNet34: [3, 4, 6, 3]
        var config34 = new ResNetConfiguration(ResNetVariant.ResNet34, 10);
        Assert.Equal(new[] { 3, 4, 6, 3 }, config34.BlockCounts);

        // ResNet50: [3, 4, 6, 3]
        var config50 = new ResNetConfiguration(ResNetVariant.ResNet50, 10);
        Assert.Equal(new[] { 3, 4, 6, 3 }, config50.BlockCounts);

        // ResNet101: [3, 4, 23, 3]
        var config101 = new ResNetConfiguration(ResNetVariant.ResNet101, 10);
        Assert.Equal(new[] { 3, 4, 23, 3 }, config101.BlockCounts);

        // ResNet152: [3, 8, 36, 3]
        var config152 = new ResNetConfiguration(ResNetVariant.ResNet152, 10);
        Assert.Equal(new[] { 3, 8, 36, 3 }, config152.BlockCounts);
    }

    [Fact]
    public void ResNetConfiguration_Expansion_CorrectForVariants()
    {
        // BasicBlock expansion = 1
        Assert.Equal(1, new ResNetConfiguration(ResNetVariant.ResNet18, 10).Expansion);
        Assert.Equal(1, new ResNetConfiguration(ResNetVariant.ResNet34, 10).Expansion);

        // BottleneckBlock expansion = 4
        Assert.Equal(4, new ResNetConfiguration(ResNetVariant.ResNet50, 10).Expansion);
        Assert.Equal(4, new ResNetConfiguration(ResNetVariant.ResNet101, 10).Expansion);
        Assert.Equal(4, new ResNetConfiguration(ResNetVariant.ResNet152, 10).Expansion);
    }

    [Fact]
    public void ResNetConfiguration_BaseChannels_AreCorrect()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 10);
        Assert.Equal(new[] { 64, 128, 256, 512 }, config.BaseChannels);
    }

    [Fact]
    public void ResNetConfiguration_CreateResNet50_CreatesCorrectConfig()
    {
        var config = ResNetConfiguration.CreateResNet50(1000);

        Assert.Equal(ResNetVariant.ResNet50, config.Variant);
        Assert.Equal(1000, config.NumClasses);
    }

    [Fact]
    public void ResNetConfiguration_CreateForCIFAR_CreatesCorrectConfig()
    {
        var config = ResNetConfiguration.CreateForCIFAR(ResNetVariant.ResNet18, 10);

        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.Equal(10, config.NumClasses);
        Assert.Equal(32, config.InputHeight);
        Assert.Equal(32, config.InputWidth);
    }

    [Fact]
    public void ResNetConfiguration_CreateLightweight_CreatesCorrectConfig()
    {
        var config = ResNetConfiguration.CreateLightweight(100);

        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.Equal(100, config.NumClasses);
    }

    [Fact]
    public void ResNetConfiguration_CreateForTesting_CreatesMinimalConfig()
    {
        var config = ResNetConfiguration.CreateForTesting(10);

        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.Equal(10, config.NumClasses);
        Assert.Equal(32, config.InputHeight);
        Assert.Equal(32, config.InputWidth);
    }

    [Fact]
    public void ResNetConfiguration_RejectsInvalidNumClasses()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 0));
        Assert.Contains("Number of classes must be greater than 0", ex.Message);
    }

    [Fact]
    public void ResNetConfiguration_RejectsInvalidInputHeight()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputHeight: 0));
        Assert.Contains("Input height must be greater than 0", ex.Message);
    }

    [Fact]
    public void ResNetConfiguration_RejectsInvalidInputWidth()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputWidth: 0));
        Assert.Contains("Input width must be greater than 0", ex.Message);
    }

    [Fact]
    public void ResNetConfiguration_RejectsInvalidInputChannels()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputChannels: 0));
        Assert.Contains("Input channels must be greater than 0", ex.Message);
    }

    [Fact]
    public void ResNetConfiguration_RejectsTooSmallInputHeight()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputHeight: 16));
        Assert.Contains("at least 32x32 pixels", ex.Message);
    }

    [Fact]
    public void ResNetConfiguration_RejectsTooSmallInputWidth()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 10, inputWidth: 16));
        Assert.Contains("at least 32x32 pixels", ex.Message);
    }

    #endregion

    #region BenchmarkingOptions Tests

    [Fact]
    public void BenchmarkingOptions_DefaultValues_AreCorrect()
    {
        var options = new BenchmarkingOptions();

        Assert.Empty(options.Suites);
        Assert.Null(options.SampleSize);
        Assert.Null(options.Seed);
        Assert.False(options.CiMode);
        Assert.Equal(BenchmarkReportDetailLevel.Summary, options.DetailLevel);
        Assert.Equal(BenchmarkFailurePolicy.FailFast, options.FailurePolicy);
        Assert.True(options.AttachReportToResult);
        Assert.Null(options.Leaf);
        Assert.Null(options.Vision);
        Assert.Null(options.Tabular);
        Assert.Null(options.Text);
    }

    [Fact]
    public void BenchmarkingOptions_CanSetAllProperties()
    {
        var options = new BenchmarkingOptions
        {
            Suites = new[] { BenchmarkSuite.LEAF },
            SampleSize = 100,
            Seed = 42,
            CiMode = true,
            DetailLevel = BenchmarkReportDetailLevel.Detailed,
            FailurePolicy = BenchmarkFailurePolicy.ContinueAndAttachReport,
            AttachReportToResult = false,
            Leaf = new LeafFederatedBenchmarkOptions(),
            Vision = new FederatedVisionBenchmarkOptions(),
            Tabular = new FederatedTabularBenchmarkOptions(),
            Text = new FederatedTextBenchmarkOptions()
        };

        Assert.Single(options.Suites);
        Assert.Equal(BenchmarkSuite.LEAF, options.Suites[0]);
        Assert.Equal(100, options.SampleSize);
        Assert.Equal(42, options.Seed);
        Assert.True(options.CiMode);
        Assert.Equal(BenchmarkReportDetailLevel.Detailed, options.DetailLevel);
        Assert.Equal(BenchmarkFailurePolicy.ContinueAndAttachReport, options.FailurePolicy);
        Assert.False(options.AttachReportToResult);
        Assert.NotNull(options.Leaf);
        Assert.NotNull(options.Vision);
        Assert.NotNull(options.Tabular);
        Assert.NotNull(options.Text);
    }

    #endregion

    #region CurriculumLearningOptions Tests

    [Fact]
    public void CurriculumLearningOptions_DefaultValues_AreCorrect()
    {
        var options = new CurriculumLearningOptions<double, double[], double>();

        Assert.Equal(CurriculumScheduleType.Linear, options.ScheduleType);
        Assert.Null(options.DifficultyEstimator);
        Assert.Null(options.NumPhases);
        Assert.Null(options.TotalEpochs);
        Assert.Null(options.InitialDataFraction);
        Assert.Null(options.FinalDataFraction);
        Assert.Null(options.EarlyStopping);
        Assert.Null(options.SelfPaced);
        Assert.Null(options.CompetenceBased);
        Assert.Null(options.RecalculateDifficulties);
        Assert.Null(options.DifficultyRecalculationFrequency);
        Assert.Null(options.NormalizeDifficulties);
        Assert.Null(options.ShuffleWithinPhase);
        Assert.Null(options.UseDifficultyWeighting);
        Assert.Null(options.BatchSize);
        Assert.Null(options.RandomSeed);
        Assert.Equal(CurriculumVerbosity.Normal, options.Verbosity);
    }

    [Fact]
    public void CurriculumLearningOptions_CanSetAllProperties()
    {
        var options = new CurriculumLearningOptions<double, double[], double>
        {
            ScheduleType = CurriculumScheduleType.Exponential,
            DifficultyEstimator = DifficultyEstimatorType.LossBased,
            NumPhases = 10,
            TotalEpochs = 100,
            InitialDataFraction = 0.1,
            FinalDataFraction = 1.0,
            RecalculateDifficulties = true,
            DifficultyRecalculationFrequency = 5,
            NormalizeDifficulties = true,
            ShuffleWithinPhase = true,
            UseDifficultyWeighting = true,
            BatchSize = 64,
            RandomSeed = 42,
            Verbosity = CurriculumVerbosity.Verbose
        };

        Assert.Equal(CurriculumScheduleType.Exponential, options.ScheduleType);
        Assert.Equal(DifficultyEstimatorType.LossBased, options.DifficultyEstimator);
        Assert.Equal(10, options.NumPhases);
        Assert.Equal(100, options.TotalEpochs);
        Assert.Equal(0.1, options.InitialDataFraction);
        Assert.Equal(1.0, options.FinalDataFraction);
        Assert.True(options.RecalculateDifficulties);
        Assert.Equal(5, options.DifficultyRecalculationFrequency);
        Assert.True(options.NormalizeDifficulties);
        Assert.True(options.ShuffleWithinPhase);
        Assert.True(options.UseDifficultyWeighting);
        Assert.Equal(64, options.BatchSize);
        Assert.Equal(42, options.RandomSeed);
        Assert.Equal(CurriculumVerbosity.Verbose, options.Verbosity);
    }

    [Fact]
    public void CurriculumLearningOptions_SupportsAllScheduleTypes()
    {
        foreach (CurriculumScheduleType scheduleType in Enum.GetValues(typeof(CurriculumScheduleType)))
        {
            var options = new CurriculumLearningOptions<double, double[], double>
            {
                ScheduleType = scheduleType
            };
            Assert.Equal(scheduleType, options.ScheduleType);
        }
    }

    [Fact]
    public void CurriculumLearningOptions_SupportsAllDifficultyEstimatorTypes()
    {
        foreach (DifficultyEstimatorType estimator in Enum.GetValues(typeof(DifficultyEstimatorType)))
        {
            var options = new CurriculumLearningOptions<double, double[], double>
            {
                DifficultyEstimator = estimator
            };
            Assert.Equal(estimator, options.DifficultyEstimator);
        }
    }

    [Fact]
    public void CurriculumEarlyStoppingOptions_DefaultValues_AreCorrect()
    {
        var options = new CurriculumEarlyStoppingOptions();

        Assert.Null(options.Enabled);
        Assert.Null(options.Patience);
        Assert.Null(options.MinDelta);
    }

    [Fact]
    public void CurriculumEarlyStoppingOptions_CanSetAllProperties()
    {
        var options = new CurriculumEarlyStoppingOptions
        {
            Enabled = true,
            Patience = 5,
            MinDelta = 0.001
        };

        Assert.True(options.Enabled);
        Assert.Equal(5, options.Patience);
        Assert.Equal(0.001, options.MinDelta);
    }

    [Fact]
    public void SelfPacedOptions_DefaultValues_AreCorrect()
    {
        var options = new SelfPacedOptions();

        Assert.Null(options.InitialLambda);
        Assert.Null(options.MaxLambda);
        Assert.Null(options.LambdaGrowthRate);
        Assert.Equal(SelfPaceRegularizer.Hard, options.Regularizer);
    }

    [Fact]
    public void SelfPacedOptions_CanSetAllProperties()
    {
        var options = new SelfPacedOptions
        {
            InitialLambda = 0.5,
            MaxLambda = 20.0,
            LambdaGrowthRate = 0.2,
            Regularizer = SelfPaceRegularizer.Linear
        };

        Assert.Equal(0.5, options.InitialLambda);
        Assert.Equal(20.0, options.MaxLambda);
        Assert.Equal(0.2, options.LambdaGrowthRate);
        Assert.Equal(SelfPaceRegularizer.Linear, options.Regularizer);
    }

    [Fact]
    public void CompetenceBasedOptions_DefaultValues_AreCorrect()
    {
        var options = new CompetenceBasedOptions();

        Assert.Null(options.CompetenceThreshold);
        Assert.Equal(CompetenceMetricType.Combined, options.MetricType);
        Assert.Null(options.PatienceEpochs);
        Assert.Null(options.MinImprovement);
        Assert.Null(options.SmoothingFactor);
    }

    [Fact]
    public void CompetenceBasedOptions_CanSetAllProperties()
    {
        var options = new CompetenceBasedOptions
        {
            CompetenceThreshold = 0.95,
            MetricType = CompetenceMetricType.Accuracy,
            PatienceEpochs = 10,
            MinImprovement = 0.002,
            SmoothingFactor = 0.5
        };

        Assert.Equal(0.95, options.CompetenceThreshold);
        Assert.Equal(CompetenceMetricType.Accuracy, options.MetricType);
        Assert.Equal(10, options.PatienceEpochs);
        Assert.Equal(0.002, options.MinImprovement);
        Assert.Equal(0.5, options.SmoothingFactor);
    }

    #endregion

    #region RLEpisodeMetrics and RLStepMetrics Tests

    [Fact]
    public void RLEpisodeMetrics_CanBeInstantiated()
    {
        var metrics = new RLEpisodeMetrics<double>();
        Assert.NotNull(metrics);
    }

    [Fact]
    public void RLStepMetrics_CanBeInstantiated()
    {
        var metrics = new RLStepMetrics<double>();
        Assert.NotNull(metrics);
    }

    [Fact]
    public void RLTrainingSummary_CanBeInstantiated()
    {
        var summary = new RLTrainingSummary<double>();
        Assert.NotNull(summary);
    }

    #endregion

    #region Generic Type Tests

    [Fact]
    public void RLTrainingOptions_WorksWithDifferentNumericTypes()
    {
        // Test with double
        var doubleOptions = new RLTrainingOptions<double>();
        Assert.NotNull(doubleOptions);

        // Test with float
        var floatOptions = new RLTrainingOptions<float>();
        Assert.NotNull(floatOptions);

        // Test with decimal
        var decimalOptions = new RLTrainingOptions<decimal>();
        Assert.NotNull(decimalOptions);
    }

    [Fact]
    public void ExplorationScheduleConfig_WorksWithDifferentNumericTypes()
    {
        // Test with double
        var doubleConfig = new ExplorationScheduleConfig<double>();
        Assert.Equal(1.0, doubleConfig.InitialEpsilon);

        // Test with float
        var floatConfig = new ExplorationScheduleConfig<float>();
        Assert.Equal(1.0f, floatConfig.InitialEpsilon, 0.0001f);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void RLTrainingOptions_WithAllConfigs_WorksTogether()
    {
        // Create a complete RL training configuration
        var options = new RLTrainingOptions<double>
        {
            Episodes = 500,
            MaxStepsPerEpisode = 200,
            WarmupSteps = 100,
            BatchSize = 32,
            CheckpointConfig = new RLCheckpointConfig
            {
                CheckpointDirectory = "./my_checkpoints",
                SaveEveryEpisodes = 50,
                KeepBestN = 5
            },
            EarlyStoppingConfig = new RLEarlyStoppingConfig<double>
            {
                PatienceEpisodes = 50,
                MinImprovement = 0.01,
                RewardThreshold = 200.0
            },
            ExplorationSchedule = new ExplorationScheduleConfig<double>
            {
                InitialEpsilon = 1.0,
                FinalEpsilon = 0.01,
                DecaySteps = 50000,
                DecayType = ExplorationDecayType.Linear
            }
        };

        // Verify all nested configs are accessible
        Assert.NotNull(options.CheckpointConfig);
        Assert.Equal("./my_checkpoints", options.CheckpointConfig.CheckpointDirectory);

        Assert.NotNull(options.EarlyStoppingConfig);
        Assert.Equal(200.0, options.EarlyStoppingConfig.RewardThreshold);

        Assert.NotNull(options.ExplorationSchedule);
        Assert.Equal(50000, options.ExplorationSchedule.DecaySteps);
    }

    [Fact]
    public void CurriculumLearningOptions_WithAllConfigs_WorksTogether()
    {
        var options = new CurriculumLearningOptions<double, double[], double>
        {
            ScheduleType = CurriculumScheduleType.SelfPaced,
            NumPhases = 5,
            TotalEpochs = 50,
            EarlyStopping = new CurriculumEarlyStoppingOptions
            {
                Enabled = true,
                Patience = 10,
                MinDelta = 0.001
            },
            SelfPaced = new SelfPacedOptions
            {
                InitialLambda = 0.1,
                MaxLambda = 10.0,
                LambdaGrowthRate = 0.1,
                Regularizer = SelfPaceRegularizer.Logarithmic
            }
        };

        Assert.NotNull(options.EarlyStopping);
        Assert.True(options.EarlyStopping.Enabled);

        Assert.NotNull(options.SelfPaced);
        Assert.Equal(0.1, options.SelfPaced.InitialLambda);
        Assert.Equal(SelfPaceRegularizer.Logarithmic, options.SelfPaced.Regularizer);
    }

    [Fact]
    public void BenchmarkingOptions_WithFederatedConfigs_WorksTogether()
    {
        var options = new BenchmarkingOptions
        {
            Suites = new[] { BenchmarkSuite.LEAF, BenchmarkSuite.FEMNIST },
            CiMode = true,
            Seed = 42,
            Leaf = new LeafFederatedBenchmarkOptions(),
            Vision = new FederatedVisionBenchmarkOptions()
        };

        Assert.Equal(2, options.Suites.Length);
        Assert.True(options.CiMode);
        Assert.NotNull(options.Leaf);
        Assert.NotNull(options.Vision);
    }

    #endregion
}
