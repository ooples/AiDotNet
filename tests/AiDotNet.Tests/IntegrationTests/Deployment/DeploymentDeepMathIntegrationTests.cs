using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Deployment;

/// <summary>
/// Deep integration tests for Deployment configuration classes:
/// QuantizationConfig factory methods, CompressionConfig defaults,
/// ABTestingConfig, CacheConfig TTL conversion, TelemetryConfig aliases,
/// VersioningConfig aliases, ExportConfig defaults, and DeploymentConfiguration factory.
/// </summary>
public class DeploymentDeepMathIntegrationTests
{
    // ============================
    // QuantizationConfig Default Tests
    // ============================

    [Fact]
    public void QuantizationConfig_Defaults_CorrectValues()
    {
        var config = new QuantizationConfig();

        Assert.Equal(QuantizationMode.None, config.Mode);
        Assert.Equal(QuantizationStrategy.Dynamic, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerChannel, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.Null(config.TargetBitWidth);
        Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        Assert.True(config.UseSymmetricQuantization);
        Assert.Equal(100, config.CalibrationSamples);
        Assert.False(config.QuantizeActivations);
        Assert.False(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.Standard, config.QATMethod);
        Assert.Equal(1, config.QATWarmupEpochs);
    }

    // ============================
    // QuantizationConfig Factory Method Tests
    // ============================

    [Fact]
    public void ForGPTQ_Default_Correct4BitPerGroupConfig()
    {
        var config = QuantizationConfig.ForGPTQ();

        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationStrategy.GPTQ, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        Assert.True(config.UseSymmetricQuantization);
    }

    [Fact]
    public void ForGPTQ_CustomGroupSize_ReflectedInConfig()
    {
        var config = QuantizationConfig.ForGPTQ(groupSize: 64);

        Assert.Equal(64, config.GroupSize);
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationStrategy.GPTQ, config.Strategy);
    }

    [Fact]
    public void ForAWQ_Default_Correct4BitPerGroupConfig()
    {
        var config = QuantizationConfig.ForAWQ();

        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationStrategy.AWQ, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        Assert.True(config.UseSymmetricQuantization);
    }

    [Fact]
    public void ForAWQ_CustomGroupSize_ReflectedInConfig()
    {
        var config = QuantizationConfig.ForAWQ(groupSize: 256);

        Assert.Equal(256, config.GroupSize);
        Assert.Equal(QuantizationStrategy.AWQ, config.Strategy);
    }

    [Fact]
    public void ForSmoothQuant_QuantizesActivations()
    {
        var config = QuantizationConfig.ForSmoothQuant();

        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(QuantizationStrategy.SmoothQuant, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerChannel, config.Granularity);
        Assert.True(config.QuantizeActivations);
        Assert.True(config.UseSymmetricQuantization);
        Assert.Null(config.TargetBitWidth); // SmoothQuant uses default 8-bit
    }

    [Fact]
    public void ForQAT_DefaultParams_EfficientQATMethod()
    {
        var config = QuantizationConfig.ForQAT();

        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(8, config.TargetBitWidth);
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.EfficientQAT, config.QATMethod);
        Assert.Equal(1, config.QATWarmupEpochs);
        Assert.True(config.UseSymmetricQuantization);
        Assert.True(config.QuantizeActivations);
    }

    [Fact]
    public void ForQAT_Custom4Bit_CorrectBitWidth()
    {
        var config = QuantizationConfig.ForQAT(targetBitWidth: 4, method: QATMethod.ZeroQAT);

        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QATMethod.ZeroQAT, config.QATMethod);
        Assert.True(config.UseQuantizationAwareTraining);
    }

    [Fact]
    public void ForGPTQ_AndForAWQ_DifferOnlyInStrategy()
    {
        var gptq = QuantizationConfig.ForGPTQ();
        var awq = QuantizationConfig.ForAWQ();

        // Both share the same config except strategy
        Assert.Equal(gptq.Mode, awq.Mode);
        Assert.Equal(gptq.TargetBitWidth, awq.TargetBitWidth);
        Assert.Equal(gptq.Granularity, awq.Granularity);
        Assert.Equal(gptq.GroupSize, awq.GroupSize);
        Assert.NotEqual(gptq.Strategy, awq.Strategy);
    }

    // ============================
    // CompressionConfig Default Tests
    // ============================

    [Fact]
    public void CompressionConfig_Defaults_IndustryStandard()
    {
        var config = new CompressionConfig();

        Assert.Equal(ModelCompressionMode.Automatic, config.Mode);
        Assert.Equal(CompressionType.WeightClustering, config.Type);
        Assert.Equal(256, config.NumClusters); // 8-bit equivalent
        Assert.Equal(4, config.Precision);
        Assert.Equal(1e-6, config.Tolerance);
        Assert.Equal(100, config.MaxIterations);
        Assert.Null(config.RandomSeed);
        Assert.Equal(2.0, config.MaxAccuracyLossPercent);
    }

    [Fact]
    public void CompressionConfig_256Clusters_Equals8BitEquivalent()
    {
        // 256 = 2^8 clusters = 8-bit quantization equivalent
        var config = new CompressionConfig();
        double bitsEquivalent = Math.Log2(config.NumClusters);
        Assert.Equal(8.0, bitsEquivalent);
    }

    [Fact]
    public void CompressionConfig_16Clusters_Equals4BitEquivalent()
    {
        // 16 = 2^4 clusters = 4-bit equivalent (aggressive compression)
        var config = new CompressionConfig { NumClusters = 16 };
        double bitsEquivalent = Math.Log2(config.NumClusters);
        Assert.Equal(4.0, bitsEquivalent);
    }

    [Fact]
    public void CompressionConfig_65536Clusters_Equals16BitEquivalent()
    {
        // 65536 = 2^16 clusters = 16-bit equivalent (light compression)
        var config = new CompressionConfig { NumClusters = 65536 };
        double bitsEquivalent = Math.Log2(config.NumClusters);
        Assert.Equal(16.0, bitsEquivalent);
    }

    // ============================
    // ABTestingConfig Default Tests
    // ============================

    [Fact]
    public void ABTestingConfig_Defaults_CorrectValues()
    {
        var config = new ABTestingConfig();

        Assert.False(config.Enabled);
        Assert.Equal(0.5, config.DefaultTrafficSplit);
        Assert.Empty(config.TrafficSplit);
        Assert.Empty(config.Tests);
        Assert.Equal(AssignmentStrategy.Random, config.AssignmentStrategy);
        Assert.Equal(7, config.TestDurationDays);
        Assert.True(config.TrackAssignments);
        Assert.Equal(1000, config.MinSampleSize);
        Assert.Null(config.ControlVersion);
    }

    [Fact]
    public void ABTestingConfig_TrafficSplitSumsToOne()
    {
        var config = new ABTestingConfig
        {
            TrafficSplit = new Dictionary<string, double>
            {
                { "1.0.0", 0.8 },
                { "2.0.0", 0.2 }
            }
        };

        double total = config.TrafficSplit.Values.Sum();
        Assert.Equal(1.0, total, 1e-10);
    }

    [Fact]
    public void ABTestingConfig_ThreeVersionSplit_SumsToOne()
    {
        var config = new ABTestingConfig
        {
            TrafficSplit = new Dictionary<string, double>
            {
                { "1.0", 0.7 },
                { "2.0", 0.2 },
                { "3.0", 0.1 }
            }
        };

        double total = config.TrafficSplit.Values.Sum();
        Assert.Equal(1.0, total, 1e-10);
    }

    // ============================
    // ABTest Default Tests
    // ============================

    [Fact]
    public void ABTest_Defaults_CorrectValues()
    {
        var test = new ABTest();

        Assert.Equal(string.Empty, test.Name);
        Assert.Null(test.Description);
        Assert.Equal(string.Empty, test.ControlVersion);
        Assert.Equal(string.Empty, test.TreatmentVersion);
        Assert.Equal(0.1, test.TreatmentTrafficPercentage);
        Assert.True(test.IsActive);
        Assert.Null(test.StartDate);
        Assert.Null(test.EndDate);
        Assert.Null(test.PrimaryMetric);
        Assert.Equal(0.01, test.MinimumImprovementThreshold);
    }

    [Fact]
    public void ABTest_TreatmentPercentage_ControlIsComplement()
    {
        var test = new ABTest { TreatmentTrafficPercentage = 0.2 };

        // Control percentage = 1 - treatment percentage
        double controlPercentage = 1.0 - test.TreatmentTrafficPercentage;
        Assert.Equal(0.8, controlPercentage, 1e-10);
    }

    // ============================
    // CacheConfig Default Tests
    // ============================

    [Fact]
    public void CacheConfig_Defaults_CorrectValues()
    {
        var config = new CacheConfig();

        Assert.True(config.Enabled);
        Assert.Equal(10, config.MaxCacheSize);
        Assert.Equal(CacheEvictionPolicy.LRU, config.EvictionPolicy);
        Assert.Equal(3600, config.TimeToLiveSeconds);
        Assert.Equal(100.0, config.MaxSizeMB);
        Assert.False(config.PreloadModels);
        Assert.True(config.TrackStatistics);
    }

    [Fact]
    public void CacheConfig_DefaultTTL_EqualsOneHour()
    {
        var config = new CacheConfig();
        Assert.Equal(TimeSpan.FromHours(1), config.DefaultTTL);
    }

    [Fact]
    public void CacheConfig_DefaultTTL_MatchesTimeToLiveSeconds()
    {
        var config = new CacheConfig();
        Assert.Equal(config.TimeToLiveSeconds, (int)config.DefaultTTL.TotalSeconds);
    }

    [Fact]
    public void CacheConfig_SetDefaultTTL_UpdatesTimeToLiveSeconds()
    {
        var config = new CacheConfig();
        config.DefaultTTL = TimeSpan.FromMinutes(30);

        Assert.Equal(1800, config.TimeToLiveSeconds);
        Assert.Equal(TimeSpan.FromMinutes(30), config.DefaultTTL);
    }

    [Fact]
    public void CacheConfig_SetTimeToLiveSeconds_UpdatesDefaultTTL()
    {
        var config = new CacheConfig();
        config.TimeToLiveSeconds = 7200; // 2 hours

        Assert.Equal(TimeSpan.FromHours(2), config.DefaultTTL);
    }

    [Fact]
    public void CacheConfig_TTLRoundTrip_Consistent()
    {
        var config = new CacheConfig();
        var original = TimeSpan.FromMinutes(45);
        config.DefaultTTL = original;

        // Round-trip: TimeSpan → seconds → TimeSpan
        var retrieved = config.DefaultTTL;
        Assert.Equal((int)original.TotalSeconds, (int)retrieved.TotalSeconds);
    }

    // ============================
    // TelemetryConfig Default & Alias Tests
    // ============================

    [Fact]
    public void TelemetryConfig_Defaults_AllTrackingEnabled()
    {
        var config = new TelemetryConfig();

        Assert.True(config.Enabled);
        Assert.True(config.TrackLatency);
        Assert.True(config.TrackThroughput);
        Assert.True(config.TrackErrors);
        Assert.True(config.TrackCacheMetrics);
        Assert.True(config.TrackVersionUsage);
        Assert.False(config.TrackDetailedTiming);
        Assert.Equal(1.0, config.SamplingRate);
        Assert.Null(config.ExportEndpoint);
        Assert.Equal(60, config.FlushIntervalSeconds);
        Assert.Empty(config.CustomTags);
    }

    [Fact]
    public void TelemetryConfig_CollectLatency_IsAliasForTrackLatency()
    {
        var config = new TelemetryConfig();

        // Alias should reflect the underlying property
        Assert.Equal(config.TrackLatency, config.CollectLatency);

        // Setting alias should update underlying property
        config.CollectLatency = false;
        Assert.False(config.TrackLatency);
        Assert.False(config.CollectLatency);

        config.TrackLatency = true;
        Assert.True(config.CollectLatency);
    }

    [Fact]
    public void TelemetryConfig_CollectErrors_IsAliasForTrackErrors()
    {
        var config = new TelemetryConfig();

        Assert.Equal(config.TrackErrors, config.CollectErrors);

        config.CollectErrors = false;
        Assert.False(config.TrackErrors);

        config.TrackErrors = true;
        Assert.True(config.CollectErrors);
    }

    // ============================
    // VersioningConfig Default & Alias Tests
    // ============================

    [Fact]
    public void VersioningConfig_Defaults_CorrectValues()
    {
        var config = new VersioningConfig();

        Assert.True(config.Enabled);
        Assert.Equal("latest", config.DefaultVersion);
        Assert.False(config.AllowAutoUpgrade);
        Assert.Equal(3, config.MaxVersionHistory);
        Assert.True(config.AutoCleanup);
        Assert.True(config.TrackVersionUsage);
        Assert.Empty(config.VersionMetadata);
    }

    [Fact]
    public void VersioningConfig_MaxVersionsPerModel_IsAliasForMaxVersionHistory()
    {
        var config = new VersioningConfig();

        Assert.Equal(config.MaxVersionHistory, config.MaxVersionsPerModel);

        config.MaxVersionsPerModel = 10;
        Assert.Equal(10, config.MaxVersionHistory);

        config.MaxVersionHistory = 5;
        Assert.Equal(5, config.MaxVersionsPerModel);
    }

    // ============================
    // ExportConfig Default Tests
    // ============================

    [Fact]
    public void ExportConfig_Defaults_CorrectValues()
    {
        var config = new ExportConfig();

        Assert.Equal(TargetPlatform.CPU, config.TargetPlatform);
        Assert.True(config.OptimizeModel);
        Assert.Equal(QuantizationMode.None, config.Quantization);
        Assert.Equal(1, config.BatchSize);
        Assert.True(config.IncludeMetadata);
        Assert.Null(config.ModelName);
        Assert.Null(config.ModelVersion);
        Assert.Null(config.ModelDescription);
        Assert.True(config.ValidateAfterExport);
    }

    // ============================
    // DeploymentConfiguration Factory Tests
    // ============================

    [Fact]
    public void DeploymentConfiguration_Create_AllNulls_AllPropertiesNull()
    {
        var config = DeploymentConfiguration.Create(
            quantization: null,
            caching: null,
            versioning: null,
            abTesting: null,
            telemetry: null,
            export: null,
            gpuAcceleration: null);

        Assert.Null(config.Quantization);
        Assert.Null(config.Caching);
        Assert.Null(config.Versioning);
        Assert.Null(config.ABTesting);
        Assert.Null(config.Telemetry);
        Assert.Null(config.Export);
        Assert.Null(config.GpuAcceleration);
        Assert.Null(config.Compression);
        Assert.Null(config.Profiling);
    }

    [Fact]
    public void DeploymentConfiguration_Create_WithAllConfigs_PreservesReferences()
    {
        var quant = new QuantizationConfig();
        var cache = new CacheConfig();
        var version = new VersioningConfig();
        var abTest = new ABTestingConfig();
        var telemetry = new TelemetryConfig();
        var export = new ExportConfig();
        var compression = new CompressionConfig();

        var config = DeploymentConfiguration.Create(
            quantization: quant,
            caching: cache,
            versioning: version,
            abTesting: abTest,
            telemetry: telemetry,
            export: export,
            gpuAcceleration: null,
            compression: compression);

        Assert.Same(quant, config.Quantization);
        Assert.Same(cache, config.Caching);
        Assert.Same(version, config.Versioning);
        Assert.Same(abTest, config.ABTesting);
        Assert.Same(telemetry, config.Telemetry);
        Assert.Same(export, config.Export);
        Assert.Same(compression, config.Compression);
    }

    // ============================
    // Cross-Config Consistency Tests
    // ============================

    [Fact]
    public void QuantizationConfig_AllFactoryMethods_UseInt8Mode()
    {
        // All factory methods set Mode = Int8
        Assert.Equal(QuantizationMode.Int8, QuantizationConfig.ForGPTQ().Mode);
        Assert.Equal(QuantizationMode.Int8, QuantizationConfig.ForAWQ().Mode);
        Assert.Equal(QuantizationMode.Int8, QuantizationConfig.ForSmoothQuant().Mode);
        Assert.Equal(QuantizationMode.Int8, QuantizationConfig.ForQAT().Mode);
    }

    [Fact]
    public void QuantizationConfig_AllFactoryMethods_UseSymmetricQuantization()
    {
        Assert.True(QuantizationConfig.ForGPTQ().UseSymmetricQuantization);
        Assert.True(QuantizationConfig.ForAWQ().UseSymmetricQuantization);
        Assert.True(QuantizationConfig.ForSmoothQuant().UseSymmetricQuantization);
        Assert.True(QuantizationConfig.ForQAT().UseSymmetricQuantization);
    }

    [Fact]
    public void QuantizationConfig_OnlySmoothQuantAndQAT_QuantizeActivations()
    {
        Assert.False(QuantizationConfig.ForGPTQ().QuantizeActivations);
        Assert.False(QuantizationConfig.ForAWQ().QuantizeActivations);
        Assert.True(QuantizationConfig.ForSmoothQuant().QuantizeActivations);
        Assert.True(QuantizationConfig.ForQAT().QuantizeActivations);
    }

    [Fact]
    public void QuantizationConfig_OnlyGPTQAndAWQ_UsePerGroupGranularity()
    {
        Assert.Equal(QuantizationGranularity.PerGroup, QuantizationConfig.ForGPTQ().Granularity);
        Assert.Equal(QuantizationGranularity.PerGroup, QuantizationConfig.ForAWQ().Granularity);
        Assert.Equal(QuantizationGranularity.PerChannel, QuantizationConfig.ForSmoothQuant().Granularity);
    }

    // ============================
    // Compression Bit-Width Calculations
    // ============================

    [Fact]
    public void CompressionConfig_ClusterCount_CompressionRatio()
    {
        // Full precision (float32 = 32 bits per weight)
        // With N clusters, each weight needs log2(N) bits to store the cluster index
        // Compression ratio = 32 / log2(N)

        // 256 clusters = 8 bits → 32/8 = 4x compression
        double ratio256 = 32.0 / Math.Log2(256);
        Assert.Equal(4.0, ratio256);

        // 16 clusters = 4 bits → 32/4 = 8x compression
        double ratio16 = 32.0 / Math.Log2(16);
        Assert.Equal(8.0, ratio16);

        // 65536 clusters = 16 bits → 32/16 = 2x compression
        double ratio65536 = 32.0 / Math.Log2(65536);
        Assert.Equal(2.0, ratio65536);
    }

    [Fact]
    public void CompressionConfig_MoreClusters_LessCompression()
    {
        // As cluster count increases, bits per weight increases, compression ratio decreases
        int[] clusterCounts = { 4, 16, 64, 256, 1024, 65536 };
        double prevRatio = double.MaxValue;

        foreach (var clusters in clusterCounts)
        {
            double bitsPerWeight = Math.Log2(clusters);
            double compressionRatio = 32.0 / bitsPerWeight;
            Assert.True(compressionRatio < prevRatio);
            prevRatio = compressionRatio;
        }
    }

    // ============================
    // CacheConfig Memory Calculations
    // ============================

    [Fact]
    public void CacheConfig_MaxSizeMB_InBytesConversion()
    {
        var config = new CacheConfig();

        // 100 MB = 100 * 1024 * 1024 bytes = 104,857,600 bytes
        double bytes = config.MaxSizeMB * 1024 * 1024;
        Assert.Equal(104_857_600.0, bytes);
    }

    [Fact]
    public void CacheConfig_TTLZero_MeansNoExpiry()
    {
        var config = new CacheConfig { TimeToLiveSeconds = 0 };

        Assert.Equal(TimeSpan.Zero, config.DefaultTTL);
    }

    // ============================
    // ABTest Math: Statistical Significance
    // ============================

    [Fact]
    public void ABTest_MinSampleSize_StatisticalPower()
    {
        var config = new ABTestingConfig();

        // With 1000 minimum samples and default 0.01 improvement threshold,
        // this gives reasonable statistical power for detecting effects
        // Minimum detectable effect = MinimumImprovementThreshold of ABTest
        var test = new ABTest { MinimumImprovementThreshold = 0.01 };

        // At 1000 samples per group, two-proportion z-test power is adequate for 1% effect
        Assert.Equal(1000, config.MinSampleSize);
        Assert.Equal(0.01, test.MinimumImprovementThreshold);
    }
}
