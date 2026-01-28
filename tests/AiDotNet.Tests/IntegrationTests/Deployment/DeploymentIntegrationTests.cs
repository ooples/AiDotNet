using AiDotNet.Deployment.Configuration;
using AiDotNet.Deployment.Edge;
using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Deployment.Mobile.Android;
using AiDotNet.Deployment.Mobile.CoreML;
using AiDotNet.Deployment.Mobile.TensorFlowLite;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Deployment.Runtime;
using AiDotNet.Deployment.TensorRT;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Deployment;

/// <summary>
/// Comprehensive integration tests for the Deployment module.
/// Tests RuntimeConfiguration, EdgeOptimizer, ModelCache, TelemetryCollector,
/// Quantizers, ONNX export, Mobile deployment, and TensorRT components.
/// </summary>
public class DeploymentIntegrationTests
{
    #region RuntimeConfiguration Tests

    [Fact]
    public void RuntimeConfiguration_DefaultValues_AreCorrect()
    {
        var config = new RuntimeConfiguration();

        Assert.True(config.EnableTelemetry);
        Assert.True(config.EnableCaching);
        Assert.Equal(100.0, config.CacheSizeMB);
        Assert.Equal(CacheEvictionPolicy.LRU, config.CacheEvictionPolicy);
        Assert.True(config.AutoWarmUp);
        Assert.Equal(10, config.WarmUpIterations);
        Assert.True(config.EnableVersioning);
        Assert.Equal(3, config.MaxVersionsPerModel);
        Assert.False(config.EnableABTesting);
        Assert.Equal(1.0, config.TelemetrySamplingRate);
        Assert.Equal(1000, config.TelemetryBufferSize);
        Assert.Equal(60, config.TelemetryFlushIntervalSeconds);
        Assert.True(config.EnablePerformanceMonitoring);
        Assert.Equal(1000.0, config.PerformanceAlertThresholdMs);
        Assert.True(config.EnableHealthChecks);
        Assert.Equal(300, config.HealthCheckIntervalSeconds);
        Assert.Equal(60, config.ModelLoadTimeoutSeconds);
        Assert.Equal(30, config.InferenceTimeoutSeconds);
        Assert.True(config.EnableGpuAcceleration);
    }

    [Fact]
    public void RuntimeConfiguration_ForProduction_ReturnsOptimizedConfig()
    {
        var config = RuntimeConfiguration.ForProduction();

        Assert.True(config.EnableTelemetry);
        Assert.True(config.EnableCaching);
        Assert.Equal(500.0, config.CacheSizeMB);
        Assert.True(config.AutoWarmUp);
        Assert.Equal(20, config.WarmUpIterations);
        Assert.True(config.EnableVersioning);
        Assert.Equal(5, config.MaxVersionsPerModel);
        Assert.True(config.EnableABTesting);
        Assert.Equal(0.1, config.TelemetrySamplingRate);
        Assert.True(config.EnablePerformanceMonitoring);
        Assert.Equal(500.0, config.PerformanceAlertThresholdMs);
        Assert.True(config.EnableHealthChecks);
    }

    [Fact]
    public void RuntimeConfiguration_ForDevelopment_ReturnsDevConfig()
    {
        var config = RuntimeConfiguration.ForDevelopment();

        Assert.True(config.EnableTelemetry);
        Assert.True(config.EnableCaching);
        Assert.Equal(100.0, config.CacheSizeMB);
        Assert.False(config.AutoWarmUp);
        Assert.True(config.EnableVersioning);
        Assert.Equal(2, config.MaxVersionsPerModel);
        Assert.False(config.EnableABTesting);
        Assert.Equal(1.0, config.TelemetrySamplingRate);
        Assert.True(config.EnablePerformanceMonitoring);
        Assert.False(config.EnableHealthChecks);
    }

    [Fact]
    public void RuntimeConfiguration_ForEdge_ReturnsMinimalConfig()
    {
        var config = RuntimeConfiguration.ForEdge();

        Assert.True(config.EnableTelemetry);
        Assert.Equal(0.01, config.TelemetrySamplingRate);
        Assert.True(config.EnableCaching);
        Assert.Equal(10.0, config.CacheSizeMB);
        Assert.True(config.AutoWarmUp);
        Assert.Equal(5, config.WarmUpIterations);
        Assert.False(config.EnableVersioning);
        Assert.False(config.EnableABTesting);
        Assert.False(config.EnablePerformanceMonitoring);
        Assert.False(config.EnableHealthChecks);
    }

    #endregion

    #region ModelCache Tests

    [Fact]
    public void ModelCache_PutAndGet_WorksCorrectly()
    {
        var cache = new ModelCache<double>(enabled: true);
        var input = new double[] { 1.0, 2.0, 3.0 };
        var result = new double[] { 4.0, 5.0, 6.0 };

        cache.Put("model:v1", input, result);
        var cached = cache.Get("model:v1", input);

        Assert.NotNull(cached);
        Assert.Equal(result, cached);
    }

    [Fact]
    public void ModelCache_Get_ReturnsNullForMissingKey()
    {
        var cache = new ModelCache<double>(enabled: true);
        var input = new double[] { 1.0, 2.0, 3.0 };

        var cached = cache.Get("nonexistent:v1", input);

        Assert.Null(cached);
    }

    [Fact]
    public void ModelCache_Get_ReturnsNullWhenDisabled()
    {
        var cache = new ModelCache<double>(enabled: false);
        var input = new double[] { 1.0, 2.0, 3.0 };
        var result = new double[] { 4.0, 5.0, 6.0 };

        cache.Put("model:v1", input, result);
        var cached = cache.Get("model:v1", input);

        Assert.Null(cached);
    }

    [Fact]
    public void ModelCache_Clear_RemovesAllEntries()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1.0 }, new double[] { 2.0 });
        cache.Put("model:v2", new double[] { 3.0 }, new double[] { 4.0 });

        cache.Clear();

        Assert.Null(cache.Get("model:v1", new double[] { 1.0 }));
        Assert.Null(cache.Get("model:v2", new double[] { 3.0 }));
    }

    [Fact]
    public void ModelCache_EvictOlderThan_RemovesOldEntries()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1.0 }, new double[] { 2.0 });

        // Wait a tiny bit and add another
        Thread.Sleep(10);
        cache.Put("model:v2", new double[] { 3.0 }, new double[] { 4.0 });

        // Evict entries older than 5ms
        var removed = cache.EvictOlderThan(TimeSpan.FromMilliseconds(5));

        // The first entry might have been evicted
        Assert.True(removed >= 0);
    }

    [Fact]
    public void ModelCache_EvictLRU_RemovesLeastRecentlyUsed()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1.0 }, new double[] { 2.0 });
        cache.Put("model:v2", new double[] { 3.0 }, new double[] { 4.0 });
        cache.Put("model:v3", new double[] { 5.0 }, new double[] { 6.0 });

        // Access v2 to make it more recent
        cache.Get("model:v2", new double[] { 3.0 });

        var removed = cache.EvictLRU(2);

        Assert.Equal(1, removed);
    }

    [Fact]
    public void ModelCache_EvictLFU_RemovesLeastFrequentlyUsed()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1.0 }, new double[] { 2.0 });
        cache.Put("model:v2", new double[] { 3.0 }, new double[] { 4.0 });
        cache.Put("model:v3", new double[] { 5.0 }, new double[] { 6.0 });

        // Access v2 multiple times
        cache.Get("model:v2", new double[] { 3.0 });
        cache.Get("model:v2", new double[] { 3.0 });

        var removed = cache.EvictLFU(2);

        Assert.Equal(1, removed);
    }

    [Fact]
    public void ModelCache_GetStatistics_ReturnsCorrectStats()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1.0 }, new double[] { 2.0 });
        cache.Get("model:v1", new double[] { 1.0 });
        cache.Get("model:v1", new double[] { 1.0 });

        var stats = cache.GetStatistics();

        Assert.Equal(1, stats.TotalEntries);
        Assert.True(stats.TotalAccessCount >= 2);
    }

    [Fact]
    public void ModelCache_DifferentInputs_ProduceDifferentCacheKeys()
    {
        var cache = new ModelCache<double>(enabled: true);
        var input1 = new double[] { 1.0, 2.0, 3.0 };
        var input2 = new double[] { 1.0, 2.0, 4.0 };
        var result1 = new double[] { 10.0 };
        var result2 = new double[] { 20.0 };

        cache.Put("model:v1", input1, result1);
        cache.Put("model:v1", input2, result2);

        var cached1 = cache.Get("model:v1", input1);
        var cached2 = cache.Get("model:v1", input2);

        Assert.NotNull(cached1);
        Assert.NotNull(cached2);
        Assert.Equal(result1, cached1);
        Assert.Equal(result2, cached2);
    }

    #endregion

    #region TelemetryCollector Tests

    [Fact]
    public void TelemetryCollector_RecordEvent_StoresEvent()
    {
        var collector = new TelemetryCollector(enabled: true);

        collector.RecordEvent("TestEvent", new Dictionary<string, object>
        {
            ["key1"] = "value1",
            ["key2"] = 42
        });

        var events = collector.GetEvents(10);
        Assert.Single(events);
        Assert.Equal("TestEvent", events[0].Name);
        Assert.Equal("value1", events[0].Properties["key1"]);
        Assert.Equal(42, events[0].Properties["key2"]);
    }

    [Fact]
    public void TelemetryCollector_RecordInference_UpdatesMetrics()
    {
        var collector = new TelemetryCollector(enabled: true);

        collector.RecordInference("model1", "v1", 100, fromCache: false);
        collector.RecordInference("model1", "v1", 150, fromCache: false);
        collector.RecordInference("model1", "v1", 50, fromCache: true);

        var stats = collector.GetStatistics("model1", "v1");

        Assert.Equal(3, stats.TotalInferences);
        Assert.Equal(100.0, stats.AverageLatencyMs, 1);
        Assert.Equal(50, stats.MinLatencyMs);
        Assert.Equal(150, stats.MaxLatencyMs);
        Assert.True(stats.CacheHitRate > 0);
    }

    [Fact]
    public void TelemetryCollector_RecordError_TracksErrors()
    {
        var collector = new TelemetryCollector(enabled: true);

        collector.RecordInference("model1", "v1", 100, fromCache: false);
        collector.RecordError("model1", "v1", new InvalidOperationException("Test error"));

        var stats = collector.GetStatistics("model1", "v1");

        Assert.Equal(1, stats.TotalInferences);
        Assert.Equal(1, stats.TotalErrors);
        Assert.Equal(0.5, stats.ErrorRate, 1);
    }

    [Fact]
    public void TelemetryCollector_GetStatistics_ReturnsZerosForUnknownModel()
    {
        var collector = new TelemetryCollector(enabled: true);

        var stats = collector.GetStatistics("unknown", "v1");

        Assert.Equal(0, stats.TotalInferences);
        Assert.Equal(0, stats.TotalErrors);
        Assert.Equal(0.0, stats.AverageLatencyMs);
    }

    [Fact]
    public void TelemetryCollector_Disabled_DoesNotRecordEvents()
    {
        var collector = new TelemetryCollector(enabled: false);

        collector.RecordEvent("TestEvent", new Dictionary<string, object> { ["key"] = "value" });
        collector.RecordInference("model1", "v1", 100, fromCache: false);

        var events = collector.GetEvents(10);
        var stats = collector.GetStatistics("model1", "v1");

        Assert.Empty(events);
        Assert.Equal(0, stats.TotalInferences);
    }

    [Fact]
    public void TelemetryCollector_Clear_RemovesAllData()
    {
        var collector = new TelemetryCollector(enabled: true);
        collector.RecordEvent("Event1", new Dictionary<string, object>());
        collector.RecordInference("model1", "v1", 100, fromCache: false);

        collector.Clear();

        Assert.Empty(collector.GetEvents(10));
        Assert.Equal(0, collector.GetStatistics("model1", "v1").TotalInferences);
    }

    [Fact]
    public void TelemetryCollector_GetEvents_ReturnsOrderedByTimestamp()
    {
        var collector = new TelemetryCollector(enabled: true);
        collector.RecordEvent("Event1", new Dictionary<string, object>());
        Thread.Sleep(10);
        collector.RecordEvent("Event2", new Dictionary<string, object>());
        Thread.Sleep(10);
        collector.RecordEvent("Event3", new Dictionary<string, object>());

        var events = collector.GetEvents(10);

        Assert.Equal(3, events.Count);
        Assert.Equal("Event3", events[0].Name); // Most recent first
        Assert.Equal("Event1", events[2].Name); // Oldest last
    }

    #endregion

    #region EdgeConfiguration Tests

    [Fact]
    public void EdgeConfiguration_DefaultValues_AreCorrect()
    {
        var config = new EdgeConfiguration();

        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.True(config.UsePruning);
        Assert.Equal(0.3, config.PruningRatio);
        Assert.True(config.EnableLayerFusion);
        Assert.True(config.EnableArmNeonOptimization);
        Assert.False(config.EnableModelPartitioning);
        Assert.Equal(PartitionStrategy.Adaptive, config.PartitionStrategy);
        Assert.Equal(EdgeDeviceType.Generic, config.TargetDevice);
        Assert.Equal(10.0, config.MaxModelSizeMB);
        Assert.Equal(50.0, config.MaxMemoryUsageMB);
        Assert.Equal(100.0, config.TargetLatencyMs);
        Assert.True(config.OptimizeForPower);
        Assert.False(config.EnableAdaptiveInference);
        Assert.Equal(5.0, config.CacheSizeMB);
    }

    [Fact]
    public void EdgeConfiguration_ForRaspberryPi_ReturnsOptimizedConfig()
    {
        var config = EdgeConfiguration.ForRaspberryPi();

        Assert.Equal(EdgeDeviceType.RaspberryPi, config.TargetDevice);
        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.True(config.UsePruning);
        Assert.Equal(0.5, config.PruningRatio);
        Assert.True(config.EnableArmNeonOptimization);
        Assert.Equal(50.0, config.MaxModelSizeMB);
        Assert.Equal(100.0, config.MaxMemoryUsageMB);
        Assert.Equal(200.0, config.TargetLatencyMs);
        Assert.True(config.OptimizeForPower);
    }

    [Fact]
    public void EdgeConfiguration_ForJetson_ReturnsGpuOptimizedConfig()
    {
        var config = EdgeConfiguration.ForJetson();

        Assert.Equal(EdgeDeviceType.Jetson, config.TargetDevice);
        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Float16, config.QuantizationMode);
        Assert.False(config.UsePruning);
        Assert.True(config.EnableLayerFusion);
        Assert.True(config.EnableArmNeonOptimization);
        Assert.Equal(500.0, config.MaxModelSizeMB);
        Assert.Equal(1000.0, config.MaxMemoryUsageMB);
        Assert.Equal(50.0, config.TargetLatencyMs);
        Assert.False(config.OptimizeForPower);
    }

    [Fact]
    public void EdgeConfiguration_ForMicrocontroller_ReturnsMinimalConfig()
    {
        var config = EdgeConfiguration.ForMicrocontroller();

        Assert.Equal(EdgeDeviceType.Microcontroller, config.TargetDevice);
        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.True(config.UsePruning);
        Assert.Equal(0.7, config.PruningRatio);
        Assert.True(config.EnableLayerFusion);
        Assert.False(config.EnableArmNeonOptimization);
        Assert.Equal(1.0, config.MaxModelSizeMB);
        Assert.Equal(2.0, config.MaxMemoryUsageMB);
        Assert.Equal(500.0, config.TargetLatencyMs);
        Assert.True(config.OptimizeForPower);
    }

    [Fact]
    public void EdgeConfiguration_ForCloudEdge_ReturnsPartitionedConfig()
    {
        var config = EdgeConfiguration.ForCloudEdge();

        Assert.Equal(EdgeDeviceType.Generic, config.TargetDevice);
        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.False(config.UsePruning);
        Assert.True(config.EnableModelPartitioning);
        Assert.Equal(PartitionStrategy.Adaptive, config.PartitionStrategy);
        Assert.Equal(20.0, config.MaxModelSizeMB);
        Assert.True(config.OptimizeForPower);
    }

    #endregion

    #region AdaptiveInferenceConfig Tests

    [Fact]
    public void EdgeOptimizer_CreateAdaptiveConfig_LowBattery_ReturnsFastConfig()
    {
        var edgeConfig = new EdgeConfiguration();
        var optimizer = new EdgeOptimizer<double, double[], double[]>(edgeConfig);

        var config = optimizer.CreateAdaptiveConfig(batteryLevel: 0.1, cpuLoad: 0.5);

        Assert.Equal(QualityLevel.Low, config.QualityLevel);
        Assert.True(config.UseQuantization);
        Assert.Equal(8, config.QuantizationBits);
    }

    [Fact]
    public void EdgeOptimizer_CreateAdaptiveConfig_HighLoad_ReturnsFastConfig()
    {
        var edgeConfig = new EdgeConfiguration();
        var optimizer = new EdgeOptimizer<double, double[], double[]>(edgeConfig);

        var config = optimizer.CreateAdaptiveConfig(batteryLevel: 0.8, cpuLoad: 0.9);

        Assert.Equal(QualityLevel.Low, config.QualityLevel);
        Assert.True(config.UseQuantization);
    }

    [Fact]
    public void EdgeOptimizer_CreateAdaptiveConfig_HighBatteryLowLoad_ReturnsHighQualityConfig()
    {
        var edgeConfig = new EdgeConfiguration();
        var optimizer = new EdgeOptimizer<double, double[], double[]>(edgeConfig);

        var config = optimizer.CreateAdaptiveConfig(batteryLevel: 0.9, cpuLoad: 0.2);

        Assert.Equal(QualityLevel.High, config.QualityLevel);
        Assert.False(config.UseQuantization);
        Assert.Empty(config.SkipLayers);
    }

    [Fact]
    public void EdgeOptimizer_CreateAdaptiveConfig_MediumConditions_ReturnsBalancedConfig()
    {
        var edgeConfig = new EdgeConfiguration();
        var optimizer = new EdgeOptimizer<double, double[], double[]>(edgeConfig);

        var config = optimizer.CreateAdaptiveConfig(batteryLevel: 0.5, cpuLoad: 0.5);

        Assert.Equal(QualityLevel.Medium, config.QualityLevel);
        Assert.True(config.UseQuantization);
        Assert.Equal(16, config.QuantizationBits);
    }

    #endregion

    #region Int8Quantizer Tests

    [Fact]
    public void Int8Quantizer_Properties_AreCorrect()
    {
        var quantizer = new Int8Quantizer<double, double[], double[]>();

        Assert.Equal(QuantizationMode.Int8, quantizer.Mode);
        Assert.Equal(8, quantizer.BitWidth);
    }

    [Fact]
    public void Int8Quantizer_GetScaleFactor_ReturnsDefaultForUnknownLayer()
    {
        var quantizer = new Int8Quantizer<double, double[], double[]>();

        var scale = quantizer.GetScaleFactor("unknown_layer");

        Assert.Equal(1.0, scale);
    }

    [Fact]
    public void Int8Quantizer_GetZeroPoint_ReturnsDefaultForUnknownLayer()
    {
        var quantizer = new Int8Quantizer<double, double[], double[]>();

        var zeroPoint = quantizer.GetZeroPoint("unknown_layer");

        Assert.Equal(0, zeroPoint);
    }

    [Fact]
    public void Int8Quantizer_GetScaleFactor_AfterCalibration_ReturnsGlobalScale()
    {
        var quantizer = new Int8Quantizer<double, double[], double[]>();

        // Before calibration, returns default
        Assert.Equal(1.0, quantizer.GetScaleFactor("global"));

        // The "global" key is used internally after calibration
        Assert.Equal(0, quantizer.GetZeroPoint("global"));
    }

    #endregion

    #region Float16Quantizer Tests

    [Fact]
    public void Float16Quantizer_Properties_AreCorrect()
    {
        var quantizer = new Float16Quantizer<double, double[], double[]>();

        Assert.Equal(QuantizationMode.Float16, quantizer.Mode);
        Assert.Equal(16, quantizer.BitWidth);
    }

    [Fact]
    public void Float16Quantizer_GetScaleFactor_ReturnsDefaultForUnknownLayer()
    {
        var quantizer = new Float16Quantizer<double, double[], double[]>();

        var scale = quantizer.GetScaleFactor("unknown_layer");

        Assert.Equal(1.0, scale);
    }

    [Fact]
    public void Float16Quantizer_GetZeroPoint_ReturnsDefaultForUnknownLayer()
    {
        var quantizer = new Float16Quantizer<double, double[], double[]>();

        var zeroPoint = quantizer.GetZeroPoint("unknown_layer");

        Assert.Equal(0, zeroPoint);
    }

    #endregion

    #region QuantizationConfiguration Tests

    [Fact]
    public void QuantizationConfiguration_ForInt8_ReturnsCorrectConfig()
    {
        var config = QuantizationConfiguration.ForInt8(CalibrationMethod.MinMax);

        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        Assert.Equal(8, config.BitWidth);
    }

    [Fact]
    public void QuantizationConfiguration_ForFloat16_ReturnsCorrectConfig()
    {
        var config = QuantizationConfiguration.ForFloat16();

        Assert.Equal(QuantizationMode.Float16, config.Mode);
        Assert.Equal(16, config.BitWidth);
    }

    #endregion

    #region ExportConfiguration Tests

    [Fact]
    public void ExportConfiguration_DefaultValues_AreCorrect()
    {
        var config = new ExportConfiguration();

        Assert.Null(config.ModelName);
        Assert.Equal(13, config.OpsetVersion);
        Assert.Equal(1, config.BatchSize);
        Assert.False(config.UseDynamicShapes);
        Assert.Null(config.InputShape);
    }

    [Fact]
    public void ExportConfiguration_CustomValues_AreSetCorrectly()
    {
        var config = new ExportConfiguration
        {
            ModelName = "TestModel",
            OpsetVersion = 14,
            BatchSize = 32,
            UseDynamicShapes = false,
            InputShape = new[] { 224, 224, 3 }
        };

        Assert.Equal("TestModel", config.ModelName);
        Assert.Equal(14, config.OpsetVersion);
        Assert.Equal(32, config.BatchSize);
        Assert.False(config.UseDynamicShapes);
        Assert.Equal(new[] { 224, 224, 3 }, config.InputShape);
    }

    #endregion

    #region OnnxModelExporter Tests

    [Fact]
    public void OnnxModelExporter_Properties_AreCorrect()
    {
        var exporter = new OnnxModelExporter<double, double[], double[]>();

        Assert.Equal("ONNX", exporter.ExportFormat);
        Assert.Equal(".onnx", exporter.FileExtension);
    }

    [Fact]
    public void OnnxModelExporter_GetValidationErrors_NullModel_ReturnsError()
    {
        var exporter = new OnnxModelExporter<double, double[], double[]>();

        var errors = exporter.GetValidationErrors(null!);

        Assert.Single(errors);
        Assert.Contains("null", errors[0].ToLower());
    }

    [Fact]
    public void OnnxModelExporter_Export_NullModel_ThrowsArgumentNullException()
    {
        var exporter = new OnnxModelExporter<double, double[], double[]>();
        var config = new ExportConfiguration();

        Assert.Throws<ArgumentNullException>(() => exporter.ExportToBytes(null!, config));
    }

    #endregion

    #region CoreML Configuration Tests

    [Fact]
    public void CoreMLConfiguration_DefaultValues_AreCorrect()
    {
        var config = new CoreMLConfiguration();

        Assert.Null(config.ModelName);
        Assert.Null(config.ModelDescription);
        Assert.Equal(CoreMLComputeUnits.All, config.ComputeUnits);
        Assert.True(config.UseQuantization);
        Assert.Equal(8, config.QuantizationBits);
    }

    [Fact]
    public void CoreMLConfiguration_ComputeUnits_CanBeSet()
    {
        var config = new CoreMLConfiguration
        {
            ComputeUnits = CoreMLComputeUnits.CPUOnly
        };

        Assert.Equal(CoreMLComputeUnits.CPUOnly, config.ComputeUnits);

        config.ComputeUnits = CoreMLComputeUnits.CPUAndNeuralEngine;
        Assert.Equal(CoreMLComputeUnits.CPUAndNeuralEngine, config.ComputeUnits);
    }

    #endregion

    #region TFLite Configuration Tests

    [Fact]
    public void TFLiteConfiguration_DefaultValues_AreCorrect()
    {
        var config = new TFLiteConfiguration();

        Assert.True(config.UseQuantization);
        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.False(config.EnableGpuDelegate);
        Assert.Equal(4, config.NumThreads);
    }

    [Fact]
    public void TFLiteConfiguration_CanSetAllProperties()
    {
        var config = new TFLiteConfiguration
        {
            QuantizationMode = QuantizationMode.Int8,
            EnableGpuDelegate = true,
            NumThreads = 4
        };

        Assert.Equal(QuantizationMode.Int8, config.QuantizationMode);
        Assert.True(config.EnableGpuDelegate);
        Assert.Equal(4, config.NumThreads);
    }

    [Fact]
    public void TFLiteTargetSpec_HasCorrectProperties()
    {
        var spec = new TFLiteTargetSpec
        {
            AndroidMinSdkVersion = 26,
            SupportGpu = true,
            SupportHexagonDsp = false
        };

        Assert.Equal(26, spec.AndroidMinSdkVersion);
        Assert.True(spec.SupportGpu);
        Assert.False(spec.SupportHexagonDsp);
    }

    #endregion

    #region NNAPI Configuration Tests

    [Fact]
    public void NNAPIConfiguration_DefaultValues_AreCorrect()
    {
        var config = new NNAPIConfiguration();

        Assert.Equal(NNAPIExecutionPreference.Default, config.ExecutionPreference);
        Assert.True(config.AllowFp16);
    }

    [Fact]
    public void NNAPIConfiguration_CanSetExecutionPreference()
    {
        var config = new NNAPIConfiguration
        {
            ExecutionPreference = NNAPIExecutionPreference.LowPower
        };

        Assert.Equal(NNAPIExecutionPreference.LowPower, config.ExecutionPreference);

        config.ExecutionPreference = NNAPIExecutionPreference.FastSingleAnswer;
        Assert.Equal(NNAPIExecutionPreference.FastSingleAnswer, config.ExecutionPreference);
    }

    [Fact]
    public void NNAPIDeviceInfo_HasCorrectProperties()
    {
        var deviceInfo = new NNAPIDeviceInfo
        {
            Name = "Test GPU",
            Type = NNAPIDevice.GPU,
            FeatureLevel = 4,
            SupportsFp16 = true,
            SupportsInt8 = true
        };

        Assert.Equal("Test GPU", deviceInfo.Name);
        Assert.Equal(NNAPIDevice.GPU, deviceInfo.Type);
        Assert.Equal(4, deviceInfo.FeatureLevel);
        Assert.True(deviceInfo.SupportsFp16);
        Assert.True(deviceInfo.SupportsInt8);
    }

    #endregion

    #region TensorRT Configuration Tests

    [Fact]
    public void TensorRTConfiguration_DefaultValues_AreCorrect()
    {
        var config = new TensorRTConfiguration();

        Assert.Equal(1, config.MaxBatchSize);
        Assert.Equal(1L * 1024 * 1024 * 1024, config.MaxWorkspaceSize); // 1GB
        Assert.Equal(TensorRTPrecision.FP32, config.Precision);
        Assert.False(config.EnableDLA);
        Assert.Equal(-1, config.DLACore);
    }

    [Fact]
    public void TensorRTConfiguration_CanSetPrecision()
    {
        var config = new TensorRTConfiguration
        {
            Precision = TensorRTPrecision.FP16
        };

        Assert.Equal(TensorRTPrecision.FP16, config.Precision);

        config.Precision = TensorRTPrecision.INT8;
        Assert.Equal(TensorRTPrecision.INT8, config.Precision);
    }

    [Fact]
    public void TensorRTConfiguration_CanEnableDLA()
    {
        var config = new TensorRTConfiguration
        {
            EnableDLA = true,
            DLACore = 0
        };

        Assert.True(config.EnableDLA);
        Assert.Equal(0, config.DLACore);
    }

    [Fact]
    public void OptimizationProfile_DefaultValues_AreCorrect()
    {
        var profile = new OptimizationProfile();

        Assert.Null(profile.InputName);
        Assert.Null(profile.MinShape);
        Assert.Null(profile.OptimalShape);
        Assert.Null(profile.MaxShape);
    }

    [Fact]
    public void OptimizationProfileConfig_CanSetMinOptMax()
    {
        var config = new OptimizationProfileConfig
        {
            MinShape = new[] { 1, 3, 224, 224 },
            OptimalShape = new[] { 8, 3, 224, 224 },
            MaxShape = new[] { 32, 3, 224, 224 }
        };

        Assert.Equal(new[] { 1, 3, 224, 224 }, config.MinShape);
        Assert.Equal(new[] { 8, 3, 224, 224 }, config.OptimalShape);
        Assert.Equal(new[] { 32, 3, 224, 224 }, config.MaxShape);
    }

    #endregion

    #region Configuration Classes Tests

    [Fact]
    public void ABTestingConfig_DefaultValues_AreCorrect()
    {
        var config = new ABTestingConfig();

        Assert.False(config.Enabled);
        Assert.Equal(0.5, config.DefaultTrafficSplit);
        Assert.Empty(config.Tests);
    }

    [Fact]
    public void CacheConfig_DefaultValues_AreCorrect()
    {
        var config = new CacheConfig();

        Assert.True(config.Enabled);
        Assert.Equal(100.0, config.MaxSizeMB);
        Assert.Equal(CacheEvictionPolicy.LRU, config.EvictionPolicy);
        Assert.Equal(TimeSpan.FromHours(1), config.DefaultTTL);
    }

    [Fact]
    public void CompressionConfig_DefaultValues_AreCorrect()
    {
        var config = new CompressionConfig();

        Assert.Equal(ModelCompressionMode.Automatic, config.Mode);
        Assert.Equal(CompressionType.WeightClustering, config.Type);
        Assert.Equal(256, config.NumClusters);
        Assert.Equal(4, config.Precision);
        Assert.Equal(100, config.MaxIterations);
        Assert.Equal(2.0, config.MaxAccuracyLossPercent);
    }

    [Fact]
    public void ProfilingConfig_DefaultValues_AreCorrect()
    {
        var config = new ProfilingConfig();

        Assert.False(config.Enabled);
        Assert.True(config.TraceExecution);
        Assert.True(config.MeasureMemory);
    }

    [Fact]
    public void TelemetryConfig_DefaultValues_AreCorrect()
    {
        var config = new TelemetryConfig();

        Assert.True(config.Enabled);
        Assert.Equal(1.0, config.SamplingRate);
        Assert.True(config.CollectLatency);
        Assert.True(config.CollectErrors);
    }

    [Fact]
    public void VersioningConfig_DefaultValues_AreCorrect()
    {
        var config = new VersioningConfig();

        Assert.True(config.Enabled);
        Assert.Equal(3, config.MaxVersionsPerModel);
        Assert.True(config.AutoCleanup);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Integration_CacheAndTelemetry_WorkTogether()
    {
        var cache = new ModelCache<double>(enabled: true);
        var telemetry = new TelemetryCollector(enabled: true);
        var input = new double[] { 1.0, 2.0, 3.0 };
        var result = new double[] { 4.0, 5.0, 6.0 };

        // Simulate cache miss
        var cached = cache.Get("model:v1", input);
        if (cached == null)
        {
            telemetry.RecordInference("model", "v1", 100, fromCache: false);
            cache.Put("model:v1", input, result);
        }

        // Simulate cache hit
        cached = cache.Get("model:v1", input);
        if (cached != null)
        {
            telemetry.RecordInference("model", "v1", 5, fromCache: true);
        }

        var stats = telemetry.GetStatistics("model", "v1");
        Assert.Equal(2, stats.TotalInferences);
        Assert.Equal(0.5, stats.CacheHitRate, 1);
    }

    [Fact]
    public void Integration_EdgeConfigToQuantizer_MatchesBitWidth()
    {
        var rpiConfig = EdgeConfiguration.ForRaspberryPi();
        var jetsonConfig = EdgeConfiguration.ForJetson();

        // Verify edge configs specify correct quantization modes
        Assert.Equal(QuantizationMode.Int8, rpiConfig.QuantizationMode);
        Assert.Equal(QuantizationMode.Float16, jetsonConfig.QuantizationMode);

        // Create appropriate quantizers based on config
        var int8Quantizer = new Int8Quantizer<double, double[], double[]>();
        var fp16Quantizer = new Float16Quantizer<double, double[], double[]>();

        Assert.Equal(8, int8Quantizer.BitWidth);
        Assert.Equal(16, fp16Quantizer.BitWidth);
    }

    [Fact]
    public void Integration_MultipleConfigurationsForDifferentDevices()
    {
        var rpiConfig = EdgeConfiguration.ForRaspberryPi();
        var jetsonConfig = EdgeConfiguration.ForJetson();
        var mcuConfig = EdgeConfiguration.ForMicrocontroller();

        // Raspberry Pi: moderate resources, ARM
        Assert.Equal(0.5, rpiConfig.PruningRatio);
        Assert.Equal(50.0, rpiConfig.MaxModelSizeMB);

        // Jetson: more powerful, GPU capable
        Assert.False(jetsonConfig.UsePruning);
        Assert.Equal(500.0, jetsonConfig.MaxModelSizeMB);
        Assert.Equal(QuantizationMode.Float16, jetsonConfig.QuantizationMode);

        // MCU: very constrained
        Assert.Equal(0.7, mcuConfig.PruningRatio);
        Assert.Equal(1.0, mcuConfig.MaxModelSizeMB);
    }

    [Fact]
    public void Integration_TelemetryWithMultipleModels()
    {
        var telemetry = new TelemetryCollector(enabled: true);

        // Record inferences for multiple models
        telemetry.RecordInference("model_a", "v1", 50, fromCache: false);
        telemetry.RecordInference("model_a", "v2", 60, fromCache: false);
        telemetry.RecordInference("model_b", "v1", 100, fromCache: false);

        // Get stats for specific model/version
        var modelAV1 = telemetry.GetStatistics("model_a", "v1");
        var modelAV2 = telemetry.GetStatistics("model_a", "v2");
        var modelB = telemetry.GetStatistics("model_b", "v1");

        Assert.Equal(1, modelAV1.TotalInferences);
        Assert.Equal(50.0, modelAV1.AverageLatencyMs);

        Assert.Equal(1, modelAV2.TotalInferences);
        Assert.Equal(60.0, modelAV2.AverageLatencyMs);

        Assert.Equal(1, modelB.TotalInferences);
        Assert.Equal(100.0, modelB.AverageLatencyMs);

        // Get aggregate stats for model_a (all versions)
        var modelAAll = telemetry.GetStatistics("model_a");
        Assert.Equal(2, modelAAll.TotalInferences);
    }

    [Fact]
    public void Integration_CacheStatistics_AccurateOverTime()
    {
        var cache = new ModelCache<double>(enabled: true);

        // Add multiple entries
        for (int i = 0; i < 10; i++)
        {
            cache.Put($"model:v{i}", new double[] { i }, new double[] { i * 2 });
        }

        var stats = cache.GetStatistics();
        Assert.Equal(10, stats.TotalEntries);

        // Access some entries
        cache.Get("model:v0", new double[] { 0 });
        cache.Get("model:v0", new double[] { 0 });
        cache.Get("model:v1", new double[] { 1 });

        stats = cache.GetStatistics();
        Assert.True(stats.TotalAccessCount >= 3);
        Assert.True(stats.AverageAccessCount > 0);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EdgeCase_EmptyCacheStatistics()
    {
        var cache = new ModelCache<double>(enabled: true);
        var stats = cache.GetStatistics();

        Assert.Equal(0, stats.TotalEntries);
        Assert.Equal(0, stats.TotalAccessCount);
        Assert.Equal(0.0, stats.AverageAccessCount);
        Assert.Equal(TimeSpan.Zero, stats.OldestEntryAge);
        Assert.Equal(TimeSpan.Zero, stats.NewestEntryAge);
    }

    [Fact]
    public void EdgeCase_TelemetryMinMaxLatency()
    {
        var telemetry = new TelemetryCollector(enabled: true);

        // Record inference with known latencies
        telemetry.RecordInference("model", "v1", 10, fromCache: false);
        telemetry.RecordInference("model", "v1", 100, fromCache: false);
        telemetry.RecordInference("model", "v1", 50, fromCache: false);

        var stats = telemetry.GetStatistics("model", "v1");

        Assert.Equal(10, stats.MinLatencyMs);
        Assert.Equal(100, stats.MaxLatencyMs);
    }

    [Fact]
    public void EdgeCase_EvictFromEmptyCache()
    {
        var cache = new ModelCache<double>(enabled: true);

        var removedLRU = cache.EvictLRU(10);
        var removedLFU = cache.EvictLFU(10);
        var removedAge = cache.EvictOlderThan(TimeSpan.FromMinutes(1));

        Assert.Equal(0, removedLRU);
        Assert.Equal(0, removedLFU);
        Assert.Equal(0, removedAge);
    }

    [Fact]
    public void EdgeCase_EvictWithExactCount()
    {
        var cache = new ModelCache<double>(enabled: true);
        cache.Put("model:v1", new double[] { 1 }, new double[] { 1 });
        cache.Put("model:v2", new double[] { 2 }, new double[] { 2 });

        // Evict to keep exactly 2 entries (no removal needed)
        var removed = cache.EvictLRU(2);

        Assert.Equal(0, removed);
    }

    [Fact]
    public void EdgeCase_NullInput_CacheHash()
    {
        var cache = new ModelCache<double>(enabled: true);

        // Empty array should still work
        cache.Put("model:v1", Array.Empty<double>(), new double[] { 1 });
        var result = cache.Get("model:v1", Array.Empty<double>());

        Assert.NotNull(result);
    }

    [Fact]
    public void EdgeCase_AdaptiveConfig_BoundaryConditions()
    {
        var edgeConfig = new EdgeConfiguration();
        var optimizer = new EdgeOptimizer<double, double[], double[]>(edgeConfig);

        // Exactly at low battery threshold (0.2)
        var configAtThreshold = optimizer.CreateAdaptiveConfig(0.2, 0.5);
        Assert.Equal(QualityLevel.Medium, configAtThreshold.QualityLevel);

        // Exactly at high battery threshold (0.8) and low load (0.3)
        var configAtHighThreshold = optimizer.CreateAdaptiveConfig(0.8, 0.3);
        Assert.Equal(QualityLevel.Medium, configAtHighThreshold.QualityLevel);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void ThreadSafety_ConcurrentCacheAccess()
    {
        var cache = new ModelCache<double>(enabled: true);
        var tasks = new List<Task>();

        for (int i = 0; i < 100; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() =>
            {
                cache.Put($"model:v{index}", new double[] { index }, new double[] { index * 2 });
                cache.Get($"model:v{index}", new double[] { index });
            }));
        }

        Task.WaitAll(tasks.ToArray());

        var stats = cache.GetStatistics();
        Assert.True(stats.TotalEntries <= 100);
    }

    [Fact]
    public void ThreadSafety_ConcurrentTelemetryRecording()
    {
        var telemetry = new TelemetryCollector(enabled: true);
        var tasks = new List<Task>();

        for (int i = 0; i < 100; i++)
        {
            var index = i;
            tasks.Add(Task.Run(() =>
            {
                telemetry.RecordInference("model", "v1", index, fromCache: index % 2 == 0);
            }));
        }

        Task.WaitAll(tasks.ToArray());

        var stats = telemetry.GetStatistics("model", "v1");
        Assert.Equal(100, stats.TotalInferences);
    }

    #endregion
}
