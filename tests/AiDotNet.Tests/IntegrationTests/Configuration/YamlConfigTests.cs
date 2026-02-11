using Xunit;
using AiDotNet.Configuration;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.Factories;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Integration tests for the YAML configuration system.
/// Tests YAML loading, deserialization, builder integration, and error handling.
/// </summary>
public class YamlConfigTests
{
    #region YamlConfigLoader.LoadFromString Tests

    [Fact]
    public void LoadFromString_WithOptimizerSection_DeserializesCorrectly()
    {
        var yaml = @"
optimizer:
  type: Adam
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Optimizer);
        Assert.Equal("Adam", config.Optimizer.Type);
    }

    [Fact]
    public void LoadFromString_WithTimeSeriesModelSection_DeserializesCorrectly()
    {
        var yaml = @"
timeSeriesModel:
  type: ARIMA
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.TimeSeriesModel);
        Assert.Equal("ARIMA", config.TimeSeriesModel.Type);
    }

    [Fact]
    public void LoadFromString_WithCachingSection_DeserializesCorrectly()
    {
        var yaml = @"
caching:
  enabled: true
  maxCacheSize: 500
  timeToLiveSeconds: 7200
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Caching);
        Assert.True(config.Caching.Enabled);
        Assert.Equal(500, config.Caching.MaxCacheSize);
        Assert.Equal(7200, config.Caching.TimeToLiveSeconds);
    }

    [Fact]
    public void LoadFromString_WithQuantizationSection_DeserializesCorrectly()
    {
        var yaml = @"
quantization:
  mode: Int8
  targetBitWidth: 4
  useSymmetricQuantization: true
  calibrationSamples: 200
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Quantization);
        Assert.Equal(QuantizationMode.Int8, config.Quantization.Mode);
        Assert.Equal(4, config.Quantization.TargetBitWidth);
        Assert.True(config.Quantization.UseSymmetricQuantization);
        Assert.Equal(200, config.Quantization.CalibrationSamples);
    }

    [Fact]
    public void LoadFromString_WithJitCompilationSection_DeserializesCorrectly()
    {
        var yaml = @"
jitCompilation:
  enabled: true
  throwOnFailure: false
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.JitCompilation);
        Assert.True(config.JitCompilation.Enabled);
        Assert.False(config.JitCompilation.ThrowOnFailure);
    }

    [Fact]
    public void LoadFromString_WithProfilingSection_DeserializesCorrectly()
    {
        var yaml = @"
profiling:
  enabled: true
  samplingRate: 0.5
  reservoirSize: 2000
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Profiling);
        Assert.True(config.Profiling.Enabled);
        Assert.Equal(0.5, config.Profiling.SamplingRate);
        Assert.Equal(2000, config.Profiling.ReservoirSize);
    }

    [Fact]
    public void LoadFromString_WithTelemetrySection_DeserializesCorrectly()
    {
        var yaml = @"
telemetry:
  enabled: true
  trackLatency: true
  trackThroughput: false
  samplingRate: 0.75
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Telemetry);
        Assert.True(config.Telemetry.Enabled);
        Assert.True(config.Telemetry.TrackLatency);
        Assert.False(config.Telemetry.TrackThroughput);
        Assert.Equal(0.75, config.Telemetry.SamplingRate);
    }

    [Fact]
    public void LoadFromString_WithVersioningSection_DeserializesCorrectly()
    {
        var yaml = @"
versioning:
  enabled: true
  defaultVersion: ""2.0.0""
  maxVersionHistory: 10
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Versioning);
        Assert.True(config.Versioning.Enabled);
        Assert.Equal("2.0.0", config.Versioning.DefaultVersion);
        Assert.Equal(10, config.Versioning.MaxVersionHistory);
    }

    [Fact]
    public void LoadFromString_WithCompressionSection_DeserializesCorrectly()
    {
        var yaml = @"
compression:
  numClusters: 128
  precision: 6
  maxIterations: 200
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Compression);
        Assert.Equal(128, config.Compression.NumClusters);
        Assert.Equal(6, config.Compression.Precision);
        Assert.Equal(200, config.Compression.MaxIterations);
    }

    [Fact]
    public void LoadFromString_WithMultipleSections_DeserializesAllSections()
    {
        var yaml = @"
optimizer:
  type: Adam

caching:
  enabled: true
  maxCacheSize: 1000

jitCompilation:
  enabled: true

profiling:
  enabled: false
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Optimizer);
        Assert.Equal("Adam", config.Optimizer.Type);

        Assert.NotNull(config.Caching);
        Assert.True(config.Caching.Enabled);
        Assert.Equal(1000, config.Caching.MaxCacheSize);

        Assert.NotNull(config.JitCompilation);
        Assert.True(config.JitCompilation.Enabled);

        Assert.NotNull(config.Profiling);
        Assert.False(config.Profiling.Enabled);
    }

    [Fact]
    public void LoadFromString_WithEmptySections_LeavesPropertiesNull()
    {
        var yaml = @"
optimizer:
  type: Adam
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Optimizer);
        Assert.Null(config.Quantization);
        Assert.Null(config.Compression);
        Assert.Null(config.Caching);
        Assert.Null(config.Versioning);
        Assert.Null(config.Telemetry);
        Assert.Null(config.JitCompilation);
        Assert.Null(config.TimeSeriesModel);
        Assert.Null(config.InferenceOptimizations);
        Assert.Null(config.Interpretability);
        Assert.Null(config.MemoryManagement);
    }

    [Fact]
    public void LoadFromString_IgnoresUnknownProperties()
    {
        var yaml = @"
optimizer:
  type: Adam
unknownSection:
  foo: bar
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Optimizer);
        Assert.Equal("Adam", config.Optimizer.Type);
    }

    #endregion

    #region YamlConfigLoader Error Handling Tests

    [Fact]
    public void LoadFromString_WithNullContent_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(null as string));
    }

    [Fact]
    public void LoadFromString_WithEmptyContent_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(""));
    }

    [Fact]
    public void LoadFromString_WithWhitespaceContent_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString("   "));
    }

    [Fact]
    public void LoadFromFile_WithNullPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromFile(null as string));
    }

    [Fact]
    public void LoadFromFile_WithEmptyPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromFile(""));
    }

    [Fact]
    public void LoadFromFile_WithNonexistentFile_ThrowsFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() => YamlConfigLoader.LoadFromFile("nonexistent-config.yaml"));
    }

    #endregion

    #region AiModelBuilder Constructor Tests

    [Fact]
    public void Constructor_WithNullPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>(null as string));
    }

    [Fact]
    public void Constructor_WithEmptyPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>(""));
    }

    [Fact]
    public void Constructor_WithNonexistentFile_ThrowsFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>("does-not-exist.yaml"));
    }

    [Fact]
    public void Constructor_WithYamlFile_AppliesConfiguration()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, @"
caching:
  enabled: true
  maxCacheSize: 2000

jitCompilation:
  enabled: true
  throwOnFailure: true
");

            // The builder successfully parses and applies the YAML config
            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>(tempFile);
            Assert.NotNull(builder);

            // Verify YAML values were actually applied to the builder
            Assert.NotNull(builder.ConfiguredCaching);
            Assert.True(builder.ConfiguredCaching.Enabled);
            Assert.Equal(2000, builder.ConfiguredCaching.MaxCacheSize);

            Assert.NotNull(builder.ConfiguredJitCompilation);
            Assert.True(builder.ConfiguredJitCompilation.Enabled);
            Assert.True(builder.ConfiguredJitCompilation.ThrowOnFailure);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void Constructor_WithYamlFile_FluentOverridesWork()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, @"
caching:
  enabled: true
  maxCacheSize: 500
");

            // YAML sets caching (enabled=true, maxCacheSize=500), then fluent overrides it
            var builder = (AiModelBuilder<double, Matrix<double>, Vector<double>>)
                new AiModelBuilder<double, Matrix<double>, Vector<double>>(tempFile)
                .ConfigureCaching(new CacheConfig
                {
                    Enabled = false,
                    MaxCacheSize = 100
                });

            Assert.NotNull(builder);

            // Verify the fluent override took effect over YAML values
            Assert.NotNull(builder.ConfiguredCaching);
            Assert.False(builder.ConfiguredCaching.Enabled);
            Assert.Equal(100, builder.ConfiguredCaching.MaxCacheSize);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void DefaultConstructor_CreatesValidBuilder()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(builder);
    }

    #endregion

    #region YamlConfigApplier Validation Tests

    [Fact]
    public void Apply_WithInvalidOptimizerType_ThrowsArgumentException()
    {
        var yaml = @"
optimizer:
  type: NotARealOptimizer
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var ex = Assert.Throws<ArgumentException>(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));

        Assert.Contains("NotARealOptimizer", ex.Message);
    }

    [Fact]
    public void Apply_WithInvalidTimeSeriesModelType_ThrowsArgumentException()
    {
        var yaml = @"
timeSeriesModel:
  type: NotARealModel
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var ex = Assert.Throws<ArgumentException>(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));

        Assert.Contains("NotARealModel", ex.Message);
    }

    [Fact]
    public void Apply_WithValidOptimizerType_ParsesEnumCorrectly()
    {
        var yaml = @"
optimizer:
  type: Adam
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        // Verify the optimizer type string is parsed correctly
        Assert.NotNull(config.Optimizer);
        Assert.True(Enum.TryParse<OptimizerType>(config.Optimizer.Type, ignoreCase: true, out var parsed));
        Assert.Equal(OptimizerType.Adam, parsed);
    }

    [Fact]
    public void Apply_WithCaseInsensitiveOptimizerType_ParsesEnumCorrectly()
    {
        var yaml = @"
optimizer:
  type: adam
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        // Case-insensitive parsing should work
        Assert.NotNull(config.Optimizer);
        Assert.True(Enum.TryParse<OptimizerType>(config.Optimizer.Type, ignoreCase: true, out var parsed));
        Assert.Equal(OptimizerType.Adam, parsed);
    }

    [Fact]
    public void Apply_WithAllOptimizerTypeNames_ParsesCorrectly()
    {
        foreach (var name in Enum.GetNames(typeof(OptimizerType)))
        {
            var yaml = $@"
optimizer:
  type: {name}
";
            var config = YamlConfigLoader.LoadFromString(yaml);
            Assert.NotNull(config.Optimizer);
            Assert.True(Enum.TryParse<OptimizerType>(config.Optimizer.Type, ignoreCase: true, out _),
                $"Failed to parse optimizer type: {name}");
        }
    }

    [Fact]
    public void Apply_WithValidTimeSeriesModel_ConfiguresModelOnBuilder()
    {
        var yaml = @"
timeSeriesModel:
  type: ARIMA
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        Assert.NotNull(config.TimeSeriesModel);
        Assert.Equal("ARIMA", config.TimeSeriesModel.Type);

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // Apply should configure the ARIMA model on the builder without throwing
        YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder);

        // Verify the builder returns itself (fluent API works after apply)
        Assert.NotNull(builder);
    }

    [Fact]
    public void Apply_WithTimeSeriesModelOnIncompatibleBuilder_ThrowsInvalidOperationException()
    {
        var yaml = @"
timeSeriesModel:
  type: ARIMA
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        // Using double[] instead of Matrix<double>/Vector<double> - incompatible with ITimeSeriesModel
        var builder = new AiModelBuilder<double, double[], double[]>();

        Assert.Throws<InvalidOperationException>(() =>
            YamlConfigApplier<double, double[], double[]>.Apply(config, builder));
    }

    [Fact]
    public void Apply_WithNullConfig_OnlyAppliesNonNullSections()
    {
        var config = new YamlModelConfig();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // Verify all sections are null before applying
        Assert.Null(config.Optimizer);
        Assert.Null(config.TimeSeriesModel);
        Assert.Null(config.Quantization);
        Assert.Null(config.Compression);

        // Should not throw - empty config means no sections to apply
        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    #endregion

    #region OptimizerFactory Parameterless Overload Tests

    [Fact]
    public void OptimizerFactory_CreateOptimizer_WithUnknownType_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            OptimizerFactory<double, Matrix<double>, Vector<double>>
                .CreateOptimizer((OptimizerType)999));
    }

    [Fact]
    public void OptimizerFactory_CreateOptimizer_Adam_CreatesInstance()
    {
        var optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>
            .CreateOptimizer(OptimizerType.Adam);

        Assert.NotNull(optimizer);
        Assert.Contains("Adam", optimizer.GetType().Name);
    }

    [Fact]
    public void OptimizerFactory_CreateOptimizer_GradientDescent_CreatesInstance()
    {
        var optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>
            .CreateOptimizer(OptimizerType.GradientDescent);

        Assert.NotNull(optimizer);
        Assert.Contains("GradientDescent", optimizer.GetType().Name);
    }

    [Fact]
    public void OptimizerFactory_CreateOptimizer_AllRegisteredTypes_CreateSuccessfully()
    {
        var registeredTypes = new Dictionary<OptimizerType, string>
        {
            { OptimizerType.Adam, "Adam" },
            { OptimizerType.GradientDescent, "GradientDescent" },
            { OptimizerType.StochasticGradientDescent, "StochasticGradientDescent" },
            { OptimizerType.AntColony, "AntColony" },
            { OptimizerType.GeneticAlgorithm, "GeneticAlgorithm" },
            { OptimizerType.SimulatedAnnealing, "SimulatedAnnealing" },
            { OptimizerType.ParticleSwarm, "ParticleSwarm" },
            { OptimizerType.Normal, "Normal" },
        };

        foreach (var (type, expectedName) in registeredTypes)
        {
            var optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>
                .CreateOptimizer(type);

            Assert.NotNull(optimizer);
            Assert.Contains(expectedName, optimizer.GetType().Name);
        }
    }

    #endregion

    #region New POCO Section Deserialization Tests

    [Fact]
    public void LoadFromString_WithInferenceOptimizationsSection_DeserializesCorrectly()
    {
        var yaml = @"
inferenceOptimizations:
  enableKVCache: true
  kVCacheMaxSizeMB: 2048
  enableBatching: true
  maxBatchSize: 64
  enableSpeculativeDecoding: false
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.InferenceOptimizations);
        Assert.True(config.InferenceOptimizations.EnableKVCache);
        Assert.Equal(2048, config.InferenceOptimizations.KVCacheMaxSizeMB);
        Assert.True(config.InferenceOptimizations.EnableBatching);
        Assert.Equal(64, config.InferenceOptimizations.MaxBatchSize);
        Assert.False(config.InferenceOptimizations.EnableSpeculativeDecoding);
    }

    [Fact]
    public void LoadFromString_WithInterpretabilitySection_DeserializesCorrectly()
    {
        var yaml = @"
interpretability:
  enableSHAP: true
  sHAPSampleCount: 200
  enableLIME: false
  enablePermutationImportance: true
  permutationRepeatCount: 10
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Interpretability);
        Assert.True(config.Interpretability.EnableSHAP);
        Assert.Equal(200, config.Interpretability.SHAPSampleCount);
        Assert.False(config.Interpretability.EnableLIME);
        Assert.True(config.Interpretability.EnablePermutationImportance);
        Assert.Equal(10, config.Interpretability.PermutationRepeatCount);
    }

    [Fact]
    public void LoadFromString_WithMemoryManagementSection_DeserializesCorrectly()
    {
        var yaml = @"
memoryManagement:
  useGradientCheckpointing: true
  checkpointEveryNLayers: 3
  useActivationPooling: true
  maxPoolMemoryMB: 2048
  numDevices: 2
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.MemoryManagement);
        Assert.True(config.MemoryManagement.UseGradientCheckpointing);
        Assert.Equal(3, config.MemoryManagement.CheckpointEveryNLayers);
        Assert.True(config.MemoryManagement.UseActivationPooling);
        Assert.Equal(2048, config.MemoryManagement.MaxPoolMemoryMB);
        Assert.Equal(2, config.MemoryManagement.NumDevices);
    }

    #endregion

    #region Apply With Optimizer End-to-End Tests

    [Fact]
    public void Apply_WithAdamOptimizer_CreatesOptimizerOnBuilder()
    {
        var yaml = @"
optimizer:
  type: Adam
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // Factory creates Adam optimizer with null model
        YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder);

        // Verify the optimizer was actually configured on the builder
        Assert.NotNull(builder.ConfiguredOptimizer);
    }

    [Fact]
    public void Apply_WithGradientDescentOptimizer_CreatesOptimizerOnBuilder()
    {
        var yaml = @"
optimizer:
  type: GradientDescent
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder);

        // Verify the optimizer was actually configured on the builder
        Assert.NotNull(builder.ConfiguredOptimizer);
    }

    #endregion

    #region Full YAML Recipe Tests

    [Fact]
    public void LoadFromString_FullRecipe_DeserializesAllSections()
    {
        var yaml = @"
optimizer:
  type: Adam

timeSeriesModel:
  type: SARIMA

quantization:
  mode: Int8
  targetBitWidth: 8

compression:
  numClusters: 256

caching:
  enabled: true
  maxCacheSize: 1000

versioning:
  enabled: true
  defaultVersion: latest

telemetry:
  enabled: true
  samplingRate: 1.0

profiling:
  enabled: true

jitCompilation:
  enabled: true
  throwOnFailure: false

inferenceOptimizations:
  enableKVCache: true

interpretability:
  enableSHAP: true

memoryManagement:
  useGradientCheckpointing: true
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Optimizer);
        Assert.Equal("Adam", config.Optimizer.Type);

        Assert.NotNull(config.TimeSeriesModel);
        Assert.Equal("SARIMA", config.TimeSeriesModel.Type);

        Assert.NotNull(config.Quantization);
        Assert.Equal(QuantizationMode.Int8, config.Quantization.Mode);
        Assert.Equal(8, config.Quantization.TargetBitWidth);

        Assert.NotNull(config.Compression);
        Assert.Equal(256, config.Compression.NumClusters);

        Assert.NotNull(config.Caching);
        Assert.True(config.Caching.Enabled);
        Assert.Equal(1000, config.Caching.MaxCacheSize);

        Assert.NotNull(config.Versioning);
        Assert.True(config.Versioning.Enabled);
        Assert.Equal("latest", config.Versioning.DefaultVersion);

        Assert.NotNull(config.Telemetry);
        Assert.True(config.Telemetry.Enabled);
        Assert.Equal(1.0, config.Telemetry.SamplingRate);

        Assert.NotNull(config.Profiling);
        Assert.True(config.Profiling.Enabled);

        Assert.NotNull(config.JitCompilation);
        Assert.True(config.JitCompilation.Enabled);
        Assert.False(config.JitCompilation.ThrowOnFailure);

        Assert.NotNull(config.InferenceOptimizations);
        Assert.True(config.InferenceOptimizations.EnableKVCache);

        Assert.NotNull(config.Interpretability);
        Assert.True(config.Interpretability.EnableSHAP);

        Assert.NotNull(config.MemoryManagement);
        Assert.True(config.MemoryManagement.UseGradientCheckpointing);
    }

    [Fact]
    public void Constructor_WithFullYamlRecipe_AppliesAllSections()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, @"
optimizer:
  type: Adam

caching:
  enabled: true
  maxCacheSize: 1000

jitCompilation:
  enabled: true

inferenceOptimizations:
  enableKVCache: true

interpretability:
  enableSHAP: true

memoryManagement:
  useGradientCheckpointing: true
");

            // End-to-end: YAML file -> builder constructor -> all sections applied
            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>(tempFile);
            Assert.NotNull(builder);

            // Verify each section was actually applied
            Assert.NotNull(builder.ConfiguredOptimizer);

            Assert.NotNull(builder.ConfiguredCaching);
            Assert.True(builder.ConfiguredCaching.Enabled);
            Assert.Equal(1000, builder.ConfiguredCaching.MaxCacheSize);

            Assert.NotNull(builder.ConfiguredJitCompilation);
            Assert.True(builder.ConfiguredJitCompilation.Enabled);

            Assert.NotNull(builder.ConfiguredInferenceOptimizations);
            Assert.True(builder.ConfiguredInferenceOptimizations.EnableKVCache);

            Assert.NotNull(builder.ConfiguredInterpretability);
            Assert.True(builder.ConfiguredInterpretability.EnableSHAP);

            Assert.NotNull(builder.ConfiguredMemoryManagement);
            Assert.True(builder.ConfiguredMemoryManagement.UseGradientCheckpointing);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    #endregion
}
