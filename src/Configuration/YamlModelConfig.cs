using AiDotNet.Deployment.Configuration;
using AiDotNet.Engines;
using AiDotNet.MixedPrecision;
using AiDotNet.Models.Options;
using AiDotNet.Reasoning.Models;
using AiDotNet.Training.Memory;

namespace AiDotNet.Configuration;

/// <summary>
/// Root POCO that YAML/JSON config files deserialize into.
/// Each section maps to either an enum-based factory selection or a direct config POCO.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class represents the structure of a YAML configuration file
/// that can be used to set up an AI model builder. Instead of writing C# code for every setting,
/// you can define your configuration in a YAML file and load it automatically.</para>
///
/// <para>Supported sections:</para>
/// <list type="bullet">
/// <item><description><b>Optimizer:</b> Select optimizer type (e.g., Adam, SGD)</description></item>
/// <item><description><b>TimeSeriesModel:</b> Select time series model type (e.g., ARIMA, SARIMA)</description></item>
/// <item><description><b>Deployment configs:</b> Quantization, compression, caching, versioning, etc.</description></item>
/// <item><description><b>Infrastructure configs:</b> JIT compilation, mixed precision, reasoning, benchmarking</description></item>
/// </list>
/// </remarks>
public class YamlModelConfig
{
    /// <summary>
    /// Optimizer selection section. Specify the optimizer type by name.
    /// </summary>
    public YamlOptimizerSection? Optimizer { get; set; }

    /// <summary>
    /// Time series model selection section. Specify the model type by name.
    /// </summary>
    public YamlTimeSeriesModelSection? TimeSeriesModel { get; set; }

    /// <summary>
    /// Model quantization configuration for compressing models with lower precision.
    /// </summary>
    public QuantizationConfig? Quantization { get; set; }

    /// <summary>
    /// Model compression configuration for reducing model size.
    /// </summary>
    public CompressionConfig? Compression { get; set; }

    /// <summary>
    /// Model caching configuration for storing loaded models in memory.
    /// </summary>
    public CacheConfig? Caching { get; set; }

    /// <summary>
    /// Model versioning configuration for managing multiple model versions.
    /// </summary>
    public VersioningConfig? Versioning { get; set; }

    /// <summary>
    /// A/B testing configuration for comparing model versions.
    /// </summary>
    public ABTestingConfig? AbTesting { get; set; }

    /// <summary>
    /// Telemetry configuration for tracking inference metrics.
    /// </summary>
    public TelemetryConfig? Telemetry { get; set; }

    /// <summary>
    /// Export configuration for exporting models to different formats.
    /// </summary>
    public ExportConfig? Export { get; set; }

    /// <summary>
    /// GPU acceleration configuration for hardware-accelerated training and inference.
    /// </summary>
    public GpuAccelerationConfig? GpuAcceleration { get; set; }

    /// <summary>
    /// Performance profiling configuration.
    /// </summary>
    public ProfilingConfig? Profiling { get; set; }

    /// <summary>
    /// JIT compilation configuration for accelerated inference.
    /// </summary>
    public JitCompilationConfig? JitCompilation { get; set; }

    /// <summary>
    /// Mixed precision training configuration.
    /// </summary>
    public MixedPrecisionConfig? MixedPrecision { get; set; }

    /// <summary>
    /// Reasoning strategy configuration.
    /// </summary>
    public ReasoningConfig? Reasoning { get; set; }

    /// <summary>
    /// Benchmarking configuration for running standardized benchmark suites.
    /// </summary>
    public BenchmarkingOptions? Benchmarking { get; set; }

    /// <summary>
    /// Inference optimization configuration for KV caching, batching, and speculative decoding.
    /// </summary>
    public InferenceOptimizationConfig? InferenceOptimizations { get; set; }

    /// <summary>
    /// Interpretability configuration for model explainability (SHAP, LIME, etc.).
    /// </summary>
    public InterpretabilityOptions? Interpretability { get; set; }

    /// <summary>
    /// Training memory management configuration (gradient checkpointing, activation pooling, model sharding).
    /// </summary>
    public TrainingMemoryConfig? MemoryManagement { get; set; }
}

/// <summary>
/// YAML section for selecting an optimizer type by name.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Use this to specify which optimizer to use in your YAML config.
/// The type should match one of the <see cref="Enums.OptimizerType"/> enum values (case-insensitive).</para>
/// <para><b>Example YAML:</b></para>
/// <code>
/// optimizer:
///   type: "Adam"
/// </code>
/// </remarks>
public class YamlOptimizerSection
{
    /// <summary>
    /// The optimizer type name. Must match an <see cref="Enums.OptimizerType"/> enum value (case-insensitive).
    /// </summary>
    public string Type { get; set; } = string.Empty;
}

/// <summary>
/// YAML section for selecting a time series model type by name.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Use this to specify which time series model to use in your YAML config.
/// The type should match one of the <see cref="Enums.TimeSeriesModelType"/> enum values (case-insensitive).</para>
/// <para><b>Example YAML:</b></para>
/// <code>
/// timeSeriesModel:
///   type: "ARIMA"
/// </code>
/// </remarks>
public class YamlTimeSeriesModelSection
{
    /// <summary>
    /// The time series model type name. Must match a <see cref="Enums.TimeSeriesModelType"/> enum value (case-insensitive).
    /// </summary>
    public string Type { get; set; } = string.Empty;
}
