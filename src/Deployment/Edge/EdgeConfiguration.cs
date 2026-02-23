using AiDotNet.Deployment.Export;

namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Configuration for edge device deployment optimization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> EdgeConfiguration provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class EdgeConfiguration
{
    /// <summary>
    /// Gets or sets whether to use quantization (default: true).
    /// </summary>
    public bool UseQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization mode for edge devices.
    /// </summary>
    public QuantizationMode QuantizationMode { get; set; } = QuantizationMode.Int8;

    /// <summary>
    /// Gets or sets whether to use pruning (default: true).
    /// </summary>
    public bool UsePruning { get; set; } = true;

    /// <summary>
    /// Gets or sets the pruning ratio (0.0 to 1.0, default: 0.3 = 30% sparsity).
    /// </summary>
    public double PruningRatio { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to enable layer fusion (default: true).
    /// </summary>
    public bool EnableLayerFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable ARM NEON optimization (default: true).
    /// </summary>
    public bool EnableArmNeonOptimization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable model partitioning for cloud+edge (default: false).
    /// </summary>
    public bool EnableModelPartitioning { get; set; } = false;

    /// <summary>
    /// Gets or sets the partition strategy.
    /// </summary>
    public PartitionStrategy PartitionStrategy { get; set; } = PartitionStrategy.Adaptive;

    /// <summary>
    /// Gets or sets the target device type.
    /// </summary>
    public EdgeDeviceType TargetDevice { get; set; } = EdgeDeviceType.Generic;

    /// <summary>
    /// Gets or sets the maximum model size in megabytes (default: 10 MB).
    /// </summary>
    public double MaxModelSizeMB { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the maximum memory usage in megabytes (default: 50 MB).
    /// </summary>
    public double MaxMemoryUsageMB { get; set; } = 50.0;

    /// <summary>
    /// Gets or sets the target inference latency in milliseconds (default: 100 ms).
    /// </summary>
    public double TargetLatencyMs { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets whether to optimize for power consumption (default: true).
    /// </summary>
    public bool OptimizeForPower { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable adaptive inference (default: false).
    /// </summary>
    public bool EnableAdaptiveInference { get; set; } = false;

    /// <summary>
    /// Gets or sets the cache size for intermediate results in MB (default: 5 MB).
    /// </summary>
    public double CacheSizeMB { get; set; } = 5.0;

    /// <summary>
    /// Creates a configuration for Raspberry Pi.
    /// </summary>
    public static EdgeConfiguration ForRaspberryPi()
    {
        return new EdgeConfiguration
        {
            TargetDevice = EdgeDeviceType.RaspberryPi,
            UseQuantization = true,
            QuantizationMode = QuantizationMode.Int8,
            UsePruning = true,
            PruningRatio = 0.5,
            EnableArmNeonOptimization = true,
            MaxModelSizeMB = 50.0,
            MaxMemoryUsageMB = 100.0,
            TargetLatencyMs = 200.0,
            OptimizeForPower = true
        };
    }

    /// <summary>
    /// Creates a configuration for NVIDIA Jetson devices.
    /// </summary>
    public static EdgeConfiguration ForJetson()
    {
        return new EdgeConfiguration
        {
            TargetDevice = EdgeDeviceType.Jetson,
            UseQuantization = true,
            QuantizationMode = QuantizationMode.Float16,
            UsePruning = false,
            EnableLayerFusion = true,
            EnableArmNeonOptimization = true,
            MaxModelSizeMB = 500.0,
            MaxMemoryUsageMB = 1000.0,
            TargetLatencyMs = 50.0,
            OptimizeForPower = false
        };
    }

    /// <summary>
    /// Creates a configuration for microcontrollers (MCU).
    /// </summary>
    public static EdgeConfiguration ForMicrocontroller()
    {
        return new EdgeConfiguration
        {
            TargetDevice = EdgeDeviceType.Microcontroller,
            UseQuantization = true,
            QuantizationMode = QuantizationMode.Int8,
            UsePruning = true,
            PruningRatio = 0.7,
            EnableLayerFusion = true,
            EnableArmNeonOptimization = false,
            MaxModelSizeMB = 1.0,
            MaxMemoryUsageMB = 2.0,
            TargetLatencyMs = 500.0,
            OptimizeForPower = true
        };
    }

    /// <summary>
    /// Creates a configuration for cloud+edge partitioned deployment.
    /// </summary>
    public static EdgeConfiguration ForCloudEdge()
    {
        return new EdgeConfiguration
        {
            TargetDevice = EdgeDeviceType.Generic,
            UseQuantization = true,
            QuantizationMode = QuantizationMode.Int8,
            UsePruning = false,
            EnableModelPartitioning = true,
            PartitionStrategy = PartitionStrategy.Adaptive,
            MaxModelSizeMB = 20.0,
            OptimizeForPower = true
        };
    }
}
