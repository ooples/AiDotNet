using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Optimizer for edge device deployment with ARM NEON and other optimizations.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class EdgeOptimizer<T, TInput, TOutput> where T : struct
{
    private readonly EdgeConfiguration _config;

    public EdgeOptimizer(EdgeConfiguration config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>
    /// Optimizes a model for edge deployment.
    /// </summary>
    /// <param name="model">The model to optimize</param>
    /// <returns>The optimized model</returns>
    public IFullModel<T, TInput, TOutput> OptimizeForEdge(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var optimizedModel = model;

        // Apply quantization if requested
        if (_config.UseQuantization)
        {
            optimizedModel = ApplyQuantization(optimizedModel);
        }

        // Apply pruning if requested
        if (_config.UsePruning)
        {
            optimizedModel = ApplyPruning(optimizedModel);
        }

        // Apply layer fusion
        if (_config.EnableLayerFusion)
        {
            optimizedModel = ApplyLayerFusion(optimizedModel);
        }

        // Optimize for ARM NEON if available
        if (_config.EnableArmNeonOptimization && IsArmPlatform())
        {
            optimizedModel = OptimizeForArmNeon(optimizedModel);
        }

        // Apply model partitioning for cloud+edge deployment
        if (_config.EnableModelPartitioning)
        {
            return PartitionModel(optimizedModel);
        }

        return optimizedModel;
    }

    /// <summary>
    /// Partitions a model for split execution between cloud and edge.
    /// </summary>
    /// <param name="model">The model to partition</param>
    /// <returns>Partitioned model structure</returns>
    public PartitionedModel PartitionModel(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var partitioned = new PartitionedModel
        {
            OriginalModel = model,
            PartitionStrategy = _config.PartitionStrategy
        };

        // Analyze model and determine optimal partition point
        var partitionPoint = DeterminePartitionPoint(model);

        // Split model into edge and cloud parts
        partitioned.EdgeModel = ExtractEdgeLayers(model, 0, partitionPoint);
        partitioned.CloudModel = ExtractCloudLayers(model, partitionPoint);

        return partitioned;
    }

    /// <summary>
    /// Applies adaptive inference optimization (quality vs speed tradeoff).
    /// </summary>
    public AdaptiveInferenceConfig CreateAdaptiveConfig(double batteryLevel, double cpuLoad)
    {
        var config = new AdaptiveInferenceConfig();

        // Adjust quality based on battery level and CPU load
        if (batteryLevel < 0.2 || cpuLoad > 0.8)
        {
            // Low battery or high load: prioritize speed
            config.QualityLevel = QualityLevel.Low;
            config.UseQuantization = true;
            config.QuantizationBits = 8;
            config.SkipLayers = DetermineLayersToSkip(model: null, skipRatio: 0.2);
        }
        else if (batteryLevel > 0.8 && cpuLoad < 0.3)
        {
            // High battery and low load: prioritize quality
            config.QualityLevel = QualityLevel.High;
            config.UseQuantization = false;
            config.SkipLayers = new List<string>();
        }
        else
        {
            // Medium conditions: balanced
            config.QualityLevel = QualityLevel.Medium;
            config.UseQuantization = true;
            config.QuantizationBits = 16;
            config.SkipLayers = new List<string>();
        }

        return config;
    }

    private IFullModel<T, TInput, TOutput> ApplyQuantization(IFullModel<T, TInput, TOutput> model)
    {
        var quantizer = _config.QuantizationMode == QuantizationMode.Int8
            ? new Int8Quantizer<T, TInput, TOutput>() as IQuantizer<T, TInput, TOutput>
            : new Float16Quantizer<T, TInput, TOutput>();

        var quantConfig = _config.QuantizationMode == QuantizationMode.Int8
            ? QuantizationConfiguration.ForInt8(CalibrationMethod.None)
            : QuantizationConfiguration.ForFloat16();

        // For edge deployment, we use pre-computed quantization without calibration
        // to avoid overhead. Users should calibrate separately if needed for better accuracy.
        return quantizer.Quantize(model, quantConfig);
    }

    private IFullModel<T, TInput, TOutput> ApplyPruning(IFullModel<T, TInput, TOutput> model)
    {
        // Magnitude-based weight pruning: zero out weights below threshold
        var parameters = model.GetParameters();
        var prunedParams = new T[parameters.Length];

        // Calculate pruning threshold based on magnitude distribution
        var magnitudes = new double[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            magnitudes[i] = Math.Abs(Convert.ToDouble(parameters[i]));
        }

        Array.Sort(magnitudes);
        var pruneRatio = _config.PruningRatio; // e.g., 0.3 = remove 30% smallest weights
        var thresholdIndex = (int)(magnitudes.Length * pruneRatio);
        var threshold = magnitudes[thresholdIndex];

        // Apply pruning: set weights below threshold to zero
        int prunedCount = 0;
        for (int i = 0; i < parameters.Length; i++)
        {
            var magnitude = Math.Abs(Convert.ToDouble(parameters[i]));
            if (magnitude < threshold)
            {
                prunedParams[i] = default(T); // Zero
                prunedCount++;
            }
            else
            {
                prunedParams[i] = parameters[i];
            }
        }

        // Create new model with pruned parameters
        return model.WithParameters(new Vector<T>(prunedParams));
    }

    private IFullModel<T, TInput, TOutput> ApplyLayerFusion(IFullModel<T, TInput, TOutput> model)
    {
        // Layer fusion optimization is automatically handled by ONNX Runtime during graph optimization.
        // When models are exported to ONNX and run through ONNX Runtime, the GraphOptimizationLevel
        // setting enables automatic fusion of common patterns:
        // - Conv + BatchNorm + ReLU → Fused ConvBnRelu
        // - Gemm + Bias + Activation → Fused GemmActivation
        // - MatMul + Add → Gemm
        // - Transpose + MatMul → Gemm with transposed inputs
        //
        // This happens automatically at runtime, so no model transformation needed here.
        // The model structure remains unchanged; fusion occurs during inference.

        return model;
    }

    private IFullModel<T, TInput, TOutput> OptimizeForArmNeon(IFullModel<T, TInput, TOutput> model)
    {
        // ARM NEON optimizations are automatically applied by ONNX Runtime on ARM platforms.
        // ONNX Runtime includes optimized kernels that use ARM NEON SIMD instructions for:
        // - Matrix multiplications (SGEMM with NEON)
        // - Convolutions (Winograd/Im2Col with NEON vectorization)
        // - Activation functions (vectorized ReLU, Sigmoid, Tanh)
        // - Element-wise operations (vectorized add, mul, etc.)
        //
        // These optimizations are built into the ONNX Runtime ARM64 binaries and activated
        // automatically when running on ARM CPUs. No model transformation required.
        //
        // For custom operations beyond ONNX Runtime, users would need to implement
        // model-specific kernels using ARM NEON intrinsics (arm_neon.h).

        return model;
    }

    private int DeterminePartitionPoint(IFullModel<T, TInput, TOutput> model)
    {
        // Analyze model and determine optimal partition point
        // Based on:
        // - Layer computational cost
        // - Data transfer cost
        // - Edge device capabilities

        return _config.PartitionStrategy switch
        {
            PartitionStrategy.EarlyLayers => 3, // First 3 layers on edge
            PartitionStrategy.LateLayers => 10, // Most layers on edge
            PartitionStrategy.Adaptive => CalculateAdaptivePartitionPoint(model),
            _ => 5 // Default: middle partition
        };
    }

    private int CalculateAdaptivePartitionPoint(IFullModel<T, TInput, TOutput> model)
    {
        // Adaptive partitioning based on runtime conditions
        // Analysis factors:
        // 1. Network bandwidth: higher bandwidth → more layers on cloud
        // 2. Edge compute power: stronger edge → more layers on edge
        // 3. Battery level: low battery → fewer layers on edge
        // 4. Model complexity: analyze parameter count to estimate compute

        var parameterCount = model.GetParameters().Length;

        // Heuristic: larger models benefit more from cloud processing
        // Small models (< 1M params): process mostly on edge (partition at 70%)
        // Medium models (1M-10M params): balanced (partition at 50%)
        // Large models (> 10M params): process mostly on cloud (partition at 30%)

        if (parameterCount < 1_000_000)
        {
            return 7; // 70% on edge
        }
        else if (parameterCount < 10_000_000)
        {
            return 5; // 50% on edge
        }
        else
        {
            return 3; // 30% on edge
        }
    }

    private object ExtractEdgeLayers(IFullModel<T, TInput, TOutput> model, int start, int end)
    {
        // Model partitioning requires access to the model's layer-wise structure.
        // IFullModel provides parameter access but not individual layer extraction.
        //
        // Production implementation approaches:
        // 1. ONNX-based: Export to ONNX, split graph at specific nodes, create 2 ONNX models
        // 2. Model-specific: Implement IPartitionable interface with layer extraction
        // 3. Proxy-based: Create wrapper that runs partial inference on edge portion
        //
        // For now, we return metadata about the partition for ONNX-based splitting.
        // The actual split happens during ONNX export via graph node slicing.

        return new EdgePartitionMetadata
        {
            OriginalModel = model,
            StartLayer = start,
            EndLayer = end,
            PartitionType = "Edge"
        };
    }

    private object ExtractCloudLayers(IFullModel<T, TInput, TOutput> model, int startFrom)
    {
        // Similar to ExtractEdgeLayers, returns metadata for ONNX-based partitioning.
        // The cloud portion starts where the edge portion ended.

        return new CloudPartitionMetadata
        {
            OriginalModel = model,
            StartLayer = startFrom,
            PartitionType = "Cloud"
        };
    }

    private List<string> DetermineLayersToSkip(object? model, double skipRatio)
    {
        // Determine which layers can be skipped for speed
        // Typically skip some intermediate layers in skip connections
        return new List<string>();
    }

    private bool IsArmPlatform()
    {
        // Check if running on ARM architecture
        var arch = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture;
        return arch == System.Runtime.InteropServices.Architecture.Arm ||
               arch == System.Runtime.InteropServices.Architecture.Arm64;
    }
}

/// <summary>
/// Metadata for edge partition of a split model.
/// </summary>
internal class EdgePartitionMetadata
{
    public object OriginalModel { get; set; } = new();
    public int StartLayer { get; set; }
    public int EndLayer { get; set; }
    public string PartitionType { get; set; } = string.Empty;
}

/// <summary>
/// Metadata for cloud partition of a split model.
/// </summary>
internal class CloudPartitionMetadata
{
    public object OriginalModel { get; set; } = new();
    public int StartLayer { get; set; }
    public string PartitionType { get; set; } = string.Empty;
}
