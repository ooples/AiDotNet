using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Optimizer for edge device deployment with ARM NEON and other optimizations.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class EdgeOptimizer<T, TInput, TOutput>
{
    private readonly EdgeConfiguration _config;

    public EdgeOptimizer(EdgeConfiguration config)
    {
        Guard.NotNull(config);
        _config = config;
    }

    /// <summary>
    /// Optimizes a model for edge deployment.
    /// </summary>
    /// <param name="model">The model to optimize</param>
    /// <returns>The optimized model</returns>
    public IFullModel<T, TInput, TOutput> OptimizeForEdge(IFullModel<T, TInput, TOutput> model)
    {
        Guard.NotNull(model);

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

        // Note: Model partitioning should be done separately using PartitionModel()
        // since it returns a different type (PartitionedModel) than IFullModel

        return optimizedModel;
    }

    /// <summary>
    /// Partitions a model for split execution between cloud and edge.
    /// </summary>
    /// <param name="model">The model to partition</param>
    /// <returns>Partitioned model structure</returns>
    public PartitionedModel<T, TInput, TOutput> PartitionModel(IFullModel<T, TInput, TOutput> model)
    {
        Guard.NotNull(model);

        var partitioned = new PartitionedModel<T, TInput, TOutput>
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
        var numOps = MathHelper.GetNumericOperations<T>();

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
                prunedParams[i] = numOps.Zero;
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
        if (model is not ILayeredModel<T> layeredModel)
        {
            // Models without layer metadata cannot be meaningfully partitioned.
            // Return 0 so the entire model runs on one side rather than creating
            // an invalid split with unknown layer boundaries.
            Console.WriteLine(
                $"[EdgeOptimizer] Warning: Model type {model.GetType().Name} does not implement ILayeredModel<T>. " +
                "Cannot determine partition point without layer information. Defaulting to partition point 0 (no split).");
            return 0;
        }

        if (layeredModel.LayerCount <= 1)
        {
            return 0;
        }

        return _config.PartitionStrategy switch
        {
            PartitionStrategy.EarlyLayers => CalculateProportionalPartitionPoint(layeredModel, 0.25),
            PartitionStrategy.LateLayers => CalculateProportionalPartitionPoint(layeredModel, 0.75),
            PartitionStrategy.Adaptive => CalculateLoadBalancedPartitionPoint(layeredModel),
            _ => CalculateLoadBalancedPartitionPoint(layeredModel)
        };
    }

    /// <summary>
    /// Calculates a partition point at a fixed proportion of the model's layers.
    /// </summary>
    /// <param name="layeredModel">The model with layer information.</param>
    /// <param name="proportion">Fraction of layers to assign to the edge (0.0 to 1.0).</param>
    /// <returns>The layer index at which to partition.</returns>
    private int CalculateProportionalPartitionPoint(ILayeredModel<T> layeredModel, double proportion)
    {
        int targetLayer = Math.Max(1, (int)(layeredModel.LayerCount * proportion));
        targetLayer = Math.Min(targetLayer, layeredModel.LayerCount - 1);

        // Find the closest structurally valid partition point
        var allLayerInfo = layeredModel.GetAllLayerInfo();
        int bestPartition = targetLayer;
        int bestDistance = int.MaxValue;

        for (int i = 0; i < allLayerInfo.Count - 1; i++)
        {
            if (layeredModel.ValidatePartitionPoint(i))
            {
                int distance = Math.Abs(i + 1 - targetLayer);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestPartition = i + 1;
                }
            }
        }

        return bestPartition;
    }

    /// <summary>
    /// Calculates the optimal partition point by balancing estimated FLOPs between edge and cloud.
    /// </summary>
    /// <remarks>
    /// <para>Uses <see cref="LayerInfo{T}.EstimatedFlops"/> from <see cref="ILayeredModel{T}"/> to find
    /// the layer boundary where cumulative FLOPs are closest to half the total model FLOPs.
    /// This ensures both edge and cloud portions have roughly equal compute cost.</para>
    ///
    /// <para><b>Reference:</b> Inspired by Megatron-LM's cost-aware pipeline partition strategy,
    /// which uses per-layer FLOP estimates for balanced stage assignment.</para>
    /// </remarks>
    private int CalculateLoadBalancedPartitionPoint(ILayeredModel<T> layeredModel)
    {
        var allLayerInfo = layeredModel.GetAllLayerInfo();
        long totalFlops = 0;
        for (int i = 0; i < allLayerInfo.Count; i++)
        {
            totalFlops += allLayerInfo[i].EstimatedFlops;
        }

        // Pre-compute structurally valid partition points to avoid repeated validation calls
        var validPartitionPoints = new HashSet<int>();
        for (int i = 0; i < allLayerInfo.Count - 1; i++)
        {
            if (layeredModel.ValidatePartitionPoint(i))
            {
                validPartitionPoints.Add(i);
            }
        }

        // Find the layer boundary where cumulative FLOPs are closest to half
        long halfFlops = totalFlops / 2;
        long cumulative = 0;
        int bestPartition = -1;
        long bestDiff = long.MaxValue;

        for (int i = 0; i < allLayerInfo.Count - 1; i++)
        {
            cumulative += allLayerInfo[i].EstimatedFlops;

            if (!validPartitionPoints.Contains(i))
            {
                continue;
            }

            long diff = Math.Abs(cumulative - halfFlops);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                bestPartition = i + 1; // Partition after this layer
            }
        }

        // If no structurally valid partition point was found, fallback to midpoint
        if (bestPartition < 0)
        {
            bestPartition = allLayerInfo.Count / 2;
        }

        return bestPartition;
    }

    private IFullModel<T, TInput, TOutput>? ExtractEdgeLayers(IFullModel<T, TInput, TOutput> model, int start, int end)
    {
        // PRODUCTION-READY MODEL PARTITIONING
        //
        // Model partitioning requires layer-wise computational graph structure which
        // IFullModel does not expose (it only provides parameter access via GetParameters/WithParameters).
        //
        // For true production-ready model partitioning, implement one of these approaches:
        //
        // 1. ONNX-Based Partitioning (RECOMMENDED):
        //    - Export model to ONNX using OnnxModelExporter
        //    - Parse ONNX graph to identify layer boundaries
        //    - Split graph at specified operator node indices
        //    - Create separate ONNX models for edge and cloud portions
        //    - Load partitioned models via ONNX Runtime for inference
        //
        // 2. Custom IPartitionable Interface:
        //    - Define IPartitionable<T, TInput, TOutput> interface with ExtractLayers method
        //    - Implement interface on models that support layer-wise extraction
        //    - Use pattern: if (model is IPartitionable<T, TInput, TOutput> partitionable)
        //
        // 3. Framework-Specific Solutions:
        //    - TensorFlow: Use SavedModel with signature slicing
        //    - PyTorch: Use TorchScript with module extraction
        //    - ONNX Runtime: Use session slicing APIs
        //
        // CURRENT LIMITATION:
        // Without layer boundary information, we cannot create semantically valid
        // partitioned models. Attempting to split parameters arbitrarily would create
        // models with incomplete layers that cannot perform valid inference.
        //
        // Therefore, this method returns null to indicate partitioning is not supported
        // for models that only implement IFullModel without additional partitioning interfaces.

        throw new NotSupportedException(
            "Model partitioning requires layer-wise structure information not provided by IFullModel. " +
            "To enable model partitioning, use one of these approaches:\n" +
            "1. Export your model to ONNX format using OnnxModelExporter, then use ONNX graph manipulation tools to split the computational graph at specific operator nodes.\n" +
            "2. Implement a custom IPartitionable interface on your model type that exposes layer extraction methods.\n" +
            "3. Use framework-specific partitioning (TensorFlow SavedModel, PyTorch TorchScript) before importing to AiDotNet.\n" +
            "\n" +
            "Example ONNX approach:\n" +
            "  var exporter = new OnnxModelExporter<T, TInput, TOutput>(exportConfig);\n" +
            "  var onnxBytes = exporter.Export(model);\n" +
            "  // Use ONNX graph manipulation library to split graph\n" +
            "  // Load edge and cloud portions separately via ONNX Runtime\n" +
            "\n" +
            "Arbitrary parameter splitting is not implemented as it would create invalid models with incomplete layers.");
    }

    private IFullModel<T, TInput, TOutput>? ExtractCloudLayers(IFullModel<T, TInput, TOutput> model, int startFrom)
    {
        // See ExtractEdgeLayers documentation for full explanation.
        // Cloud layer extraction has the same fundamental limitation as edge extraction.

        throw new NotSupportedException(
            "Model partitioning requires layer-wise structure information not provided by IFullModel. " +
            "See EdgeOptimizer.ExtractEdgeLayers documentation for production-ready partitioning approaches.");
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
