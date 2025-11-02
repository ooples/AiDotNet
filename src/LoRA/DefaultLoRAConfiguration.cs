using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.LoRA;

/// <summary>
/// Default LoRA configuration that applies LoRA to Dense and FullyConnected layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This configuration implements a simple strategy: wrap all DenseLayer and FullyConnectedLayer
/// instances with StandardLoRAAdapter (the generic LoRA implementation), and leave all other layer
/// types unchanged. This is the most common use case for LoRA in neural networks.
/// </para>
/// <para>
/// <b>Available LoRA Variants:</b> AiDotNet includes 32 cutting-edge LoRA variants for different use cases:
/// - StandardLoRAAdapter: Generic LoRA for all layer types
/// - QLoRAAdapter: 4-bit quantization for 75% memory reduction
/// - DoRAAdapter: Weight decomposition (+3.7% on LLaMA-7B)
/// - AdaLoRAAdapter: Adaptive rank allocation
/// - VeRAAdapter: Shared matrices (10x fewer parameters)
/// - LoRAPlusAdapter: Dual learning rates (2x faster convergence)
/// - LoHaAdapter: Hadamard products for CNNs
/// - LoKrAdapter: Kronecker products (57x compression)
/// - DyLoRAAdapter: Dynamic rank training
/// - RoSAAdapter: Robust to distribution shifts
/// - DVoRAAdapter: DoRA+VeRA hybrid
/// - LoRAFAAdapter: Frozen A matrix (50% reduction)
/// - DeltaLoRAAdapter: Delta-based updates with momentum
/// - LoRADropAdapter: Dropout regularization
/// - PiSSAAdapter: SVD initialization (NeurIPS 2024)
/// - GLoRAAdapter: Weight + activation adaptation
/// - LongLoRAAdapter: Context length extension
/// - MultiLoRAAdapter: Multi-task learning with routing
/// - XLoRAAdapter: Mixture of experts
/// - TiedLoRAAdapter: Weight tying (90% reduction)
/// - ReLoRAAdapter: Restart mechanism prevents forgetting
/// - LoftQAdapter: Alternating quantization+LoRA
/// - QALoRAAdapter: Quantization-aware training
/// - VBLoRAAdapter: Vector banks (2024)
/// - SLoRAAdapter: Scalable serving (1000+ adapters)
/// - MoRAAdapter: High-rank updates for knowledge tasks
/// - LoRAXSAdapter: Extreme efficiency (100x compression)
/// - FloraAdapter: Gradient compression view
/// - ChainLoRAAdapter: Sequential task chaining
/// - HRAAdapter: Hybrid low-rank + sparse
/// - LoRETTAAdapter: Tensor-train decomposition
/// - NOLAAdapter: Random basis (20x compression)
///
/// To use a specific variant, create a custom ILoRAConfiguration implementation or
/// directly instantiate the desired adapter type.
/// </para>
/// <para><b>For Beginners:</b> This is a ready-to-use LoRA configuration for most common scenarios.
///
/// When you apply this configuration to a model:
/// - All Dense layers get wrapped with LoRA adapters
/// - All FullyConnected layers get wrapped with LoRA adapters
/// - All other layers (convolutional, pooling, etc.) pass through unchanged
///
/// This is perfect for:
/// - Fine-tuning pre-trained models on new tasks
/// - Adapting large language models with limited resources
/// - Training multiple task-specific adapters for the same base model
///
/// Example usage:
/// ```csharp
/// // Create a configuration with rank=8, alpha=8, and frozen base layers
/// var loraConfig = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, freezeBaseLayer: true);
///
/// // Apply to all layers in your model
/// var adaptedLayers = model.Layers.Select(layer => loraConfig.ApplyLoRA(layer)).ToList();
/// ```
///
/// The configuration respects these parameters:
/// - Rank: Controls compression (fewer parameters = lower rank)
/// - Alpha: Controls adaptation strength (typically same as rank)
/// - FreezeBaseLayer: Whether to freeze original weights (true for efficiency)
/// </para>
/// </remarks>
public class DefaultLoRAConfiguration<T> : ILoRAConfiguration<T>
{
    /// <summary>
    /// Gets the rank of the low-rank decomposition to use for adapted layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines the number of parameters in the LoRA adaptation.
    /// Lower rank = fewer parameters = more efficient but less flexible.
    /// </para>
    /// <para>
    /// Common values:
    /// - 1-4: Minimal parameters, very efficient
    /// - 8: Good default balance
    /// - 16-32: More flexibility
    /// - 64+: Approaching full fine-tuning
    /// </para>
    /// </remarks>
    public int Rank { get; }

    /// <summary>
    /// Gets the scaling factor (alpha) for LoRA adaptations.
    /// </summary>
    /// <remarks>
    /// Alpha controls how strongly LoRA adaptations affect outputs.
    /// Common practice: alpha = rank (for scaling factor of 1.0)
    /// Set to -1 to use rank as alpha (automatic scaling).
    /// </remarks>
    public double Alpha { get; }

    /// <summary>
    /// Gets whether base layers should be frozen during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true (typical), only LoRA parameters are trained while base layer
    /// weights remain frozen. This dramatically reduces memory and compute requirements.
    /// </para>
    /// <para>
    /// When false, both base layer and LoRA parameters are trained. This uses more
    /// resources but may achieve better results in some scenarios.
    /// </para>
    /// </remarks>
    public bool FreezeBaseLayer { get; }

    /// <summary>
    /// Initializes a new DefaultLoRAConfiguration with the specified parameters.
    /// </summary>
    /// <param name="rank">The rank of the low-rank decomposition (must be positive).</param>
    /// <param name="alpha">The scaling factor for LoRA contributions (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze base layers during training (default: true).</param>
    /// <exception cref="ArgumentException">Thrown when rank is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a configuration that will be applied to your model's layers.
    ///
    /// Parameters explained:
    /// - rank: How many "compression channels" to use (8 is a good starting point)
    /// - alpha: How strong the LoRA effect is (use -1 to auto-set to rank value)
    /// - freezeBaseLayer: Whether to lock original weights (true = more efficient, recommended)
    ///
    /// Example configurations:
    /// ```csharp
    /// // Efficient configuration for limited resources
    /// var efficient = new DefaultLoRAConfiguration&lt;double&gt;(rank: 4, alpha: 4, freezeBaseLayer: true);
    ///
    /// // Balanced configuration (most common)
    /// var balanced = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, freezeBaseLayer: true);
    ///
    /// // Higher capacity configuration
    /// var highCapacity = new DefaultLoRAConfiguration&lt;double&gt;(rank: 16, alpha: 16, freezeBaseLayer: true);
    ///
    /// // Full fine-tuning with LoRA structure (not frozen)
    /// var fullFineTune = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, freezeBaseLayer: false);
    /// ```
    /// </para>
    /// </remarks>
    public DefaultLoRAConfiguration(int rank, double alpha = -1, bool freezeBaseLayer = true)
    {
        if (rank <= 0)
        {
            throw new ArgumentException("Rank must be positive", nameof(rank));
        }

        Rank = rank;
        Alpha = alpha;
        FreezeBaseLayer = freezeBaseLayer;
    }

    /// <summary>
    /// Applies LoRA adaptation to a layer if it's a Dense or FullyConnected layer.
    /// </summary>
    /// <param name="layer">The layer to potentially adapt with LoRA.</param>
    /// <returns>
    /// A StandardLoRAAdapter wrapping the layer if it's a DenseLayer or FullyConnectedLayer,
    /// otherwise returns the original layer unchanged.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method examines the layer type and wraps it with StandardLoRAAdapter if it's
    /// a Dense or FullyConnected layer. All other layer types pass through unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides whether to add LoRA to each layer.
    ///
    /// Decision logic:
    /// - If the layer is DenseLayer → Wrap it with StandardLoRAAdapter
    /// - If the layer is FullyConnectedLayer → Wrap it with StandardLoRAAdapter
    /// - If the layer is anything else → Return it unchanged
    ///
    /// This selective approach means:
    /// - You get parameter-efficient fine-tuning where it matters most (dense layers)
    /// - Other layers (like convolutions or activations) work normally
    /// - The model structure remains compatible with existing code
    ///
    /// Example:
    /// ```csharp
    /// var config = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8);
    ///
    /// // Dense layer gets adapted
    /// var denseLayer = new DenseLayer&lt;double&gt;(100, 50);
    /// var adaptedDense = config.ApplyLoRA(denseLayer); // Returns StandardLoRAAdapter
    ///
    /// // Convolutional layer passes through unchanged
    /// var convLayer = new Conv2DLayer&lt;double&gt;(...);
    /// var unchanged = config.ApplyLoRA(convLayer); // Returns original convLayer
    /// ```
    /// </para>
    /// </remarks>
    public ILayer<T> ApplyLoRA(ILayer<T> layer)
    {
        if (layer == null)
        {
            throw new ArgumentNullException(nameof(layer));
        }

        // Check if this is a Dense or FullyConnected layer
        if (layer is DenseLayer<T> || layer is FullyConnectedLayer<T>)
        {
            // Wrap with StandardLoRAAdapter (the generic LoRA implementation)
            return new StandardLoRAAdapter<T>(layer, Rank, Alpha, FreezeBaseLayer);
        }

        // Return other layer types unchanged
        return layer;
    }
}
