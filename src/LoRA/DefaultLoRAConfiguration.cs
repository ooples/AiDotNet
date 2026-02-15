using System;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.LoRA;

/// <summary>
/// Default LoRA configuration that applies LoRA to all layers with trainable weight matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This configuration implements an intelligent strategy: wrap all layers that have trainable
/// weight matrices with StandardLoRAAdapter, and leave utility layers (activation, pooling, etc.)
/// unchanged. This maximizes the benefits of LoRA across all applicable layer types.
/// </para>
/// <para>
/// <b>Supported Layer Types (30+ layer types):</b>
/// - Dense/Linear layers (Dense, FullyConnected, FeedForward)
/// - Convolutional layers (all Conv variants including depthwise, separable, dilated, etc.)
/// - Recurrent layers (LSTM, GRU, ConvLSTM, Bidirectional)
/// - Attention layers (Attention, MultiHeadAttention, SelfAttention)
/// - Transformer layers (Encoder, Decoder)
/// - Embedding layers (Embedding, PatchEmbedding)
/// - Specialized layers (Highway, GatedLinearUnit, SqueezeAndExcitation, Capsule, CRF, etc.)
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
/// To use a specific variant, pass a factory function to the constructor.
/// Example: new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, adapterFactory: (layer, r, a, f) =&gt; new QLoRAAdapter&lt;double&gt;(layer, r, a, f))
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
    private readonly Type _adapterType;

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
    /// <param name="loraAdapter">Optional LoRA adapter to use. Defaults to StandardLoRAAdapter if null.</param>
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
    /// // Standard LoRA (default)
    /// var standard = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8);
    ///
    /// // QLoRA for 4-bit quantization (75% memory reduction)
    /// var qloraAdapter = new QLoRAAdapter&lt;double&gt;(null, 8, 8, true);
    /// var qlora = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, loraAdapter: qloraAdapter);
    ///
    /// // DoRA for improved weight decomposition (+3.7% accuracy on LLaMA-7B)
    /// var doraAdapter = new DoRAAdapter&lt;double&gt;(null, 8, 8, true);
    /// var dora = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, loraAdapter: doraAdapter);
    ///
    /// // VeRA for extreme parameter efficiency (10x fewer parameters)
    /// var veraAdapter = new VeRAAdapter&lt;double&gt;(null, 8, 8, true);
    /// var vera = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8, alpha: 8, loraAdapter: veraAdapter);
    /// ```
    /// </para>
    /// </remarks>
    public DefaultLoRAConfiguration(
        int rank,
        double alpha = -1,
        bool freezeBaseLayer = true,
        ILoRAAdapter<T>? loraAdapter = null)
    {
        if (rank <= 0)
        {
            throw new ArgumentException("Rank must be positive", nameof(rank));
        }

        Rank = rank;
        Alpha = alpha;
        FreezeBaseLayer = freezeBaseLayer;
        _adapterType = loraAdapter?.GetType() ?? typeof(StandardLoRAAdapter<T>);
    }

    /// <summary>
    /// Applies LoRA adaptation to layers with trainable weight matrices.
    /// </summary>
    /// <param name="layer">The layer to potentially adapt with LoRA.</param>
    /// <returns>
    /// A StandardLoRAAdapter wrapping the layer if it has trainable weights,
    /// otherwise returns the original layer unchanged.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method examines the layer type and wraps it with StandardLoRAAdapter if it's
    /// a layer type that benefits from LoRA adaptation (has trainable weight matrices).
    /// </para>
    /// <para><b>Supported Layer Types:</b>
    /// - <b>Dense/Linear:</b> DenseLayer, FullyConnectedLayer, FeedForwardLayer
    /// - <b>Convolutional:</b> ConvolutionalLayer, DeconvolutionalLayer, DepthwiseSeparableConvolutionalLayer,
    ///   DilatedConvolutionalLayer, SeparableConvolutionalLayer, SubpixelConvolutionalLayer
    /// - <b>Recurrent:</b> LSTMLayer, GRULayer, RecurrentLayer, ConvLSTMLayer, BidirectionalLayer
    /// - <b>Attention:</b> AttentionLayer, MultiHeadAttentionLayer, SelfAttentionLayer
    /// - <b>Transformer:</b> TransformerEncoderLayer, TransformerDecoderLayer
    /// - <b>Embedding:</b> EmbeddingLayer, PatchEmbeddingLayer
    /// - <b>Specialized:</b> LocallyConnectedLayer, HighwayLayer, GatedLinearUnitLayer, SqueezeAndExcitationLayer
    /// - <b>Advanced:</b> CapsuleLayer, PrimaryCapsuleLayer, DigitCapsuleLayer, ConditionalRandomFieldLayer
    ///
    /// <b>Excluded Layer Types:</b>
    /// - Activation, Pooling, Dropout, Flatten, Reshape, Normalization (no trainable weights)
    /// - GraphConvolutionalLayer (requires specialized adapter that implements IGraphConvolutionLayer)
    /// </para>
    /// <para><b>For Beginners:</b> This method decides whether to add LoRA to each layer.
    ///
    /// Decision logic:
    /// - If the layer has trainable weight matrices → Wrap it with StandardLoRAAdapter
    /// - If the layer is just doing math operations (activation, pooling, etc.) → Return unchanged
    ///
    /// This intelligent approach means:
    /// - LoRA is applied to all layers that can benefit from it
    /// - Works with Dense, Convolutional, Recurrent, Attention, and Transformer layers
    /// - Utility layers (pooling, dropout, etc.) pass through unchanged
    ///
    /// Example:
    /// ```csharp
    /// var config = new DefaultLoRAConfiguration&lt;double&gt;(rank: 8);
    ///
    /// // Dense layer gets adapted
    /// var denseLayer = new DenseLayer&lt;double&gt;(100, 50);
    /// var adapted1 = config.ApplyLoRA(denseLayer); // Returns StandardLoRAAdapter
    ///
    /// // Convolutional layer gets adapted
    /// var convLayer = new ConvolutionalLayer&lt;double&gt;(...);
    /// var adapted2 = config.ApplyLoRA(convLayer); // Returns StandardLoRAAdapter
    ///
    /// // Attention layer gets adapted
    /// var attnLayer = new MultiHeadAttentionLayer&lt;double&gt;(...);
    /// var adapted3 = config.ApplyLoRA(attnLayer); // Returns StandardLoRAAdapter
    ///
    /// // Pooling layer passes through (no weights to adapt)
    /// var poolLayer = new MaxPoolingLayer&lt;double&gt;(...);
    /// var unchanged = config.ApplyLoRA(poolLayer); // Returns original poolLayer
    /// ```
    /// </para>
    /// </remarks>
    public ILayer<T> ApplyLoRA(ILayer<T> layer)
    {
        if (layer == null)
        {
            throw new ArgumentNullException(nameof(layer));
        }

        // Check if this is a layer type that benefits from LoRA adaptation
        // (layers with trainable weight matrices)

        // Dense/Linear layers
        if (layer is DenseLayer<T> || layer is FullyConnectedLayer<T> || layer is FeedForwardLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Convolutional layers
        if (layer is ConvolutionalLayer<T> || layer is DeconvolutionalLayer<T> ||
            layer is DepthwiseSeparableConvolutionalLayer<T> || layer is DilatedConvolutionalLayer<T> ||
            layer is SeparableConvolutionalLayer<T> || layer is SubpixelConvolutionalLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Recurrent layers (LSTM, GRU, etc.)
        if (layer is LSTMLayer<T> || layer is GRULayer<T> || layer is RecurrentLayer<T> ||
            layer is ConvLSTMLayer<T> || layer is BidirectionalLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Attention layers
        if (layer is AttentionLayer<T> || layer is MultiHeadAttentionLayer<T> || layer is SelfAttentionLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Transformer layers
        if (layer is TransformerEncoderLayer<T> || layer is TransformerDecoderLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Embedding layers
        if (layer is EmbeddingLayer<T> || layer is PatchEmbeddingLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Specialized layers with trainable weights
        if (layer is LocallyConnectedLayer<T> || layer is HighwayLayer<T> ||
            layer is GatedLinearUnitLayer<T> || layer is SqueezeAndExcitationLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Graph convolutional layers - use specialized GraphConvolutionalLoRAAdapter
        // which implements IGraphConvolutionLayer<T> and properly delegates graph methods
        if (layer is IGraphConvolutionLayer<T>)
        {
            return new GraphConvolutionalLoRAAdapter<T>(layer, Rank, Alpha, FreezeBaseLayer);
        }

        // Capsule layers
        if (layer is CapsuleLayer<T> || layer is PrimaryCapsuleLayer<T> || layer is DigitCapsuleLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // CRF and other advanced layers
        if (layer is ConditionalRandomFieldLayer<T>)
        {
            return CreateAdapter(layer);
        }

        // Return layers without trainable weights unchanged
        // (Activation, Pooling, Dropout, Flatten, Reshape, Normalization, etc.)
        return layer;
    }

    /// <summary>
    /// Determines whether a layer should have LoRA applied based on its <see cref="LayerCategory"/>.
    /// </summary>
    /// <remarks>
    /// <para>This method provides a category-based alternative to the type-checking approach in
    /// <see cref="ApplyLoRA"/>. It uses the <see cref="LayerCategory"/> enum from
    /// <see cref="ILayeredModel{T}"/> metadata to make decisions, enabling LoRA injection
    /// without knowing the concrete layer type.</para>
    ///
    /// <para><b>Categories that receive LoRA adapters:</b></para>
    /// <list type="bullet">
    /// <item><description><see cref="LayerCategory.Dense"/>: Dense/linear layers (most common target)</description></item>
    /// <item><description><see cref="LayerCategory.Convolution"/>: All convolutional variants</description></item>
    /// <item><description><see cref="LayerCategory.Attention"/>: Attention and transformer layers</description></item>
    /// <item><description><see cref="LayerCategory.Recurrent"/>: LSTM, GRU, and other RNN layers</description></item>
    /// <item><description><see cref="LayerCategory.Embedding"/>: Token and patch embedding layers</description></item>
    /// <item><description><see cref="LayerCategory.FeedForward"/>: MLP blocks and mixture-of-experts</description></item>
    /// <item><description><see cref="LayerCategory.Graph"/>: Graph neural network layers</description></item>
    /// <item><description><see cref="LayerCategory.Residual"/>: Residual blocks with trainable weights</description></item>
    /// </list>
    ///
    /// <para><b>Categories that do NOT receive LoRA adapters:</b></para>
    /// <list type="bullet">
    /// <item><description><see cref="LayerCategory.Activation"/>: No trainable parameters</description></item>
    /// <item><description><see cref="LayerCategory.Pooling"/>: No trainable parameters</description></item>
    /// <item><description><see cref="LayerCategory.Normalization"/>: Few parameters, sensitive to modification</description></item>
    /// <item><description><see cref="LayerCategory.Regularization"/>: Dropout/noise layers</description></item>
    /// <item><description><see cref="LayerCategory.Structural"/>: Reshape/flatten/concat layers</description></item>
    /// <item><description><see cref="LayerCategory.Input"/>: Input layers</description></item>
    /// </list>
    /// </remarks>
    /// <param name="category">The layer category to check.</param>
    /// <returns>True if layers of this category should receive LoRA adapters.</returns>
    public static bool ShouldApplyLoRA(LayerCategory category)
    {
        return category switch
        {
            LayerCategory.Dense => true,
            LayerCategory.Convolution => true,
            LayerCategory.Attention => true,
            LayerCategory.Recurrent => true,
            LayerCategory.Embedding => true,
            LayerCategory.FeedForward => true,
            LayerCategory.Graph => true,
            LayerCategory.Residual => true,
            _ => false
        };
    }

    /// <summary>
    /// Applies LoRA adapters to all eligible layers in a layered model, returning
    /// the list of adapted layers.
    /// </summary>
    /// <remarks>
    /// <para>This method iterates through all layers in the model using <see cref="ILayeredModel{T}"/>
    /// metadata and applies LoRA adapters based on <see cref="LayerCategory"/>. Layers that don't
    /// benefit from LoRA (activation, pooling, normalization, etc.) are returned unchanged.</para>
    ///
    /// <para><b>For Beginners:</b> This method automatically identifies which layers in your
    /// neural network should be adapted with LoRA and wraps them. You don't need to manually
    /// specify which layers to adapt - the method uses the layer category metadata to decide.</para>
    /// </remarks>
    /// <param name="layeredModel">The model with layer-level access.</param>
    /// <returns>A list of layers where eligible layers have been wrapped with LoRA adapters.</returns>
    public IReadOnlyList<ILayer<T>> ApplyLoRAToModel(ILayeredModel<T> layeredModel)
    {
        var allLayerInfo = layeredModel.GetAllLayerInfo();
        var adaptedLayers = new List<ILayer<T>>(allLayerInfo.Count);

        foreach (var info in allLayerInfo)
        {
            if (ShouldApplyLoRA(info.Category) && info.IsTrainable)
            {
                adaptedLayers.Add(ApplyLoRA(info.Layer));
            }
            else
            {
                adaptedLayers.Add(info.Layer);
            }
        }

        return adaptedLayers;
    }

    /// <summary>
    /// Creates an instance of the configured LoRA adapter for the given layer.
    /// </summary>
    /// <param name="layer">The layer to wrap with a LoRA adapter.</param>
    /// <returns>A new LoRA adapter wrapping the given layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no compatible constructor is found.</exception>
    /// <remarks>
    /// <para>
    /// This method attempts to create an adapter using a constructor with signature
    /// (ILayer&lt;T&gt;, int, double, bool) for (layer, rank, alpha, freezeBaseLayer).
    /// </para>
    /// <para>
    /// If the adapter type does not have this exact constructor signature (e.g., QLoRAAdapter
    /// which has additional quantization parameters), this method will throw with a helpful
    /// error message explaining how to properly instantiate that adapter type.
    /// </para>
    /// </remarks>
    private ILayer<T> CreateAdapter(ILayer<T> layer)
    {
        // Try to find a constructor matching (ILayer<T>, int, double, bool)
        var constructors = _adapterType.GetConstructors();

        foreach (var ctor in constructors)
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length >= 4)
            {
                // Check first 4 parameters match expected types
                bool firstFourMatch =
                    typeof(ILayer<T>).IsAssignableFrom(parameters[0].ParameterType) &&
                    parameters[1].ParameterType == typeof(int) &&
                    parameters[2].ParameterType == typeof(double) &&
                    parameters[3].ParameterType == typeof(bool);

                if (firstFourMatch)
                {
                    // Check remaining parameters have defaults
                    bool allRemainingHaveDefaults = true;
                    for (int i = 4; i < parameters.Length; i++)
                    {
                        if (!parameters[i].HasDefaultValue)
                        {
                            allRemainingHaveDefaults = false;
                            break;
                        }
                    }

                    if (allRemainingHaveDefaults)
                    {
                        // Build parameter array with defaults for remaining parameters
                        object?[] args = new object?[parameters.Length];
                        args[0] = layer;
                        args[1] = Rank;
                        args[2] = Alpha;
                        args[3] = FreezeBaseLayer;

                        for (int i = 4; i < parameters.Length; i++)
                        {
                            args[i] = parameters[i].DefaultValue;
                        }

                        return (ILayer<T>)ctor.Invoke(args)!;
                    }
                }
            }
        }

        // If no compatible constructor found, provide helpful error message
        throw new InvalidOperationException(
            $"Adapter type '{_adapterType.Name}' does not have a compatible constructor signature. " +
            $"Expected signature: (ILayer<T> layer, int rank, double alpha, bool freezeBaseLayer, ...) " +
            $"where the 4th parameter is bool (freezeBaseLayer). " +
            $"Some adapters like QLoRAAdapter have different parameter orders and require explicit instantiation. " +
            $"Use DefaultLoRAConfiguration with StandardLoRAAdapter (default) or pass a pre-created adapter instance.");
    }
}
