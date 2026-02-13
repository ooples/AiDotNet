using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A builder class that helps create and configure Mixture-of-Experts layers with sensible defaults.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This builder simplifies the creation of Mixture-of-Experts layers by providing convenient methods
/// with research-backed default values. It follows best practices from MoE literature to ensure
/// good initial configuration for most use cases.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a guided recipe for creating an MoE layer.
///
/// Instead of manually specifying every detail of your MoE layer (which experts to use,
/// how to route between them, whether to use load balancing, etc.), this builder provides
/// good default choices based on research and best practices.
///
/// It's like having a cooking recipe that says "preheat to 350°F" instead of making you
/// figure out the right temperature yourself. You can still customize if needed, but the
/// defaults work well for most cases.
/// </para>
/// </remarks>
public class MixtureOfExpertsBuilder<T>
{
    private int _numExperts = 4;
    private int _inputDim = 128;
    private int _outputDim = 128;
    private int _expertHiddenDim = 512;
    private int _topK = 0; // 0 means use all experts (soft routing)
    private bool _useLoadBalancing = true;
    private T _loadBalancingWeight;
    private IActivationFunction<T>? _expertActivation;
    private IActivationFunction<T>? _outputActivation;
    private bool _useIntermediateLayer = true;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="MixtureOfExpertsBuilder{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The builder is initialized with sensible default values based on research:
    /// - 4 experts (balance between capacity and computation)
    /// - Soft routing (all experts active, good for smaller models)
    /// - Load balancing enabled with weight 0.01 (prevents expert collapse)
    /// - ReLU activation for experts (standard, well-tested choice)
    /// - Identity activation for output (let downstream layers add non-linearity)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Creates a new MoE builder with smart default settings.
    ///
    /// The defaults are chosen to work well in most situations:
    /// - Not too many experts (4): Fast training and inference
    /// - Not too few experts (4): Enough specialization capacity
    /// - Load balancing: Ensures all experts get used
    /// - ReLU activation: The most popular choice, works well in practice
    ///
    /// These defaults are based on what researchers have found works best in practice.
    /// You can change any of these later if you have specific needs.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _loadBalancingWeight = _numOps.FromDouble(0.01); // Default: 1% weight
        _expertActivation = new ReLUActivation<T>();
        _outputActivation = new IdentityActivation<T>();
    }

    /// <summary>
    /// Sets the number of expert networks in the MoE layer.
    /// </summary>
    /// <param name="numExperts">The number of experts (must be at least 2).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Common values in research: 4-16 for small/medium models, 32-128 for large models.
    /// More experts = more capacity but also more computation and memory.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Sets how many specialist networks to create.
    ///
    /// Guidelines:
    /// - 2-4 experts: Good for small models or limited compute
    /// - 4-8 experts: Sweet spot for most applications
    /// - 8-16 experts: For larger, more complex tasks
    /// - 16+ experts: For very large scale models (use with TopK for efficiency)
    ///
    /// More experts allow more specialization, but:
    /// - Take longer to train
    /// - Use more memory
    /// - May need load balancing to prevent some being unused
    ///
    /// Start with 4-8 and adjust based on your results.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithExperts(int numExperts)
    {
        if (numExperts < 2)
        {
            throw new ArgumentException("Must have at least 2 experts.", nameof(numExperts));
        }
        _numExperts = numExperts;
        return this;
    }

    /// <summary>
    /// Sets the input and output dimensions for the MoE layer.
    /// </summary>
    /// <param name="inputDim">The input dimension.</param>
    /// <param name="outputDim">The output dimension.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// For transformer-style architectures, input and output dimensions are typically the same (residual connections).
    /// For bottleneck architectures, output might be smaller than input (dimensionality reduction).
    /// For expansion architectures, output might be larger than input (feature expansion).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Sets the size of data coming in and going out.
    ///
    /// Common patterns:
    /// - Same size (128→128): Maintains dimensionality, easy to stack multiple MoE layers
    /// - Bottleneck (512→128): Compresses information, reduces computation in later layers
    /// - Expansion (128→512): Expands features, increases representational capacity
    ///
    /// Most transformer-based models use the same input and output dimensions,
    /// which makes it easy to stack many MoE layers together.
    ///
    /// Example: If your previous layer outputs 256 features and your next layer expects
    /// 256 features, use WithDimensions(256, 256).
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithDimensions(int inputDim, int outputDim)
    {
        if (inputDim <= 0 || outputDim <= 0)
        {
            throw new ArgumentException("Dimensions must be positive.");
        }
        _inputDim = inputDim;
        _outputDim = outputDim;
        return this;
    }

    /// <summary>
    /// Sets the hidden dimension for the expert networks (for 2-layer experts).
    /// </summary>
    /// <param name="hiddenDim">The hidden dimension for expert networks.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Research often uses 4x the model dimension as hidden dimension in MoE layers.
    /// For example, if your model uses 128-dimensional embeddings, use hiddenDim=512.
    /// This is known as the "feed-forward expansion factor" in transformer literature.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Sets how large the "middle" of each expert is.
    ///
    /// Each expert is like a mini-network: Input → Hidden → Output
    /// The hidden layer is where the expert does its "thinking."
    ///
    /// Common practice: Make hidden dimension 4x the input dimension
    /// - Input 128 → Hidden 512 → Output 128
    /// - Input 256 → Hidden 1024 → Output 256
    ///
    /// Why 4x?
    /// - Gives experts enough capacity to learn complex patterns
    /// - Based on extensive research (used in BERT, GPT, etc.)
    /// - Good balance between capacity and efficiency
    ///
    /// You can go lower (2x) for smaller models or higher (8x) for more capacity.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithExpertHiddenDim(int hiddenDim)
    {
        if (hiddenDim <= 0)
        {
            throw new ArgumentException("Hidden dimension must be positive.", nameof(hiddenDim));
        }
        _expertHiddenDim = hiddenDim;
        return this;
    }

    /// <summary>
    /// Sets the hidden dimension expansion factor for expert networks.
    /// </summary>
    /// <param name="expansion">The expansion factor (hidden dim = input dim * expansion).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The actual expert hidden dimension will be calculated as InputDim * expansion.
    /// Common values: 2-4 for moderate capacity, 4-8 for high capacity.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithHiddenExpansion(int expansion)
    {
        if (expansion < 1)
        {
            throw new ArgumentException("Expansion factor must be at least 1.", nameof(expansion));
        }
        _expertHiddenDim = _inputDim * expansion;
        return this;
    }

    /// <summary>
    /// Configures Top-K sparse routing.
    /// </summary>
    /// <param name="k">The number of top experts to activate per input (0 = use all experts).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Top-K routing dramatically improves efficiency by activating only K experts per input.
    /// Common values: K=1 or K=2 for large models, K=0 (all experts) for smaller models.
    /// Research shows K=2 often provides the best accuracy/efficiency tradeoff.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Controls how many experts process each input.
    ///
    /// Options:
    /// - TopK = 0 (default): All experts process every input (soft routing)
    ///   * Pros: Maximum quality, all experts contribute
    ///   * Cons: Slower, uses more memory
    ///   * Best for: Small models (4-8 experts), when quality is critical
    ///
    /// - TopK = 1: Only the best expert for each input
    ///   * Pros: Very fast, minimal computation
    ///   * Cons: Less capacity, experts must specialize strongly
    ///   * Best for: Very large models (32+ experts), inference speed critical
    ///
    /// - TopK = 2 (recommended for large models): Top 2 experts per input
    ///   * Pros: Good balance of quality and speed
    ///   * Cons: Still more computation than TopK=1
    ///   * Best for: Medium to large models (8-32 experts)
    ///
    /// Example: With 8 experts and TopK=2, you use only 25% of the computation!
    ///
    /// Rule of thumb:
    /// - 4-8 experts: Use TopK=0 (all)
    /// - 8-16 experts: Use TopK=2
    /// - 16+ experts: Use TopK=1 or TopK=2
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithTopK(int k)
    {
        if (k < 0 || k > _numExperts)
        {
            throw new ArgumentException($"TopK must be between 0 and {_numExperts}.", nameof(k));
        }
        _topK = k;
        return this;
    }

    /// <summary>
    /// Configures load balancing to encourage even expert utilization.
    /// </summary>
    /// <param name="enabled">Whether to enable load balancing.</param>
    /// <param name="weight">The weight for the load balancing loss (typically 0.01-0.1).</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Load balancing prevents "expert collapse" where all inputs are routed to a small subset of experts.
    /// The default weight of 0.01 is based on the Switch Transformer paper and works well in most cases.
    /// Increase to 0.05-0.1 if you observe severe imbalance, decrease to 0.001-0.005 if it hurts accuracy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Ensures all experts get used roughly equally.
    ///
    /// The Problem:
    /// Without load balancing, the router might send all inputs to just 1-2 experts,
    /// leaving others unused. This wastes capacity and prevents specialization.
    ///
    /// The Solution:
    /// Load balancing adds a small penalty when experts are used unevenly,
    /// encouraging the router to spread inputs across all experts.
    ///
    /// Weight Guidelines:
    /// - 0.01 (default): Gentle encouragement, rarely hurts accuracy
    /// - 0.05: Moderate encouragement, use if you see significant imbalance
    /// - 0.1: Strong encouragement, may slightly reduce accuracy but ensures balance
    /// - 0.001: Very gentle, use if load balancing seems to hurt performance
    ///
    /// When to use:
    /// - Always use for training (enabled by default)
    /// - Disable for inference/testing (the builder does this automatically)
    ///
    /// Monitoring:
    /// Check GetAuxiliaryLossDiagnostics() during training to see if experts
    /// are balanced. Ideally, all experts should be used 10-30% of the time
    /// (with 4 experts, each should get ~25%).
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithLoadBalancing(bool enabled = true, double weight = 0.01)
    {
        _useLoadBalancing = enabled;
        _loadBalancingWeight = _numOps.FromDouble(weight);
        return this;
    }

    /// <summary>
    /// Sets the activation function for experts.
    /// </summary>
    /// <param name="activation">The activation function to use in expert networks.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Common choices: ReLU (default, fast and stable), GELU (used in transformers, smoother),
    /// Swish/SiLU (good performance, slightly more computation).
    /// ReLU is the safest default choice for most applications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Sets what mathematical function experts use for non-linearity.
    ///
    /// Popular choices:
    /// - ReLU (default): Fast, stable, works in most cases
    ///   * Use when: You want safe, reliable performance
    ///   * Used in: Most computer vision, many NLP models
    ///
    /// - GELU: Smoother than ReLU, used in modern transformers
    ///   * Use when: Building transformer-based models
    ///   * Used in: BERT, GPT, most modern language models
    ///
    /// - Swish/SiLU: Smooth and performs well
    ///   * Use when: You want slightly better performance
    ///   * Trade-off: A bit slower than ReLU
    ///
    /// - Tanh: Classic choice, outputs -1 to 1
    ///   * Use when: You need bounded outputs
    ///   * Used in: LSTMs, some older architectures
    ///
    /// If unsure, stick with the default (ReLU). It's the most tested and reliable.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithExpertActivation(IActivationFunction<T> activation)
    {
        Guard.NotNull(activation);
        _expertActivation = activation;
        return this;
    }

    /// <summary>
    /// Sets the activation function for the MoE layer output.
    /// </summary>
    /// <param name="activation">The activation function to apply after combining expert outputs.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// The default is Identity (no activation), which is appropriate when the MoE layer is used
    /// as a drop-in replacement for a feed-forward layer in architectures with residual connections.
    /// Use ReLU or other activations if you want non-linearity at this point.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Sets what happens to the combined output of all experts.
    ///
    /// Typical choices:
    /// - Identity (default): No change to the output
    ///   * Use when: MoE is part of a residual block (most transformer architectures)
    ///   * Reasoning: Downstream layers will add their own activations
    ///
    /// - ReLU: Applies non-linearity to the final output
    ///   * Use when: MoE is a standalone layer without residual connections
    ///   * Common in: Feed-forward networks, some CNN architectures
    ///
    /// In most modern architectures (like transformers), you want Identity here
    /// because the architecture already has non-linearity elsewhere.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithOutputActivation(IActivationFunction<T> activation)
    {
        Guard.NotNull(activation);
        _outputActivation = activation;
        return this;
    }

    /// <summary>
    /// Configures whether experts should use an intermediate (hidden) layer.
    /// </summary>
    /// <param name="useIntermediateLayer">True to use 2-layer experts, false for single-layer experts.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Two-layer experts (Input → Hidden → Output) provide more capacity and are standard in research.
    /// Single-layer experts (Input → Output) are faster and use less memory, suitable for simpler tasks.
    /// Default is true (two-layer) as this matches most research implementations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Controls how complex each expert network is.
    ///
    /// Two-layer experts (default: true):
    /// - Structure: Input → Hidden → Output
    /// - Pros: More capacity to learn complex patterns
    /// - Cons: Slower, uses more memory
    /// - Use when: You have a complex task or enough compute
    /// - Example: Input(128) → Hidden(512) → Output(128)
    ///
    /// Single-layer experts (false):
    /// - Structure: Input → Output (direct connection)
    /// - Pros: Faster, less memory, easier to train
    /// - Cons: Less capacity for complex patterns
    /// - Use when: Simpler task or limited compute
    /// - Example: Input(128) → Output(128)
    ///
    /// Rule of thumb:
    /// - Complex tasks (language, vision): Use two-layer (true)
    /// - Simple tasks (regression, small classification): Can use single-layer (false)
    /// - When unsure: Stick with default (true)
    /// </para>
    /// </remarks>
    public MixtureOfExpertsBuilder<T> WithIntermediateLayer(bool useIntermediateLayer)
    {
        _useIntermediateLayer = useIntermediateLayer;
        return this;
    }

    /// <summary>
    /// Builds the Mixture-of-Experts layer with the configured settings.
    /// </summary>
    /// <returns>A configured MixtureOfExpertsLayer instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates all the expert networks, the routing network, and assembles them into
    /// a complete MoE layer. It uses the configuration specified via the builder methods, falling
    /// back to sensible defaults for any unspecified settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Creates the actual MoE layer with all your settings.
    ///
    /// What happens when you call Build():
    /// 1. Creates the routing network (decides which experts to use)
    /// 2. Creates all the expert networks with your specified architecture
    /// 3. Connects everything together into one MoE layer
    /// 4. Initializes all parameters with good starting values
    ///
    /// After calling Build(), you get a complete, ready-to-use MoE layer that you can:
    /// - Add to your neural network architecture
    /// - Train with your data
    /// - Use for inference
    ///
    /// Example:
    /// <code>
    /// var moeLayer = new MixtureOfExpertsBuilder&lt;float&gt;()
    ///     .WithExperts(8)
    ///     .WithDimensions(256, 256)
    ///     .WithTopK(2)
    ///     .WithLoadBalancing(true, 0.01)
    ///     .Build();
    /// </code>
    ///
    /// This creates an MoE layer with 8 experts, where each input uses only the top 2 experts,
    /// and load balancing ensures all experts get used equally during training.
    /// </para>
    /// </remarks>
    public MixtureOfExpertsLayer<T> Build()
    {
        // Create experts
        var experts = new List<ILayer<T>>();
        for (int i = 0; i < _numExperts; i++)
        {
            experts.Add(CreateExpert());
        }

        // Create router (output dimension = number of experts)
        var router = new DenseLayer<T>(_inputDim, _numExperts, (IActivationFunction<T>?)new IdentityActivation<T>());

        // Create MoE layer
        var moeLayer = new MixtureOfExpertsLayer<T>(
            experts,
            router,
            new[] { _inputDim },
            new[] { _outputDim },
            _topK,
            _outputActivation,
            _useLoadBalancing,
            _loadBalancingWeight);

        return moeLayer;
    }

    /// <summary>
    /// Creates a single expert network based on the configured settings.
    /// </summary>
    /// <returns>An expert layer containing the configured architecture.</returns>
    private ExpertLayer<T> CreateExpert()
    {
        var layers = new List<ILayer<T>>();

        if (_useIntermediateLayer)
        {
            // Two-layer expert: Input → Hidden → Output
            layers.Add(new DenseLayer<T>(_inputDim, _expertHiddenDim, (IActivationFunction<T>?)_expertActivation));
            layers.Add(new DenseLayer<T>(_expertHiddenDim, _outputDim, (IActivationFunction<T>?)new IdentityActivation<T>()));
        }
        else
        {
            // Single-layer expert: Input → Output
            layers.Add(new DenseLayer<T>(_inputDim, _outputDim, (IActivationFunction<T>?)_expertActivation));
        }

        return new ExpertLayer<T>(layers, new[] { _inputDim }, new[] { _outputDim });
    }
}
