using System.Buffers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a multi-head attention layer for neural networks, a key component in transformer architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Multi-head attention is like having multiple "experts" look at the same information
/// from different perspectives. Each "head" focuses on different parts of the input, allowing the model
/// to capture various relationships in the data simultaneously. This is similar to how you might ask
/// several friends for advice on a decision - each person might notice different important factors.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This layer is not thread-safe. Each layer instance maintains internal state
/// during forward and backward passes. If you need concurrent execution, use separate layer instances
/// per thread or synchronize access to shared instances.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "2, 4")]
public partial class MultiHeadAttentionLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (attention regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention regularization includes entropy regularization per head and head diversity penalties.
    /// This prevents attention collapse and encourages heads to learn different patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This helps ensure attention heads learn diverse patterns.
    ///
    /// Multi-head attention works best when each head specializes in different aspects:
    /// - Without regularization: Heads might learn redundant patterns
    /// - With regularization: Each head focuses on unique relationships
    ///
    /// Two types of regularization:
    /// 1. Entropy: Prevents attention from being too sharp (focused on one position)
    /// 2. Diversity: Prevents heads from being too similar to each other
    ///
    /// This helps the model:
    /// - Learn more robust representations
    /// - Utilize all attention heads effectively
    /// - Improve generalization to new data
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the attention entropy auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much attention entropy regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much we encourage diverse attention patterns.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced entropy regularization
    /// - 0.001-0.003: Light regularization
    /// - 0.008-0.01: Strong regularization
    ///
    /// Higher values encourage more distributed attention.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight for head diversity penalty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This encourages different heads to learn different patterns.
    ///
    /// Common values:
    /// - 0.01 (default): Moderate diversity encouragement
    /// - 0.005-0.008: Light diversity
    /// - 0.015-0.02: Strong diversity
    /// </para>
    /// </remarks>
    public T HeadDiversityWeight { get; set; }

    private T _lastEntropyLoss;
    private T _lastDiversityLoss;
    private List<Tensor<T>>? _lastHeadOutputs = null;

    // Positional encoding support
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    /// <summary>
    /// Gets or sets whether causal masking is applied during attention computation.
    /// When true, positions can only attend to earlier positions (autoregressive behavior).
    /// When false, attention is bidirectional (encoder-style).
    /// </summary>
    public bool UseCausalMask { get; set; }

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets the RoPE theta parameter if RoPE is configured, or the default 10000.0.
    /// </summary>
    public double RoPETheta => _ropeLayer?.Theta ?? 10000.0;

    // Cached projected Q, K, V for backward pass (4D: [batch, heads, seq, head_dim])
    private Tensor<T>? _lastProjectedQueries = null;
    private Tensor<T>? _lastProjectedKeys = null;
    private Tensor<T>? _lastProjectedValues = null;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput2D;
    private Tensor<T>? _gpuQ;
    private Tensor<T>? _gpuK;
    private Tensor<T>? _gpuV;
    private Tensor<T>? _gpuContextFlat;
    private Tensor<T>? _gpuAttentionWeights;
    private int _gpuBatchSize;
    private int _gpuSeqLength;
    private int _gpuEmbeddingDim;

    /// <summary>
    /// Tensor of weights for transforming input into query representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _queryWeights;

    /// <summary>
    /// Tensor of weights for transforming input into key representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _keyWeights;

    /// <summary>
    /// Tensor of weights for transforming input into value representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _valueWeights;

    /// <summary>
    /// Tensor of weights for the final output projection.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _outputWeights;

    /// <summary>
    /// Tensor of biases added to the final output.
    /// Shape: [embeddingDimension]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _outputBias;

    /// <summary>
    /// True once <see cref="EnsureInitialized"/> has allocated and populated the
    /// weight/bias tensors. Needed for the lazy-initialization path introduced to
    /// keep DiT-scale transformer stacks from eagerly allocating ~1 GB of Q/K/V/O
    /// projection tensors at construction time.
    /// </summary>
    private bool _isInitialized = true;

    /// <summary>
    /// Cached embedding dimension so <see cref="EnsureInitialized"/> knows what shape to
    /// allocate for the weight tensors when taking the lazy path. InputShape[1] carries
    /// the same value but reading the field is clearer and avoids array indexing in a
    /// hot path.
    /// </summary>
    private readonly int _embeddingDimension;

    /// <summary>
    /// Cached input from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    private Tensor<T>? _lastQueryInput;
    private Tensor<T>? _lastKeyInput;
    private Tensor<T>? _lastValueInput;

    /// <summary>
    /// Cached output from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached pre-activation output (before activation function) for computing
    /// activation derivative correctly. GELU and other activations need the
    /// pre-activation input to compute derivatives, not the post-activation output.
    /// </summary>
    private Tensor<T>? _lastPreActivationOutput;

    /// <summary>
    /// Cached attention context (pre-projection input) for computing output weights gradient.
    /// </summary>
    private Tensor<T>? _lastAttentionContext;

    /// <summary>
    /// Cached attention scores from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastAttentionScores;

    /// <summary>
    /// Tensor storing gradients for query weights calculated during backward pass.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T>? _queryWeightsGradient;

    /// <summary>
    /// Tensor storing gradients for key weights calculated during backward pass.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T>? _keyWeightsGradient;

    /// <summary>
    /// Tensor storing gradients for value weights calculated during backward pass.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T>? _valueWeightsGradient;

    /// <summary>
    /// Tensor storing gradients for output weights calculated during backward pass.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T>? _outputWeightsGradient;

    /// <summary>
    /// Tensor storing gradients for output bias calculated during backward pass.
    /// Shape: [embeddingDimension]
    /// </summary>
    private Tensor<T>? _outputBiasGradient;

    /// <summary>
    /// The number of attention heads in this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of this as the number of "experts" or different perspectives
    /// that will analyze the same input data.
    /// </remarks>
    private readonly int _headCount;

    /// <summary>
    /// The size of each attention head.
    /// </summary>
    private readonly int _headDimension;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Indicates whether this layer supports GPU-resident execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the number of attention heads in this layer.
    /// </summary>
    public int HeadCount => _headCount;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <remarks>
    /// Multi-head attention parameters are stored in multiple internal tensors (Q/K/V/O projections + output bias).
    /// </remarks>
    public override int ParameterCount => _isInitialized
        // After EnsureInitialized has run the live tensor lengths are authoritative.
        ? _queryWeights.Length + _keyWeights.Length + _valueWeights.Length + _outputWeights.Length + _outputBias.Length
        // Lazy path: four dim×dim projection matrices + one bias vector of size dim.
        // Reads the field without forcing the expensive tensor allocation, which is
        // the whole point of staying lazy for existence checks.
        : (4 * _embeddingDimension * _embeddingDimension) + _embeddingDimension;

    /// <summary>
    /// Gets the query projection weights tensor for JIT compilation.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the key projection weights tensor for JIT compilation.
    /// </summary>
    public Tensor<T> GetKeyWeights() => _keyWeights;

    /// <summary>
    /// Gets the value projection weights tensor for JIT compilation.
    /// </summary>
    public Tensor<T> GetValueWeights() => _valueWeights;

    /// <summary>
    /// Gets the output projection weights tensor for JIT compilation.
    /// </summary>
    public Tensor<T> GetOutputWeights() => _outputWeights;

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="activationFunction">The activation function to apply (defaults to identity function if null).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the attention mechanism with:
    /// - sequenceLength: How many items are in your sequence (like words in a sentence)
    /// - embeddingDimension: How much information is stored about each item
    /// - headCount: How many different "perspectives" or "experts" will analyze the data
    /// </para>
    /// </remarks>
    public MultiHeadAttentionLayer(
        int headCount,
        int headDimension,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, headCount * headDimension }, new[] { -1, headCount * headDimension },
               activationFunction ?? new IdentityActivation<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        HeadDiversityWeight = NumOps.FromDouble(0.01);
        _lastEntropyLoss = NumOps.Zero;
        _lastDiversityLoss = NumOps.Zero;

        if (headCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(headCount),
                $"headCount must be positive, got {headCount}.");
        if (headDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDimension),
                $"headDimension must be positive, got {headDimension}.");

        _headCount = headCount;
        _headDimension = headDimension;
        _embeddingDimension = headCount * headDimension;
        InitializationStrategy = initializationStrategy;

        // Always lazy now. Q/K/V/O projection weights are allocated on first forward
        // (or via EnsureInitialized when GetParameters is called before forward).
        _queryWeights = new Tensor<T>([0, 0]);
        _keyWeights = new Tensor<T>([0, 0]);
        _valueWeights = new Tensor<T>([0, 0]);
        _outputWeights = new Tensor<T>([0, 0]);
        _outputBias = new Tensor<T>([0]);
        _isInitialized = false;
    }

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply (required to disambiguate from IActivationFunction overload).</param>
    public MultiHeadAttentionLayer(
        int headCount,
        int headDimension,
        IVectorActivationFunction<T> vectorActivationFunction,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, headCount * headDimension }, new[] { -1, headCount * headDimension }, vectorActivationFunction)
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        HeadDiversityWeight = NumOps.FromDouble(0.01);
        _lastEntropyLoss = NumOps.Zero;
        _lastDiversityLoss = NumOps.Zero;

        if (headCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(headCount));
        if (headDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDimension));

        _headCount = headCount;
        _headDimension = headDimension;
        _embeddingDimension = headCount * headDimension;
        InitializationStrategy = initializationStrategy;

        _queryWeights = new Tensor<T>([0, 0]);
        _keyWeights = new Tensor<T>([0, 0]);
        _valueWeights = new Tensor<T>([0, 0]);
        _outputWeights = new Tensor<T>([0, 0]);
        _outputBias = new Tensor<T>([0]);
        _isInitialized = false;
    }

    /// <summary>
    /// Resolves shape on first forward; passthrough since output equals input shape.
    /// Validates that input.Shape[^1] == headCount * headDimension.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 2)
            throw new ArgumentException(
                $"MultiHeadAttentionLayer requires rank>=2 input; got rank {rank}.",
                nameof(input));

        int actualEmbed = input.Shape[rank - 1];
        if (actualEmbed != _embeddingDimension)
            throw new ArgumentException(
                $"MultiHeadAttentionLayer was constructed with embeddingDimension={_embeddingDimension} " +
                $"(headCount={_headCount} * headDimension={_headDimension}), but input.Shape[^1]={actualEmbed}.",
                nameof(input));

        var shape = input.Shape.ToArray();
        ResolveShapes(shape, shape);
    }

    /// <summary>
    /// Configures positional encoding for this attention layer.
    /// </summary>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <param name="ropeTheta">Base frequency for RoPE (default: 10000.0).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for pre-computation (default: 2048).</param>
    public void ConfigurePositionalEncoding(
        PositionalEncodingType encodingType,
        double ropeTheta = 10000.0,
        int maxSequenceLength = 2048)
    {
        PositionalEncoding = encodingType;
        _ropeLayer = null;
        _alibiLayer = null;

        switch (encodingType)
        {
            case PositionalEncodingType.Rotary:
                _ropeLayer = new RotaryPositionalEncodingLayer<T>(
                    maxSequenceLength, _headDimension, ropeTheta);
                break;
            case PositionalEncodingType.ALiBi:
                _alibiLayer = new ALiBiPositionalBiasLayer<T>(_headCount, maxSequenceLength);
                break;
            case PositionalEncodingType.None:
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported positional encoding type for MultiHeadAttentionLayer: {encodingType}.",
                    nameof(encodingType));
        }
    }

    /// <summary>
    /// Ensures the Q/K/V/O/bias tensors are allocated and populated. Cheap no-op
    /// after the first call. Takes <see cref="LayerBase{T}.InitializationLock"/> so
    /// concurrent forward/parameter queries from different threads observe a single
    /// initialization event.
    /// </summary>
    // Not an override of EnsureInitialized — the TrainableParameterGenerator
    // auto-emits an EnsureInitialized override on MHA because _ropeLayer /
    // _alibiLayer are sub-layers that need sub-layer registration. Our lazy-init
    // logic lives in this differently-named private helper and gets explicitly
    // called from Forward / GetParameters / SetParameters.
    private void EnsureWeightsAllocated()
    {
        if (_isInitialized) return;

        lock (InitializationLock)
        {
            if (_isInitialized) return;

            _queryWeights = new Tensor<T>([_embeddingDimension, _embeddingDimension]);
            _keyWeights = new Tensor<T>([_embeddingDimension, _embeddingDimension]);
            _valueWeights = new Tensor<T>([_embeddingDimension, _embeddingDimension]);
            _outputWeights = new Tensor<T>([_embeddingDimension, _embeddingDimension]);
            _outputBias = new Tensor<T>([_embeddingDimension]);

            if (InitializationStrategy is not null && !InitializationStrategy.IsLazy)
            {
                InitializationStrategy.InitializeWeights(_queryWeights, _embeddingDimension, _embeddingDimension);
                InitializationStrategy.InitializeWeights(_keyWeights, _embeddingDimension, _embeddingDimension);
                InitializationStrategy.InitializeWeights(_valueWeights, _embeddingDimension, _embeddingDimension);
                InitializationStrategy.InitializeWeights(_outputWeights, _embeddingDimension, _embeddingDimension);
                InitializationStrategy.InitializeBiases(_outputBias);
            }
            else
            {
                // Fall back to the existing Xavier + zero-bias init; also covers the
                // "lazy strategy" case where the strategy only advertises the deferral
                // contract but expects the layer's own parameter setup to run.
                InitializeParameters();
            }

            RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);

            _isInitialized = true;
        }
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier scale based on query weight shape
        int rows = _queryWeights.Shape[0];
        int cols = _queryWeights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (rows + cols)));

        // Use SimdRandom for vectorized initialization (4x faster than Vector.CreateRandom
        // which uses LockedRandom with per-element locking)
        var rng = new SimdRandom();
        FillTensorRandomScaled(_queryWeights, rng, scale);
        FillTensorRandomScaled(_keyWeights, rng, scale);
        FillTensorRandomScaled(_valueWeights, rng, scale);
        FillTensorRandomScaled(_outputWeights, rng, scale);

        // Initialize bias tensor to zeros
        _outputBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Fills a tensor with scaled random values in [-0.5, 0.5] * scale using SimdRandom.
    /// </summary>
    private void FillTensorRandomScaled(Tensor<T> tensor, SimdRandom rng, T scale)
    {
        var span = tensor.Data.Span;
        int total = span.Length;
        if (total == 0) return; // zero-sized tensor: nothing to fill
        double scaleD = NumOps.ToDouble(scale);

        // For double/float, write via a rented temp array + array-level reinterpret so
        // the SIMD-batched xoshiro256** path still applies. Span<T> can't be
        // reinterpreted across generic T (MemoryMarshal.Cast needs T:struct;
        // Unsafe.As<Span<T>,...> rejects the ref-struct; MemoryMarshal.CreateSpan is
        // missing on net471). Arrays are reference types so Unsafe.As<double[], T[]>
        // is unconstrained and safe when T == double/float at runtime.
        //
        // Use ArrayPool to avoid allocating a fresh full-sized buffer on every init
        // (attention weight tensors can be multi-million elements across many layers).
        if (typeof(T) == typeof(double))
        {
            var buffer = ArrayPool<double>.Shared.Rent(total);
            try
            {
                rng.NextDoubles(buffer.AsSpan(0, total));
                for (int i = 0; i < total; i++)
                    buffer[i] = (buffer[i] - 0.5) * scaleD;
                var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<double[], T[]>(ref buffer);
                reinterpreted.AsSpan(0, total).CopyTo(span);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(buffer);
            }
        }
        else if (typeof(T) == typeof(float))
        {
            var buffer = ArrayPool<float>.Shared.Rent(total);
            try
            {
                rng.NextFloats(buffer.AsSpan(0, total));
                float scaleF = (float)scaleD;
                for (int i = 0; i < total; i++)
                    buffer[i] = (buffer[i] - 0.5f) * scaleF;
                var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<float[], T[]>(ref buffer);
                reinterpreted.AsSpan(0, total).CopyTo(span);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
        else
        {
            const int batchSize = 4096;
            var tempBuf = ArrayPool<double>.Shared.Rent(Math.Min(total, batchSize));
            try
            {
                int offset = 0;
                while (offset < total)
                {
                    int chunk = Math.Min(batchSize, total - offset);
                    rng.NextDoubles(tempBuf.AsSpan(0, chunk));
                    for (int j = 0; j < chunk; j++)
                        span[offset + j] = NumOps.FromDouble((tempBuf[j] - 0.5) * scaleD);
                    offset += chunk;
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(tempBuf);
            }
        }
    }

    /// <summary>
    /// Returns layer-specific metadata required for cloning/serialization.
    /// </summary>
    /// <remarks>
    /// Multi-head attention requires the configured head count to reconstruct the layer correctly from shapes alone.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HeadCount"] = _headCount.ToString();
        metadata["PositionalEncoding"] = PositionalEncoding.ToString();
        return metadata;
    }

    /// <summary>
    /// Computes the auxiliary loss for attention regularization (entropy + head diversity).
    /// </summary>
    /// <returns>The computed attention regularization auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes two types of regularization:
    /// 1. Attention Entropy: Encourages attention to be distributed (not too peaked)
    /// 2. Head Diversity: Encourages different heads to learn different patterns
    /// Formula: L = entropy_weight * Σ_heads H(attention) + diversity_weight * Σ_pairs CosineSim(head_i, head_j)
    /// </para>
    /// <para><b>For Beginners:</b> This calculates penalties to improve attention quality.
    ///
    /// Attention regularization works by:
    /// 1. Measuring attention entropy for each head (prevents over-focusing)
    /// 2. Measuring similarity between different heads (prevents redundancy)
    /// 3. Combining these into a single auxiliary loss
    ///
    /// This helps because:
    /// - Prevents attention from collapsing to single positions
    /// - Ensures different heads specialize in different patterns
    /// - Improves model robustness and interpretability
    ///
    /// The auxiliary loss is minimized during training alongside the main task loss.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastAttentionScores == null)
        {
            _lastEntropyLoss = NumOps.Zero;
            _lastDiversityLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;

        // 1. Compute entropy regularization per head
        // H = -Σ(p * log(p)) for attention weights
        // We want to maximize entropy (minimize -H), so we minimize Σ(p * log(p))
        // Use GPU-accelerated tensor operations
        T epsilon = NumOps.FromDouble(1e-10);

        // Clamp values to avoid log(0)
        var clamped = Engine.TensorMax(_lastAttentionScores, epsilon);
        var logP = Engine.TensorLog(clamped);
        var pLogP = Engine.TensorMultiply(clamped, logP);

        // Sum over all elements (Batch, Head, Seq, Seq)
        T sumPLogP = Engine.TensorSum(pLogP);

        // Entropy = -sumPLogP. We want to maximize Entropy, so minimize -Entropy = sumPLogP.
        // Wait, original code minimized -H.
        // H = -sum(p log p). -H = sum(p log p).
        // So we minimize sum(p log p).

        // Actually, higher entropy = more uniform.
        // If we want to prevent collapse (too peaked), we want high entropy.
        // Loss = -Entropy = sum(p log p).
        // p log p is negative (since p < 1). Sum is negative.
        // Entropy is positive.
        // -Entropy is negative.
        // Minimizing a negative number -> making it more negative -> increasing magnitude of entropy -> increasing entropy.

        // Original code calculated totalNegativeEntropy = -H. And returned weighted loss.
        // So returning sumPLogP is correct.

        T totalNegativeEntropy = sumPLogP; // This is -Entropy

        _lastEntropyLoss = totalNegativeEntropy;
        totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(AuxiliaryLossWeight, totalNegativeEntropy));

        // 2. Compute head diversity penalty
        // Penalize high cosine similarity between head outputs
        if (_lastHeadOutputs != null && _lastHeadOutputs.Count > 0)
        {
            // Stacked heads tensor: [H, B, S, D]
            var headStack = _lastHeadOutputs[0];
            int flattenDim = headStack.Shape[1] * headStack.Shape[2] * headStack.Shape[3];

            // Flatten heads to [H, K]
            var headsFlat = headStack.Reshape([_headCount, flattenDim]);

            // Normalize each head vector
            var squared = Engine.TensorMultiply(headsFlat, headsFlat);
            var normSquared = Engine.ReduceSum(squared, new[] { 1 }, keepDims: true); // [H,1]
            var norm = Engine.TensorSqrt(normSquared);
            var normSafe = Engine.TensorMax(norm, NumOps.FromDouble(1e-12));
            var normalized = Engine.TensorDivide(headsFlat, normSafe); // broadcast divide

            // Cosine similarity matrix: [H, H]
            var normalizedT = Engine.TensorTranspose(normalized);
            var cosine = Engine.TensorMatMul(normalized, normalizedT);

            // Sum off-diagonal entries
            T sumAll = Engine.TensorSum(cosine);
            T sumDiag = NumOps.FromDouble(_headCount); // diag ~1 after normalization
            T offDiagSum = NumOps.Subtract(sumAll, sumDiag);
            T pairCount = NumOps.FromDouble(_headCount * (_headCount - 1)); // counts upper+lower

            var diversityPenalty = NumericalStabilityHelper.SafeDiv(offDiagSum, pairCount);

            _lastDiversityLoss = diversityPenalty;
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(HeadDiversityWeight, diversityPenalty));
        }

        return totalLoss;
    }

    /// <summary>
    /// Computes cosine similarity between two tensors.
    /// </summary>
    private T ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        // GPU-accelerated cosine similarity using tensor operations
        // Use Engine.TensorMultiply for element-wise multiplication and TensorSum for reduction
        var dotTensor = Engine.TensorMultiply(a, b);
        T dotProduct = Engine.TensorSum(dotTensor);

        // Compute norms using GPU-accelerated tensor operations
        var normATensor = Engine.TensorMultiply(a, a);
        var normBTensor = Engine.TensorMultiply(b, b);

        T normA = NumOps.Sqrt(Engine.TensorSum(normATensor));
        T normB = NumOps.Sqrt(Engine.TensorSum(normBTensor));

        T denominator = NumOps.Multiply(normA, normB);
        return NumericalStabilityHelper.SafeDiv(dotProduct, denominator);
    }

    /// <summary>
    /// Gets diagnostic information about the attention regularization auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about attention regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about attention regularization, including
    /// entropy loss, diversity loss, and configuration parameters.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how attention regularization is working.
    ///
    /// The diagnostics include:
    /// - Total entropy loss (how distributed attention patterns are)
    /// - Total diversity loss (how different heads are from each other)
    /// - Weights applied to each loss component
    /// - Whether regularization is enabled
    /// - Number of attention heads
    ///
    /// This helps you:
    /// - Monitor if attention is becoming too sharp or redundant
    /// - Debug issues with head specialization
    /// - Understand the impact of regularization on learning
    ///
    /// You can use this information to adjust regularization weights for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalEntropyLoss", System.Convert.ToString(_lastEntropyLoss) ?? "0" },
            { "TotalDiversityLoss", System.Convert.ToString(_lastDiversityLoss) ?? "0" },
            { "EntropyWeight", System.Convert.ToString(AuxiliaryLossWeight) ?? "0.005" },
            { "DiversityWeight", System.Convert.ToString(HeadDiversityWeight) ?? "0.01" },
            { "UseAttentionRegularization", UseAuxiliaryLoss.ToString() },
            { "NumberOfHeads", _headCount.ToString() },
            { "AttentionScoresCached", (_lastAttentionScores != null).ToString() },
            { "HeadOutputsCached", (_lastHeadOutputs != null).ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Declares named input ports for this multi-input layer.
    /// </summary>
    private IReadOnlyList<LayerPort>? _inputPortsCache;
    public override IReadOnlyList<LayerPort> InputPorts =>
        _inputPortsCache ??=
        [
            new LayerPort("query", GetInputShape()),
            new LayerPort("key", GetInputShape(), Required: false),
            new LayerPort("value", GetInputShape(), Required: false)
        ];

    /// <summary>
    /// Named multi-input forward pass.
    /// </summary>
    public override Tensor<T> Forward(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (!inputs.TryGetValue("query", out var query) || query == null)
            throw new ArgumentException("MultiHeadAttentionLayer requires a 'query' input.", nameof(inputs));
        inputs.TryGetValue("key", out var key);
        inputs.TryGetValue("value", out var value);
        // Default resolution: K defaults to V (if provided), then query.
        // V defaults to K (if provided), then query.
        // This keeps K/V aligned per standard transformer convention.
        var resolvedKey = key ?? value ?? query;
        var resolvedValue = value ?? key ?? query;
        return ForwardInternal(query, resolvedKey, resolvedValue);
    }

    /// <summary>
    /// Performs the forward pass of the multi-head attention layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after applying multi-head attention.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass is where the layer processes the input data. 
    /// Here's what happens:
    /// 1. The input is transformed into three different representations: queries, keys, and values
    /// 2. These are split into multiple "heads" (different perspectives)
    /// 3. Each head calculates how much attention to pay to different parts of the input
    /// 4. The results from all heads are combined to create the final output
    /// 
    /// Think of it like this: If you're reading a book, you might pay attention to different aspects
    /// like characters, plot, and setting all at once. Each "head" is like focusing on one of these aspects.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return ForwardInternal(input, input, input);
    }

    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 1) return ForwardInternal(inputs[0], inputs[0], inputs[0]);
        if (inputs.Length == 2) return ForwardInternal(inputs[0], inputs[1], inputs[1]); // Q, K=V (Cross Attention)
        if (inputs.Length == 3) return ForwardInternal(inputs[0], inputs[1], inputs[2]); // Q, K, V
        throw new ArgumentException("MultiHeadAttentionLayer supports 1, 2, or 3 inputs.");
    }

    private int[] _originalQueryShape = [];
    private int[] _originalKeyShape = [];
    private int[] _originalValueShape = [];

    private Tensor<T> ForwardInternal(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        // If the layer was constructed with a lazy init strategy, the Q/K/V/O
        // weight tensors are still at shape [0,0]. Materialize them now, before any
        // of the projection matmuls run — same guarantee DenseLayer gives. We also
        // still call the generated EnsureInitialized so sub-layer registration
        // happens (the ROPE/ALiBi fields this layer owns).
        EnsureWeightsAllocated();
        EnsureInitialized();

        // Industry standard: Support any-rank tensors (like PyTorch's MultiheadAttention)
        // Last two dimensions are [sequence, embedding_dim]
        // All preceding dimensions are treated as batch dimensions
        // Examples:
        //   2D [seq, dim] -> batch=1, seq, dim
        //   3D [batch, seq, dim] -> batch, seq, dim
        //   4D [batch1, batch2, seq, dim] -> batch1*batch2, seq, dim
        //   5D [b1, b2, b3, seq, dim] -> b1*b2*b3, seq, dim

        _originalQueryShape = query._shape;
        _originalKeyShape = key._shape;
        _originalValueShape = value._shape;

        // Handle 1D input by reshaping to 2D [1, dim]
        bool was1D = query.Rank == 1;
        if (was1D)
        {
            // Treat 1D [dim] as single token sequence [1, dim]
            query = Engine.Reshape(query, [1, query.Shape[0]]);
            key = Engine.Reshape(key, [1, key.Shape[0]]);
            value = Engine.Reshape(value, [1, value.Shape[0]]);
        }

        // Flatten all batch dimensions to get 3D [batch, seq, dim]
        int seqLenQ = query.Shape[^2];
        int dimQ = query.Shape[^1];
        int batchQ = 1;
        for (int i = 0; i < query.Rank - 2; i++)
            batchQ *= query.Shape[i];
        if (query.Rank == 2) batchQ = 1; // 2D case: [seq, dim] -> [1, seq, dim]

        int seqLenK = key.Shape[^2];
        int dimK = key.Shape[^1];
        int batchK = 1;
        for (int i = 0; i < key.Rank - 2; i++)
            batchK *= key.Shape[i];
        if (key.Rank == 2) batchK = 1;

        int seqLenV = value.Shape[^2];
        int dimV = value.Shape[^1];
        int batchV = 1;
        for (int i = 0; i < value.Rank - 2; i++)
            batchV *= value.Shape[i];
        if (value.Rank == 2) batchV = 1;

        // Reshape to 3D for processing
        query = Engine.Reshape(query, [batchQ, seqLenQ, dimQ]);
        key = Engine.Reshape(key, [batchK, seqLenK, dimK]);
        value = Engine.Reshape(value, [batchV, seqLenV, dimV]);

        _lastInput = query;
        _lastQueryInput = query;
        _lastKeyInput = key;
        _lastValueInput = value;

        int batchSize = query.Shape[0];
        int seqLengthQ = query.Shape[1];
        int embeddingDimension = query.Shape[2];
        int seqLengthKV = key.Shape[1];

        // 1. Project Input to Q, K, V
        // Validate that input embedding dimension matches weights
        int weightsEmbedDim = _queryWeights.Shape[0];
        if (embeddingDimension != weightsEmbedDim)
        {
            throw new ArgumentException(
                $"Input embedding dimension ({embeddingDimension}) does not match weight dimension ({weightsEmbedDim}). " +
                $"Query shape: [{string.Join(", ", query._shape)}], Weights shape: [{string.Join(", ", _queryWeights._shape)}]");
        }

        var q2D = Engine.Reshape(query, [batchSize * seqLengthQ, embeddingDimension]);
        var k2D = Engine.Reshape(key, [batchSize * seqLengthKV, embeddingDimension]);
        var v2D = Engine.Reshape(value, [batchSize * seqLengthKV, embeddingDimension]);

        var Q_flat = Engine.TensorMatMul(q2D, _queryWeights);
        var K_flat = Engine.TensorMatMul(k2D, _keyWeights);
        var V_flat = Engine.TensorMatMul(v2D, _valueWeights);

        // Reshape and Transpose to [Batch, HeadCount, Seq, HeadDim]
        int targetQElems = batchSize * seqLengthQ * _headCount * _headDimension;
        if (Q_flat.Length != targetQElems)
        {
            throw new ArgumentException(
                $"Q_flat reshape mismatch: Q_flat has {Q_flat.Length} elements, " +
                $"but target shape [{batchSize}, {seqLengthQ}, {_headCount}, {_headDimension}] needs {targetQElems}. " +
                $"Q_flat shape: [{string.Join(", ", Q_flat._shape)}], " +
                $"q2D shape: [{string.Join(", ", q2D._shape)}], " +
                $"_queryWeights shape: [{string.Join(", ", _queryWeights._shape)}]");
        }

        // Every shape op must go through Engine so the gradient tape records the
        // transformation — direct Tensor<T>.Transpose bypasses the tape and breaks
        // gradient flow through Q/K/V projections and back to the weight tensors.
        var queriesReshaped = Engine.Reshape(Q_flat, [batchSize, seqLengthQ, _headCount, _headDimension]);
        var keysReshaped = Engine.Reshape(K_flat, [batchSize, seqLengthKV, _headCount, _headDimension]);
        var valuesReshaped = Engine.Reshape(V_flat, [batchSize, seqLengthKV, _headCount, _headDimension]);
        var queries = Engine.TensorPermute(queriesReshaped, new[] { 0, 2, 1, 3 });
        var keys = Engine.TensorPermute(keysReshaped, new[] { 0, 2, 1, 3 });
        var values = Engine.TensorPermute(valuesReshaped, new[] { 0, 2, 1, 3 });

        // Apply RoPE to Q and K if configured
        if (_ropeLayer != null)
        {
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
        }

        // Cache projected Q, K, V for backward pass (4D: [batch, heads, seq, head_dim])
        _lastProjectedQueries = queries;
        _lastProjectedKeys = keys;
        _lastProjectedValues = values;

        // 2. Compute Scaled Dot-Product Attention
        // ScaledDotProductAttention computes: softmax(Q @ K^T / scale) @ V
        // Input shapes: [batch, heads, seq, head_dim]
        // Output shape: [batch, heads, seq_q, head_dim]

        // Compute Scaled Dot-Product Attention (with optional ALiBi bias)
        Tensor<T> context_4D;
        Tensor<T> attentionWeights4D;

        if (_alibiLayer != null)
        {
            // Use FlashAttention with ALiBi bias injection
            var aliBiBias = _alibiLayer.ComputeBias(seqLengthQ, seqLengthKV, useCausalMask: UseCausalMask);
            var flashConfig = FlashAttentionConfig.Default;
            flashConfig.ReturnAttentionWeights = true;
            var (flashOutput, flashWeights) = FlashAttention<T>.Forward(queries, keys, values, flashConfig, attentionBias: aliBiBias);
            context_4D = flashOutput;
            attentionWeights4D = flashWeights ?? new Tensor<T>(new[] { batchSize, _headCount, seqLengthQ, seqLengthKV });
        }
        else
        {
            context_4D = Engine.ScaledDotProductAttention(
                queries, keys, values,
                mask: null,
                scale: 1.0 / Math.Sqrt(_headDimension),
                out attentionWeights4D);
        }

        // Cache attention weights for backward pass
        _lastAttentionScores = attentionWeights4D;

        // 3. Cache Head Outputs — via Engine so the tape records the transpose.
        var permutedForCache = Engine.TensorPermute(context_4D, new[] { 1, 0, 2, 3 }); // [H, B, S, D]
        _lastHeadOutputs = new List<Tensor<T>> { permutedForCache }; // store stacked heads

        // 5. Concatenate and Project Output
        // [B, H, S, D] -> [B, S, H, D] -> [B, S, E] (Engine op keeps tape connected)
        var context_transposed = Engine.TensorPermute(context_4D, new[] { 0, 2, 1, 3 });
        var context_flat = Engine.Reshape(context_transposed, [batchSize * seqLengthQ, embeddingDimension]);

        // Cache pre-projection context for weight gradient computation in backward pass
        _lastAttentionContext = Engine.Reshape(context_transposed, [batchSize, seqLengthQ, embeddingDimension]);

        var output_flat = Engine.TensorMatMul(context_flat, _outputWeights);
        var output_reshaped = Engine.Reshape(output_flat, [batchSize, seqLengthQ, embeddingDimension]);

        var biasBroadcast = Engine.Reshape(_outputBias, [1, 1, embeddingDimension]);
        var outputWithBias = Engine.TensorBroadcastAdd(output_reshaped, biasBroadcast);
        var result = ApplyActivation(outputWithBias);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
            _lastPreActivationOutput = outputWithBias;
        }

        // Reshape output back to original batch dimensions
        // Output is currently [flatBatch, seq, dim], need to reshape to [origBatch..., seq, dim]
        if (_originalQueryShape.Length == 1)
        {
            // 1D input -> 1D output [dim]
            return Engine.Reshape(result, [embeddingDimension]);
        }

        int[] outputShape = new int[_originalQueryShape.Length];
        for (int i = 0; i < _originalQueryShape.Length - 2; i++)
        {
            outputShape[i] = _originalQueryShape[i];
        }
        outputShape[^2] = seqLengthQ;
        outputShape[^1] = embeddingDimension;

        return Engine.Reshape(result, outputShape);
    }


    /// <summary>
    /// GPU-resident forward pass for multi-head attention.
    /// Performs all projections and attention computation on GPU without downloading intermediate results.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        // Materialize lazy Q/K/V/O weights before GPU projection matmuls.
        // Without this, a fresh lazy-init MHA entering the GPU path would
        // use [0,0] placeholder tensors and produce wrong outputs.
        EnsureInitialized();

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Handle input shape - flatten to 3D [batch, seq, embedding]
        int[] inputShape = input._shape;
        int seqLength, embeddingDimension, batchSize;

        if (inputShape.Length == 2)
        {
            // 2D input: [seq, embedding] -> treat as batch=1
            batchSize = 1;
            seqLength = inputShape[0];
            embeddingDimension = inputShape[1];
        }
        else if (inputShape.Length >= 3)
        {
            // 3D+ input: flatten batch dimensions
            batchSize = 1;
            for (int i = 0; i < inputShape.Length - 2; i++)
                batchSize *= inputShape[i];
            seqLength = inputShape[^2];
            embeddingDimension = inputShape[^1];
        }
        else
        {
            throw new ArgumentException("Input must be at least 2D [seq, embedding]");
        }

        // 1. Reshape input to 3D for processing
        var input3D = gpuEngine.ReshapeGpu(input, new[] { batchSize, seqLength, embeddingDimension });

        // 2. Project to Q, K, V using batched matrix multiplication
        // Input: [batch, seq, embedding], Weights: [embedding, embedding]
        // Output: [batch, seq, embedding]
        var queries = gpuEngine.BatchedMatMulGpu(input3D, _queryWeights);
        var keys = gpuEngine.BatchedMatMulGpu(input3D, _keyWeights);
        var values = gpuEngine.BatchedMatMulGpu(input3D, _valueWeights);

        // 3. Reshape to [batch, seq, heads, headDim]
        var qReshaped = gpuEngine.ReshapeGpu(queries, new[] { batchSize, seqLength, _headCount, _headDimension });
        var kReshaped = gpuEngine.ReshapeGpu(keys, new[] { batchSize, seqLength, _headCount, _headDimension });
        var vReshaped = gpuEngine.ReshapeGpu(values, new[] { batchSize, seqLength, _headCount, _headDimension });

        // 4. Transpose to [batch, heads, seq, headDim] for attention
        var qPermuted = gpuEngine.PermuteGpu(qReshaped, new[] { 0, 2, 1, 3 });
        var kPermuted = gpuEngine.PermuteGpu(kReshaped, new[] { 0, 2, 1, 3 });
        var vPermuted = gpuEngine.PermuteGpu(vReshaped, new[] { 0, 2, 1, 3 });

        // 5. Compute scaled dot-product attention
        // Use overload that returns attention weights during training for backward pass
        double scale = 1.0 / Math.Sqrt(_headDimension);
        Tensor<T> attentionOutput;
        Tensor<T>? attentionWeightsGpu = null;

        if (IsTrainingMode)
        {
            // Training mode: get attention weights for backward pass
            attentionOutput = gpuEngine.ScaledDotProductAttentionGpu(
                qPermuted, kPermuted, vPermuted, scale, out attentionWeightsGpu);
        }
        else
        {
            // Inference mode: no need for attention weights
            attentionOutput = gpuEngine.ScaledDotProductAttentionGpu(qPermuted, kPermuted, vPermuted, scale);
        }

        // 6. Transpose back to [batch, seq, heads, headDim]
        var contextPermuted = gpuEngine.PermuteGpu(attentionOutput, new[] { 0, 2, 1, 3 });

        // 7. Reshape to [batch, seq, embedding]
        var contextFlat = gpuEngine.ReshapeGpu(contextPermuted, new[] { batchSize, seqLength, embeddingDimension });

        // 8. Apply output projection
        var outputProjected = gpuEngine.BatchedMatMulGpu(contextFlat, _outputWeights);

        // 9. Add output bias
        var outputWithBias = gpuEngine.AddBiasGpu(outputProjected, _outputBias);

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU-resident backward pass
            // Reshape input3D to 2D for backward pass weight gradients
            _gpuInput2D = gpuEngine.ReshapeGpu(input3D, new[] { batchSize * seqLength, embeddingDimension });
            _gpuQ = qPermuted;
            _gpuK = kPermuted;
            _gpuV = vPermuted;
            _gpuContextFlat = contextFlat;
            _gpuAttentionWeights = attentionWeightsGpu;
            _gpuBatchSize = batchSize;
            _gpuSeqLength = seqLength;
            _gpuEmbeddingDim = embeddingDimension;

            // Also cache CPU tensors for fallback backward pass
            _lastInput = input3D;

            // Cache projected Q, K, V for backward pass
            _lastProjectedQueries = qPermuted;
            _lastProjectedKeys = kPermuted;
            _lastProjectedValues = vPermuted;

            // Cache attention context for output weights gradient
            _lastAttentionContext = contextFlat;

            // Cache attention weights for backward pass
            _lastAttentionScores = attentionWeightsGpu;

            _lastOutput = outputWithBias;
        }

        // 10. Reshape back to original batch dimensions if needed
        if (inputShape.Length != 3 || inputShape[0] != batchSize)
        {
            int[] outputShape = new int[inputShape.Length];
            for (int i = 0; i < inputShape.Length - 2; i++)
                outputShape[i] = inputShape[i];
            outputShape[^2] = seqLength;
            outputShape[^1] = embeddingDimension;
            return gpuEngine.ReshapeGpu(outputWithBias, outputShape);
        }

        return outputWithBias;
    }


    private Tensor<T>? _queryWeightsVelocity;
    private Tensor<T>? _keyWeightsVelocity;
    private Tensor<T>? _valueWeightsVelocity;
    private Tensor<T>? _outputWeightsVelocity;
    private Tensor<T>? _outputBiasVelocity;

    /// <summary>
    /// Updates the layer's parameters (weights and biases) using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much to adjust the parameters.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is like adjusting a recipe based on feedback. The learning rate 
    /// is how bold we are with our changes - a higher rate means bigger adjustments, while a lower 
    /// rate means more cautious, smaller adjustments. The gradients tell us which direction to adjust 
    /// each parameter to improve the network's performance.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null || _outputWeightsGradient == null || _outputBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_queryWeightsVelocity == null)
            {
                _queryWeightsVelocity = new Tensor<T>(_queryWeights._shape);
                _queryWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_queryWeightsVelocity, PersistentTensorRole.OptimizerState);

                _keyWeightsVelocity = new Tensor<T>(_keyWeights._shape);
                _keyWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_keyWeightsVelocity, PersistentTensorRole.OptimizerState);

                _valueWeightsVelocity = new Tensor<T>(_valueWeights._shape);
                _valueWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_valueWeightsVelocity, PersistentTensorRole.OptimizerState);

                _outputWeightsVelocity = new Tensor<T>(_outputWeights._shape);
                _outputWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_outputWeightsVelocity, PersistentTensorRole.OptimizerState);

                _outputBiasVelocity = new Tensor<T>(_outputBias._shape);
                _outputBiasVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_outputBiasVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_queryWeights, _queryWeightsGradient, _queryWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_keyWeights, _keyWeightsGradient, _keyWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_valueWeights, _valueWeightsGradient, _valueWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_outputWeights, _outputWeightsGradient, _outputWeightsVelocity!, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_outputBias, _outputBiasGradient, _outputBiasVelocity!, lr, 0.0f, 0.0f);
        }
        else
        {
            // Update weights using tensor operations (production-ready pattern - no conversions)
            _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
            _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
            _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
            _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
            _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));

            // Notify GPU that tensor data has changed
            Engine.InvalidatePersistentTensor(_queryWeights);
            Engine.InvalidatePersistentTensor(_keyWeights);
            Engine.InvalidatePersistentTensor(_valueWeights);
            Engine.InvalidatePersistentTensor(_outputWeights);
            Engine.InvalidatePersistentTensor(_outputBias);
        }
    }

    /// <summary>
    /// Extracts all parameters (weights and biases) from the layer into a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method collects all the layer's adjustable values (weights and biases) 
    /// into a single list. Think of it like taking inventory of all the ingredients in a recipe.
    /// This is useful for saving the model's state or for optimization algorithms that need to 
    /// work with all parameters at once.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Materialize lazy-init tensors before copying their data — a fresh DiT
        // block that's never seen a Forward() call otherwise returns an empty
        // Vector here, which breaks any caller that concatenates per-layer
        // parameter vectors (including NeuralNetworkBase.GetParameters itself).
        EnsureWeightsAllocated();
        // Bulk copy from contiguous tensor storage — avoids ToArray() double-copy
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_queryWeights.Data),
            Vector<T>.FromMemory(_keyWeights.Data),
            Vector<T>.FromMemory(_valueWeights.Data),
            Vector<T>.FromMemory(_outputWeights.Data),
            Vector<T>.FromMemory(_outputBias.Data));
    }

    /// <summary>
    /// Sets all parameters (weights and biases) of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set in the layer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the opposite of GetParameters - it takes a list of values 
    /// and distributes them back into the layer's weights and biases. It's like restocking all the 
    /// ingredients in your kitchen from a single shopping bag, putting each item in its proper place.
    /// This is useful when loading a saved model or when optimization algorithms have computed 
    /// improved parameter values.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // SetParameters is the logical mirror of GetParameters — if the caller is
        // loading a snapshot into a lazily-constructed layer, we still need the real
        // tensors allocated first or every shape read below is 0.
        EnsureWeightsAllocated();
        // Calculate total number of parameters using tensor shape
        int qRows = _queryWeights.Shape[0], qCols = _queryWeights.Shape[1];
        int kRows = _keyWeights.Shape[0], kCols = _keyWeights.Shape[1];
        int vRows = _valueWeights.Shape[0], vCols = _valueWeights.Shape[1];
        int oRows = _outputWeights.Shape[0], oCols = _outputWeights.Shape[1];
        int biasLen = _outputBias.Shape[0];

        int totalParams = qRows * qCols + kRows * kCols + vRows * vCols + oRows * oCols + biasLen;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        var qLen = qRows * qCols;
        var kLen = kRows * kCols;
        var vLen = vRows * vCols;
        var oLen = oRows * oCols;

        // Bulk copy in-place — preserves engine persistent tensor references
        var src = parameters.AsSpan();
        int idx = 0;
        src.Slice(idx, qLen).CopyTo(_queryWeights.Data.Span); idx += qLen;
        src.Slice(idx, kLen).CopyTo(_keyWeights.Data.Span); idx += kLen;
        src.Slice(idx, vLen).CopyTo(_valueWeights.Data.Span); idx += vLen;
        src.Slice(idx, oLen).CopyTo(_outputWeights.Data.Span); idx += oLen;
        src.Slice(idx, biasLen).CopyTo(_outputBias.Data.Span);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    /// <summary>
    /// Resets the internal state of the multi-head attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all cached values from previous forward and backward passes,
    /// effectively resetting the layer to its initial state but keeping the learned weights.
    /// This is useful when starting a new training sequence or when you want to clear
    /// any temporary data without losing the layer's learned parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this like clearing your scratch paper after solving a math problem.
    /// You're keeping all the knowledge you've gained (the weights), but you're getting rid of
    /// all the intermediate calculations (cached values) to make room for new work.
    /// 
    /// This is particularly important in neural networks because:
    /// 1. It frees up memory by removing data we no longer need
    /// 2. It ensures that each new input is processed with a "clean slate"
    /// 3. It prevents old calculations from affecting new ones, which could lead to incorrect results
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null)
            return new Vector<T>(ParameterCount);
        // Bulk copy from contiguous tensor storage — avoids ToArray() double-copy
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_queryWeightsGradient.Data),
            Vector<T>.FromMemory(_keyWeightsGradient.Data),
            Vector<T>.FromMemory(_valueWeightsGradient.Data),
            _outputWeightsGradient != null ? Vector<T>.FromMemory(_outputWeightsGradient.Data) : new Vector<T>(_outputWeights.Length),
            _outputBiasGradient != null ? Vector<T>.FromMemory(_outputBiasGradient.Data) : new Vector<T>(_outputBias.Length));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null;
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastPreActivationOutput = null;
        _lastAttentionContext = null;
        _lastAttentionScores = null;
        _lastHeadOutputs = null;  // Clear per-head output cache
        _lastProjectedQueries = null;
        _lastProjectedKeys = null;
        _lastProjectedValues = null;

        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;

        // Clear GPU cached tensors
        _gpuInput2D = null;
        _gpuQ = null;
        _gpuK = null;
        _gpuV = null;
        _gpuContextFlat = null;
        _gpuAttentionWeights = null;
    }
}
