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
public class MultiHeadAttentionLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
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

    // Cached projected Q, K, V for backward pass (4D: [batch, heads, seq, head_dim])
    private Tensor<T>? _lastProjectedQueries = null;
    private Tensor<T>? _lastProjectedKeys = null;
    private Tensor<T>? _lastProjectedValues = null;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput2D;
    private IGpuTensor<T>? _gpuQ;
    private IGpuTensor<T>? _gpuK;
    private IGpuTensor<T>? _gpuV;
    private IGpuTensor<T>? _gpuContextFlat;
    private IGpuTensor<T>? _gpuAttentionWeights;
    private int _gpuBatchSize;
    private int _gpuSeqLength;
    private int _gpuEmbeddingDim;

    /// <summary>
    /// Tensor of weights for transforming input into query representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T> _queryWeights;

    /// <summary>
    /// Tensor of weights for transforming input into key representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T> _keyWeights;

    /// <summary>
    /// Tensor of weights for transforming input into value representations.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T> _valueWeights;

    /// <summary>
    /// Tensor of weights for the final output projection.
    /// Shape: [embeddingDimension, embeddingDimension]
    /// </summary>
    private Tensor<T> _outputWeights;

    /// <summary>
    /// Tensor of biases added to the final output.
    /// Shape: [embeddingDimension]
    /// </summary>
    private Tensor<T> _outputBias;

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
    public override int ParameterCount =>
        _queryWeights.Length +
        _keyWeights.Length +
        _valueWeights.Length +
        _outputWeights.Length +
        _outputBias.Length;

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
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IActivationFunction<T>? activationFunction = null)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], activationFunction ?? new IdentityActivation<T>())
    {
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        HeadDiversityWeight = NumOps.FromDouble(0.01);
        _lastEntropyLoss = NumOps.Zero;
        _lastDiversityLoss = NumOps.Zero;

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;

        // Initialize weight tensors - production-ready pattern (no Matrix/Vector types)
        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Creates a new multi-head attention layer with the specified dimensions and head count.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each element in the sequence.</param>
    /// <param name="headCount">The number of attention heads to use.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply (required to disambiguate from IActivationFunction overload).</param>
    public MultiHeadAttentionLayer(int sequenceLength, int embeddingDimension, int headCount, IVectorActivationFunction<T> vectorActivationFunction)
        : base([sequenceLength, embeddingDimension], [sequenceLength, embeddingDimension], vectorActivationFunction)
    {
        // Initialize auxiliary loss fields first so compiler knows they're set
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        HeadDiversityWeight = NumOps.FromDouble(0.01);
        _lastEntropyLoss = NumOps.Zero;
        _lastDiversityLoss = NumOps.Zero;

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;

        // Initialize weight tensors - production-ready pattern (no Matrix/Vector types)
        _queryWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _keyWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _valueWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputWeights = new Tensor<T>([embeddingDimension, embeddingDimension]);
        _outputBias = new Tensor<T>([embeddingDimension]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
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

        _queryWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_queryWeights.Shape, Vector<T>.CreateRandom(rows * cols, -0.5, 0.5)),
            scale);
        _keyWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_keyWeights.Shape, Vector<T>.CreateRandom(rows * cols, -0.5, 0.5)),
            scale);
        _valueWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_valueWeights.Shape, Vector<T>.CreateRandom(rows * cols, -0.5, 0.5)),
            scale);
        _outputWeights = Engine.TensorMultiplyScalar(
            new Tensor<T>(_outputWeights.Shape, Vector<T>.CreateRandom(rows * cols, -0.5, 0.5)),
            scale);

        // Initialize bias tensor to zeros
        _outputBias.Fill(NumOps.Zero);
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
        // Industry standard: Support any-rank tensors (like PyTorch's MultiheadAttention)
        // Last two dimensions are [sequence, embedding_dim]
        // All preceding dimensions are treated as batch dimensions
        // Examples:
        //   2D [seq, dim] -> batch=1, seq, dim
        //   3D [batch, seq, dim] -> batch, seq, dim
        //   4D [batch1, batch2, seq, dim] -> batch1*batch2, seq, dim
        //   5D [b1, b2, b3, seq, dim] -> b1*b2*b3, seq, dim

        _originalQueryShape = query.Shape;
        _originalKeyShape = key.Shape;
        _originalValueShape = value.Shape;

        // Handle 1D input by reshaping to 2D [1, dim]
        bool was1D = query.Rank == 1;
        if (was1D)
        {
            // Treat 1D [dim] as single token sequence [1, dim]
            query = query.Reshape([1, query.Shape[0]]);
            key = key.Reshape([1, key.Shape[0]]);
            value = value.Reshape([1, value.Shape[0]]);
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
        query = query.Reshape(batchQ, seqLenQ, dimQ);
        key = key.Reshape(batchK, seqLenK, dimK);
        value = value.Reshape(batchV, seqLenV, dimV);

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
                $"Query shape: [{string.Join(", ", query.Shape)}], Weights shape: [{string.Join(", ", _queryWeights.Shape)}]");
        }

        var q2D = query.Reshape(batchSize * seqLengthQ, embeddingDimension);
        var k2D = key.Reshape(batchSize * seqLengthKV, embeddingDimension);
        var v2D = value.Reshape(batchSize * seqLengthKV, embeddingDimension);

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
                $"Q_flat shape: [{string.Join(", ", Q_flat.Shape)}], " +
                $"q2D shape: [{string.Join(", ", q2D.Shape)}], " +
                $"_queryWeights shape: [{string.Join(", ", _queryWeights.Shape)}]");
        }

        var queries = Q_flat.Reshape(batchSize, seqLengthQ, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });
        var keys = K_flat.Reshape(batchSize, seqLengthKV, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });
        var values = V_flat.Reshape(batchSize, seqLengthKV, _headCount, _headDimension).Transpose(new[] { 0, 2, 1, 3 });

        // Cache projected Q, K, V for backward pass (4D: [batch, heads, seq, head_dim])
        _lastProjectedQueries = queries;
        _lastProjectedKeys = keys;
        _lastProjectedValues = values;

        // 2. Compute Scaled Dot-Product Attention
        // ScaledDotProductAttention computes: softmax(Q @ K^T / scale) @ V
        // Input shapes: [batch, heads, seq, head_dim]
        // Output shape: [batch, heads, seq_q, head_dim]
        var context_4D = Engine.ScaledDotProductAttention(
            queries, keys, values,
            mask: null,
            scale: 1.0 / Math.Sqrt(_headDimension),
            out var attentionWeights4D);

        // Cache attention weights for backward pass
        _lastAttentionScores = attentionWeights4D;

        // 3. Cache Head Outputs
        var permutedForCache = context_4D.Transpose(new[] { 1, 0, 2, 3 }); // [H, B, S, D]
        _lastHeadOutputs = new List<Tensor<T>> { permutedForCache }; // store stacked heads

        // 5. Concatenate and Project Output
        // [B, H, S, D] -> [B, S, H, D] -> [B, S, E]
        var context_transposed = context_4D.Transpose(new[] { 0, 2, 1, 3 });
        var context_flat = context_transposed.Reshape(batchSize * seqLengthQ, embeddingDimension);

        // Cache pre-projection context for weight gradient computation in backward pass
        _lastAttentionContext = context_transposed.Reshape(batchSize, seqLengthQ, embeddingDimension);

        var output_flat = Engine.TensorMatMul(context_flat, _outputWeights);
        var output_reshaped = output_flat.Reshape(batchSize, seqLengthQ, embeddingDimension);

        var biasBroadcast = _outputBias.Reshape(1, 1, embeddingDimension);
        var outputWithBias = Engine.TensorBroadcastAdd(output_reshaped, biasBroadcast);
        var result = ApplyActivation(outputWithBias);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Reshape output back to original batch dimensions
        // Output is currently [flatBatch, seq, dim], need to reshape to [origBatch..., seq, dim]
        if (_originalQueryShape.Length == 1)
        {
            // 1D input -> 1D output [dim]
            return result.Reshape([embeddingDimension]);
        }

        int[] outputShape = new int[_originalQueryShape.Length];
        for (int i = 0; i < _originalQueryShape.Length - 2; i++)
        {
            outputShape[i] = _originalQueryShape[i];
        }
        outputShape[^2] = seqLengthQ;
        outputShape[^1] = embeddingDimension;

        return result.Reshape(outputShape);
    }

    /// <summary>
    /// GPU-resident forward pass for multi-head attention.
    /// Performs all projections and attention computation on GPU without downloading intermediate results.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Handle input shape - flatten to 3D [batch, seq, embedding]
        int[] inputShape = input.Shape;
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
        IGpuTensor<T> attentionOutput;
        IGpuTensor<T>? attentionWeightsGpu = null;

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
            _lastInput = input3D.ToTensor();

            // Cache projected Q, K, V for backward pass
            _lastProjectedQueries = qPermuted.ToTensor();
            _lastProjectedKeys = kPermuted.ToTensor();
            _lastProjectedValues = vPermuted.ToTensor();

            // Cache attention context for output weights gradient
            _lastAttentionContext = contextFlat.ToTensor();

            // Cache attention weights for backward pass
            _lastAttentionScores = attentionWeightsGpu?.ToTensor();

            _lastOutput = outputWithBias.ToTensor();
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

    /// <summary>
    /// Performs the backward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient of the loss w.r.t. output.</param>
    /// <returns>GPU-resident gradient of the loss w.r.t. input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_gpuInput2D == null || _gpuQ == null || _gpuK == null || _gpuV == null ||
            _gpuContextFlat == null || _gpuAttentionWeights == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        int batchSize = _gpuBatchSize;
        int seqLength = _gpuSeqLength;
        int embeddingDimension = _gpuEmbeddingDim;

        // Reshape output gradient to 3D if needed
        IGpuTensor<T> outputGrad3D = outputGradient;
        if (outputGradient.Shape.Length != 3)
        {
            outputGrad3D = gpuEngine.ReshapeGpu(outputGradient, new[] { batchSize, seqLength, embeddingDimension });
        }

        // 1. Bias gradient: sum over batch and sequence dimensions
        var biasSumBatch = gpuEngine.SumAxisGpu(outputGrad3D, 0);
        var biasSum = gpuEngine.SumAxisGpu(biasSumBatch, 0);
        _outputBiasGradient = biasSum.ToTensor();

        // 2. Output weights gradient: context^T @ grad_output
        var outputGrad2D = gpuEngine.ReshapeGpu(outputGrad3D, new[] { batchSize * seqLength, embeddingDimension });
        var contextFlat2D = gpuEngine.ReshapeGpu(_gpuContextFlat, new[] { batchSize * seqLength, embeddingDimension });
        var contextT = gpuEngine.TransposeGpu(contextFlat2D);
        var dOutputWeights = gpuEngine.MatMulGpuTensors(contextT, outputGrad2D);
        _outputWeightsGradient = dOutputWeights.ToTensor();

        // 3. Gradient through output projection: grad @ output_weights^T
        var outputWeightsT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_outputWeights), GpuTensorRole.Weight);
        var dContext = gpuEngine.MatMulGpuTensors(outputGrad2D, outputWeightsT);
        var dContext3D = gpuEngine.ReshapeGpu(dContext, new[] { batchSize, seqLength, embeddingDimension });

        // 4. Reshape context gradient to [batch, seq, heads, headDim] and permute to [batch, heads, seq, headDim]
        var dContextReshaped = gpuEngine.ReshapeGpu(dContext3D, new[] { batchSize, seqLength, _headCount, _headDimension });
        var dOutput4D = gpuEngine.PermuteGpu(dContextReshaped, new[] { 0, 2, 1, 3 });

        // 5. Use GPU ScaledDotProductAttentionBackward for efficient gradient computation
        double scale = 1.0 / Math.Sqrt(_headDimension);
        var (dQ_4D, dK_4D, dV_4D) = gpuEngine.ScaledDotProductAttentionBackwardGpu(
            dOutput4D, _gpuQ, _gpuK, _gpuV, _gpuAttentionWeights, scale, isCausal: false);

        // 6. Permute gradients from [batch, heads, seq, headDim] to [batch, seq, heads, headDim]
        var dQ_transposed = gpuEngine.PermuteGpu(dQ_4D, new[] { 0, 2, 1, 3 });
        var dK_transposed = gpuEngine.PermuteGpu(dK_4D, new[] { 0, 2, 1, 3 });
        var dV_transposed = gpuEngine.PermuteGpu(dV_4D, new[] { 0, 2, 1, 3 });

        // 7. Reshape to [batch*seq, embedding]
        var dQ = gpuEngine.ReshapeGpu(dQ_transposed, new[] { batchSize * seqLength, embeddingDimension });
        var dK = gpuEngine.ReshapeGpu(dK_transposed, new[] { batchSize * seqLength, embeddingDimension });
        var dV = gpuEngine.ReshapeGpu(dV_transposed, new[] { batchSize * seqLength, embeddingDimension });

        // 8. Q, K, V weight gradients: input2D^T @ dQ/dK/dV
        var input2D_T = gpuEngine.TransposeGpu(_gpuInput2D);
        var dQueryWeights = gpuEngine.MatMulGpuTensors(input2D_T, dQ);
        var dKeyWeights = gpuEngine.MatMulGpuTensors(input2D_T, dK);
        var dValueWeights = gpuEngine.MatMulGpuTensors(input2D_T, dV);

        // Download weight gradients to CPU (needed for UpdateParameters)
        _queryWeightsGradient = dQueryWeights.ToTensor();
        _keyWeightsGradient = dKeyWeights.ToTensor();
        _valueWeightsGradient = dValueWeights.ToTensor();

        // 9. Input gradient: dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
        var wqT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_queryWeights), GpuTensorRole.Weight);
        var wkT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_keyWeights), GpuTensorRole.Weight);
        var wvT = gpuEngine.UploadToGpu(Engine.TensorTranspose(_valueWeights), GpuTensorRole.Weight);

        var dInputFromQ = gpuEngine.MatMulGpuTensors(dQ, wqT);
        var dInputFromK = gpuEngine.MatMulGpuTensors(dK, wkT);
        var dInputFromV = gpuEngine.MatMulGpuTensors(dV, wvT);

        var dInput2D = gpuEngine.AddGpu(gpuEngine.AddGpu(dInputFromQ, dInputFromK), dInputFromV);

        // 10. Reshape back to original input shape
        var inputGradient = gpuEngine.ReshapeGpu(dInput2D, new[] { batchSize, seqLength, embeddingDimension });

        return inputGradient;
    }

    /// <summary>
    /// Performs the backward pass of the multi-head attention layer, calculating gradients for learning.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to be passed to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass is how neural networks learn. Think of it like figuring out
    /// which parts of a recipe need adjustment after tasting the final dish:
    ///
    /// 1. We first check how our output differs from what was expected (the gradient)
    /// 2. Then we trace backward through all the calculations we did in the forward pass
    /// 3. We determine how much each weight contributed to any errors
    /// 4. These contributions become our gradients, which we'll use to update the weights
    ///
    /// The complex matrix operations are just a mathematical way of figuring out
    /// "if I change this weight a little bit, how much would it improve the output?"
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Flatten any-rank gradient to 3D for processing (like forward pass)
        int seqLen = outputGradient.Shape[^2];
        int dim = outputGradient.Shape[^1];
        int batch = 1;
        for (int i = 0; i < outputGradient.Rank - 2; i++)
            batch *= outputGradient.Shape[i];
        if (outputGradient.Rank == 2) batch = 1;

        outputGradient = outputGradient.Reshape(batch, seqLen, dim);

        var result = UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);

        // Reshape gradient back to original input shape
        return result.Reshape(_originalQueryShape);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastAttentionScores == null || _lastAttentionContext == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_lastProjectedQueries == null || _lastProjectedKeys == null || _lastProjectedValues == null)
            throw new InvalidOperationException("Projected Q, K, V must be cached from forward pass.");

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        // Compute attention output gradient using tensor transpose
        // dAttentionContext = dOut @ Wo^T
        var attentionOutputGradient = activationGradient.Multiply(_outputWeights.Transpose([1, 0]));

        // Compute output weights gradient using pre-projection context (not post-activation output)
        // Weight gradient = input^T @ gradient, where input is the pre-projection attention context
        _outputWeightsGradient = _lastAttentionContext.Transpose([0, 2, 1]).Multiply(activationGradient).Sum([0]).Reshape([embeddingDimension, embeddingDimension]);

        // Compute output bias gradient - keep as Tensor<T> (no conversion)
        _outputBiasGradient = activationGradient.Sum([0, 1]);

        // Reshape attention output gradient to 4D: [batch, heads, seq, head_dim]
        attentionOutputGradient = attentionOutputGradient.Reshape([batchSize, sequenceLength, _headCount, _headDimension]).Transpose([0, 2, 1, 3]);

        // Use Engine.ScaledDotProductAttentionBackward for efficient gradient computation
        // Computes dQ, dK, dV gradients through the attention operation
        Engine.ScaledDotProductAttentionBackward(
            attentionOutputGradient,
            _lastProjectedQueries,
            _lastProjectedKeys,
            _lastProjectedValues,
            _lastAttentionScores,
            1.0 / Math.Sqrt(_headDimension),
            out var queriesGradient4D,
            out var keysGradient4D,
            out var valuesGradient4D);

        // Reshape gradients from 4D to 3D: [batch, seq, embed]
        var queriesGradient = queriesGradient4D.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        var keysGradient = keysGradient4D.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);
        var valuesGradient = valuesGradient4D.Transpose([0, 2, 1, 3]).Reshape([batchSize, sequenceLength, embeddingDimension]);

        // Compute weight gradients - keep as Tensor<T> (no conversion)
        // dWq = Input^T @ dQ_flat
        _queryWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(queriesGradient).Sum([0]).Reshape([embeddingDimension, embeddingDimension]);
        _keyWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(keysGradient).Sum([0]).Reshape([embeddingDimension, embeddingDimension]);
        _valueWeightsGradient = _lastInput.Transpose([0, 2, 1]).Multiply(valuesGradient).Sum([0]).Reshape([embeddingDimension, embeddingDimension]);

        // Compute input gradient using tensor transpose
        // dInput = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
        var inputGradient = queriesGradient.Multiply(_queryWeights.Transpose([1, 0]))
                            .Add(keysGradient.Multiply(_keyWeights.Transpose([1, 0])))
                            .Add(valuesGradient.Multiply(_valueWeights.Transpose([1, 0])));

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients by building a computation graph
    /// that mirrors the multi-head attention forward pass operations. It supports both Self-Attention and Cross-Attention.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastQueryInput == null || _lastKeyInput == null || _lastValueInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        var qInputNode = Autodiff.TensorOperations<T>.Variable(_lastQueryInput, "query", requiresGradient: true);
        var kInputNode = Autodiff.TensorOperations<T>.Variable(_lastKeyInput, "key", requiresGradient: true);
        var vInputNode = Autodiff.TensorOperations<T>.Variable(_lastValueInput, "value", requiresGradient: true);

        var qWeightsNode = Autodiff.TensorOperations<T>.Variable(_queryWeights, "QWeights", requiresGradient: true);
        var kWeightsNode = Autodiff.TensorOperations<T>.Variable(_keyWeights, "KWeights", requiresGradient: true);
        var vWeightsNode = Autodiff.TensorOperations<T>.Variable(_valueWeights, "VWeights", requiresGradient: true);
        var oWeightsNode = Autodiff.TensorOperations<T>.Variable(_outputWeights, "OWeights", requiresGradient: true);
        var oBiasNode = Autodiff.TensorOperations<T>.Variable(_outputBias, "OBias", requiresGradient: true);

        int batchSize = _lastQueryInput.Shape[0];
        int seqLengthQ = _lastQueryInput.Shape[1];
        int embeddingDimension = _lastQueryInput.Shape[2];
        int seqLengthKV = _lastKeyInput.Shape[1];

        // 2. Project Input to Q, K, V
        // Reshape to 2D [B*S, E] for matrix multiply
        var qInput2D = Autodiff.TensorOperations<T>.Reshape(qInputNode, batchSize * seqLengthQ, embeddingDimension);
        var kInput2D = Autodiff.TensorOperations<T>.Reshape(kInputNode, batchSize * seqLengthKV, embeddingDimension);
        var vInput2D = Autodiff.TensorOperations<T>.Reshape(vInputNode, batchSize * seqLengthKV, embeddingDimension);

        var qFlat = Autodiff.TensorOperations<T>.MatrixMultiply(qInput2D, qWeightsNode);
        var kFlat = Autodiff.TensorOperations<T>.MatrixMultiply(kInput2D, kWeightsNode);
        var vFlat = Autodiff.TensorOperations<T>.MatrixMultiply(vInput2D, vWeightsNode);

        // Reshape to [B, S, H, D]
        var qReshaped = Autodiff.TensorOperations<T>.Reshape(qFlat, batchSize, seqLengthQ, _headCount, _headDimension);
        var kReshaped = Autodiff.TensorOperations<T>.Reshape(kFlat, batchSize, seqLengthKV, _headCount, _headDimension);
        var vReshaped = Autodiff.TensorOperations<T>.Reshape(vFlat, batchSize, seqLengthKV, _headCount, _headDimension);

        // Permute to [B, H, S, D]
        var qHeads = Autodiff.TensorOperations<T>.Permute(qReshaped, 0, 2, 1, 3);
        var kHeads = Autodiff.TensorOperations<T>.Permute(kReshaped, 0, 2, 1, 3);
        var vHeads = Autodiff.TensorOperations<T>.Permute(vReshaped, 0, 2, 1, 3);

        // 3. Attention Scores: Q @ K.T
        // Permute K to [B, H, D, S] for multiplication
        var kHeadsT = Autodiff.TensorOperations<T>.Permute(kHeads, 0, 1, 3, 2);

        // [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]
        var scores = Autodiff.TensorOperations<T>.MatrixMultiply(qHeads, kHeadsT);

        // Scale
        T scaleFactor = NumOps.Sqrt(NumOps.FromDouble(_headDimension));
        T scaleValue = NumericalStabilityHelper.SafeDiv(NumOps.One, scaleFactor);
        var scaleTensor = new Tensor<T>(new int[] { 1 });
        scaleTensor[0] = scaleValue;
        var scaleNode = Autodiff.TensorOperations<T>.Constant(scaleTensor, "scale");

        var scaledScores = Autodiff.TensorOperations<T>.ElementwiseMultiply(scores, scaleNode);

        // Softmax
        var attentionWeights = Autodiff.TensorOperations<T>.Softmax(scaledScores);

        // 4. Output: Weights @ V
        // [B, H, Sq, Sk] @ [B, H, Sk, D] -> [B, H, Sq, D]
        var attentionOutput = Autodiff.TensorOperations<T>.MatrixMultiply(attentionWeights, vHeads);

        // 5. Merge Heads
        // Permute to [B, Sq, H, D]
        var contextPermuted = Autodiff.TensorOperations<T>.Permute(attentionOutput, 0, 2, 1, 3);

        // Reshape to [B*Sq, E]
        var contextFlat = Autodiff.TensorOperations<T>.Reshape(contextPermuted, batchSize * seqLengthQ, embeddingDimension);

        // 6. Output Projection
        var outputFlat = Autodiff.TensorOperations<T>.MatrixMultiply(contextFlat, oWeightsNode);

        // Reshape to [B, Sq, E]
        var outputReshaped = Autodiff.TensorOperations<T>.Reshape(outputFlat, batchSize, seqLengthQ, embeddingDimension);

        // Add Bias (broadcast)
        var biasReshaped = Autodiff.TensorOperations<T>.Reshape(oBiasNode, 1, 1, embeddingDimension);
        var outputWithBias = Autodiff.TensorOperations<T>.Add(outputReshaped, biasReshaped);

        // Apply Activation
        var finalOutput = ApplyActivationToGraph(outputWithBias);

        // 7. Set Gradient
        finalOutput.Gradient = outputGradient;

        // 8. Topo Sort & Backward
        finalOutput.Backward();

        // 9. Store Gradients
        _queryWeightsGradient = qWeightsNode.Gradient;
        _keyWeightsGradient = kWeightsNode.Gradient;
        _valueWeightsGradient = vWeightsNode.Gradient;
        _outputWeightsGradient = oWeightsNode.Gradient;
        _outputBiasGradient = oBiasNode.Gradient;

        // Return gradient w.r.t Query input (as expected by TransformerDecoderLayer)
        // Note: Gradients for Key and Value (Encoder outputs) are computed in kInputNode.Gradient and vInputNode.Gradient
        // but ILayer only returns one gradient tensor.
        return qInputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
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
                _queryWeightsVelocity = new Tensor<T>(_queryWeights.Shape);
                _queryWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_queryWeightsVelocity, PersistentTensorRole.OptimizerState);

                _keyWeightsVelocity = new Tensor<T>(_keyWeights.Shape);
                _keyWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_keyWeightsVelocity, PersistentTensorRole.OptimizerState);

                _valueWeightsVelocity = new Tensor<T>(_valueWeights.Shape);
                _valueWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_valueWeightsVelocity, PersistentTensorRole.OptimizerState);

                _outputWeightsVelocity = new Tensor<T>(_outputWeights.Shape);
                _outputWeightsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_outputWeightsVelocity, PersistentTensorRole.OptimizerState);

                _outputBiasVelocity = new Tensor<T>(_outputBias.Shape);
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
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeights.ToArray()),
            new Vector<T>(_keyWeights.ToArray()),
            new Vector<T>(_valueWeights.ToArray()),
            new Vector<T>(_outputWeights.ToArray()),
            new Vector<T>(_outputBias.ToArray()));
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

        int index = 0;

        var qLen = qRows * qCols;
        var kLen = kRows * kCols;
        var vLen = vRows * vCols;
        var oLen = oRows * oCols;

        _queryWeights = new Tensor<T>([qRows, qCols], parameters.Slice(index, qLen));
        index += qLen;

        _keyWeights = new Tensor<T>([kRows, kCols], parameters.Slice(index, kLen));
        index += kLen;

        _valueWeights = new Tensor<T>([vRows, vCols], parameters.Slice(index, vLen));
        index += vLen;

        _outputWeights = new Tensor<T>([oRows, oCols], parameters.Slice(index, oLen));
        index += oLen;

        _outputBias = new Tensor<T>([biasLen], parameters.Slice(index, biasLen));

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
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
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

    /// <summary>
    /// Exports the multi-head attention layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the multi-head attention operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, sequenceLength, embeddingDimension]
    /// 2. Creates constant nodes for Q, K, V, and output projection weights
    /// 3. Applies multi-head attention using TensorOperations<T>.MultiHeadAttention()
    /// 4. Returns the final output with output projection applied
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of multi-head attention for JIT.
    ///
    /// JIT compilation converts multi-head attention into optimized native code.
    /// Multi-head attention is like having multiple "experts" analyzing the input:
    /// - Each head learns to focus on different aspects (syntax, semantics, context)
    /// - Heads process in parallel for efficiency
    /// - Results are combined through output projection
    ///
    /// The process:
    /// 1. Project input to queries, keys, values using learned weights
    /// 2. Split projections into multiple heads (e.g., 8 heads)
    /// 3. Each head computes scaled dot-product attention independently
    /// 4. Concatenate all head outputs
    /// 5. Apply final output projection
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize parallel processing across heads
    /// - Fuse projection operations
    /// - Generate efficient memory layouts for multi-head computation
    /// - Optimize attention score computation and softmax
    ///
    /// This is the core mechanism in BERT, GPT, T5, and all modern Transformers.
    /// JIT compilation provides 5-10x speedup for this complex operation.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer parameters are not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Initialize the layer first.");

        if (_queryWeights == null || _keyWeights == null || _valueWeights == null || _outputWeights == null)
            throw new InvalidOperationException("Layer projection weights not initialized. Train or initialize the model first.");

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // MultiHeadAttentionLayer expects input shape: [sequenceLength, embeddingDimension]
        // For attention, we use: [batch, sequenceLength, embeddingDimension]
        var embeddingDim = InputShape[1];
        var seqLength = InputShape[0];
        var symbolicInput = new Tensor<T>(new int[] { 1, seqLength, embeddingDim });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for projection weights - weights are already Tensor<T> (no conversion needed)
        var wqNode = TensorOperations<T>.Constant(_queryWeights, "Wq");
        var wkNode = TensorOperations<T>.Constant(_keyWeights, "Wk");
        var wvNode = TensorOperations<T>.Constant(_valueWeights, "Wv");
        var woNode = TensorOperations<T>.Constant(_outputWeights, "Wo");

        // Apply multi-head attention
        // For self-attention: query, key, value all come from the same input
        var output = TensorOperations<T>.MultiHeadAttention(
            query: inputNode,
            key: inputNode,
            value: inputNode,
            numHeads: _headCount,
            wQ: wqNode,
            wK: wkNode,
            wV: wvNode,
            wO: woNode);

        return output;
    }

    /// <summary>
    /// Gets whether this multi-head attention layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer parameters are initialized.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Query, Key, Value projection weights are initialized
    /// - Output projection weights are initialized
    /// - The multi-head structure is properly configured
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - All projection weight matrices are initialized (Wq, Wk, Wv, Wo)
    /// - The number of attention heads is configured
    ///
    /// Multi-head attention is one of the most expensive operations in modern deep learning:
    /// - Used extensively in Transformers (BERT has 144 attention layers, GPT-3 has 96)
    /// - Each forward pass computes attention scores for all position pairs (O(n²))
    /// - Multiple heads process in parallel
    ///
    /// JIT compilation provides significant speedup (5-10x) by optimizing:
    /// - Parallel matrix multiplications for all heads
    /// - Attention score computation across heads
    /// - Softmax operations
    /// - Head concatenation and output projection
    /// - Memory access patterns for cache efficiency
    ///
    /// This optimization is critical for:
    /// - Real-time NLP applications (translation, summarization, chat)
    /// - Large language models (GPT, BERT, T5)
    /// - Vision Transformers processing high-resolution images
    /// - Any application using Transformer architecture
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Multi-head attention supports JIT if all projection weight tensors are initialized
            return _queryWeights != null && _keyWeights != null &&
                   _valueWeights != null && _outputWeights != null &&
                   _queryWeights.Shape.Length >= 2 && _queryWeights.Shape[0] > 0 &&
                   _keyWeights.Shape.Length >= 2 && _keyWeights.Shape[0] > 0 &&
                   _valueWeights.Shape.Length >= 2 && _valueWeights.Shape[0] > 0 &&
                   _outputWeights.Shape.Length >= 2 && _outputWeights.Shape[0] > 0;
        }
    }
}
