
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Attention;

/// <summary>
/// A multi-head attention layer using the Flash Attention algorithm for memory-efficient computation.
/// </summary>
/// <remarks>
/// <para>
/// FlashAttentionLayer provides the same functionality as MultiHeadAttentionLayer but uses the
/// Flash Attention algorithm which is 2-4x faster and uses significantly less memory.
/// It can be used as a drop-in replacement in transformer architectures.
/// </para>
/// <para><b>For Beginners:</b> This is like MultiHeadAttentionLayer but faster and more memory-efficient.
///
/// Flash Attention is a breakthrough algorithm that makes transformers much faster:
/// - Standard attention: O(N^2) memory, slow for long sequences
/// - Flash Attention: O(N) memory, 2-4x faster
///
/// Use this layer when:
/// - Training with long sequences (1024+ tokens)
/// - Training large models with limited GPU memory
/// - You need faster training/inference
///
/// The output is mathematically identical to standard attention - only the computation is different.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
internal class FlashAttentionLayer<T> : LayerBase<T>
{
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly FlashAttentionConfig _config;

    // Projection weights
    private Matrix<T> _queryWeights;
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastAttentionOutput;

    // Gradients
    private Matrix<T>? _queryWeightsGradient;
    private Matrix<T>? _keyWeightsGradient;
    private Matrix<T>? _valueWeightsGradient;
    private Matrix<T>? _outputWeightsGradient;
    private Vector<T>? _outputBiasGradient;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int HeadCount => _headCount;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the Flash Attention configuration.
    /// </summary>
    public FlashAttentionConfig Config => _config;

    /// <summary>
    /// Creates a new Flash Attention layer with the specified dimensions.
    /// </summary>
    /// <param name="sequenceLength">The length of the input sequence.</param>
    /// <param name="embeddingDimension">The dimension of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="config">Optional Flash Attention configuration.</param>
    /// <param name="activationFunction">Optional activation function (defaults to identity).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a Flash Attention layer.
    ///
    /// Parameters:
    /// - sequenceLength: How many tokens/words in your sequence (e.g., 512, 1024, 4096)
    /// - embeddingDimension: Size of each token's representation (e.g., 768 for BERT, 4096 for GPT-3)
    /// - headCount: Number of attention heads (e.g., 12 for BERT-base, 96 for GPT-3)
    ///
    /// The embeddingDimension must be divisible by headCount.
    /// Each head will have dimension = embeddingDimension / headCount.
    /// </para>
    /// </remarks>
    public FlashAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        FlashAttentionConfig? config = null,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).",
                nameof(headCount));
        }

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _config = config ?? FlashAttentionConfig.Default;

        // Initialize projection weights
        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Creates a new Flash Attention layer with vector activation function.
    /// </summary>
    public FlashAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        FlashAttentionConfig? config,
        IVectorActivationFunction<T>? vectorActivationFunction)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            vectorActivationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).",
                nameof(headCount));
        }

        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _config = config ?? FlashAttentionConfig.Default;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes projection weights using Xavier/Glorot initialization.
    /// </summary>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_queryWeights.Rows + _queryWeights.Columns)));

        InitializeMatrix(_queryWeights, scale);
        InitializeMatrix(_keyWeights, scale);
        InitializeMatrix(_valueWeights, scale);
        InitializeMatrix(_outputWeights, scale);

        // Initialize bias to zero
        _outputBias = Vector<T>.CreateDefault(_outputBias.Length, NumOps.Zero);
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass using Flash Attention.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, sequenceLength, embeddingDimension].</param>
    /// <returns>Output tensor of the same shape as input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where the Flash Attention computation happens.
    ///
    /// The forward pass:
    /// 1. Projects input to Query, Key, Value using learned weights
    /// 2. Reshapes into multiple heads
    /// 3. Applies Flash Attention (the fast, memory-efficient algorithm)
    /// 4. Concatenates heads and projects output
    ///
    /// Flash Attention computes the same result as standard attention but:
    /// - Never materializes the full N x N attention matrix
    /// - Processes in tiles that fit in fast cache memory
    /// - Uses online softmax for numerical stability
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape[1];
        int embeddingDimension = input.Shape[2];

        // Project input to Q, K, V
        var queries = input.Multiply(_queryWeights);
        var keys = input.Multiply(_keyWeights);
        var values = input.Multiply(_valueWeights);

        // Reshape to [batch, heads, seq, headDim]
        queries = queries.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        keys = keys.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);
        values = values.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        // Cache for backward pass
        _lastQuery = queries;
        _lastKey = keys;
        _lastValue = values;

        // Apply Flash Attention
        var (attentionOutput, _) = FlashAttention<T>.Forward(queries, keys, values, _config);

        _lastAttentionOutput = attentionOutput;

        // Reshape back to [batch, seq, embedding]
        attentionOutput = attentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);

        // Output projection
        var output = attentionOutput.Multiply(_outputWeights).Add(_outputBias);

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass using Flash Attention's memory-efficient gradient computation.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastQuery == null ||
            _lastKey == null || _lastValue == null || _lastAttentionOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int sequenceLength = _lastInput.Shape[1];
        int embeddingDimension = _lastInput.Shape[2];

        // Apply activation derivative
        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Gradient through output projection
        var attentionOutputGradient = activationGradient.Multiply(_outputWeights.Transpose());

        // Compute output weights gradient
        var attentionOutputFlat = _lastAttentionOutput.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);
        _outputWeightsGradient = ComputeWeightGradient(attentionOutputFlat, activationGradient);
        _outputBiasGradient = activationGradient.Sum([0, 1]).ToVector();

        // Reshape gradient for attention backward
        attentionOutputGradient = attentionOutputGradient.Reshape(batchSize, sequenceLength, _headCount, _headDimension).Transpose([0, 2, 1, 3]);

        // Flash Attention backward pass with recomputation
        var (gradQuery, gradKey, gradValue) = FlashAttention<T>.Backward(
            attentionOutputGradient,
            _lastQuery,
            _lastKey,
            _lastValue,
            _lastAttentionOutput,
            _config);

        // Reshape gradients back to [batch, seq, embedding]
        gradQuery = gradQuery.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);
        gradKey = gradKey.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);
        gradValue = gradValue.Transpose([0, 2, 1, 3]).Reshape(batchSize, sequenceLength, embeddingDimension);

        // Compute projection weight gradients
        _queryWeightsGradient = ComputeWeightGradient(_lastInput, gradQuery);
        _keyWeightsGradient = ComputeWeightGradient(_lastInput, gradKey);
        _valueWeightsGradient = ComputeWeightGradient(_lastInput, gradValue);

        // Compute input gradient
        var inputGradient = gradQuery.Multiply(_queryWeights.Transpose())
            .Add(gradKey.Multiply(_keyWeights.Transpose()))
            .Add(gradValue.Multiply(_valueWeights.Transpose()));

        return inputGradient;
    }

    /// <summary>
    /// Computes weight gradient from input and output gradient.
    /// </summary>
    private Matrix<T> ComputeWeightGradient(Tensor<T> input, Tensor<T> gradient)
    {
        // Sum over batch dimension: input^T @ gradient
        var inputT = input.Transpose([0, 2, 1]);
        var grad = inputT.Multiply(gradient);
        return grad.Sum([0]).ToMatrix();
    }

    /// <summary>
    /// Updates parameters using computed gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null ||
            _valueWeightsGradient == null || _outputWeightsGradient == null ||
            _outputBiasGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        _queryWeights = _queryWeights.Subtract(_queryWeightsGradient.Multiply(learningRate));
        _keyWeights = _keyWeights.Subtract(_keyWeightsGradient.Multiply(learningRate));
        _valueWeights = _valueWeights.Subtract(_valueWeightsGradient.Multiply(learningRate));
        _outputWeights = _outputWeights.Subtract(_outputWeightsGradient.Multiply(learningRate));
        _outputBias = _outputBias.Subtract(_outputBiasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all layer parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        int totalParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy all weight matrices
        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    parameters[index++] = matrix[i, j];
                }
            }
        }

        // Copy bias
        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all layer parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");
        }

        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = parameters[index++];
                }
            }
        }

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the layer's internal state.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastAttentionOutput = null;

        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation
    {
        get
        {
            return _queryWeights != null && _keyWeights != null &&
                   _valueWeights != null && _outputWeights != null &&
                   _queryWeights.Rows > 0;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input
        var seqLen = InputShape[0];
        var embDim = InputShape[1];
        var symbolicInput = new Tensor<T>(new[] { 1, seqLen, embDim });
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Convert weights to tensors
        var wqTensor = MatrixToTensor(_queryWeights);
        var wkTensor = MatrixToTensor(_keyWeights);
        var wvTensor = MatrixToTensor(_valueWeights);
        var woTensor = MatrixToTensor(_outputWeights);

        var wqNode = Autodiff.TensorOperations<T>.Constant(wqTensor, "Wq");
        var wkNode = Autodiff.TensorOperations<T>.Constant(wkTensor, "Wk");
        var wvNode = Autodiff.TensorOperations<T>.Constant(wvTensor, "Wv");
        var woNode = Autodiff.TensorOperations<T>.Constant(woTensor, "Wo");

        // Multi-head attention using TensorOperations
        var output = Autodiff.TensorOperations<T>.MultiHeadAttention(
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

    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(new[] { matrix.Rows, matrix.Columns });
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                tensor[i, j] = matrix[i, j];
            }
        }
        return tensor;
    }

    /// <summary>
    /// Gets diagnostic information about the layer.
    /// </summary>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        diagnostics["HeadCount"] = _headCount.ToString();
        diagnostics["HeadDimension"] = _headDimension.ToString();
        diagnostics["UseCausalMask"] = _config.UseCausalMask.ToString();
        diagnostics["BlockSizeQ"] = _config.BlockSizeQ.ToString();
        diagnostics["BlockSizeKV"] = _config.BlockSizeKV.ToString();
        diagnostics["RecomputeInBackward"] = _config.RecomputeInBackward.ToString();
        diagnostics["Precision"] = _config.Precision.ToString();

        return diagnostics;
    }

    /// <summary>
    /// Gets the query projection weights (for external access/debugging).
    /// </summary>
    public Matrix<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the key projection weights.
    /// </summary>
    public Matrix<T> GetKeyWeights() => _keyWeights;

    /// <summary>
    /// Gets the value projection weights.
    /// </summary>
    public Matrix<T> GetValueWeights() => _valueWeights;

    /// <summary>
    /// Gets the output projection weights.
    /// </summary>
    public Matrix<T> GetOutputWeights() => _outputWeights;

    internal override Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>
        {
            ["HeadCount"] = _headCount.ToString(),
            ["UseCausalMask"] = _config.UseCausalMask.ToString()
        };
    }
}
