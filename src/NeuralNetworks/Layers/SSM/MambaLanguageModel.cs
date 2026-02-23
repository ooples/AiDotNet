using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements a full Mamba language model: token embedding + N MambaBlocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// This assembles the complete Mamba architecture as described in the original paper:
/// <code>
///   1. Token Embedding: token indices → dense vectors [batch, seqLen, modelDim]
///   2. N × MambaBlock: selective scan processing with residual connections
///   3. RMS Normalization: final layer normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// The model supports autoregressive generation via <see cref="SSMStateCache{T}"/>. During inference,
/// previously computed hidden states are cached so only the new token needs to be processed per step,
/// giving O(1) per-token generation cost (compared to O(n) for Transformers without KV-cache).
/// </para>
/// <para><b>For Beginners:</b> This is a complete text generation model built entirely from Mamba blocks.
///
/// How it works:
/// 1. Each word/token is converted to a vector of numbers (embedding)
/// 2. These vectors pass through several Mamba layers, which understand context
/// 3. The final output is a probability distribution over all possible next words
///
/// What makes it special:
/// - Linear time complexity: processes 10x longer text in the same time as Transformers
/// - Constant memory per token during generation (via state caching)
/// - Competitive quality with Transformer models of similar size
///
/// Real-world examples: Falcon Mamba 7B, Mamba-2 models, and various research models.
/// </para>
/// <para>
/// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
/// https://arxiv.org/abs/2312.00752
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MambaLanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _expandFactor;

    // Token embedding: [vocabSize, modelDim]
    private Tensor<T> _embeddingWeights;

    // Stack of Mamba blocks
    private readonly MambaBlock<T>[] _blocks;

    // Final RMS normalization
    private Tensor<T> _finalNormGamma;

    // LM head (output projection): [modelDim, vocabSize]
    private Tensor<T> _lmHeadWeights;
    private Tensor<T> _lmHeadBias;

    // State cache for autoregressive generation
    private SSMStateCache<T>? _stateCache;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEmbedded;
    private Tensor<T>? _lastNormedOutput;
    private Tensor<T>[]? _lastBlockInputs;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _embeddingWeightsGradient;
    private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _lmHeadWeightsGradient;
    private Tensor<T>? _lmHeadBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the total number of different words/tokens the model can recognize.
    /// Common values are 32000 (LLaMA), 50257 (GPT-2), or 100000+ for multilingual models.</para>
    /// </remarks>
    public int VocabSize => _vocabSize;

    /// <summary>
    /// Gets the model dimension (d_model).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The width of the model's hidden representation. Larger values allow the model
    /// to capture more complex patterns but require more memory and computation.</para>
    /// </remarks>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of Mamba layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The depth of the model. More layers allow the model to learn more
    /// abstract patterns, but add to both parameter count and inference time.</para>
    /// </remarks>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the SSM state dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The memory capacity per position in each Mamba layer. Larger values
    /// let the model remember more context at each step. 16 is the standard default.</para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The total count of numbers the model can adjust during training.
    /// More parameters generally means more capacity but also more memory and compute required.</para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int count = _embeddingWeights.Length;
            foreach (var block in _blocks)
                count += block.ParameterCount;
            count += _finalNormGamma.Length;
            count += _lmHeadWeights.Length + _lmHeadBias.Length;
            return count;
        }
    }

    /// <summary>
    /// Creates a new Mamba language model.
    /// </summary>
    /// <param name="vocabSize">
    /// Size of the token vocabulary. Typical: 32000 (LLaMA), 50257 (GPT-2).
    /// <para><b>For Beginners:</b> How many different words/tokens the model knows.
    /// Common tokenizers use 32K-50K tokens.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> Width of the hidden representation. Larger = more capacity.
    /// Mamba-130M uses 768, Mamba-1.4B uses 2048.</para>
    /// </param>
    /// <param name="numLayers">
    /// Number of Mamba blocks. Default: 4.
    /// <para><b>For Beginners:</b> Depth of the network. More layers = deeper understanding.
    /// Mamba-130M uses 24 layers, Mamba-1.4B uses 48 layers.</para>
    /// </param>
    /// <param name="stateDimension">
    /// SSM state dimension (N). Default: 16.
    /// <para><b>For Beginners:</b> Memory capacity per position. 16 is the standard default.</para>
    /// </param>
    /// <param name="expandFactor">
    /// Expansion factor for inner dimension. Default: 2.
    /// <para><b>For Beginners:</b> How much the dimension expands inside each Mamba block. 2 is standard.</para>
    /// </param>
    /// <param name="maxSeqLength">Maximum sequence length. Default: 512.</param>
    /// <param name="activationFunction">Optional activation function on final output.</param>
    /// <exception cref="ArgumentException">When parameters are invalid.</exception>
    public MambaLanguageModel(
        int vocabSize,
        int modelDimension = 256,
        int numLayers = 4,
        int stateDimension = 16,
        int expandFactor = 2,
        int maxSeqLength = 512,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [maxSeqLength, vocabSize],
            [maxSeqLength, vocabSize],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (vocabSize <= 0)
            throw new ArgumentException($"Vocab size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0)
            throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (stateDimension <= 0)
            throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));
        if (expandFactor <= 0)
            throw new ArgumentException($"Expand factor ({expandFactor}) must be positive.", nameof(expandFactor));
        if (maxSeqLength <= 0)
            throw new ArgumentException($"Max sequence length ({maxSeqLength}) must be positive.", nameof(maxSeqLength));

        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _expandFactor = expandFactor;

        // Token embedding
        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension });
        InitializeTensor(_embeddingWeights);

        // Mamba blocks
        _blocks = new MambaBlock<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _blocks[i] = new MambaBlock<T>(
                maxSeqLength, modelDimension, stateDimension, expandFactor);
        }

        // Final RMS normalization
        _finalNormGamma = new Tensor<T>(new[] { modelDimension });
        _finalNormGamma.Fill(NumOps.One);

        // LM head
        _lmHeadWeights = new Tensor<T>(new[] { modelDimension, vocabSize });
        InitializeTensor(_lmHeadWeights);
        _lmHeadBias = new Tensor<T>(new[] { vocabSize });
        _lmHeadBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(
                NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int inputDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, inputDim)
            : input.Reshape(batchSize, seqLen, inputDim);

        _lastInput = input3D;

        // Validate input vocab dimension matches embedding
        if (inputDim != _vocabSize)
            throw new ArgumentException(
                $"Input last dimension ({inputDim}) must match vocab size ({_vocabSize}).",
                nameof(input));

        // Step 1: Token embedding (input is one-hot or soft indices -> dense projection)
        // input: [batch, seqLen, vocabSize], embedding: [vocabSize, modelDim]
        var inputFlat = input3D.Reshape(batchSize * seqLen, inputDim);
        var embedded = Engine.TensorMatMul(inputFlat, _embeddingWeights);
        var embedded3D = embedded.Reshape(batchSize, seqLen, _modelDimension);
        _lastEmbedded = embedded3D;

        // Step 2: Pass through Mamba blocks with residual connections
        _lastBlockInputs = new Tensor<T>[_numLayers];
        var current = embedded3D;

        for (int i = 0; i < _numLayers; i++)
        {
            _lastBlockInputs[i] = current;
            var blockOut = _blocks[i].Forward(current);
            current = Engine.TensorAdd(current, blockOut); // Residual connection
        }

        // Step 3: Final RMS normalization
        var normed = ApplyRMSNorm(current, _finalNormGamma, batchSize, seqLen);
        _lastNormedOutput = normed;

        // Step 4: LM head projection to vocab logits
        var normedFlat = normed.Reshape(batchSize * seqLen, _modelDimension);
        var logitsFlat = Engine.TensorMatMul(normedFlat, _lmHeadWeights);
        var bias2D = _lmHeadBias.Reshape(1, _vocabSize);
        logitsFlat = Engine.TensorBroadcastAdd(logitsFlat, bias2D);
        var logits3D = logitsFlat.Reshape(batchSize, seqLen, _vocabSize);

        var result = ApplyActivation(logits3D);
        _lastOutput = result;

        if (rank == 2)
            return result.Reshape(seqLen, _vocabSize);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _vocabSize;
        return result.Reshape(outputShape);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastEmbedded == null ||
            _lastNormedOutput == null || _lastBlockInputs == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int rank = outputGradient.Shape.Length;
        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, seqLen, _vocabSize)
            : outputGradient.Reshape(batchSize, seqLen, _vocabSize);

        grad3D = ApplyActivationDerivative(_lastOutput, grad3D);

        // Step 4 backward: LM head
        var gradFlat = grad3D.Reshape(batchSize * seqLen, _vocabSize);
        _lmHeadBiasGradient = Engine.ReduceSum(grad3D, new int[] { 0, 1 });

        var normedFlat = _lastNormedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _lmHeadWeightsGradient = Engine.TensorMatMul(
            normedFlat.Transpose(new[] { 1, 0 }), gradFlat);

        var dNormed = Engine.TensorMatMul(gradFlat, _lmHeadWeights.Transpose(new[] { 1, 0 }))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 3 backward: RMS norm
        var dPostBlocks = BackwardRMSNorm(dNormed, _lastBlockInputs[_numLayers - 1],
            _finalNormGamma, batchSize, seqLen, out var dFinalGamma);
        _finalNormGammaGradient = dFinalGamma;

        // Step 2 backward: Mamba blocks in reverse
        var current = dPostBlocks;
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            // Residual: gradient flows directly + through block
            var blockGrad = _blocks[i].Backward(current);
            current = Engine.TensorAdd(current, blockGrad);
        }

        // Step 1 backward: Embedding
        var embGradFlat = current.Reshape(batchSize * seqLen, _modelDimension);
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _lastInput.Shape[2]);
        _embeddingWeightsGradient = Engine.TensorMatMul(
            inputFlat.Transpose(new[] { 1, 0 }), embGradFlat);

        var dInputFlat = Engine.TensorMatMul(embGradFlat, _embeddingWeights.Transpose(new[] { 1, 0 }));
        var dInput3D = dInputFlat.Reshape(batchSize, seqLen, _lastInput.Shape[2]);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _lastInput.Shape[2]);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    private Tensor<T> ApplyRMSNorm(Tensor<T> input, Tensor<T> gamma, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(input.Shape);
        T eps = NumOps.FromDouble(1e-6);
        var gamma2D = gamma.Reshape(1, _modelDimension);

        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1);
            var squared = Engine.TensorMultiply(slice, slice);
            var meanSquared = Engine.ReduceSum(squared, new int[] { 1 });
            T divisor = NumOps.FromDouble(_modelDimension);

            var normed = new Tensor<T>(slice.Shape);
            for (int b = 0; b < batchSize; b++)
            {
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(meanSquared[new[] { b }], divisor), eps));
                for (int d = 0; d < _modelDimension; d++)
                {
                    normed[new[] { b, d }] = NumOps.Divide(slice[new[] { b, d }], rms);
                }
            }

            var scaled = Engine.TensorBroadcastMultiply(normed, gamma2D);
            output.SetSlice(1, t, scaled);
        }

        return output;
    }

    private Tensor<T> BackwardRMSNorm(Tensor<T> dOutput, Tensor<T> input, Tensor<T> gamma,
        int batchSize, int seqLen, out Tensor<T> dGamma)
    {
        var dInput = new Tensor<T>(input.Shape);
        dGamma = new Tensor<T>(new[] { _modelDimension });
        T eps = NumOps.FromDouble(1e-6);

        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1);
            var dOut = dOutput.GetSliceAlongDimension(t, 1);

            for (int b = 0; b < batchSize; b++)
            {
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T val = slice[new[] { b, d }];
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));
                }
                T meanSq = NumOps.Divide(sumSq, NumOps.FromDouble(_modelDimension));
                T rms = NumOps.Sqrt(NumOps.Add(meanSq, eps));
                T rmsInv = NumOps.Divide(NumOps.One, rms);

                for (int d = 0; d < _modelDimension; d++)
                {
                    T normed = NumOps.Multiply(slice[new[] { b, d }], rmsInv);
                    dGamma[d] = NumOps.Add(dGamma[d], NumOps.Multiply(dOut[new[] { b, d }], normed));
                }

                T dotProduct = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(dOut[new[] { b, d }],
                            NumOps.Multiply(gamma[d], NumOps.Multiply(slice[new[] { b, d }], rmsInv))));
                }

                T rms3Inv = NumOps.Divide(rmsInv, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++)
                {
                    T g = gamma[d];
                    T grad = NumOps.Multiply(NumOps.Multiply(dOut[new[] { b, d }], g), rmsInv);
                    T correction = NumOps.Multiply(
                        NumOps.Multiply(dotProduct, slice[new[] { b, d }]),
                        NumOps.Divide(rms3Inv, NumOps.FromDouble(_modelDimension)));
                    dInput[new[] { b, t, d }] = NumOps.Subtract(grad, correction);
                }
            }
        }

        return dInput;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;

        for (int i = 0; i < _embeddingWeights.Length; i++)
            parameters[index++] = _embeddingWeights[i];

        foreach (var block in _blocks)
        {
            var blockParams = block.GetParameters();
            for (int i = 0; i < blockParams.Length; i++)
                parameters[index++] = blockParams[i];
        }

        for (int i = 0; i < _finalNormGamma.Length; i++)
            parameters[index++] = _finalNormGamma[i];

        for (int i = 0; i < _lmHeadWeights.Length; i++)
            parameters[index++] = _lmHeadWeights[i];

        for (int i = 0; i < _lmHeadBias.Length; i++)
            parameters[index++] = _lmHeadBias[i];

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = ParameterCount;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;

        for (int i = 0; i < _embeddingWeights.Length; i++)
            _embeddingWeights[i] = parameters[index++];

        foreach (var block in _blocks)
        {
            var blockParams = new Vector<T>(block.ParameterCount);
            for (int i = 0; i < block.ParameterCount; i++)
                blockParams[i] = parameters[index++];
            block.SetParameters(blockParams);
        }

        for (int i = 0; i < _finalNormGamma.Length; i++)
            _finalNormGamma[i] = parameters[index++];

        for (int i = 0; i < _lmHeadWeights.Length; i++)
            _lmHeadWeights[i] = parameters[index++];

        for (int i = 0; i < _lmHeadBias.Length; i++)
            _lmHeadBias[i] = parameters[index++];
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_embeddingWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);

        _embeddingWeights = Engine.TensorAdd(_embeddingWeights,
            Engine.TensorMultiplyScalar(_embeddingWeightsGradient, negLR));

        foreach (var block in _blocks)
            block.UpdateParameters(learningRate);

        _finalNormGamma = Engine.TensorAdd(_finalNormGamma,
            Engine.TensorMultiplyScalar(_finalNormGammaGradient!, negLR));
        _lmHeadWeights = Engine.TensorAdd(_lmHeadWeights,
            Engine.TensorMultiplyScalar(_lmHeadWeightsGradient!, negLR));
        _lmHeadBias = Engine.TensorAdd(_lmHeadBias,
            Engine.TensorMultiplyScalar(_lmHeadBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEmbedded = null;
        _lastNormedOutput = null;
        _lastBlockInputs = null;
        _originalInputShape = null;
        _embeddingWeightsGradient = null;
        _finalNormGammaGradient = null;
        _lmHeadWeightsGradient = null;
        _lmHeadBiasGradient = null;

        _stateCache?.Reset();

        foreach (var block in _blocks)
            block.ResetState();
    }

    /// <summary>
    /// Initializes the state cache for autoregressive generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this before starting token-by-token generation with <see cref="GenerateStep"/>.
    /// The cache stores SSM hidden states so each new token only requires O(1) computation,
    /// rather than re-processing the entire sequence.
    /// </para>
    /// <para><b>For Beginners:</b> Before generating text one word at a time, call this method
    /// to set up the "memory" that will store the model's state between words. Without this,
    /// the model would have to re-read everything from the start for each new word.</para>
    /// </remarks>
    /// <param name="enableCompression">
    /// Whether to compress cached states to save memory. Default: false.
    /// <para><b>For Beginners:</b> Enable this when generating very long sequences to reduce
    /// memory usage at a small cost to precision.</para>
    /// </param>
    /// <param name="compressionBitWidth">Bit width for compressed states. Default: 8.</param>
    /// <returns>The initialized state cache, which is also stored internally.</returns>
    public SSMStateCache<T> InitializeStateCache(bool enableCompression = false, int compressionBitWidth = 8)
    {
        _stateCache = new SSMStateCache<T>(enableCompression, compressionBitWidth);
        for (int i = 0; i < _numLayers; i++)
        {
            _blocks[i].ResetState();
        }
        return _stateCache;
    }

    /// <summary>
    /// Gets the current state cache, or null if not initialized.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns the model's current memory cache used during
    /// token-by-token generation. Null if <see cref="InitializeStateCache"/> hasn't been called.</para>
    /// </remarks>
    public SSMStateCache<T>? StateCache => _stateCache;

    /// <summary>
    /// Performs a single autoregressive generation step, processing one token and returning logits.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method processes a single token through the full model stack, using cached hidden states
    /// from previous steps. The SSM hidden state is updated in-place in the cache after each layer.
    /// This gives O(1) per-token cost compared to O(n) for full sequence reprocessing.
    /// </para>
    /// <para>
    /// The flow for each step:
    /// 1. Embed the token: one-hot → dense vector
    /// 2. For each Mamba layer: retrieve cached state → process token → update cache
    /// 3. Apply final normalization
    /// 4. Project to vocabulary logits
    /// </para>
    /// <para><b>For Beginners:</b> This generates one word at a time. For each word:
    /// 1. Convert the current word to a vector
    /// 2. Pass it through all Mamba layers (using saved memory from previous words)
    /// 3. Output probabilities for what the next word should be
    ///
    /// The key advantage: each new word only takes a fixed amount of time to process,
    /// no matter how many words have been generated before.
    /// </para>
    /// </remarks>
    /// <param name="tokenOneHot">
    /// One-hot encoded token [vocabSize] or [1, vocabSize].
    /// <para><b>For Beginners:</b> The current word represented as a vector where only one position
    /// is 1 (the word's index) and all others are 0.</para>
    /// </param>
    /// <returns>Logits over the vocabulary [vocabSize] representing next-token probabilities.</returns>
    /// <exception cref="InvalidOperationException">If state cache has not been initialized.</exception>
    public Tensor<T> GenerateStep(Tensor<T> tokenOneHot)
    {
        if (_stateCache == null)
            throw new InvalidOperationException(
                "State cache must be initialized before generation. Call InitializeStateCache() first.");

        // Validate input shape
        int tokenDim = tokenOneHot.Shape[tokenOneHot.Rank - 1];
        if (tokenDim != _vocabSize)
            throw new ArgumentException(
                $"Token one-hot last dimension ({tokenDim}) must match vocab size ({_vocabSize}).",
                nameof(tokenOneHot));

        // Normalize input to [1, vocabSize]
        var input2D = tokenOneHot.Rank == 1
            ? tokenOneHot.Reshape(1, _vocabSize)
            : tokenOneHot;

        // Step 1: Embed the token
        var embedded = Engine.TensorMatMul(input2D, _embeddingWeights); // [1, modelDim]

        // Step 2: Process through Mamba blocks with cached state
        var current = embedded;
        for (int i = 0; i < _numLayers; i++)
        {
            var residual = current;

            // Restore cached hidden state into the block before forward
            var cachedState = _stateCache.GetSSMState(i);
            if (cachedState != null)
            {
                _blocks[i].SetHiddenState(cachedState);
            }

            // Reshape for MambaBlock: [1, 1, modelDim] (batch=1, seqLen=1, dim=modelDim)
            var block3D = current.Reshape(1, 1, _modelDimension);
            var blockOut = _blocks[i].Forward(block3D); // [1, 1, modelDim]
            var blockOut2D = blockOut.Reshape(1, _modelDimension);

            // Cache the SSM hidden state from this block for next step
            var blockState = _blocks[i].GetHiddenState();
            if (blockState != null)
            {
                // Extract the final timestep state: [batch, innerDim, stateDim]
                // Hidden states shape is [batch, seqLen+1, innerDim, stateDim], take last timestep
                var finalState = blockState.GetSliceAlongDimension(blockState.Shape[1] - 1, 1);
                _stateCache.CacheSSMState(i, finalState);
            }

            // Residual connection
            current = Engine.TensorAdd(residual, blockOut2D);
        }

        // Step 3: Final RMS normalization on [1, modelDim]
        var normed = ApplyRMSNorm(current.Reshape(1, 1, _modelDimension),
            _finalNormGamma, 1, 1);
        var normed2D = normed.Reshape(1, _modelDimension);

        // Step 4: LM head projection
        var logits = Engine.TensorMatMul(normed2D, _lmHeadWeights); // [1, vocabSize]
        var bias2D = _lmHeadBias.Reshape(1, _vocabSize);
        logits = Engine.TensorBroadcastAdd(logits, bias2D);

        var result = ApplyActivation(logits);
        return result.Reshape(_vocabSize);
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Input: [1, vocabSize] one-hot token
        var inputPlaceholder = new Tensor<T>(new int[] { 1, _vocabSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "token_input");
        inputNodes.Add(inputNode);

        // Embedding lookup (matmul with embedding matrix)
        var embWeightsNode = TensorOperations<T>.Variable(_embeddingWeights, "W_emb");
        inputNodes.Add(embWeightsNode);
        var embedded = TensorOperations<T>.MatrixMultiply(inputNode, embWeightsNode);

        // Chain Mamba block computation graphs
        var current = embedded;
        for (int i = 0; i < _numLayers; i++)
        {
            var blockInputs = new List<ComputationNode<T>> { current };
            var blockOutput = _blocks[i].ExportComputationGraph(blockInputs);
            inputNodes.AddRange(blockInputs.GetRange(1, blockInputs.Count - 1));

            // Residual connection
            current = TensorOperations<T>.Add(current, blockOutput);
        }

        // LM head projection
        var lmWeightsNode = TensorOperations<T>.Variable(_lmHeadWeights, "W_lm");
        var lmBiasNode = TensorOperations<T>.Variable(_lmHeadBias, "b_lm");
        inputNodes.Add(lmWeightsNode);
        inputNodes.Add(lmBiasNode);

        var logits = TensorOperations<T>.MatrixMultiply(current, lmWeightsNode);
        var output = TensorOperations<T>.Add(logits, lmBiasNode);

        return output;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        metadata["ExpandFactor"] = _expandFactor.ToString();
        return metadata;
    }
}
