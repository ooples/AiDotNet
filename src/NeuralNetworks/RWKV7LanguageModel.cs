using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full RWKV-7 "Goose" language model: token embedding + N RWKV7Blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// This assembles the complete RWKV-7 architecture as described in the paper:
/// <code>
///   1. Token Embedding: token indices -> dense vectors [batch, seqLen, modelDim]
///   2. N x RWKV7Block: WKV-7 time mixing + SiLU channel mixing with residual connections
///   3. RMS Normalization: final layer normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// The model supports two execution modes:
/// <list type="bullet">
///   <item><b>Parallel mode</b>: process full sequences for training (Forward/Backward)</item>
///   <item><b>Sequential mode</b>: process one token at a time for inference (GenerateStep),
///     using cached recurrent states for O(1) per-token cost</item>
/// </list>
/// </para>
/// <para>
/// RWKV-7 key innovations over previous versions:
/// <list type="bullet">
///   <item>Dynamic state evolution with learnable transition matrices a_t, b_t</item>
///   <item>State: S_t = diag(sigmoid(a_t)) * S_{t-1} + sigmoid(b_t) * outer(k_t, v_t)</item>
///   <item>Group normalization on WKV output for training stability</item>
///   <item>SiLU channel mixing replacing squared ReLU</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> This is a complete text generation model built from RWKV-7 blocks.
///
/// How it works:
/// 1. Each word/token is converted to a vector of numbers (embedding)
/// 2. These vectors pass through several RWKV-7 layers that understand context
/// 3. The final output is a probability distribution over all possible next words
///
/// What makes it special:
/// - Linear time complexity: processes longer text without quadratic slowdown
/// - Constant memory per token during generation (via recurrent state caching)
/// - Dynamic memory management: the model learns when to remember and forget
/// - Competitive quality with Transformer models of similar size
///
/// Real-world usage: RWKV-7 "Goose" models range from 0.1B to 7B+ parameters.
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RWKV7LanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly double _ffnMultiplier;

    // Token embedding: [vocabSize, modelDim]
    private Tensor<T> _embeddingWeights;

    // Stack of RWKV-7 blocks
    private readonly RWKV7Block<T>[] _blocks;

    // Final RMS normalization
    private Tensor<T> _finalNormGamma;

    // LM head (output projection): [modelDim, vocabSize]
    private Tensor<T> _lmHeadWeights;
    private Tensor<T> _lmHeadBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEmbedded;
    private Tensor<T>? _lastNormedOutput;
    private Tensor<T>? _lastPostBlocksOutput;  // Output after all blocks, input to final RMS norm
    private Tensor<T>[]? _lastBlockInputs;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _embeddingWeightsGradient;
    private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _lmHeadWeightsGradient;
    private Tensor<T>? _lmHeadBiasGradient;

    // Whether state cache is active for sequential generation
    private bool _isGenerating;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of RWKV-7 blocks.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Gets the number of attention heads per block.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the FFN expansion multiplier.</summary>
    public double FFNMultiplier => _ffnMultiplier;

    /// <summary>Gets whether the model is in sequential generation mode.</summary>
    public bool IsGenerating => _isGenerating;

    /// <inheritdoc />
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
    /// Creates a new RWKV-7 "Goose" language model.
    /// </summary>
    /// <param name="vocabSize">
    /// Size of the token vocabulary. Typical: 65536 for RWKV-7 models.
    /// <para><b>For Beginners:</b> How many different words/tokens the model knows.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> Width of the hidden representation. RWKV-7 0.1B uses 768,
    /// 1.5B uses 2048, 7B uses 4096.</para>
    /// </param>
    /// <param name="numLayers">
    /// Number of RWKV-7 blocks. Default: 4.
    /// <para><b>For Beginners:</b> Depth of the network. RWKV-7 0.1B uses 12 layers,
    /// 1.5B uses 24, 7B uses 32.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of heads per block. Default: 4. Must divide modelDimension.
    /// <para><b>For Beginners:</b> Splits the hidden representation into multiple "perspectives"
    /// for the recurrent state. Head size is typically 64.</para>
    /// </param>
    /// <param name="ffnMultiplier">
    /// FFN expansion multiplier. Default: 3.5 (RWKV-7 standard).
    /// <para><b>For Beginners:</b> How much wider the feed-forward network is compared to the
    /// model dimension. 3.5x is the RWKV-7 default.</para>
    /// </param>
    /// <param name="maxSeqLength">Maximum sequence length. Default: 512.</param>
    /// <param name="activationFunction">Optional activation function on final output.</param>
    public RWKV7LanguageModel(
        int vocabSize,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 4,
        double ffnMultiplier = 3.5,
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
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (ffnMultiplier <= 0)
            throw new ArgumentException($"FFN multiplier ({ffnMultiplier}) must be positive.", nameof(ffnMultiplier));
        if (maxSeqLength <= 0)
            throw new ArgumentException($"Max sequence length ({maxSeqLength}) must be positive.", nameof(maxSeqLength));

        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _ffnMultiplier = ffnMultiplier;

        // Token embedding
        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension });
        InitializeTensor(_embeddingWeights);

        // RWKV-7 blocks
        _blocks = new RWKV7Block<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _blocks[i] = new RWKV7Block<T>(
                maxSeqLength, modelDimension, numHeads, ffnMultiplier);
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
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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

        if (inputDim != _vocabSize)
            throw new ArgumentException(
                $"Input last dimension ({inputDim}) must match vocab size ({_vocabSize}).",
                nameof(input));

        // Step 1: Token embedding
        var inputFlat = input3D.Reshape(batchSize * seqLen, inputDim);
        var embedded = Engine.TensorMatMul(inputFlat, _embeddingWeights);
        var embedded3D = embedded.Reshape(batchSize, seqLen, _modelDimension);
        _lastEmbedded = embedded3D;

        // Step 2: Pass through RWKV-7 blocks with residual connections
        _lastBlockInputs = new Tensor<T>[_numLayers];
        var current = embedded3D;

        for (int i = 0; i < _numLayers; i++)
        {
            _lastBlockInputs[i] = current;
            var blockOut = _blocks[i].Forward(current);
            // RWKV blocks have internal residual connections, but we also keep
            // the option for an outer residual if the block output has same shape
            current = blockOut;
        }

        // Step 3: Final RMS normalization
        _lastPostBlocksOutput = current;  // Cache for BackwardRMSNorm
        var normed = ApplyRMSNorm(current, _finalNormGamma, batchSize, seqLen);
        _lastNormedOutput = normed;

        // Step 4: LM head projection
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
            _lastNormedOutput == null || _lastPostBlocksOutput == null || _lastBlockInputs == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

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
        var dPostBlocks = BackwardRMSNorm(dNormed, _lastPostBlocksOutput,
            _finalNormGamma, batchSize, seqLen, out var dFinalGamma);
        _finalNormGammaGradient = dFinalGamma;

        // Step 2 backward: RWKV-7 blocks in reverse
        var current = dPostBlocks;
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            current = _blocks[i].Backward(current);
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
                    normed[new[] { b, d }] = NumOps.Divide(slice[new[] { b, d }], rms);
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

    /// <summary>
    /// Initializes the model for sequential token-by-token generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this before starting generation with <see cref="GenerateStep"/>.
    /// Resets all block recurrent states so generation starts fresh.
    /// </para>
    /// <para><b>For Beginners:</b> Before generating text one word at a time, call this
    /// to prepare the model's memory. Each block will start with a blank slate.</para>
    /// </remarks>
    public void InitializeGeneration()
    {
        _isGenerating = true;
        foreach (var block in _blocks)
        {
            block.SetRecurrentState(null);
            block.SetPreviousToken(null);
        }
    }

    /// <summary>
    /// Performs a single autoregressive generation step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Processes a single token through the full model stack, using each block's cached recurrent
    /// state from the previous step. This gives O(1) per-token cost.
    /// </para>
    /// <para><b>For Beginners:</b> Generates one word at a time. For each word:
    /// 1. Convert the current word to a vector
    /// 2. Pass through all RWKV-7 blocks (using saved memory from previous words)
    /// 3. Output probabilities for what the next word should be
    /// </para>
    /// </remarks>
    /// <param name="tokenOneHot">
    /// One-hot encoded token [vocabSize] or [1, vocabSize].
    /// </param>
    /// <returns>Logits over the vocabulary [vocabSize].</returns>
    public Tensor<T> GenerateStep(Tensor<T> tokenOneHot)
    {
        if (!_isGenerating)
            throw new InvalidOperationException(
                "Generation mode must be initialized. Call InitializeGeneration() first.");

        int tokenDim = tokenOneHot.Shape[tokenOneHot.Rank - 1];
        if (tokenDim != _vocabSize)
            throw new ArgumentException(
                $"Token one-hot last dimension ({tokenDim}) must match vocab size ({_vocabSize}).",
                nameof(tokenOneHot));

        var input2D = tokenOneHot.Rank == 1
            ? tokenOneHot.Reshape(1, _vocabSize)
            : tokenOneHot;

        // Step 1: Embed
        var embedded = Engine.TensorMatMul(input2D, _embeddingWeights);  // [1, modelDim]

        // Step 2: Process through RWKV-7 blocks (each maintains its own recurrent state)
        var current = embedded.Reshape(1, 1, _modelDimension);  // [1, 1, modelDim]
        for (int i = 0; i < _numLayers; i++)
        {
            current = _blocks[i].Forward(current);
        }

        // Step 3: Final RMS norm
        var normed = ApplyRMSNorm(current, _finalNormGamma, 1, 1);
        var normed2D = normed.Reshape(1, _modelDimension);

        // Step 4: LM head
        var logits = Engine.TensorMatMul(normed2D, _lmHeadWeights);
        var bias2D = _lmHeadBias.Reshape(1, _vocabSize);
        logits = Engine.TensorBroadcastAdd(logits, bias2D);

        var result = ApplyActivation(logits);
        return result.Reshape(_vocabSize);
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

        if (_finalNormGammaGradient != null)
            _finalNormGamma = Engine.TensorAdd(_finalNormGamma,
                Engine.TensorMultiplyScalar(_finalNormGammaGradient, negLR));
        if (_lmHeadWeightsGradient != null)
            _lmHeadWeights = Engine.TensorAdd(_lmHeadWeights,
                Engine.TensorMultiplyScalar(_lmHeadWeightsGradient, negLR));
        if (_lmHeadBiasGradient != null)
            _lmHeadBias = Engine.TensorAdd(_lmHeadBias,
                Engine.TensorMultiplyScalar(_lmHeadBiasGradient, negLR));
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
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEmbedded = null;
        _lastNormedOutput = null;
        _lastPostBlocksOutput = null;
        _lastBlockInputs = null;
        _originalInputShape = null;
        _embeddingWeightsGradient = null;
        _finalNormGammaGradient = null;
        _lmHeadWeightsGradient = null;
        _lmHeadBiasGradient = null;
        _isGenerating = false;

        foreach (var block in _blocks)
            block.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var inputPlaceholder = new Tensor<T>(new int[] { 1, _vocabSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "rwkv7_token_input");
        inputNodes.Add(inputNode);

        var embWeightsNode = TensorOperations<T>.Variable(_embeddingWeights, "rwkv7_W_emb");
        inputNodes.Add(embWeightsNode);
        var embedded = TensorOperations<T>.MatrixMultiply(inputNode, embWeightsNode);

        var current = embedded;
        for (int i = 0; i < _numLayers; i++)
        {
            var blockInputs = new List<ComputationNode<T>> { current };
            var blockOutput = _blocks[i].ExportComputationGraph(blockInputs);
            inputNodes.AddRange(blockInputs.GetRange(1, blockInputs.Count - 1));
            current = blockOutput;  // Blocks have internal residuals; no outer residual in Forward
        }

        var lmWeightsNode = TensorOperations<T>.Variable(_lmHeadWeights, "rwkv7_W_lm");
        var lmBiasNode = TensorOperations<T>.Variable(_lmHeadBias, "rwkv7_b_lm");
        inputNodes.Add(lmWeightsNode);
        inputNodes.Add(lmBiasNode);

        var logits = TensorOperations<T>.MatrixMultiply(current, lmWeightsNode);
        return TensorOperations<T>.Add(logits, lmBiasNode);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["FFNMultiplier"] = _ffnMultiplier.ToString("F1");
        metadata["Architecture"] = "RWKV-7-Goose";
        return metadata;
    }
}
