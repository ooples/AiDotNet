using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full RWKV-6 "Finch" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// This assembles the complete RWKV-6 "Finch" architecture. Finch extends Eagle (RWKV-5) with
/// data-dependent token shifting via a LoRA-based mechanism, allowing the model to dynamically
/// adjust how much to blend current and previous tokens based on the input content:
/// <code>
///   1. Token Embedding: token indices -> dense vectors [batch, seqLen, modelDim]
///   2. N x RWKVLayer: multi-head WKV attention with dynamic token shift + channel mixing
///   3. RMS Normalization: final layer normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// RWKV-6 "Finch" key innovations over RWKV-5 "Eagle":
/// <list type="bullet">
///   <item><b>Data-dependent token shifting</b>: The mixing coefficients mu_r, mu_k, mu_v are no longer
///     fixed learned values but are computed from the input via a LoRA-style projection:
///     mu_x = sigmoid(LoRA_down(LoRA_up(x_t))). This allows the model to dynamically decide
///     how much context from the previous token to incorporate.</item>
///   <item><b>Data-dependent time decay</b>: The exponential decay factor w is also input-dependent,
///     projected from the current token through a learned linear transformation.</item>
///   <item><b>Improved training stability</b>: Better gradient flow through the dynamic mechanisms.</item>
///   <item><b>Same linear complexity</b>: Despite the added expressiveness, Finch maintains O(n) time
///     and O(1) per-token memory during generation.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Finch is the most advanced version of RWKV before Goose (v7).
///
/// How it works:
/// 1. Each word is converted to a vector (embedding)
/// 2. Multiple Finch layers process the vectors, each deciding dynamically how much to
///    pay attention to the current word vs. the previous word
/// 3. This dynamic attention is what sets Finch apart: unlike Eagle where the blending
///    weights are fixed, Finch adjusts them based on what it sees
/// 4. The output is probabilities for what the next word should be
///
/// Why the dynamic shifting matters:
/// - Some words need a lot of context from the previous word (e.g., "New" before "York")
/// - Other words are more independent (e.g., punctuation)
/// - Finch learns to adjust its blending for each situation automatically
///
/// Real-world examples: RWKV-6 Finch models from 1.6B to 7.5B parameters.
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
/// https://arxiv.org/abs/2404.05892
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FinchLanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;

    // Token embedding: [vocabSize, modelDim]
    private Tensor<T> _embeddingWeights;

    // Stack of RWKV layers configured for Finch (multi-head + data-dependent shifts)
    private readonly RWKVLayer<T>[] _blocks;

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
    private Tensor<T>? _lastPostBlocksOutput;
    private Tensor<T>[]? _lastBlockInputs;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _embeddingWeightsGradient;
    private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _lmHeadWeightsGradient;
    private Tensor<T>? _lmHeadBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Finch blocks.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Gets the number of attention heads per block.</summary>
    public int NumHeads => _numHeads;

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
    /// Creates a new RWKV-6 "Finch" language model.
    /// </summary>
    /// <param name="vocabSize">
    /// Size of the token vocabulary. Typical: 65536 for RWKV-6 models.
    /// <para><b>For Beginners:</b> How many different words/tokens the model knows.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> Width of the hidden representation. Finch 1.6B uses 2048,
    /// 3B uses 2560, 7.5B uses 4096.</para>
    /// </param>
    /// <param name="numLayers">
    /// Number of RWKV blocks. Default: 4.
    /// <para><b>For Beginners:</b> Depth of the network. Finch 1.6B uses 24 layers,
    /// 3B uses 32, 7.5B uses 32.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads per block. Default: 8. Must divide modelDimension.
    /// <para><b>For Beginners:</b> Multiple heads track different information types. Head dimension
    /// is typically 64 for Finch models.</para>
    /// </param>
    /// <param name="maxSeqLength">Maximum sequence length. Default: 512.</param>
    /// <param name="activationFunction">Optional activation function on final output.</param>
    public FinchLanguageModel(
        int vocabSize,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 8,
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
        if (maxSeqLength <= 0)
            throw new ArgumentException($"Max sequence length ({maxSeqLength}) must be positive.", nameof(maxSeqLength));

        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;

        // Token embedding
        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension });
        InitializeTensor(_embeddingWeights);

        // Finch layers: multi-headed with data-dependent mixing (RWKVLayer supports this)
        _blocks = new RWKVLayer<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _blocks[i] = new RWKVLayer<T>(
                maxSeqLength, modelDimension, numHeads);
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
        if (rank < 2)
            throw new ArgumentException(
                $"Input must be at least rank 2 [seqLen, vocabSize], got rank {rank}.",
                nameof(input));
        int seqLen = input.Shape[rank - 2];
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

        // Step 2: Pass through Finch blocks (each has internal residual connections)
        _lastBlockInputs = new Tensor<T>[_numLayers];
        var current = embedded3D;

        for (int i = 0; i < _numLayers; i++)
        {
            _lastBlockInputs[i] = current;
            current = _blocks[i].Forward(current);
        }

        // Step 3: Final RMS normalization
        _lastPostBlocksOutput = current;
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

        // Step 2 backward: Finch blocks in reverse
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
                            NumOps.Multiply(gamma[d], slice[new[] { b, d }])));
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
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "finch_token_input");
        inputNodes.Add(inputNode);

        var embWeightsNode = TensorOperations<T>.Variable(_embeddingWeights, "finch_W_emb");
        inputNodes.Add(embWeightsNode);
        var embedded = TensorOperations<T>.MatrixMultiply(inputNode, embWeightsNode);

        var current = embedded;
        for (int i = 0; i < _numLayers; i++)
        {
            var blockInputs = new List<ComputationNode<T>> { current };
            var blockOutput = _blocks[i].ExportComputationGraph(blockInputs);
            inputNodes.AddRange(blockInputs.GetRange(1, blockInputs.Count - 1));
            current = blockOutput;
        }

        var finalNormGammaNode = TensorOperations<T>.Variable(_finalNormGamma, "finch_final_norm_gamma");
        inputNodes.Add(finalNormGammaNode);
        var normed = TensorOperations<T>.ElementwiseMultiply(current, finalNormGammaNode);

        var lmWeightsNode = TensorOperations<T>.Variable(_lmHeadWeights, "finch_W_lm");
        var lmBiasNode = TensorOperations<T>.Variable(_lmHeadBias, "finch_b_lm");
        inputNodes.Add(lmWeightsNode);
        inputNodes.Add(lmBiasNode);

        var logits = TensorOperations<T>.MatrixMultiply(normed, lmWeightsNode);
        return TensorOperations<T>.Add(logits, lmBiasNode);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["Architecture"] = "RWKV-6-Finch";
        return metadata;
    }
}
