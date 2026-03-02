using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full Mamba-2 language model: token embedding + N Mamba2Blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Mamba-2 improves upon the original Mamba architecture by replacing the selective scan with a
/// structured state space duality (SSD) formulation that enables more efficient hardware utilization:
/// <code>
///   1. Token Embedding: token indices -> dense vectors [batch, seqLen, modelDim]
///   2. N x Mamba2Block: SSD-based selective scan with multi-head structure
///   3. RMS Normalization: final layer normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// Mamba-2 key innovations over Mamba-1:
/// <list type="bullet">
///   <item><b>Structured State Space Duality (SSD)</b>: Reformulates the selective scan as a structured
///     matrix multiplication, enabling 2-8x faster training on modern hardware</item>
///   <item><b>Multi-head SSM</b>: Splits the state into multiple heads for parallel processing</item>
///   <item><b>Larger state dimension</b>: Typically N=64-128 vs N=16 in Mamba-1</item>
///   <item><b>Chunk-wise parallel scan</b>: Processes sequences in chunks for better GPU utilization</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Mamba-2 is a faster version of Mamba that processes text more efficiently.
///
/// Think of it like upgrading from a single-lane highway to a multi-lane highway:
/// - Mamba-1 processes everything in one stream
/// - Mamba-2 splits the work into multiple parallel lanes (heads)
/// - Each lane handles part of the information independently
/// - This makes training 2-8x faster while maintaining the same quality
///
/// Real-world examples: Mamba-2 models from 130M to 2.7B parameters, Codestral Mamba 7B.
/// </para>
/// <para>
/// <b>Reference:</b> Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
/// Through Structured State Space Duality", 2024. https://arxiv.org/abs/2405.21060
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Mamba2LanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _numHeads;

    private Tensor<T> _embeddingWeights;
    private readonly Mamba2Block<T>[] _blocks;
    private Tensor<T> _finalNormGamma;
    private Tensor<T> _lmHeadWeights;
    private Tensor<T> _lmHeadBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEmbedded;
    private Tensor<T>? _lastNormedOutput;
    private Tensor<T>? _lastPostBlocksOutput;
    private Tensor<T>[]? _lastBlockInputs;
    private int[]? _originalInputShape;

    private Tensor<T>? _embeddingWeightsGradient;
    private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _lmHeadWeightsGradient;
    private Tensor<T>? _lmHeadBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Mamba-2 blocks.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Gets the SSM state dimension.</summary>
    public int StateDimension => _stateDimension;

    /// <summary>Gets the number of SSM heads.</summary>
    public int NumHeads => _numHeads;

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            int count = _embeddingWeights.Length;
            foreach (var block in _blocks) count += block.ParameterCount;
            count += _finalNormGamma.Length + _lmHeadWeights.Length + _lmHeadBias.Length;
            return count;
        }
    }

    /// <summary>
    /// Creates a new Mamba-2 language model.
    /// </summary>
    public Mamba2LanguageModel(
        int vocabSize,
        int modelDimension = 256,
        int numLayers = 4,
        int stateDimension = 64,
        int numHeads = 8,
        int maxSeqLength = 512,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [maxSeqLength, vocabSize],
            [maxSeqLength, vocabSize],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException($"Vocab size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (stateDimension <= 0) throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));
        if (numHeads <= 0) throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));

        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _numHeads = numHeads;

        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension });
        InitializeTensor(_embeddingWeights);

        _blocks = new Mamba2Block<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
            _blocks[i] = new Mamba2Block<T>(maxSeqLength, modelDimension, stateDimension, numHeads);

        _finalNormGamma = new Tensor<T>(new[] { modelDimension });
        _finalNormGamma.Fill(NumOps.One);

        _lmHeadWeights = new Tensor<T>(new[] { modelDimension, vocabSize });
        InitializeTensor(_lmHeadWeights);
        _lmHeadBias = new Tensor<T>(new[] { vocabSize });
        _lmHeadBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        int fanIn = tensor.Shape[0], fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;
        if (rank < 2) throw new ArgumentException($"Input must be at least rank 2, got rank {rank}.", nameof(input));
        int seqLen = input.Shape[rank - 2], inputDim = input.Shape[rank - 1];
        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++) batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2 ? input.Reshape(1, seqLen, inputDim) : input.Reshape(batchSize, seqLen, inputDim);
        _lastInput = input3D;
        if (inputDim != _vocabSize) throw new ArgumentException($"Input last dimension ({inputDim}) must match vocab size ({_vocabSize}).", nameof(input));

        var inputFlat = input3D.Reshape(batchSize * seqLen, inputDim);
        var embedded = Engine.TensorMatMul(inputFlat, _embeddingWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastEmbedded = embedded;

        _lastBlockInputs = new Tensor<T>[_numLayers];
        var current = embedded;
        for (int i = 0; i < _numLayers; i++)
        {
            _lastBlockInputs[i] = current;
            var blockOut = _blocks[i].Forward(current);
            current = Engine.TensorAdd(current, blockOut);
        }

        _lastPostBlocksOutput = current;
        var normed = ApplyRMSNorm(current, _finalNormGamma, batchSize, seqLen);
        _lastNormedOutput = normed;

        var normedFlat = normed.Reshape(batchSize * seqLen, _modelDimension);
        var logitsFlat = Engine.TensorMatMul(normedFlat, _lmHeadWeights);
        logitsFlat = Engine.TensorBroadcastAdd(logitsFlat, _lmHeadBias.Reshape(1, _vocabSize));
        var result = ApplyActivation(logitsFlat.Reshape(batchSize, seqLen, _vocabSize));
        _lastOutput = result;

        if (rank == 2) return result.Reshape(seqLen, _vocabSize);
        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++) outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen; outputShape[rank - 1] = _vocabSize;
        return result.Reshape(outputShape);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastEmbedded == null ||
            _lastNormedOutput == null || _lastPostBlocksOutput == null || _lastBlockInputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0], seqLen = _lastInput.Shape[1];
        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, seqLen, _vocabSize) : outputGradient.Reshape(batchSize, seqLen, _vocabSize);
        grad3D = ApplyActivationDerivative(_lastOutput, grad3D);

        var gradFlat = grad3D.Reshape(batchSize * seqLen, _vocabSize);
        _lmHeadBiasGradient = Engine.ReduceSum(grad3D, new int[] { 0, 1 });
        var normedFlat = _lastNormedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _lmHeadWeightsGradient = Engine.TensorMatMul(normedFlat.Transpose(new[] { 1, 0 }), gradFlat);

        var dNormed = Engine.TensorMatMul(gradFlat, _lmHeadWeights.Transpose(new[] { 1, 0 })).Reshape(batchSize, seqLen, _modelDimension);
        var dPostBlocks = BackwardRMSNorm(dNormed, _lastPostBlocksOutput, _finalNormGamma, batchSize, seqLen, out var dFinalGamma);
        _finalNormGammaGradient = dFinalGamma;

        var current = dPostBlocks;
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            var blockGrad = _blocks[i].Backward(current);
            current = Engine.TensorAdd(current, blockGrad);
        }

        var embGradFlat = current.Reshape(batchSize * seqLen, _modelDimension);
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _lastInput.Shape[2]);
        _embeddingWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose(new[] { 1, 0 }), embGradFlat);
        var dInputFlat = Engine.TensorMatMul(embGradFlat, _embeddingWeights.Transpose(new[] { 1, 0 }));
        var dInput3D = dInputFlat.Reshape(batchSize, seqLen, _lastInput.Shape[2]);

        if (_originalInputShape != null && _originalInputShape.Length == 2) return dInput3D.Reshape(seqLen, _lastInput.Shape[2]);
        if (_originalInputShape != null) return dInput3D.Reshape(_originalInputShape);
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
            output.SetSlice(1, t, Engine.TensorBroadcastMultiply(normed, gamma2D));
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
                for (int d = 0; d < _modelDimension; d++) { T val = slice[new[] { b, d }]; sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(sumSq, NumOps.FromDouble(_modelDimension)), eps));
                T rmsInv = NumOps.Divide(NumOps.One, rms);
                for (int d = 0; d < _modelDimension; d++)
                    dGamma[d] = NumOps.Add(dGamma[d], NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(slice[new[] { b, d }], rmsInv)));
                T dotProduct = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(gamma[d], slice[new[] { b, d }])));
                T rms3Inv = NumOps.Divide(rmsInv, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++)
                {
                    T grad = NumOps.Multiply(NumOps.Multiply(dOut[new[] { b, d }], gamma[d]), rmsInv);
                    T correction = NumOps.Multiply(NumOps.Multiply(dotProduct, slice[new[] { b, d }]), NumOps.Divide(rms3Inv, NumOps.FromDouble(_modelDimension)));
                    dInput[new[] { b, t, d }] = NumOps.Subtract(grad, correction);
                }
            }
        }
        return dInput;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_embeddingWeightsGradient == null) throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        T negLR = NumOps.Negate(learningRate);
        _embeddingWeights = Engine.TensorAdd(_embeddingWeights, Engine.TensorMultiplyScalar(_embeddingWeightsGradient, negLR));
        foreach (var block in _blocks) block.UpdateParameters(learningRate);
        if (_finalNormGammaGradient != null) _finalNormGamma = Engine.TensorAdd(_finalNormGamma, Engine.TensorMultiplyScalar(_finalNormGammaGradient, negLR));
        if (_lmHeadWeightsGradient != null) _lmHeadWeights = Engine.TensorAdd(_lmHeadWeights, Engine.TensorMultiplyScalar(_lmHeadWeightsGradient, negLR));
        if (_lmHeadBiasGradient != null) _lmHeadBias = Engine.TensorAdd(_lmHeadBias, Engine.TensorMultiplyScalar(_lmHeadBiasGradient, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) parameters[index++] = _embeddingWeights[i];
        foreach (var block in _blocks) { var bp = block.GetParameters(); for (int i = 0; i < bp.Length; i++) parameters[index++] = bp[i]; }
        for (int i = 0; i < _finalNormGamma.Length; i++) parameters[index++] = _finalNormGamma[i];
        for (int i = 0; i < _lmHeadWeights.Length; i++) parameters[index++] = _lmHeadWeights[i];
        for (int i = 0; i < _lmHeadBias.Length; i++) parameters[index++] = _lmHeadBias[i];
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount) throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int index = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) _embeddingWeights[i] = parameters[index++];
        foreach (var block in _blocks) { var bp = new Vector<T>(block.ParameterCount); for (int i = 0; i < block.ParameterCount; i++) bp[i] = parameters[index++]; block.SetParameters(bp); }
        for (int i = 0; i < _finalNormGamma.Length; i++) _finalNormGamma[i] = parameters[index++];
        for (int i = 0; i < _lmHeadWeights.Length; i++) _lmHeadWeights[i] = parameters[index++];
        for (int i = 0; i < _lmHeadBias.Length; i++) _lmHeadBias[i] = parameters[index++];
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null; _lastOutput = null; _lastEmbedded = null;
        _lastNormedOutput = null; _lastPostBlocksOutput = null; _lastBlockInputs = null;
        _originalInputShape = null; _embeddingWeightsGradient = null;
        _finalNormGammaGradient = null; _lmHeadWeightsGradient = null; _lmHeadBiasGradient = null;
        foreach (var block in _blocks) block.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
        var inputNode = TensorOperations<T>.Variable(new Tensor<T>(new int[] { 1, _vocabSize }), "mamba2_token_input");
        inputNodes.Add(inputNode);
        var embNode = TensorOperations<T>.Variable(_embeddingWeights, "mamba2_W_emb");
        inputNodes.Add(embNode);
        var current = TensorOperations<T>.MatrixMultiply(inputNode, embNode);
        for (int i = 0; i < _numLayers; i++)
        {
            var blockInputs = new List<ComputationNode<T>> { current };
            var blockOutput = _blocks[i].ExportComputationGraph(blockInputs);
            inputNodes.AddRange(blockInputs.GetRange(1, blockInputs.Count - 1));
            current = TensorOperations<T>.Add(current, blockOutput);
        }
        var normNode = TensorOperations<T>.Variable(_finalNormGamma, "mamba2_final_norm");
        inputNodes.Add(normNode);
        current = TensorOperations<T>.ElementwiseMultiply(current, normNode);
        var lmW = TensorOperations<T>.Variable(_lmHeadWeights, "mamba2_W_lm");
        var lmB = TensorOperations<T>.Variable(_lmHeadBias, "mamba2_b_lm");
        inputNodes.Add(lmW); inputNodes.Add(lmB);
        return TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(current, lmW), lmB);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["StateDimension"] = _stateDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["Architecture"] = "Mamba-2";
        return metadata;
    }
}
