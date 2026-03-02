using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full xLSTM language model: token embedding + N ExtendedLSTMLayer blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// xLSTM (Extended LSTM) modernizes the classic LSTM architecture with exponential gating, new memory
/// structures, and residual block stacking to achieve competitive language modeling performance:
/// <code>
///   1. Token Embedding: token indices -> dense vectors [batch, seqLen, modelDim]
///   2. N x ExtendedLSTMLayer: sLSTM/mLSTM blocks with exponential gating and matrix memory
///   3. RMS Normalization: final layer normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// xLSTM key innovations:
/// <list type="bullet">
///   <item><b>Exponential gating</b>: Replaces sigmoid gates with exponential gates for stronger gradient flow</item>
///   <item><b>sLSTM</b>: Scalar LSTM with new memory mixing and exponential gates</item>
///   <item><b>mLSTM</b>: Matrix LSTM with matrix-valued cell state for richer memory</item>
///   <item><b>Residual stacking</b>: Pre-normalization residual blocks for deep networks</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> xLSTM is a modern version of the classic LSTM that was the standard
/// for sequence modeling before Transformers. By adding stronger gates and richer memory, xLSTM
/// achieves quality competitive with Transformers and Mamba while maintaining linear-time inference.
/// </para>
/// <para>
/// <b>Reference:</b> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
/// https://arxiv.org/abs/2405.04517
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class XLSTMLanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;

    private Tensor<T> _embeddingWeights;
    private readonly ExtendedLSTMLayer<T>[] _blocks;
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
    /// <summary>Gets the number of xLSTM blocks.</summary>
    public int NumLayers => _numLayers;
    /// <summary>Gets the number of heads.</summary>
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
    /// Creates a new xLSTM language model.
    /// </summary>
    public XLSTMLanguageModel(
        int vocabSize,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 8,
        int maxSeqLength = 512,
        IActivationFunction<T>? activationFunction = null)
        : base([maxSeqLength, vocabSize], [maxSeqLength, vocabSize],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException($"Vocab size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (numHeads <= 0) throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));

        _vocabSize = vocabSize; _modelDimension = modelDimension; _numLayers = numLayers; _numHeads = numHeads;

        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension });
        InitializeTensor(_embeddingWeights);

        _blocks = new ExtendedLSTMLayer<T>[numLayers];
        for (int i = 0; i < numLayers; i++)
            _blocks[i] = new ExtendedLSTMLayer<T>(maxSeqLength, modelDimension, numHeads);

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

        var embedded = Engine.TensorMatMul(input3D.Reshape(batchSize * seqLen, inputDim), _embeddingWeights).Reshape(batchSize, seqLen, _modelDimension);
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

        var logitsFlat = Engine.TensorMatMul(normed.Reshape(batchSize * seqLen, _modelDimension), _lmHeadWeights);
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
        if (_lastInput == null || _lastOutput == null || _lastNormedOutput == null || _lastPostBlocksOutput == null || _lastBlockInputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        int batchSize = _lastInput.Shape[0], seqLen = _lastInput.Shape[1];
        var grad3D = outputGradient.Rank == 2 ? outputGradient.Reshape(1, seqLen, _vocabSize) : outputGradient.Reshape(batchSize, seqLen, _vocabSize);
        grad3D = ApplyActivationDerivative(_lastOutput, grad3D);

        var gradFlat = grad3D.Reshape(batchSize * seqLen, _vocabSize);
        _lmHeadBiasGradient = Engine.ReduceSum(grad3D, new int[] { 0, 1 });
        _lmHeadWeightsGradient = Engine.TensorMatMul(_lastNormedOutput.Reshape(batchSize * seqLen, _modelDimension).Transpose(new[] { 1, 0 }), gradFlat);
        var dNormed = Engine.TensorMatMul(gradFlat, _lmHeadWeights.Transpose(new[] { 1, 0 })).Reshape(batchSize, seqLen, _modelDimension);
        var dPostBlocks = BackwardRMSNorm(dNormed, _lastPostBlocksOutput, _finalNormGamma, batchSize, seqLen, out var dFinalGamma);
        _finalNormGammaGradient = dFinalGamma;

        var current = dPostBlocks;
        for (int i = _numLayers - 1; i >= 0; i--) { var blockGrad = _blocks[i].Backward(current); current = Engine.TensorAdd(current, blockGrad); }

        var embGradFlat = current.Reshape(batchSize * seqLen, _modelDimension);
        _embeddingWeightsGradient = Engine.TensorMatMul(_lastInput.Reshape(batchSize * seqLen, _lastInput.Shape[2]).Transpose(new[] { 1, 0 }), embGradFlat);
        var dInput3D = Engine.TensorMatMul(embGradFlat, _embeddingWeights.Transpose(new[] { 1, 0 })).Reshape(batchSize, seqLen, _lastInput.Shape[2]);
        if (_originalInputShape != null && _originalInputShape.Length == 2) return dInput3D.Reshape(seqLen, _lastInput.Shape[2]);
        if (_originalInputShape != null) return dInput3D.Reshape(_originalInputShape);
        return dInput3D;
    }

    private Tensor<T> ApplyRMSNorm(Tensor<T> input, Tensor<T> gamma, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(input.Shape); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1);
            var normed = new Tensor<T>(slice.Shape);
            for (int b = 0; b < batchSize; b++)
            {
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++) { T v = slice[new[] { b, d }]; sumSq = NumOps.Add(sumSq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(sumSq, NumOps.FromDouble(_modelDimension)), eps));
                for (int d = 0; d < _modelDimension; d++)
                    normed[new[] { b, d }] = NumOps.Multiply(NumOps.Divide(slice[new[] { b, d }], rms), gamma[d]);
            }
            output.SetSlice(1, t, normed);
        }
        return output;
    }

    private Tensor<T> BackwardRMSNorm(Tensor<T> dOutput, Tensor<T> input, Tensor<T> gamma, int batchSize, int seqLen, out Tensor<T> dGamma)
    {
        var dInput = new Tensor<T>(input.Shape); dGamma = new Tensor<T>(new[] { _modelDimension }); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < seqLen; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1); var dOut = dOutput.GetSliceAlongDimension(t, 1);
            for (int b = 0; b < batchSize; b++)
            {
                T sumSq = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++) { T v = slice[new[] { b, d }]; sumSq = NumOps.Add(sumSq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(sumSq, NumOps.FromDouble(_modelDimension)), eps));
                T rmsInv = NumOps.Divide(NumOps.One, rms);
                for (int d = 0; d < _modelDimension; d++)
                    dGamma[d] = NumOps.Add(dGamma[d], NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(slice[new[] { b, d }], rmsInv)));
                T dot = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++) dot = NumOps.Add(dot, NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(gamma[d], slice[new[] { b, d }])));
                T rms3Inv = NumOps.Divide(rmsInv, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++)
                    dInput[new[] { b, t, d }] = NumOps.Subtract(NumOps.Multiply(NumOps.Multiply(dOut[new[] { b, d }], gamma[d]), rmsInv),
                        NumOps.Multiply(NumOps.Multiply(dot, slice[new[] { b, d }]), NumOps.Divide(rms3Inv, NumOps.FromDouble(_modelDimension))));
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
        var p = new Vector<T>(ParameterCount); int idx = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) p[idx++] = _embeddingWeights[i];
        foreach (var b in _blocks) { var bp = b.GetParameters(); for (int i = 0; i < bp.Length; i++) p[idx++] = bp[i]; }
        for (int i = 0; i < _finalNormGamma.Length; i++) p[idx++] = _finalNormGamma[i];
        for (int i = 0; i < _lmHeadWeights.Length; i++) p[idx++] = _lmHeadWeights[i];
        for (int i = 0; i < _lmHeadBias.Length; i++) p[idx++] = _lmHeadBias[i];
        return p;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount) throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int idx = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) _embeddingWeights[i] = parameters[idx++];
        foreach (var b in _blocks) { var bp = new Vector<T>(b.ParameterCount); for (int i = 0; i < b.ParameterCount; i++) bp[i] = parameters[idx++]; b.SetParameters(bp); }
        for (int i = 0; i < _finalNormGamma.Length; i++) _finalNormGamma[i] = parameters[idx++];
        for (int i = 0; i < _lmHeadWeights.Length; i++) _lmHeadWeights[i] = parameters[idx++];
        for (int i = 0; i < _lmHeadBias.Length; i++) _lmHeadBias[i] = parameters[idx++];
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null; _lastOutput = null; _lastEmbedded = null; _lastNormedOutput = null;
        _lastPostBlocksOutput = null; _lastBlockInputs = null; _originalInputShape = null;
        _embeddingWeightsGradient = null; _finalNormGammaGradient = null;
        _lmHeadWeightsGradient = null; _lmHeadBiasGradient = null;
        foreach (var block in _blocks) block.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
        var inNode = TensorOperations<T>.Variable(new Tensor<T>(new int[] { 1, _vocabSize }), "xlstm_input");
        inputNodes.Add(inNode);
        var embNode = TensorOperations<T>.Variable(_embeddingWeights, "xlstm_W_emb"); inputNodes.Add(embNode);
        var current = TensorOperations<T>.MatrixMultiply(inNode, embNode);
        for (int i = 0; i < _numLayers; i++)
        {
            var bi = new List<ComputationNode<T>> { current }; var bo = _blocks[i].ExportComputationGraph(bi);
            inputNodes.AddRange(bi.GetRange(1, bi.Count - 1)); current = TensorOperations<T>.Add(current, bo);
        }
        var normN = TensorOperations<T>.Variable(_finalNormGamma, "xlstm_norm"); inputNodes.Add(normN);
        current = TensorOperations<T>.ElementwiseMultiply(current, normN);
        var wN = TensorOperations<T>.Variable(_lmHeadWeights, "xlstm_W_lm"); var bN = TensorOperations<T>.Variable(_lmHeadBias, "xlstm_b_lm");
        inputNodes.Add(wN); inputNodes.Add(bN);
        return TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(current, wN), bN);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var m = base.GetMetadata(); m["VocabSize"] = _vocabSize.ToString(); m["ModelDimension"] = _modelDimension.ToString();
        m["NumLayers"] = _numLayers.ToString(); m["NumHeads"] = _numHeads.ToString(); m["Architecture"] = "xLSTM"; return m;
    }
}
