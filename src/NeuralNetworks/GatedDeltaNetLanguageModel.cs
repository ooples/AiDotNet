using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full Gated DeltaNet language model: token embedding + N GatedDeltaNetLayer blocks + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Gated DeltaNet combines linear attention with gated delta rules for efficient sequence modeling.
/// The delta rule update allows the model to both write new associations and erase old ones in its
/// memory, unlike standard linear attention which can only accumulate.
/// </para>
/// <para><b>For Beginners:</b> Gated DeltaNet is like a smart notepad that can both add new notes
/// and erase outdated ones, giving it better memory management than simpler models.
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GatedDeltaNetLanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;

    private Tensor<T> _embeddingWeights;
    private readonly GatedDeltaNetLayer<T>[] _blocks;
    private Tensor<T> _finalNormGamma;
    private Tensor<T> _lmHeadWeights;
    private Tensor<T> _lmHeadBias;

    private Tensor<T>? _lastInput; private Tensor<T>? _lastOutput; private Tensor<T>? _lastEmbedded;
    private Tensor<T>? _lastNormedOutput; private Tensor<T>? _lastPostBlocksOutput;
    private Tensor<T>[]? _lastBlockInputs; private int[]? _originalInputShape;
    private Tensor<T>? _embeddingWeightsGradient; private Tensor<T>? _finalNormGammaGradient;
    private Tensor<T>? _lmHeadWeightsGradient; private Tensor<T>? _lmHeadBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;
    public int VocabSize => _vocabSize;
    public int ModelDimension => _modelDimension;
    public int NumLayers => _numLayers;
    public int NumHeads => _numHeads;

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            int c = _embeddingWeights.Length;
            foreach (var b in _blocks) c += b.ParameterCount;
            return c + _finalNormGamma.Length + _lmHeadWeights.Length + _lmHeadBias.Length;
        }
    }

    public GatedDeltaNetLanguageModel(int vocabSize, int modelDimension = 256, int numLayers = 4,
        int numHeads = 8, int maxSeqLength = 512, IActivationFunction<T>? activationFunction = null)
        : base([maxSeqLength, vocabSize], [maxSeqLength, vocabSize],
            activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException($"Vocab size must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers must be positive.", nameof(numLayers));
        if (numHeads <= 0) throw new ArgumentException($"Number of heads must be positive.", nameof(numHeads));

        _vocabSize = vocabSize; _modelDimension = modelDimension; _numLayers = numLayers; _numHeads = numHeads;
        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension }); InitTensor(_embeddingWeights);
        _blocks = new GatedDeltaNetLayer<T>[numLayers];
        for (int i = 0; i < numLayers; i++) _blocks[i] = new GatedDeltaNetLayer<T>(maxSeqLength, modelDimension, numHeads);
        _finalNormGamma = new Tensor<T>(new[] { modelDimension }); _finalNormGamma.Fill(NumOps.One);
        _lmHeadWeights = new Tensor<T>(new[] { modelDimension, vocabSize }); InitTensor(_lmHeadWeights);
        _lmHeadBias = new Tensor<T>(new[] { vocabSize }); _lmHeadBias.Fill(NumOps.Zero);
    }

    private void InitTensor(Tensor<T> t)
    {
        T s = NumOps.Sqrt(NumOps.FromDouble(2.0 / (t.Shape[0] + t.Shape[1])));
        for (int i = 0; i < t.Length; i++) t[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), s);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;
        if (rank < 2) throw new ArgumentException($"Input must be at least rank 2.", nameof(input));
        int seqLen = input.Shape[rank - 2], inputDim = input.Shape[rank - 1];
        int bs = 1; for (int d = 0; d < rank - 2; d++) bs *= input.Shape[d]; if (rank < 3) bs = 1;
        var in3D = rank == 2 ? input.Reshape(1, seqLen, inputDim) : input.Reshape(bs, seqLen, inputDim);
        _lastInput = in3D;
        if (inputDim != _vocabSize) throw new ArgumentException($"Input dim ({inputDim}) must match vocab ({_vocabSize}).", nameof(input));

        var emb = Engine.TensorMatMul(in3D.Reshape(bs * seqLen, inputDim), _embeddingWeights).Reshape(bs, seqLen, _modelDimension);
        _lastEmbedded = emb;
        _lastBlockInputs = new Tensor<T>[_numLayers]; var cur = emb;
        for (int i = 0; i < _numLayers; i++) { _lastBlockInputs[i] = cur; cur = Engine.TensorAdd(cur, _blocks[i].Forward(cur)); }
        _lastPostBlocksOutput = cur;
        var normed = ApplyRMSNorm(cur, _finalNormGamma, bs, seqLen); _lastNormedOutput = normed;
        var logits = Engine.TensorBroadcastAdd(Engine.TensorMatMul(normed.Reshape(bs * seqLen, _modelDimension), _lmHeadWeights), _lmHeadBias.Reshape(1, _vocabSize));
        var result = ApplyActivation(logits.Reshape(bs, seqLen, _vocabSize)); _lastOutput = result;
        if (rank == 2) return result.Reshape(seqLen, _vocabSize);
        var os = new int[rank]; for (int i = 0; i < rank - 2; i++) os[i] = input.Shape[i]; os[rank - 2] = seqLen; os[rank - 1] = _vocabSize;
        return result.Reshape(os);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastNormedOutput == null || _lastPostBlocksOutput == null || _lastBlockInputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        int bs = _lastInput.Shape[0], sl = _lastInput.Shape[1];
        var g3D = outputGradient.Rank == 2 ? outputGradient.Reshape(1, sl, _vocabSize) : outputGradient.Reshape(bs, sl, _vocabSize);
        g3D = ApplyActivationDerivative(_lastOutput, g3D);
        var gF = g3D.Reshape(bs * sl, _vocabSize);
        _lmHeadBiasGradient = Engine.ReduceSum(g3D, new int[] { 0, 1 });
        _lmHeadWeightsGradient = Engine.TensorMatMul(_lastNormedOutput.Reshape(bs * sl, _modelDimension).Transpose(new[] { 1, 0 }), gF);
        var dN = Engine.TensorMatMul(gF, _lmHeadWeights.Transpose(new[] { 1, 0 })).Reshape(bs, sl, _modelDimension);
        var dPB = BackwardRMSNorm(dN, _lastPostBlocksOutput, _finalNormGamma, bs, sl, out var dFG); _finalNormGammaGradient = dFG;
        var cur = dPB; for (int i = _numLayers - 1; i >= 0; i--) cur = Engine.TensorAdd(cur, _blocks[i].Backward(cur));
        var eGF = cur.Reshape(bs * sl, _modelDimension);
        _embeddingWeightsGradient = Engine.TensorMatMul(_lastInput.Reshape(bs * sl, _lastInput.Shape[2]).Transpose(new[] { 1, 0 }), eGF);
        var dI = Engine.TensorMatMul(eGF, _embeddingWeights.Transpose(new[] { 1, 0 })).Reshape(bs, sl, _lastInput.Shape[2]);
        if (_originalInputShape != null && _originalInputShape.Length == 2) return dI.Reshape(sl, _lastInput.Shape[2]);
        if (_originalInputShape != null) return dI.Reshape(_originalInputShape);
        return dI;
    }

    private Tensor<T> ApplyRMSNorm(Tensor<T> input, Tensor<T> gamma, int bs, int sl)
    {
        var output = new Tensor<T>(input.Shape); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < sl; t++)
        {
            var slice = input.GetSliceAlongDimension(t, 1); var normed = new Tensor<T>(slice.Shape);
            for (int b = 0; b < bs; b++)
            {
                T ssq = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) { T v = slice[new[] { b, d }]; ssq = NumOps.Add(ssq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(ssq, NumOps.FromDouble(_modelDimension)), eps));
                for (int d = 0; d < _modelDimension; d++) normed[new[] { b, d }] = NumOps.Multiply(NumOps.Divide(slice[new[] { b, d }], rms), gamma[d]);
            }
            output.SetSlice(1, t, normed);
        }
        return output;
    }

    private Tensor<T> BackwardRMSNorm(Tensor<T> dO, Tensor<T> inp, Tensor<T> gamma, int bs, int sl, out Tensor<T> dG)
    {
        var dI = new Tensor<T>(inp.Shape); dG = new Tensor<T>(new[] { _modelDimension }); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < sl; t++)
        {
            var s = inp.GetSliceAlongDimension(t, 1); var dOut = dO.GetSliceAlongDimension(t, 1);
            for (int b = 0; b < bs; b++)
            {
                T ssq = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) { T v = s[new[] { b, d }]; ssq = NumOps.Add(ssq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(ssq, NumOps.FromDouble(_modelDimension)), eps));
                T ri = NumOps.Divide(NumOps.One, rms);
                for (int d = 0; d < _modelDimension; d++) dG[d] = NumOps.Add(dG[d], NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(s[new[] { b, d }], ri)));
                T dot = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) dot = NumOps.Add(dot, NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(gamma[d], s[new[] { b, d }])));
                T r3i = NumOps.Divide(ri, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++)
                    dI[new[] { b, t, d }] = NumOps.Subtract(NumOps.Multiply(NumOps.Multiply(dOut[new[] { b, d }], gamma[d]), ri),
                        NumOps.Multiply(NumOps.Multiply(dot, s[new[] { b, d }]), NumOps.Divide(r3i, NumOps.FromDouble(_modelDimension))));
            }
        }
        return dI;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T lr)
    {
        if (_embeddingWeightsGradient == null) throw new InvalidOperationException("Backward pass must be called first.");
        T nLR = NumOps.Negate(lr);
        _embeddingWeights = Engine.TensorAdd(_embeddingWeights, Engine.TensorMultiplyScalar(_embeddingWeightsGradient, nLR));
        foreach (var b in _blocks) b.UpdateParameters(lr);
        if (_finalNormGammaGradient != null) _finalNormGamma = Engine.TensorAdd(_finalNormGamma, Engine.TensorMultiplyScalar(_finalNormGammaGradient, nLR));
        if (_lmHeadWeightsGradient != null) _lmHeadWeights = Engine.TensorAdd(_lmHeadWeights, Engine.TensorMultiplyScalar(_lmHeadWeightsGradient, nLR));
        if (_lmHeadBiasGradient != null) _lmHeadBias = Engine.TensorAdd(_lmHeadBias, Engine.TensorMultiplyScalar(_lmHeadBiasGradient, nLR));
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
    public override void SetParameters(Vector<T> p)
    {
        if (p.Length != ParameterCount) throw new ArgumentException($"Expected {ParameterCount}, got {p.Length}");
        int idx = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) _embeddingWeights[i] = p[idx++];
        foreach (var b in _blocks) { var bp = new Vector<T>(b.ParameterCount); for (int i = 0; i < b.ParameterCount; i++) bp[i] = p[idx++]; b.SetParameters(bp); }
        for (int i = 0; i < _finalNormGamma.Length; i++) _finalNormGamma[i] = p[idx++];
        for (int i = 0; i < _lmHeadWeights.Length; i++) _lmHeadWeights[i] = p[idx++];
        for (int i = 0; i < _lmHeadBias.Length; i++) _lmHeadBias[i] = p[idx++];
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null; _lastOutput = null; _lastEmbedded = null; _lastNormedOutput = null;
        _lastPostBlocksOutput = null; _lastBlockInputs = null; _originalInputShape = null;
        _embeddingWeightsGradient = null; _finalNormGammaGradient = null; _lmHeadWeightsGradient = null; _lmHeadBiasGradient = null;
        foreach (var b in _blocks) b.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
        var inN = TensorOperations<T>.Variable(new Tensor<T>(new int[] { 1, _vocabSize }), "gdnet_in"); inputNodes.Add(inN);
        var eN = TensorOperations<T>.Variable(_embeddingWeights, "gdnet_emb"); inputNodes.Add(eN);
        var cur = TensorOperations<T>.MatrixMultiply(inN, eN);
        for (int i = 0; i < _numLayers; i++)
        {
            var bi = new List<ComputationNode<T>> { cur }; var bo = _blocks[i].ExportComputationGraph(bi);
            inputNodes.AddRange(bi.GetRange(1, bi.Count - 1)); cur = TensorOperations<T>.Add(cur, bo);
        }
        var nN = TensorOperations<T>.Variable(_finalNormGamma, "gdnet_norm"); inputNodes.Add(nN);
        cur = TensorOperations<T>.ElementwiseMultiply(cur, nN);
        var wN = TensorOperations<T>.Variable(_lmHeadWeights, "gdnet_lm_w"); var bN = TensorOperations<T>.Variable(_lmHeadBias, "gdnet_lm_b");
        inputNodes.Add(wN); inputNodes.Add(bN);
        return TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(cur, wN), bN);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var m = base.GetMetadata(); m["VocabSize"] = _vocabSize.ToString(); m["ModelDimension"] = _modelDimension.ToString();
        m["NumLayers"] = _numLayers.ToString(); m["NumHeads"] = _numHeads.ToString(); m["Architecture"] = "GatedDeltaNet"; return m;
    }
}
