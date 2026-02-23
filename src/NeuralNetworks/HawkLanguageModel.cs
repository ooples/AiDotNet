using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers.SSM;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Hawk language model: embedding + N pure RGLR blocks + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Hawk is the pure-recurrent variant from Google DeepMind (companion to Griffin), using only
/// Real-Gated Linear Recurrence blocks without any attention. This gives strict O(n) complexity
/// and O(1) memory per token during generation.
/// </para>
/// <para><b>Reference:</b> De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention", 2024.</para>
/// </remarks>
public class HawkLanguageModel<T> : LayerBase<T>
{
    private readonly int _vocabSize, _modelDimension, _numLayers;
    private Tensor<T> _embeddingWeights;
    private readonly RealGatedLinearRecurrenceLayer<T>[] _blocks;
    private Tensor<T> _finalNormGamma, _lmHeadWeights, _lmHeadBias;
    private Tensor<T>? _lastInput, _lastOutput, _lastNormedOutput, _lastPostBlocksOutput;
    private Tensor<T>[]? _lastBlockInputs; private int[]? _originalInputShape;
    private Tensor<T>? _embeddingWeightsGradient, _finalNormGammaGradient, _lmHeadWeightsGradient, _lmHeadBiasGradient;

    public override bool SupportsTraining => true;
    public int VocabSize => _vocabSize; public int ModelDimension => _modelDimension; public int NumLayers => _numLayers;
    public override int ParameterCount { get { int c = _embeddingWeights.Length; foreach (var b in _blocks) c += b.ParameterCount; return c + _finalNormGamma.Length + _lmHeadWeights.Length + _lmHeadBias.Length; } }

    public HawkLanguageModel(int vocabSize, int modelDimension = 256, int numLayers = 4,
        int maxSeqLength = 512, IActivationFunction<T>? activationFunction = null)
        : base([maxSeqLength, vocabSize], [maxSeqLength, vocabSize], activationFunction ?? new ActivationFunctions.IdentityActivation<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException("Vocab size must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException("Model dimension must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException("Number of layers must be positive.", nameof(numLayers));
        _vocabSize = vocabSize; _modelDimension = modelDimension; _numLayers = numLayers;
        _embeddingWeights = new Tensor<T>(new[] { vocabSize, modelDimension }); InitT(_embeddingWeights);
        _blocks = new RealGatedLinearRecurrenceLayer<T>[numLayers];
        for (int i = 0; i < numLayers; i++) _blocks[i] = new RealGatedLinearRecurrenceLayer<T>(maxSeqLength, modelDimension);
        _finalNormGamma = new Tensor<T>(new[] { modelDimension }); _finalNormGamma.Fill(NumOps.One);
        _lmHeadWeights = new Tensor<T>(new[] { modelDimension, vocabSize }); InitT(_lmHeadWeights);
        _lmHeadBias = new Tensor<T>(new[] { vocabSize }); _lmHeadBias.Fill(NumOps.Zero);
    }

    private void InitT(Tensor<T> t) { T s = NumOps.Sqrt(NumOps.FromDouble(2.0 / (t.Shape[0] + t.Shape[1]))); for (int i = 0; i < t.Length; i++) t[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), s); }

    public override Tensor<T> Forward(Tensor<T> input) { _originalInputShape = input.Shape; int rank = input.Shape.Length;
        if (rank < 2) throw new ArgumentException("Input must be at least rank 2.", nameof(input));
        int sl = input.Shape[rank - 2], id = input.Shape[rank - 1]; int bs = 1; for (int d = 0; d < rank - 2; d++) bs *= input.Shape[d]; if (rank < 3) bs = 1;
        var in3D = rank == 2 ? input.Reshape(1, sl, id) : input.Reshape(bs, sl, id); _lastInput = in3D;
        if (id != _vocabSize) throw new ArgumentException($"Input dim ({id}) must match vocab ({_vocabSize}).", nameof(input));
        var emb = Engine.TensorMatMul(in3D.Reshape(bs * sl, id), _embeddingWeights).Reshape(bs, sl, _modelDimension);
        _lastBlockInputs = new Tensor<T>[_numLayers]; var cur = emb;
        for (int i = 0; i < _numLayers; i++) { _lastBlockInputs[i] = cur; cur = Engine.TensorAdd(cur, _blocks[i].Forward(cur)); }
        _lastPostBlocksOutput = cur; var normed = RMSNorm(cur, _finalNormGamma, bs, sl); _lastNormedOutput = normed;
        var logits = Engine.TensorBroadcastAdd(Engine.TensorMatMul(normed.Reshape(bs * sl, _modelDimension), _lmHeadWeights), _lmHeadBias.Reshape(1, _vocabSize));
        var result = ApplyActivation(logits.Reshape(bs, sl, _vocabSize)); _lastOutput = result;
        if (rank == 2) return result.Reshape(sl, _vocabSize);
        var os = new int[rank]; for (int i = 0; i < rank - 2; i++) os[i] = input.Shape[i]; os[rank - 2] = sl; os[rank - 1] = _vocabSize; return result.Reshape(os); }

    public override Tensor<T> Backward(Tensor<T> outputGradient) {
        if (_lastInput == null || _lastOutput == null || _lastNormedOutput == null || _lastPostBlocksOutput == null || _lastBlockInputs == null) throw new InvalidOperationException("Forward first.");
        int bs = _lastInput.Shape[0], sl = _lastInput.Shape[1];
        var g = outputGradient.Rank == 2 ? outputGradient.Reshape(1, sl, _vocabSize) : outputGradient.Reshape(bs, sl, _vocabSize);
        g = ApplyActivationDerivative(_lastOutput, g); var gF = g.Reshape(bs * sl, _vocabSize);
        _lmHeadBiasGradient = Engine.ReduceSum(g, new int[] { 0, 1 });
        _lmHeadWeightsGradient = Engine.TensorMatMul(_lastNormedOutput.Reshape(bs * sl, _modelDimension).Transpose(new[] { 1, 0 }), gF);
        var dN = Engine.TensorMatMul(gF, _lmHeadWeights.Transpose(new[] { 1, 0 })).Reshape(bs, sl, _modelDimension);
        var dPB = BackRMSNorm(dN, _lastPostBlocksOutput, _finalNormGamma, bs, sl, out var dFG); _finalNormGammaGradient = dFG;
        var cur = dPB; for (int i = _numLayers - 1; i >= 0; i--) cur = Engine.TensorAdd(cur, _blocks[i].Backward(cur));
        var eGF = cur.Reshape(bs * sl, _modelDimension);
        _embeddingWeightsGradient = Engine.TensorMatMul(_lastInput.Reshape(bs * sl, _lastInput.Shape[2]).Transpose(new[] { 1, 0 }), eGF);
        var dI = Engine.TensorMatMul(eGF, _embeddingWeights.Transpose(new[] { 1, 0 })).Reshape(bs, sl, _lastInput.Shape[2]);
        if (_originalInputShape != null && _originalInputShape.Length == 2) return dI.Reshape(sl, _lastInput.Shape[2]);
        return _originalInputShape != null ? dI.Reshape(_originalInputShape) : dI; }

    private Tensor<T> RMSNorm(Tensor<T> input, Tensor<T> gamma, int bs, int sl) { var output = new Tensor<T>(input.Shape); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < sl; t++) { var s = input.GetSliceAlongDimension(t, 1); var n = new Tensor<T>(s.Shape);
            for (int b = 0; b < bs; b++) { T ssq = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) { T v = s[new[] { b, d }]; ssq = NumOps.Add(ssq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(ssq, NumOps.FromDouble(_modelDimension)), eps));
                for (int d = 0; d < _modelDimension; d++) n[new[] { b, d }] = NumOps.Multiply(NumOps.Divide(s[new[] { b, d }], rms), gamma[d]); } output.SetSlice(1, t, n); } return output; }

    private Tensor<T> BackRMSNorm(Tensor<T> dO, Tensor<T> inp, Tensor<T> gamma, int bs, int sl, out Tensor<T> dG) { var dI = new Tensor<T>(inp.Shape); dG = new Tensor<T>(new[] { _modelDimension }); T eps = NumOps.FromDouble(1e-6);
        for (int t = 0; t < sl; t++) { var s = inp.GetSliceAlongDimension(t, 1); var dOut = dO.GetSliceAlongDimension(t, 1);
            for (int b = 0; b < bs; b++) { T ssq = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) { T v = s[new[] { b, d }]; ssq = NumOps.Add(ssq, NumOps.Multiply(v, v)); }
                T rms = NumOps.Sqrt(NumOps.Add(NumOps.Divide(ssq, NumOps.FromDouble(_modelDimension)), eps)); T ri = NumOps.Divide(NumOps.One, rms);
                for (int d = 0; d < _modelDimension; d++) dG[d] = NumOps.Add(dG[d], NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(s[new[] { b, d }], ri)));
                T dot = NumOps.Zero; for (int d = 0; d < _modelDimension; d++) dot = NumOps.Add(dot, NumOps.Multiply(dOut[new[] { b, d }], NumOps.Multiply(gamma[d], s[new[] { b, d }])));
                T r3 = NumOps.Divide(ri, NumOps.Multiply(rms, rms));
                for (int d = 0; d < _modelDimension; d++) dI[new[] { b, t, d }] = NumOps.Subtract(NumOps.Multiply(NumOps.Multiply(dOut[new[] { b, d }], gamma[d]), ri),
                    NumOps.Multiply(NumOps.Multiply(dot, s[new[] { b, d }]), NumOps.Divide(r3, NumOps.FromDouble(_modelDimension)))); } } return dI; }

    public override void UpdateParameters(T lr) { if (_embeddingWeightsGradient == null) throw new InvalidOperationException("Backward first."); T n = NumOps.Negate(lr);
        _embeddingWeights = Engine.TensorAdd(_embeddingWeights, Engine.TensorMultiplyScalar(_embeddingWeightsGradient, n)); foreach (var b in _blocks) b.UpdateParameters(lr);
        if (_finalNormGammaGradient != null) _finalNormGamma = Engine.TensorAdd(_finalNormGamma, Engine.TensorMultiplyScalar(_finalNormGammaGradient, n));
        if (_lmHeadWeightsGradient != null) _lmHeadWeights = Engine.TensorAdd(_lmHeadWeights, Engine.TensorMultiplyScalar(_lmHeadWeightsGradient, n));
        if (_lmHeadBiasGradient != null) _lmHeadBias = Engine.TensorAdd(_lmHeadBias, Engine.TensorMultiplyScalar(_lmHeadBiasGradient, n)); }
    public override Vector<T> GetParameters() { var p = new Vector<T>(ParameterCount); int x = 0; for (int i = 0; i < _embeddingWeights.Length; i++) p[x++] = _embeddingWeights[i];
        foreach (var b in _blocks) { var bp = b.GetParameters(); for (int i = 0; i < bp.Length; i++) p[x++] = bp[i]; } for (int i = 0; i < _finalNormGamma.Length; i++) p[x++] = _finalNormGamma[i];
        for (int i = 0; i < _lmHeadWeights.Length; i++) p[x++] = _lmHeadWeights[i]; for (int i = 0; i < _lmHeadBias.Length; i++) p[x++] = _lmHeadBias[i]; return p; }
    public override void SetParameters(Vector<T> p) { if (p.Length != ParameterCount) throw new ArgumentException($"Expected {ParameterCount}, got {p.Length}"); int x = 0;
        for (int i = 0; i < _embeddingWeights.Length; i++) _embeddingWeights[i] = p[x++]; foreach (var b in _blocks) { var bp = new Vector<T>(b.ParameterCount); for (int i = 0; i < b.ParameterCount; i++) bp[i] = p[x++]; b.SetParameters(bp); }
        for (int i = 0; i < _finalNormGamma.Length; i++) _finalNormGamma[i] = p[x++]; for (int i = 0; i < _lmHeadWeights.Length; i++) _lmHeadWeights[i] = p[x++]; for (int i = 0; i < _lmHeadBias.Length; i++) _lmHeadBias[i] = p[x++]; }
    public override void ResetState() { _lastInput = null; _lastOutput = null; _lastNormedOutput = null; _lastPostBlocksOutput = null; _lastBlockInputs = null;
        _originalInputShape = null; _embeddingWeightsGradient = null; _finalNormGammaGradient = null; _lmHeadWeightsGradient = null; _lmHeadBiasGradient = null; foreach (var b in _blocks) b.ResetState(); }
    public override bool SupportsJitCompilation => false;
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes) { if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
        var inN = TensorOperations<T>.Variable(new Tensor<T>(new int[] { 1, _vocabSize }), "hawk_in"); inputNodes.Add(inN);
        var eN = TensorOperations<T>.Variable(_embeddingWeights, "hawk_emb"); inputNodes.Add(eN); var cur = TensorOperations<T>.MatrixMultiply(inN, eN);
        for (int i = 0; i < _numLayers; i++) { var bi = new List<ComputationNode<T>> { cur }; var bo = _blocks[i].ExportComputationGraph(bi); inputNodes.AddRange(bi.GetRange(1, bi.Count - 1)); cur = TensorOperations<T>.Add(cur, bo); }
        var nN = TensorOperations<T>.Variable(_finalNormGamma, "hawk_norm"); inputNodes.Add(nN); cur = TensorOperations<T>.ElementwiseMultiply(cur, nN);
        var wN = TensorOperations<T>.Variable(_lmHeadWeights, "hawk_w"); var bN = TensorOperations<T>.Variable(_lmHeadBias, "hawk_b"); inputNodes.Add(wN); inputNodes.Add(bN);
        return TensorOperations<T>.Add(TensorOperations<T>.MatrixMultiply(cur, wN), bN); }
    internal override Dictionary<string, string> GetMetadata() { var m = base.GetMetadata(); m["VocabSize"] = _vocabSize.ToString(); m["ModelDimension"] = _modelDimension.ToString();
        m["NumLayers"] = _numLayers.ToString(); m["Architecture"] = "Hawk"; return m; }
}
