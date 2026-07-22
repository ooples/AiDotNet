using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Sparse mixture-of-experts feed-forward sublayer (Mixtral / sparse-MoE): a linear router selects the
/// top-<c>k</c> of <c>E</c> gated-SwiGLU experts per token and mixes their outputs by the renormalized
/// router weights — <c>y = Σ_{e∈topk(x)} softmax(router(x))_e · down_e(act(gate_e(x)) · up_e(x))</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Inference-oriented (loaded from a pretrained checkpoint, not trained here): the routing is a
/// non-differentiable top-k selection, so <see cref="SupportsTraining"/> is <c>false</c> and the forward
/// pass gathers each expert's routed tokens, runs that expert's SwiGLU FFN, and scatters the weighted
/// result back. Weights are the router plus, per expert, gate/up/down projections (all bias-free).
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.FeedForward)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class MoEFeedForwardLayer<T> : LayerBase<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly DenseLayer<T> _router;               // hidden -> numExperts (no bias, identity)
    private readonly DenseLayer<T>[] _gate;               // per expert: hidden -> ffn (activation)
    private readonly DenseLayer<T>[] _up;                 // per expert: hidden -> ffn (identity)
    private readonly DenseLayer<T>[] _down;               // per expert: ffn -> hidden (identity)
    private readonly int _hidden;
    private readonly int _ffnDim;
    private readonly int _numExperts;
    private readonly int _topK;

    // Optional always-on shared expert (Qwen2-MoE): its SwiGLU output, gated by sigmoid(sharedGate(x)), is
    // added to the routed output for every token. Null when the model has no shared expert (Mixtral).
    private readonly DenseLayer<T>? _sharedGate;          // hidden -> sharedFfn (activation)
    private readonly DenseLayer<T>? _sharedUp;            // hidden -> sharedFfn (identity)
    private readonly DenseLayer<T>? _sharedDown;          // sharedFfn -> hidden (identity)
    private readonly DenseLayer<T>? _sharedGateLogit;     // hidden -> 1 (sigmoid gate)
    private readonly int _sharedFfnDim;

    public override bool SupportsTraining => false;

    /// <summary>The router (gating) linear projection, hidden -&gt; numExperts (bias-free).</summary>
    public DenseLayer<T> Router => _router;

    /// <summary>The number of experts.</summary>
    public int NumExperts => _numExperts;

    /// <summary>The number of experts activated per token.</summary>
    public int TopK => _topK;

    /// <summary>The gate projection of expert <paramref name="e"/>.</summary>
    public DenseLayer<T> ExpertGate(int e) => _gate[e];

    /// <summary>The up (value) projection of expert <paramref name="e"/>.</summary>
    public DenseLayer<T> ExpertUp(int e) => _up[e];

    /// <summary>The down projection of expert <paramref name="e"/>.</summary>
    public DenseLayer<T> ExpertDown(int e) => _down[e];

    /// <summary>Whether this layer has an always-on shared expert (Qwen2-MoE).</summary>
    public bool HasSharedExpert => _sharedGate is not null;

    /// <summary>The shared expert's gate projection (null when absent).</summary>
    public DenseLayer<T>? SharedGate => _sharedGate;

    /// <summary>The shared expert's up projection (null when absent).</summary>
    public DenseLayer<T>? SharedUp => _sharedUp;

    /// <summary>The shared expert's down projection (null when absent).</summary>
    public DenseLayer<T>? SharedDown => _sharedDown;

    /// <summary>The shared expert's sigmoid gate (hidden -&gt; 1; null when absent).</summary>
    public DenseLayer<T>? SharedGateLogit => _sharedGateLogit;

    /// <summary>Creates a sparse MoE feed-forward layer.</summary>
    /// <param name="hiddenSize">Model (residual stream) dimension.</param>
    /// <param name="ffnDim">Each expert's inner (intermediate) dimension.</param>
    /// <param name="numExperts">Total expert count E.</param>
    /// <param name="topK">Experts activated per token k (1 ≤ k ≤ E).</param>
    /// <param name="activation">Gate activation (SiLU for Mixtral).</param>
    /// <param name="sharedFfnDim">If &gt; 0, adds an always-on shared expert of this inner dimension
    /// (Qwen2-MoE); its gated output, weighted by <c>sigmoid(sharedGate(x))</c>, is added for every token.</param>
    public MoEFeedForwardLayer(int hiddenSize, int ffnDim, int numExperts, int topK, IActivationFunction<T>? activation = null,
        int sharedFfnDim = 0)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK <= 0 || topK > numExperts) throw new ArgumentOutOfRangeException(nameof(topK));
        if (sharedFfnDim < 0) throw new ArgumentOutOfRangeException(nameof(sharedFfnDim));

        _hidden = hiddenSize;
        _ffnDim = ffnDim;
        _numExperts = numExperts;
        _topK = topK;
        _sharedFfnDim = sharedFfnDim;
        var act = activation ?? new SiLUActivation<T>();

        _router = new DenseLayer<T>(numExperts, activationFunction: new IdentityActivation<T>());
        _gate = new DenseLayer<T>[numExperts];
        _up = new DenseLayer<T>[numExperts];
        _down = new DenseLayer<T>[numExperts];
        RegisterSubLayer(_router);
        for (int e = 0; e < numExperts; e++)
        {
            _gate[e] = new DenseLayer<T>(ffnDim, activationFunction: act);
            _up[e] = new DenseLayer<T>(ffnDim, activationFunction: new IdentityActivation<T>());
            _down[e] = new DenseLayer<T>(hiddenSize, activationFunction: new IdentityActivation<T>());
            RegisterSubLayer(_gate[e]);
            RegisterSubLayer(_up[e]);
            RegisterSubLayer(_down[e]);
        }

        if (sharedFfnDim > 0)
        {
            _sharedGate = new DenseLayer<T>(sharedFfnDim, activationFunction: act);
            _sharedUp = new DenseLayer<T>(sharedFfnDim, activationFunction: new IdentityActivation<T>());
            _sharedDown = new DenseLayer<T>(hiddenSize, activationFunction: new IdentityActivation<T>());
            _sharedGateLogit = new DenseLayer<T>(1, activationFunction: new IdentityActivation<T>());
            RegisterSubLayer(_sharedGate);
            RegisterSubLayer(_sharedUp);
            RegisterSubLayer(_sharedDown);
            RegisterSubLayer(_sharedGateLogit);
        }
    }

    /// <summary>
    /// Forces lazy weight allocation for the router and every expert's gate/up/down projections. A warmup
    /// forward only routes tokens to the top-k experts, so the remaining experts would stay unshaped and
    /// could not be weight-loaded; call this once after construction to materialize them all.
    /// </summary>
    public void Materialize()
    {
        var dummyHidden = new Tensor<T>(new[] { 1, _hidden });
        dummyHidden.Fill(Ops.Zero);
        var dummyFfn = new Tensor<T>(new[] { 1, _ffnDim });
        dummyFfn.Fill(Ops.Zero);

        _router.Forward(dummyHidden);
        for (int e = 0; e < _numExperts; e++)
        {
            _gate[e].Forward(dummyHidden);
            _up[e].Forward(dummyHidden);
            _down[e].Forward(dummyFfn);
        }
        if (_sharedGate is not null && _sharedUp is not null && _sharedDown is not null && _sharedGateLogit is not null)
        {
            var dummyShared = new Tensor<T>(new[] { 1, _sharedFfnDim });
            dummyShared.Fill(Ops.Zero);
            _sharedGate.Forward(dummyHidden);
            _sharedUp.Forward(dummyHidden);
            _sharedDown.Forward(dummyShared);
            _sharedGateLogit.Forward(dummyHidden);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int featureDim = input.Shape[rank - 1];
        int n = 1;
        for (int i = 0; i < rank - 1; i++) n *= input.Shape[i];

        var flat = Engine.Reshape(input, new[] { n, featureDim });

        // Router logits -> per-token renormalized top-k expert weights.
        var routerOut = _router.Forward(flat); // [n, E]
        var weightPerExpert = ComputeRouting(routerOut, n);

        var output = new Tensor<T>(new[] { n, _hidden });
        output.Fill(Ops.Zero);

        for (int e = 0; e < _numExperts; e++)
        {
            // Gather the tokens routed to expert e (with their weights).
            var rows = new List<int>();
            var w = new List<T>();
            for (int tok = 0; tok < n; tok++)
            {
                var wt = weightPerExpert[tok, e];
                if (Ops.GreaterThan(wt, Ops.Zero)) { rows.Add(tok); w.Add(wt); }
            }
            if (rows.Count == 0) continue;

            int m = rows.Count;
            var gathered = new Tensor<T>(new[] { m, _hidden });
            for (int r = 0; r < m; r++)
                for (int j = 0; j < _hidden; j++)
                    gathered[r, j] = flat[rows[r], j];

            var g = _gate[e].Forward(gathered);          // act(gate(x)) : [m, ffn]
            var u = _up[e].Forward(gathered);            // up(x)        : [m, ffn]
            var prod = Engine.TensorMultiply(g, u);      // gated hidden : [m, ffn]
            var d = _down[e].Forward(prod);              // down(...)    : [m, hidden]

            for (int r = 0; r < m; r++)
            {
                int tok = rows[r];
                var weight = w[r];
                for (int j = 0; j < _hidden; j++)
                    output[tok, j] = Ops.Add(output[tok, j], Ops.Multiply(weight, d[r, j]));
            }
        }

        // Always-on shared expert (Qwen2-MoE): output += sigmoid(sharedGate(x)) * sharedSwiGLU(x) for all tokens.
        if (_sharedGate is not null && _sharedUp is not null && _sharedDown is not null && _sharedGateLogit is not null)
        {
            var sg = _sharedGate.Forward(flat);
            var su = _sharedUp.Forward(flat);
            var sh = _sharedDown.Forward(Engine.TensorMultiply(sg, su)); // [n, hidden]
            var gateLogit = _sharedGateLogit.Forward(flat);             // [n, 1]
            for (int tok = 0; tok < n; tok++)
            {
                double sig = 1.0 / (1.0 + Math.Exp(-Convert.ToDouble(gateLogit[tok, 0])));
                var sigT = Ops.FromDouble(sig);
                for (int j = 0; j < _hidden; j++)
                    output[tok, j] = Ops.Add(output[tok, j], Ops.Multiply(sigT, sh[tok, j]));
            }
        }

        return Engine.Reshape(output, input._shape);
    }

    // For each token: softmax over experts, keep the top-k, renormalize their weights (Mixtral convention).
    private Tensor<T> ComputeRouting(Tensor<T> routerOut, int n)
    {
        var weights = new Tensor<T>(new[] { n, _numExperts });
        weights.Fill(Ops.Zero);

        var logits = new double[_numExperts];
        for (int tok = 0; tok < n; tok++)
        {
            double max = double.NegativeInfinity;
            for (int e = 0; e < _numExperts; e++)
            {
                logits[e] = Convert.ToDouble(routerOut[tok, e]);
                if (logits[e] > max) max = logits[e];
            }
            double sumExp = 0.0;
            var probs = new double[_numExperts];
            for (int e = 0; e < _numExperts; e++) { probs[e] = Math.Exp(logits[e] - max); sumExp += probs[e]; }
            for (int e = 0; e < _numExperts; e++) probs[e] /= sumExp;

            // Select top-k experts by probability.
            var chosen = new int[_topK];
            var chosenProb = new double[_topK];
            var used = new bool[_numExperts];
            double topSum = 0.0;
            for (int k = 0; k < _topK; k++)
            {
                int best = -1; double bestP = double.NegativeInfinity;
                for (int e = 0; e < _numExperts; e++)
                    if (!used[e] && probs[e] > bestP) { bestP = probs[e]; best = e; }
                used[best] = true;
                chosen[k] = best; chosenProb[k] = bestP; topSum += bestP;
            }
            for (int k = 0; k < _topK; k++)
                weights[tok, chosen[k]] = Ops.FromDouble(chosenProb[k] / topSum); // renormalize
        }
        return weights;
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _router;
        for (int e = 0; e < _numExperts; e++) { yield return _gate[e]; yield return _up[e]; yield return _down[e]; }
        if (_sharedGate is not null && _sharedUp is not null && _sharedDown is not null && _sharedGateLogit is not null)
        {
            yield return _sharedGate;
            yield return _sharedUp;
            yield return _sharedDown;
            yield return _sharedGateLogit;
        }
    }

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get { long total = 0; foreach (var l in SubLayers()) total += l.ParameterCount; return total; }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var l in SubLayers()) acc = Vector<T>.Concatenate(acc, l.GetParameters());
        return acc;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");
        int offset = 0;
        foreach (var l in SubLayers())
        {
            int count = (int)l.ParameterCount;
            if (count == 0) continue;
            l.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var l in SubLayers()) acc = Vector<T>.Concatenate(acc, l.GetParameterGradients());
        return acc;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var l in SubLayers()) l.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var l in SubLayers()) l.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var l in SubLayers()) l.ResetState();
    }
}
