using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

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
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 16, 2, 1")]
// Sparse routing changes WHICH weights a token sees, never how many numbers come back. The last line of
// ForwardTraced is `Engine.Reshape(output, input._shape)` - the caller's exact shape, at any rank - and
// the accumulator it reshapes is [n, _hidden], so the trailing width is preserved too (the input's own
// width must already be _hidden for the expert projections to type). _ffnDim, _numExperts and _topK all
// size intermediates and never reach an output axis.
[ElementWiseShape(Note = "Top-k expert routing per token; every token keeps its slot and its width.")]
[AutoParameters]
public partial class MoEFeedForwardLayer<T> : LayerBase<T>, IShapeContract
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // Every child reads the token, not its predecessor's output. Chained sizing assumed otherwise and
    // built each expert against the ROUTER's numExperts-wide output, so a restore met [2, 16] where
    // the checkpoint held [8, 16] and refused it. The experts are siblings, so one declared width
    // covers the whole bank.
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T> _router;               // hidden -> numExperts (no bias, identity)
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T>[] _gate;               // per expert: hidden -> ffn (activation)
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T>[] _up;                 // per expert: hidden -> ffn (identity)
    [SubLayerInput("_ffnDim")]
    private readonly DenseLayer<T>[] _down;               // per expert: ffn -> hidden (identity)
    private readonly int _hidden;
    private readonly int _ffnDim;
    private readonly int _numExperts;
    private readonly int _topK;

    // Optional always-on shared expert (Qwen2-MoE): its SwiGLU output, gated by sigmoid(sharedGate(x)), is
    // added to the routed output for every token. Null when the model has no shared expert (Mixtral).
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T>? _sharedGate;          // hidden -> sharedFfn (activation)
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T>? _sharedUp;            // hidden -> sharedFfn (identity)
    [SubLayerInput("_sharedFfnDim")]
    private readonly DenseLayer<T>? _sharedDown;          // sharedFfn -> hidden (identity)
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T>? _sharedGateLogit;     // hidden -> 1 (sigmoid gate)
    private readonly int _sharedFfnDim;

    /// <inheritdoc />
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

    /// <summary>Construction state: the 'hiddenSize' the layer was built with.</summary>
    private readonly int _hiddenSize;

    /// <summary>Construction state: the gate 'activation' the layer was built with (SiLU when none was given). Named
    /// for the parameter so the generated factory persists it; without it a rebuild fell back to SiLU.</summary>
    private readonly IActivationFunction<T> _activation;

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
        _hiddenSize = hiddenSize;
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
        _activation = act;

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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

        T zero = Ops.Zero;
        for (int e = 0; e < _numExperts; e++)
        {
            // Gather the tokens routed to expert e (with their weights), in token order.
            Tensor<T> column;
            Tensor<int> rows;
            using (new NoGradScope<T>())
            {
                column = Engine.Reshape(Engine.TensorNarrow(weightPerExpert, 1, e, 1), new[] { n });
                var selected = Engine.TensorNonzero(Engine.TensorGreaterThan(column, zero)); // [m, 1]
                rows = Engine.Reshape(selected, new[] { selected.Shape[0] });
            }
            int m = rows.Shape[0];
            if (m == 0) continue;

            var gathered = Engine.TensorIndexSelect(flat, rows, 0); // [m, hidden]

            var g = _gate[e].Forward(gathered);          // act(gate(x)) : [m, ffn]
            var u = _up[e].Forward(gathered);            // up(x)        : [m, ffn]
            var prod = Engine.TensorMultiply(g, u);      // gated hidden : [m, ffn]
            var d = _down[e].Forward(prod);              // down(...)    : [m, hidden]

            // Scatter-add the weighted expert output back onto its tokens.
            using (new NoGradScope<T>())
            {
                var weight = Engine.Reshape(Engine.TensorIndexSelect(column, rows, 0), new[] { m, 1 });
                output = Engine.TensorIndexAdd(output, 0, rows, Engine.TensorMultiply(d, weight));
            }
        }

        // Always-on shared expert (Qwen2-MoE): output += sigmoid(sharedGate(x)) * sharedSwiGLU(x) for all tokens.
        if (_sharedGate is not null && _sharedUp is not null && _sharedDown is not null && _sharedGateLogit is not null)
        {
            var sg = _sharedGate.Forward(flat);
            var su = _sharedUp.Forward(flat);
            var sh = _sharedDown.Forward(Engine.TensorMultiply(sg, su)); // [n, hidden]
            var gateLogit = _sharedGateLogit.Forward(flat);             // [n, 1]
            using (new NoGradScope<T>())
            {
                output = Engine.TensorAdd(output, Engine.TensorMultiply(Engine.Sigmoid(gateLogit), sh));
            }
        }

        return Engine.Reshape(output, input._shape);
    }

    // For each token: softmax over experts, keep the top-k, renormalize their weights (Mixtral convention).
    private Tensor<T> ComputeRouting(Tensor<T> routerOut, int n)
    {
        // Routing weights are a fresh constant table, built off the tape.
        using var noGrad = new NoGradScope<T>();
        int experts = _numExperts;
        var probs = Engine.Softmax(routerOut, 1); // [n, E]

        // Select top-k experts by probability, ties broken toward the lower expert index (master's
        // first-max selection). Only token-by-expert storage is used:
        //   t     = the k-th largest probability per token (a value, so independent of TopK's tie order);
        //   above = experts with p > t, all kept;
        //   quota = k - |above| slots left for experts with p == t, filled in ascending index order by
        //           keeping a tied expert when its inclusive running count of ties is <= quota.
        int[] tokenByExpert = new[] { n, experts };
        var topValues = Engine.TensorTopK(probs, _topK, 1, out Tensor<int> _);           // [n, k]
        var kth = Engine.TensorBroadcastTo(
            Engine.TensorNarrow(topValues, 1, _topK - 1, 1), tokenByExpert);              // [n, E]
        var above = Engine.TensorGreaterThan(probs, kth);                                 // [n, E]
        var tied = Engine.TensorEquals(probs, kth);                                       // [n, E]
        var aboveCount = Engine.ReduceSum(above, new[] { 1 }, keepDims: true);            // [n, 1]
        var quotaPlusOne = Engine.TensorBroadcastTo(
            Engine.TensorSubtract(
                Tensor<T>.CreateDefault(new[] { n, 1 }, Ops.FromDouble(_topK + 1)), aboveCount),
            tokenByExpert);                                                               // [n, E]
        var tiedRank = Engine.TensorCumSum(tied, 1);                                      // inclusive
        var tiedKept = Engine.TensorMultiply(tied, Engine.TensorLessThan(tiedRank, quotaPlusOne));
        var chosen = Engine.TensorAdd(above, tiedKept);

        var kept = Engine.TensorMultiply(probs, chosen);
        var topSum = Engine.ReduceSum(kept, new[] { 1 }, keepDims: true);
        return Engine.TensorDivide(kept, topSum); // renormalize
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
