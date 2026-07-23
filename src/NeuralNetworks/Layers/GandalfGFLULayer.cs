using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// GANDALF feature backbone: a stack of Gated Feature Learning Units (GFLUs), the defining
/// component of GANDALF (Joseph &amp; Raj 2022, "GANDALF: Gated Adaptive Network for Deep Automated
/// Learning of Features").
/// </summary>
/// <remarks>
/// <para>
/// Each GFLU stage performs (1) <b>learnable feature selection</b> — a softmax over a per-stage
/// learnable weight produces a feature mask that emphasizes a subset of inputs; (2) a <b>gated
/// transformation</b> — the masked features go through a linear layer split into a GLU
/// (<c>value ⊙ σ(gate)</c>); and (3) a <b>gated residual update</b> of the running representation
/// (<c>h ← g ⊙ glu + (1−g) ⊙ h</c>), letting later stages build hierarchically on earlier feature
/// selections. This is what distinguishes GANDALF from plain MLPs (no learnable feature selection)
/// and from NODE (decision-tree ensembles).
/// </para>
/// <para>
/// Implemented as one composite layer held in a model's <c>Layers</c> list. Per-stage mask weights
/// are registered trainable tensors; the linear sub-layers are registered via
/// <see cref="LayerBase{T}.RegisterSubLayer"/>; the forward is all tape-recorded Engine ops, so
/// gradients reach every stage. Output: <c>[batch, numFeatures]</c> (a refined feature
/// representation a linear head maps to the prediction).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GandalfGFLULayer<T> : LayerBase<T>
{
    private readonly int _numStages;
    private int _numFeatures = -1;
    private bool _built;

    private Tensor<T>[]? _maskLogits;                  // per-stage [numFeatures] feature-mask logits
    private FullyConnectedLayer<T>[]? _inTransform;    // per-stage numFeatures -> 2*numFeatures (GLU)
    private FullyConnectedLayer<T>[]? _gateTransform;  // per-stage numFeatures -> numFeatures (residual gate)
    private Tensor<T>? _valueSelector;                 // [2F, F] constant GLU value split
    private Tensor<T>? _gateSelector;                  // [2F, F] constant GLU gate split

    /// <summary>Initializes a GFLU stack.</summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numStages">Number of GFLU stages (GANDALF default 6).</param>
    public GandalfGFLULayer(int numFeatures, int numStages = 6)
        : base(new[] { numFeatures }, new[] { numFeatures })
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (numStages <= 0) throw new ArgumentOutOfRangeException(nameof(numStages));
        _numStages = numStages;
        BuildComponents(numFeatures);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    private void BuildComponents(int numFeatures)
    {
        // On a rebuild (fed width differs from the eager ctor build), unregister the previous
        // mask tensors and sub-layers before recreating — RegisterSubLayer / RegisterTrainableParameter
        // only append, so without this a width change would leave stale entries in the registry.
        if (_built)
        {
            if (_maskLogits is not null) foreach (var m in _maskLogits) UnregisterTrainableParameter(m);
            if (_inTransform is not null) foreach (var l in _inTransform) UnregisterSubLayer(l);
            if (_gateTransform is not null) foreach (var l in _gateTransform) UnregisterSubLayer(l);
        }

        _numFeatures = numFeatures;
        var rand = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        _maskLogits = new Tensor<T>[_numStages];
        _inTransform = new FullyConnectedLayer<T>[_numStages];
        _gateTransform = new FullyConnectedLayer<T>[_numStages];
        for (int s = 0; s < _numStages; s++)
        {
            // Small random mask logits so the initial softmax mask is near-uniform but breaks symmetry.
            var logits = new Tensor<T>(new[] { numFeatures });
            for (int f = 0; f < numFeatures; f++)
                logits[f] = NumOps.FromDouble((rand.NextDouble() * 2.0 - 1.0) * 0.1);
            _maskLogits[s] = logits;
            RegisterTrainableParameter(logits, PersistentTensorRole.Weights);

            _inTransform[s] = new FullyConnectedLayer<T>(2 * numFeatures, (IActivationFunction<T>?)null);
            _gateTransform[s] = new FullyConnectedLayer<T>(numFeatures, (IActivationFunction<T>?)null);
            RegisterSubLayer(_inTransform[s]);
            RegisterSubLayer(_gateTransform[s]);
        }

        // Constant 0/1 selection matrices for the tape-safe GLU column split of the [*, 2F] linear
        // output into value/gate halves (a matmul split keeps the autodiff chain connected, unlike
        // a manual element copy).
        _valueSelector = new Tensor<T>(new[] { 2 * numFeatures, numFeatures });
        _gateSelector = new Tensor<T>(new[] { 2 * numFeatures, numFeatures });
        for (int i = 0; i < numFeatures; i++)
        {
            _valueSelector[i, i] = NumOps.One;
            _gateSelector[numFeatures + i, i] = NumOps.One;
        }

        ResolveShapes(new[] { numFeatures }, new[] { numFeatures });
        _built = true;

        // Probe forward to resolve the lazy FullyConnectedLayer shapes so ParameterCount /
        // GetParameters / Clone work before any real forward.
        var probe = new Tensor<T>(new[] { 1, numFeatures });
        Forward(probe);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int features = input.Shape[input.Rank - 1];
        if (!_built || _numFeatures != features)
        {
            BuildComponents(features);
        }

        int batch = features > 0 ? input.Length / features : 1;
        var h = Engine.Reshape(input, new[] { batch, _numFeatures });

        var one = NumOps.One;
        var minusOne = NumOps.FromDouble(-1.0);
        for (int s = 0; s < _numStages; s++)
        {
            // (1) Learnable feature selection: softmax over the per-stage mask logits, broadcast
            // over the batch and applied to the running representation.
            var maskRow = Engine.Reshape(_maskLogits![s], new[] { 1, _numFeatures });
            var mask = Engine.Softmax(maskRow);                           // [1, F]
            var feat = Engine.TensorBroadcastMultiply(h, mask);           // [batch, F]

            // (2) Gated transformation (GLU): linear -> split into value/gate halves -> value ⊙ σ(gate).
            var z = _inTransform![s].Forward(feat);                       // [batch, 2F]
            var values = Engine.TensorMatMul(z, _valueSelector!);         // [batch, F]
            var gates = Engine.TensorMatMul(z, _gateSelector!);           // [batch, F]
            var glu = Engine.TensorMultiply(values, Engine.Sigmoid(gates));

            // (3) Gated residual update: h <- g ⊙ glu + (1 - g) ⊙ h.
            var g = Engine.Sigmoid(_gateTransform![s].Forward(feat));     // [batch, F]
            var oneMinusG = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(g, minusOne), one);
            h = Engine.TensorAdd(Engine.TensorMultiply(g, glu), Engine.TensorMultiply(oneMinusG, h));
        }

        return h;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
        for (int s = 0; s < _numStages; s++)
        {
            var m = _maskLogits![s];
            for (int i = 0; i < m.Length; i++) all.Add(m[i]);
        }
        foreach (var sub in GetSubLayers())
        {
            var p = sub.GetParameters();
            for (int i = 0; i < p.Length; i++) all.Add(p[i]);
        }
        var result = new Vector<T>(all.Count);
        for (int i = 0; i < all.Count; i++) result[i] = all[i];
        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        for (int s = 0; s < _numStages; s++)
        {
            var m = _maskLogits![s];
            for (int i = 0; i < m.Length; i++) m[i] = parameters[offset++];
        }
        foreach (var sub in GetSubLayers())
        {
            int count = AiDotNet.Helpers.ParameterCountHelper.ToFlatVectorSize(sub.ParameterCount);
            if (count == 0) continue;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            sub.SetParameters(p);
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Mask logits are tape-updated (registered trainable tensors); the linear sub-layers update
        // their own parameters.
        foreach (var sub in GetSubLayers()) sub.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var sub in GetSubLayers()) sub.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var meta = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        meta["NumStages"] = _numStages.ToString(inv);
        meta["NumFeatures"] = _numFeatures.ToString(inv);
        return meta;
    }
}
