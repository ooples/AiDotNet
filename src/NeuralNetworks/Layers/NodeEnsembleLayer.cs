using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// NODE ensemble: a set of differentiable oblivious decision trees run in PARALLEL on the same
/// input, with their outputs concatenated (Popov et al. 2019, "Neural Oblivious Decision Ensembles
/// for Deep Learning on Tabular Data").
/// </summary>
/// <remarks>
/// <para>
/// Each of the <c>numTrees</c> trees is an <see cref="ObliviousDecisionTreeLayer{T}"/> that sees the
/// full feature vector and produces a <c>[batch, treeOutputDim]</c> output; the ensemble
/// concatenates them into <c>[batch, numTrees * treeOutputDim]</c>, which a linear head maps to the
/// prediction. This is the defining NODE structure — a parallel ensemble — as opposed to stacking
/// trees sequentially (which would feed one tree's output into the next, a dimension mismatch and
/// not an ensemble).
/// </para>
/// <para>
/// Implemented as one composite layer: the trees are registered via
/// <see cref="LayerBase{T}.RegisterSubLayer"/> and the forward is tape-recorded Engine ops, so
/// gradients reach every tree. Feature count adapts to the fed width via a rebuild.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NodeEnsembleLayer<T> : LayerBase<T>
{
    private readonly int _numTrees;
    private readonly int _treeDepth;
    private readonly int _treeOutputDim;
    private readonly double _initScale;

    private int _numFeatures = -1;
    private bool _built;
    private ObliviousDecisionTreeLayer<T>[]? _trees;

    /// <summary>Initializes a NODE ensemble.</summary>
    /// <param name="numFeatures">Number of input features each tree sees.</param>
    /// <param name="numTrees">Number of parallel oblivious trees.</param>
    /// <param name="treeDepth">Depth of each oblivious tree.</param>
    /// <param name="treeOutputDim">Output dimension per tree.</param>
    /// <param name="initScale">Parameter initialization scale for the trees.</param>
    public NodeEnsembleLayer(int numFeatures, int numTrees = 20, int treeDepth = 6,
        int treeOutputDim = 3, double initScale = 0.01)
        : base(new[] { numFeatures }, new[] { numTrees * treeOutputDim })
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (numTrees <= 0) throw new ArgumentOutOfRangeException(nameof(numTrees));
        if (treeOutputDim <= 0) throw new ArgumentOutOfRangeException(nameof(treeOutputDim));
        _numTrees = numTrees;
        _treeDepth = treeDepth;
        _treeOutputDim = treeOutputDim;
        _initScale = initScale;
        BuildComponents(numFeatures);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    private void BuildComponents(int numFeatures)
    {
        // On a rebuild (fed width differs from the eager ctor build), unregister the previous trees
        // before recreating — RegisterSubLayer only appends.
        if (_built && _trees is not null)
        {
            foreach (var t in _trees) UnregisterSubLayer(t);
        }

        _numFeatures = numFeatures;
        _trees = new ObliviousDecisionTreeLayer<T>[_numTrees];
        for (int i = 0; i < _numTrees; i++)
        {
            // Eager ctor (inputDim given) so each tree is shape-resolved immediately.
            _trees[i] = new ObliviousDecisionTreeLayer<T>(numFeatures, _treeDepth, _treeOutputDim, _initScale);
            RegisterSubLayer(_trees[i]);
        }

        ResolveShapes(new[] { numFeatures }, new[] { _numTrees * _treeOutputDim });
        _built = true;
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
        var x = Engine.Reshape(input, new[] { batch, _numFeatures });

        // Run every tree on the SAME input (parallel ensemble) and concatenate along the feature
        // axis into [batch, numTrees * treeOutputDim].
        var outs = new Tensor<T>[_numTrees];
        for (int i = 0; i < _numTrees; i++)
        {
            outs[i] = _trees![i].Forward(x);
        }
        return Engine.TensorConcatenate(outs, axis: 1);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
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
        meta["NumTrees"] = _numTrees.ToString(inv);
        meta["TreeDepth"] = _treeDepth.ToString(inv);
        meta["TreeOutputDim"] = _treeOutputDim.ToString(inv);
        meta["NumFeatures"] = _numFeatures.ToString(inv);
        return meta;
    }
}
