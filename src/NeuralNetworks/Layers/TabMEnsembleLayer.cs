using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// TabM ensemble MLP (Gorishniy et al. 2024, "TabM: Advancing Tabular Deep Learning with
/// Parameter-Efficient Ensembling"): an MLP whose linear layers are <see cref="BatchEnsembleLayer{T}"/>
/// (k members sharing one weight matrix via per-member rank-1 r/s adapters). The input is tiled once
/// across the k members, each member runs through the full MLP, and the per-member predictions are
/// averaged into a single output.
/// </summary>
/// <remarks>
/// <para>
/// Implemented as one composite layer (held in a model's <c>Layers</c> list like any other layer).
/// The batch is expanded to <c>[batch * k, .]</c> once by the first BatchEnsemble layer; subsequent
/// layers use <see cref="BatchEnsembleLayer{T}.ForwardExpanded"/> so the member axis persists without
/// re-tiling, and the final layer's <see cref="BatchEnsembleLayer{T}.AverageMembers"/> collapses it.
/// All sub-layers are registered via <see cref="LayerBase{T}.RegisterSubLayer"/> and the forward is
/// all tape-recorded Engine ops, so gradients flow to every member's adapters and the shared weights.
/// Feature count adapts to the fed input width via a rebuild. Output: <c>[batch, outputDim]</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
// Tabular MLP: a flat [Batch, Features] table in, one prediction row per sample out. Roles taken from
// ForwardTraced, which reads the feature width off the LAST axis and reshapes to [batch, _numFeatures].
//
// Rank 2 only, and batch is NOT optional even though a rank-1 input runs. The rank-1 path sets
// batch = 1 and still returns a rank-2 [1, outputDim] - so it is a rank-CHANGING case, not the
// batch-elided form BatchOptional describes, and folding it in would have the contract promise a
// rank-1 output this layer never produces.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TabMEnsembleLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int[] _hiddenDimensions;
    private readonly int _outputDim;
    private readonly int _numMembers;

    private int _numFeatures = -1;
    private bool _built;
    private BatchEnsembleLayer<T>[]? _ensembleLayers;

    /// <summary>
    /// Initializes a new <see cref="TabMEnsembleLayer{T}"/>.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="hiddenDimensions">Hidden layer widths of the MLP.</param>
    /// <param name="outputDim">Output dimension (per-member prediction width, averaged at the end).</param>
    /// <param name="numMembers">Number of ensemble members (k).</param>
    public TabMEnsembleLayer(
        int numFeatures,
        int[] hiddenDimensions,
        int outputDim,
        int numMembers = 8)
        : base(new[] { numFeatures }, new[] { outputDim })
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));
        if (numMembers <= 0) throw new ArgumentOutOfRangeException(nameof(numMembers));

        _hiddenDimensions = hiddenDimensions ?? Array.Empty<int>();
        _outputDim = outputDim;
        _numMembers = numMembers;

        BuildComponents(numFeatures);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written because the feature width is replaced rather than carried. From
    /// <c>BuildComponents</c>: the MLP widths are <c>[numFeatures, hidden..., _outputDim]</c> and it
    /// resolves <c>ResolveShapes(new[] { numFeatures }, new[] { _outputDim })</c>; <c>ForwardTraced</c>
    /// then returns <c>AverageMembers(current)</c>, the per-member predictions collapsed to a single
    /// <c>[batch, outputDim]</c> row block, as the type's own summary states.
    /// </para>
    /// <para>
    /// The ensemble's member axis never appears in the contract, and that is correct rather than an
    /// omission: the k members are tiled INTO the batch axis by the first
    /// <c>BatchEnsembleLayer&lt;T&gt;</c> and averaged back out by the last, so the expansion to
    /// <c>[batch * k, .]</c> lives entirely inside this layer's forward and is invisible at its edges.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputDim)</c> is read off the constructor argument, and it survives the rebuild path:
    /// <c>BuildComponents</c> re-derives every width when the fed input width changes, but
    /// <c>_outputDim</c> is readonly and is re-used unchanged. The input width is the only thing that
    /// adapts.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _outputDim <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputDim)),
        };
    }

    private void BuildComponents(int numFeatures)
    {
        // On a rebuild (the fed input width differs from the eager ctor build), unregister the
        // previous BatchEnsemble sub-layers before creating new ones. RegisterSubLayer only
        // appends, so without this a width change would leave stale children in the registry —
        // inflating ParameterCount / GetParameters ordering and the optimizer's update walk.
        if (_built && _ensembleLayers is not null)
        {
            foreach (var l in _ensembleLayers) UnregisterSubLayer(l);
        }

        _numFeatures = numFeatures;

        // Layer widths: [numFeatures, hidden..., outputDim].
        var dims = new List<int> { numFeatures };
        dims.AddRange(_hiddenDimensions);
        dims.Add(_outputDim);

        _ensembleLayers = new BatchEnsembleLayer<T>[dims.Count - 1];
        for (int l = 0; l < _ensembleLayers.Length; l++)
        {
            _ensembleLayers[l] = new BatchEnsembleLayer<T>(dims[l], dims[l + 1], _numMembers);
            RegisterSubLayer(_ensembleLayers[l]);
        }

        ResolveShapes(new[] { numFeatures }, new[] { _outputDim });
        _built = true;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int features = input.Shape[input.Rank - 1];
        if (!_built || _numFeatures != features)
        {
            BuildComponents(features);
        }

        int batch = input.Rank == 1 ? 1 : input.Shape[0];
        var x = Engine.Reshape(input, new[] { batch, _numFeatures });

        // First layer tiles the batch across members ([batch, F] -> [batch*k, h0]); subsequent
        // layers run member-aware on the already-expanded batch with a ReLU between layers.
        var current = _ensembleLayers![0].Forward(x);
        for (int l = 1; l < _ensembleLayers.Length; l++)
        {
            current = Engine.ReLU(current);
            current = _ensembleLayers[l].ForwardExpanded(current);
        }

        // Average the k members' predictions into the final [batch, outputDim].
        return _ensembleLayers[_ensembleLayers.Length - 1].AverageMembers(current);
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
        var m = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        m["OutputDim"] = _outputDim.ToString(inv);
        m["NumMembers"] = _numMembers.ToString(inv);
        m["HiddenDimensions"] = string.Join(",", _hiddenDimensions);
        return m;
    }
}
