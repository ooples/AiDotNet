using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

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
public class TabMEnsembleLayer<T> : LayerBase<T>
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
    public TabMEnsembleLayer(int numFeatures, int[] hiddenDimensions, int outputDim, int numMembers = 8)
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

    private void BuildComponents(int numFeatures)
    {
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
    public override Tensor<T> Forward(Tensor<T> input)
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
            int count = checked((int)sub.ParameterCount);
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
        var m = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        m["OutputDim"] = _outputDim.ToString(inv);
        m["NumMembers"] = _numMembers.ToString(inv);
        m["HiddenDimensions"] = string.Join(",", _hiddenDimensions);
        return m;
    }
}
