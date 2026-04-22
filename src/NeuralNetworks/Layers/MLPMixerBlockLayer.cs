using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single MLP-Mixer block: temporal-axis MLP + channel-axis MLP, each wrapped with
/// pre-norm and residual, per Tolstikhin et al. 2021 "MLP-Mixer: An all-MLP Architecture
/// for Vision" (extended to time-series patches by Ekambaram et al. 2024 for Tiny Time
/// Mixers).
/// </summary>
/// <remarks>
/// <para>
/// Operates on input tensors of shape <c>[B, numPatches, hiddenDim]</c>. The forward sequence is:
/// </para>
/// <list type="number">
/// <item><description>norm1(input) → transpose → Dense(numPatches → expanded → numPatches) [GELU] → transpose → residual</description></item>
/// <item><description>norm2(x) → Dense(hiddenDim → expanded → hiddenDim) [GELU] per-patch → residual</description></item>
/// </list>
/// <para>
/// The temporal mixer mixes information across patches (time dimension); the channel
/// mixer mixes information across hidden features at each patch position. The transpose
/// is required so a plain <see cref="DenseLayer{T}"/> (which operates on the last axis)
/// can be applied across the patch axis.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric element type.</typeparam>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.Medium, TestInputShape = "4, 8", TestConstructorArgs = "4, 8, 2")]
public class MLPMixerBlockLayer<T> : LayerBase<T>
{
    private readonly int _numPatches;
    private readonly int _hiddenDim;
    private readonly int _expansionFactor;

    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly TransposeLayer<T> _toPatchAxis;
    private readonly DenseLayer<T> _temporalMlpExpand;
    private readonly DenseLayer<T> _temporalMlpContract;
    private readonly TransposeLayer<T> _fromPatchAxis;

    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DenseLayer<T> _channelMlpExpand;
    private readonly DenseLayer<T> _channelMlpContract;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="MLPMixerBlockLayer{T}"/>.
    /// </summary>
    /// <param name="numPatches">Sequence length (axis that the temporal mixer operates on).</param>
    /// <param name="hiddenDim">Per-patch hidden dimension (axis that the channel mixer operates on).</param>
    /// <param name="expansionFactor">Expansion ratio for both mixer MLPs (hidden = dim * expansionFactor).</param>
    public MLPMixerBlockLayer(int numPatches, int hiddenDim, int expansionFactor = 2)
        : base(new[] { numPatches, hiddenDim }, new[] { numPatches, hiddenDim })
    {
        if (numPatches < 1) throw new ArgumentOutOfRangeException(nameof(numPatches));
        if (hiddenDim < 1) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (expansionFactor < 1) throw new ArgumentOutOfRangeException(nameof(expansionFactor));

        _numPatches = numPatches;
        _hiddenDim = hiddenDim;
        _expansionFactor = expansionFactor;

        int tempExpanded = numPatches * expansionFactor;
        int chanExpanded = hiddenDim * expansionFactor;

        _norm1 = new LayerNormalizationLayer<T>(hiddenDim);

        // Temporal mixer: [B, numPatches, hiddenDim] -> transpose -> [B, hiddenDim, numPatches]
        //                 -> Dense(numPatches -> tempExpanded, GELU) -> Dense(tempExpanded -> numPatches)
        //                 -> transpose -> [B, numPatches, hiddenDim]
        _toPatchAxis = new TransposeLayer<T>(new[] { numPatches, hiddenDim }, new[] { 1, 0 });
        _temporalMlpExpand = new DenseLayer<T>(
            inputSize: numPatches,
            outputSize: tempExpanded,
            activationFunction: new GELUActivation<T>());
        _temporalMlpContract = new DenseLayer<T>(
            inputSize: tempExpanded,
            outputSize: numPatches,
            activationFunction: null);
        _fromPatchAxis = new TransposeLayer<T>(new[] { hiddenDim, numPatches }, new[] { 1, 0 });

        _norm2 = new LayerNormalizationLayer<T>(hiddenDim);

        // Channel mixer: per-patch Dense(hiddenDim -> chanExpanded, GELU) -> Dense(chanExpanded -> hiddenDim).
        // DenseLayer operates on the last axis, so no transpose is needed here.
        _channelMlpExpand = new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: chanExpanded,
            activationFunction: new GELUActivation<T>());
        _channelMlpContract = new DenseLayer<T>(
            inputSize: chanExpanded,
            outputSize: hiddenDim,
            activationFunction: null);

        RegisterSubLayer(_norm1);
        RegisterSubLayer(_toPatchAxis);
        RegisterSubLayer(_temporalMlpExpand);
        RegisterSubLayer(_temporalMlpContract);
        RegisterSubLayer(_fromPatchAxis);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_channelMlpExpand);
        RegisterSubLayer(_channelMlpContract);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Temporal mixing: norm → transpose → MLP across patches → transpose → residual.
        var normed1 = _norm1.Forward(input);
        var transposed = _toPatchAxis.Forward(normed1);
        var tempHidden = _temporalMlpExpand.Forward(transposed);
        var tempOut = _temporalMlpContract.Forward(tempHidden);
        var unTransposed = _fromPatchAxis.Forward(tempOut);
        var x = Engine.TensorAdd(input, unTransposed);

        // Channel mixing: norm → per-patch MLP across hidden dim → residual.
        var normed2 = _norm2.Forward(x);
        var chanHidden = _channelMlpExpand.Forward(normed2);
        var chanOut = _channelMlpContract.Forward(chanHidden);
        var output = Engine.TensorAdd(x, chanOut);
        return output;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _temporalMlpExpand.UpdateParameters(learningRate);
        _temporalMlpContract.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _channelMlpExpand.UpdateParameters(learningRate);
        _channelMlpContract.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>>
        {
            _norm1.GetParameters(),
            _temporalMlpExpand.GetParameters(),
            _temporalMlpContract.GetParameters(),
            _norm2.GetParameters(),
            _channelMlpExpand.GetParameters(),
            _channelMlpContract.GetParameters(),
        };
        int total = 0;
        foreach (var p in parts) total += p.Length;
        var combined = new T[total];
        int offset = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < p.Length; i++)
                combined[offset + i] = p[i];
            offset += p.Length;
        }
        return new Vector<T>(combined);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm1.ResetState();
        _toPatchAxis.ResetState();
        _temporalMlpExpand.ResetState();
        _temporalMlpContract.ResetState();
        _fromPatchAxis.ResetState();
        _norm2.ResetState();
        _channelMlpExpand.ResetState();
        _channelMlpContract.ResetState();
    }
}
