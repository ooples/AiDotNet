// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements CNN14's channel-preserving frequency/time aggregation for NCHW features.
/// </summary>
/// <remarks>
/// The released PANNs CNN14 first averages the mel-frequency axis, then adds the
/// maximum and mean across time. Engine reductions are used so gradients remain on
/// the training tape. It is an architectural detail of the PANNs factory rather than a
/// general channels-last global-pooling variant, but it is public like every other layer
/// so a caller can compose or subclass it directly.
/// </remarks>
// PANNs attention-free pooling: reduces the frequency axis then the temporal axis, so a rank-4
// [Batch, Channels, Time, Frequency] spectrogram collapses to [Batch, Channels]. Read straight off the
// forward - ReduceMean(axis 3) then ReduceMean/ReduceMax(axis 2), both keepDims: false.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, TensorAxis.Frames,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class PANNsPoolingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// Hand-written because the input and output RANKS differ, and the generator only derives a
    /// contract when it can pair an output layout with a same-rank input layout - a rank-reducing
    /// layer has no such pair, so it emits nothing and the interface would go unimplemented.
    /// The relation itself is read straight off the forward: <c>ReduceMean(axis 3)</c> collapses
    /// frequency, then <c>ReduceMean</c>/<c>ReduceMax(axis 2)</c> collapse time, both with
    /// <c>keepDims: false</c>. Batch and channels are carried through untouched.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
        };
    }

    public PANNsPoolingLayer()
        : base(new[] { -1, -1, -1 }, new[] { -1 })
    {
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(
                $"PANNsPoolingLayer expects NCHW rank-4 input, got rank {input.Rank}.",
                nameof(input));
        }

        ResolveShapes(
            new[] { input.Shape[1], input.Shape[2], input.Shape[3] },
            new[] { input.Shape[1] });
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // [B,C,T,F] -> mean_F -> [B,C,T] -> (max_T + mean_T) -> [B,C].
        var frequencyMean = Engine.ReduceMean(input, new[] { 3 }, keepDims: false);
        var temporalMean = Engine.ReduceMean(frequencyMean, new[] { 2 }, keepDims: false);
        var temporalMax = Engine.ReduceMax(frequencyMean, new[] { 2 }, keepDims: false, out _);
        return Engine.TensorAdd(temporalMax, temporalMean);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void ResetState() { }
}
