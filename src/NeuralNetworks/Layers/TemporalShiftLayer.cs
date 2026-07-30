using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Temporal Shift Module: moves a fraction of the channels one step forward in time and another
/// fraction one step backward, leaving the rest in place.
/// </summary>
/// <remarks>
/// <para>Introduced by Lin et al. (TSM) and used as BSVD's training-time temporal fusion in
/// Qi et al., 2022, "Real-time Streaming Video Denoising with Bidirectional Buffers"
/// (arXiv:2207.06937). BSVD swaps these blocks for Bidirectional Buffer Blocks at inference to
/// obtain streaming operation; the two are equivalent in what they compute, differing only in
/// whether the neighbouring frames come from a buffer or from the batch.</para>
/// <para>Per the paper, <c>floor(C / r)</c> channels shift per direction, with <c>r = 8</c> the
/// default shifted-channel ratio. Shifting a minority of channels lets information cross frame
/// boundaries at zero parameter and near-zero FLOP cost, which is what makes temporal fusion
/// affordable for real-time denoising — without it the network sees each frame in isolation and
/// the model is a per-frame denoiser wearing a video name.</para>
/// <para><b>Shapes:</b> rank-4 <c>[T, C, H, W]</c> or rank-5 <c>[B, T, C, H, W]</c>. Frames that
/// would shift in from outside the clip are zero-filled, matching the standard TSM boundary.</para>
/// <para><b>Gradient tracking:</b> built from <c>Engine.TensorNarrow</c> and
/// <c>Engine.TensorConcatenate</c>, both of which stay on the autodiff tape, so the shift is
/// differentiable and carries gradients back to the frames it moved.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, ChangesShape = false, ExpectedInputRank = 4, Cost = ComputeCost.Low, TestInputShape = "2, 8, 4, 4", TestConstructorArgs = "8")]
public partial class TemporalShiftLayer<T> : LayerBase<T>
{
    private readonly int _shiftedChannelRatio;

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <summary>Initializes a new temporal shift module.</summary>
    /// <param name="shiftedChannelRatio">
    /// The paper's <c>r</c>: <c>floor(C / r)</c> channels shift per direction. Defaults to 8.
    /// </param>
    public TemporalShiftLayer(
        [LayerState] int shiftedChannelRatio = 8)
        : base(new[] { -1 }, new[] { -1 })
    {
        if (shiftedChannelRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(shiftedChannelRatio), "Shifted-channel ratio must be positive.");

        _shiftedChannelRatio = shiftedChannelRatio;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank != 4 && rank != 5)
            throw new ArgumentException($"TemporalShiftLayer expects rank-4 [T, C, H, W] or rank-5 [B, T, C, H, W], got rank {rank}.", nameof(input));

        // Temporal axis is 0 when unbatched, 1 when batched; channels follow it.
        int timeAxis = rank == 4 ? 0 : 1;
        int channelAxis = timeAxis + 1;

        int frames = input.Shape[timeAxis];
        int channels = input.Shape[channelAxis];

        int shift = channels / _shiftedChannelRatio;

        // A single frame has no neighbour to shift from, and too few channels to split means
        // there is nothing to move; either way the module is a no-op rather than an error.
        if (frames < 2 || shift < 1) return input;

        // Split channels into: [0, shift) shift forward in time, [shift, 2*shift) shift
        // backward, the remainder untouched.
        var forwardGroup = Engine.TensorNarrow(input, dim: channelAxis, start: 0, length: shift);
        var backwardGroup = Engine.TensorNarrow(input, dim: channelAxis, start: shift, length: Math.Min(shift, channels - shift));

        int consumed = shift + backwardGroup.Shape[channelAxis];
        var staticGroup = consumed < channels
            ? Engine.TensorNarrow(input, dim: channelAxis, start: consumed, length: channels - consumed)
            : null;

        var shiftedForward = ShiftAlongTime(forwardGroup, timeAxis, frames, forward: true);
        var shiftedBackward = ShiftAlongTime(backwardGroup, timeAxis, frames, forward: false);

        var parts = staticGroup is null
            ? new[] { shiftedForward, shiftedBackward }
            : new[] { shiftedForward, shiftedBackward, staticGroup };

        return Engine.TensorConcatenate(parts, axis: channelAxis);
    }

    /// <summary>
    /// Shifts a channel group one step along the temporal axis, zero-filling the frame that
    /// would come from outside the clip.
    /// </summary>
    private Tensor<T> ShiftAlongTime(Tensor<T> group, int timeAxis, int frames, bool forward)
    {
        // forward:  out[t] = in[t - 1], so drop the last frame and prepend a zero frame.
        // backward: out[t] = in[t + 1], so drop the first frame and append a zero frame.
        var kept = forward
            ? Engine.TensorNarrow(group, dim: timeAxis, start: 0, length: frames - 1)
            : Engine.TensorNarrow(group, dim: timeAxis, start: 1, length: frames - 1);

        var padShape = new int[group.Shape.Length];
        for (int i = 0; i < padShape.Length; i++) padShape[i] = group.Shape[i];
        padShape[timeAxis] = 1;
        var zeroFrame = new Tensor<T>(padShape);

        var ordered = forward
            ? new[] { zeroFrame, kept }
            : new[] { kept, zeroFrame };

        return Engine.TensorConcatenate(ordered, axis: timeAxis);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => new(0);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException("TemporalShiftLayer has no parameters.", nameof(parameters));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ShiftedChannelRatio"] = _shiftedChannelRatio.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
    }
}
