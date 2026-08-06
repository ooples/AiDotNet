using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Temporal Processor Module (TPM) from Stream-DiffVSR (Shiu et al., 2025): aligns and fuses the
/// current frame's features with the previous frame's warped features to enforce temporal coherence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> when a video is upscaled frame by frame, each frame can look good on its own
/// while the sequence still shimmers, because the model makes slightly different choices each time.
/// This module gives the decoder a memory: it takes the previous frame's features (already motion-
/// aligned to the current frame) and blends them into the current features, so detail stays put instead
/// of flickering.
/// </para>
/// <para>
/// <b>Paper.</b> Section "Temporal Processor Module (TPM)": these modules are integrated "after each
/// spatial convolutional layer in the decoder to explicitly inject temporal coherence", they "utilize
/// latent features from the current frame and warped features from the previous frame", and per Fig. 2
/// the module "aligns and fuses these features via interpolation, convolution, and weighted fusion".
/// The decoder applies them at several resolutions as a "multi-scale fusion strategy".
/// </para>
/// <para>
/// This implementation follows that description literally, in three stages:
/// </para>
/// <para>
/// 1. <b>Interpolation</b> — the previous features are bilinearly resampled to the current features'
/// spatial size. This is what lets one module type sit at every decoder resolution: the previous
/// frame's features arrive at whatever scale they were produced at.
/// </para>
/// <para>
/// 2. <b>Convolution</b> — the resampled previous features are projected to the current channel count,
/// then the concatenation of current and aligned-previous features is convolved to produce a temporal
/// correction.
/// </para>
/// <para>
/// 3. <b>Weighted fusion</b> — a learned per-position, per-channel gate (a 1x1 convolution followed by
/// a sigmoid) decides how much of that correction to apply, and the result is added to the current
/// features. The gate matters: how much to trust the previous frame is content-dependent, since at an
/// occlusion or a scene cut the warped features are simply wrong, and a fixed blend weight would drag
/// stale content into the new frame.
/// </para>
/// <para>
/// <b>First frame.</b> An online model has no previous output for the first frame. Passing
/// <c>null</c> as the previous features returns the current features unchanged, which is the correct
/// degenerate case — spatial-only reconstruction — rather than fabricating a zero "previous frame"
/// whose features would actively pull the first frame towards black.
/// </para>
/// </remarks>
public partial class TemporalProcessorModule<T> : LayerBase<T>
{
    #region Fields

    /// <summary>Projects the interpolated previous features to the current channel count.</summary>
    private ConvolutionalLayer<T>? _alignPrevious;

    /// <summary>Produces the temporal correction from [current, aligned-previous].</summary>
    private ConvolutionalLayer<T>? _fuse;

    /// <summary>Produces the per-position fusion weights (the "weighted" in weighted fusion).</summary>
    private ConvolutionalLayer<T>? _gate;

    private readonly int _kernelSize;
    private readonly int _padding;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a temporal processor module.
    /// </summary>
    /// <param name="kernelSize">Spatial extent of the fusion convolution. Default 3: temporal
    /// correction after flow warping is a local refinement, so a 3x3 neighbourhood is what the residual
    /// misalignment spans.</param>
    public TemporalProcessorModule(int kernelSize = 3)
        : base([-1, -1, -1], [-1, -1, -1])
    {
        if (kernelSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(kernelSize), kernelSize, "Kernel size must be positive.");
        }

        _kernelSize = kernelSize;
        _padding = (kernelSize - 1) / 2;

        // The convolutions are built on the first forward, not here: a convolution's output depth is
        // fixed at construction, and a TPM sits after every spatial convolution in the decoder, so the
        // channel width it must match is only known once real features arrive.
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Fuses current features with the previous frame's warped features.
    /// </summary>
    /// <param name="current">Current-frame features, <c>[C,H,W]</c> or <c>[B,C,H,W]</c>.</param>
    /// <param name="warpedPrevious">The previous frame's features, already motion-warped into this
    /// frame. May be at a different spatial size; may be <c>null</c> for the first frame.</param>
    /// <returns>Temporally stabilized features with the same shape as <paramref name="current"/>.</returns>
    public Tensor<T> Forward(Tensor<T> current, Tensor<T>? warpedPrevious)
    {
        if (current is null) throw new ArgumentNullException(nameof(current));

        // First frame: no temporal context exists, so pass the spatial features through untouched.
        if (warpedPrevious is null) return current;

        var shape = current.Shape;
        if (shape.Length is not (3 or 4))
        {
            throw new ArgumentException(
                $"TemporalProcessorModule expects [C,H,W] or [B,C,H,W]; got rank {shape.Length}.",
                nameof(current));
        }

        int channelAxis = shape.Length == 4 ? 1 : 0;
        int channels = shape[channelAxis];
        int height = shape[shape.Length - 2];
        int width = shape[shape.Length - 1];

        EnsureResolved(current, channels);

        // 1. Interpolation — bring the previous features onto the current spatial grid.
        var prev = warpedPrevious;
        int prevH = prev.Shape[prev.Shape.Length - 2];
        int prevW = prev.Shape[prev.Shape.Length - 1];
        if (prevH != height || prevW != width)
        {
            prev = Engine.Interpolate(
                prev, [height, width], InterpolateMode.Bilinear, alignCorners: false);
        }

        // 2. Convolution — project the previous features to this level's width, then convolve the
        //    concatenation to produce a temporal correction. EnsureResolved has built all three
        //    convolutions by this point.
        var align = _alignPrevious ?? throw new InvalidOperationException(
            "TemporalProcessorModule convolutions were not built; shape resolution did not run.");
        var fuse = _fuse ?? throw new InvalidOperationException(
            "TemporalProcessorModule convolutions were not built; shape resolution did not run.");
        var gate = _gate ?? throw new InvalidOperationException(
            "TemporalProcessorModule convolutions were not built; shape resolution did not run.");

        var aligned = align.Forward(prev);
        var joined = Engine.Concat([current, aligned], channelAxis);
        var correction = fuse.Forward(joined);

        // 3. Weighted fusion — a learned gate scales the correction before it is added.
        var weights = gate.Forward(joined);
        return Engine.TensorAdd(current, Engine.TensorMultiply(weights, correction));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Single-input form: no temporal context, so this is the first-frame path and returns the input
    /// unchanged. The two-argument overload is the one the decoder uses.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input) => Forward(input, null);

    /// <summary>
    /// Resolves the lazily sized sub-layers once the channel width at this decoder level is known.
    /// </summary>
    private void EnsureResolved(Tensor<T> current, int channels)
    {
        if (IsShapeResolved) return;

        int height = current.Shape[current.Shape.Length - 2];
        int width = current.Shape[current.Shape.Length - 1];

        // The align convolution maps the previous features to `channels`; fuse and gate consume the
        // 2*channels concatenation and emit `channels`.
        _alignPrevious = new ConvolutionalLayer<T>(
            outputDepth: channels, kernelSize: 1, stride: 1, padding: 0, activationFunction: null);

        _fuse = new ConvolutionalLayer<T>(
            outputDepth: channels, kernelSize: _kernelSize, stride: 1, padding: _padding,
            activationFunction: null);

        // Sigmoid keeps the fusion weight in [0, 1], so the gate can only interpolate towards the
        // temporal correction, never amplify it into an unbounded contribution.
        _gate = new ConvolutionalLayer<T>(
            outputDepth: channels, kernelSize: 1, stride: 1, padding: 0,
            activationFunction: (IActivationFunction<T>)new SigmoidActivation<T>());

        _alignPrevious.ResolveFromShape([channels, height, width]);
        _fuse.ResolveFromShape([channels * 2, height, width]);
        _gate.ResolveFromShape([channels * 2, height, width]);

        // Registration propagates the current training mode to each child.
        RegisterSubLayer(_alignPrevious);
        RegisterSubLayer(_fuse);
        RegisterSubLayer(_gate);

        ResolveShapes([channels, height, width], [channels, height, width]);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Sub-layers in one fixed order, shared by every parameter-facing override. Empty until the first
    /// forward pass builds them, so a caller that queries parameters before then sees zero rather than
    /// a crash.
    /// </summary>
    private IEnumerable<ILayer<T>> AllSubLayers()
    {
        if (_alignPrevious is null || _fuse is null || _gate is null) yield break;

        yield return _alignPrevious;
        yield return _fuse;
        yield return _gate;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
        foreach (var layer in AllSubLayers())
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) all.Add(p[i]);
        }

        return new Vector<T>([.. all]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int expected = 0;
        foreach (var layer in AllSubLayers()) expected += layer.GetParameters().Length;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters for this TemporalProcessorModule, got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;
        foreach (var layer in AllSubLayers())
        {
            int count = layer.GetParameters().Length;
            if (count == 0) continue;
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Concatenates sub-layer gradients in the same order as <see cref="GetParameters"/>. The base
    /// implementation returns only this layer's own buffer, which for a composite is an all-zero vector
    /// of the aggregate length.
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var all = new List<T>();
        foreach (var layer in AllSubLayers())
        {
            var g = layer.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) all.Add(g[i]);
        }

        return new Vector<T>([.. all]);
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        foreach (var layer in AllSubLayers()) layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in AllSubLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in AllSubLayers()) layer.ResetState();
    }

    #endregion
}
