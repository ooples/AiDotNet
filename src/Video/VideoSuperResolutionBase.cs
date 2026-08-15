using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for video super-resolution models that upscale low-resolution video to higher resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video super-resolution extends image super-resolution by exploiting temporal information
/// across multiple frames. This base class provides:
///
/// - Scale factor management (2x, 4x, 8x upscaling)
/// - Tile-based inference for memory-efficient processing of high-resolution video
/// - Bicubic upsampling as fallback/initialization
/// - Temporal consistency utilities
///
/// Derived classes implement specific architectures like BasicVSR++, RVRT, RealBasicVSR, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video super-resolution makes low-resolution video sharper and more
/// detailed. For example, it can upscale a 480p video to 4K quality. Unlike single-image
/// methods, video SR uses information from neighboring frames for better quality and
/// temporal consistency (no flickering between frames).
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "A clip of low-resolution frames.")]
// The rank-4 layouts are REQUIRED, not decorative. [TensorLayout] is declared once PER ACCEPTED RANK,
// and the first version of this contract declared only the rank-5 clip form while claiming in a note
// that a single frame was "also accepted". The sweep then reported 25 declared / 0 agreed / 25
// DECLINED for the whole family: it feeds a rank-4 image, no rank-4 input layout existed, and the
// contract correctly refused to answer. A prose note is not a declaration.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "A single low-resolution frame - the degenerate one-frame clip.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output,
    Note = "The same clip with both spatial axes multiplied by ScaleFactor. Frame count and channel "
         + "count are carried through - upscaling changes resolution, not length.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output,
    Note = "A single frame with both spatial axes multiplied by ScaleFactor.")]
public abstract class VideoSuperResolutionBase<T> : VideoNeuralNetworkBase<T>, IVideoSuperResolution<T>, IShapeContract
{
    /// <summary>
    /// The super-resolution family's law: both spatial axes scale by <see cref="ScaleFactor"/> and
    /// every other axis is carried through unchanged.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One declaration serves the family because <see cref="ScaleFactor"/> lives on this base and each
    /// model sets its own (2x, 4x, 8x). The relation is <c>Scaled</c>, not a constant: a contract that
    /// recorded "512" would be right for one input size and wrong for every other, whereas
    /// <c>Scaled(Height, ScaleFactor)</c> is correct for resolutions nobody has run.
    /// </para>
    /// <para>
    /// Stated for BOTH video ranks. Rank 5 is the clip form; rank 4 is a single frame, which the
    /// harnesses build and which a VSR model handles as a one-frame clip. Declining on rank 4 would
    /// have made the family look unverifiable when it is simply being handed the degenerate case.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => SpatialUpscaleContract(inputRank);

    /// <summary>The family law, exposed so a model with an extra axis can still reuse it.</summary>
    protected IReadOnlyList<OutputAxisContract>? SpatialUpscaleContract(int inputRank)
    {
        int scale = ScaleFactor;
        if (scale <= 0) return null;

        AxisRelation Spatial(TensorAxis axis)
            => scale == 1 ? AxisRelation.Same(axis) : AxisRelation.Scaled(axis, scale);

        return inputRank switch
        {
            4 =>
            [
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
                new OutputAxisContract(TensorAxis.Height, Spatial(TensorAxis.Height)),
                new OutputAxisContract(TensorAxis.Width, Spatial(TensorAxis.Width)),
            ],
            5 =>
            [
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Frames, AxisRelation.Same(TensorAxis.Frames)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
                new OutputAxisContract(TensorAxis.Height, Spatial(TensorAxis.Height)),
                new OutputAxisContract(TensorAxis.Width, Spatial(TensorAxis.Width)),
            ],
            _ => null,
        };
    }

    /// <summary>
    /// Gets the spatial upscaling factor (e.g., 2 for 2x, 4 for 4x).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 4 means the output is 4x larger in both width and height.
    /// For example, 480x270 input becomes 1920x1080 output.
    /// </para>
    /// </remarks>
    public int ScaleFactor { get; protected set; } = 4;

    /// <summary>
    /// Gets or sets the tile size for memory-efficient tiled processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When processing high-resolution frames, the image is split into overlapping tiles
    /// to reduce memory usage. Set to 0 to disable tiling (process full frame).
    /// Common values: 128, 256, 512.
    /// </para>
    /// </remarks>
    public int TileSize { get; protected set; }

    /// <summary>
    /// Gets or sets the overlap between adjacent tiles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Overlap helps reduce seam artifacts at tile boundaries.
    /// Typical values: 16, 32 pixels.
    /// </para>
    /// </remarks>
    public int TileOverlap { get; protected set; } = 32;

    /// <summary>
    /// Initializes a new instance of the VideoSuperResolutionBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VideoSuperResolutionBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Upscales a sequence of video frames.
    /// </summary>
    /// <param name="lowResFrames">Low-resolution frames [numFrames, channels, height, width].</param>
    /// <returns>High-resolution frames [numFrames, channels, height*scale, width*scale].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method to upscale your video.
    /// Pass in low-resolution frames and get back high-resolution frames.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Upscale(Tensor<T> lowResFrames);

    /// <summary>
    /// Estimates optical flow between two frames for temporal alignment.
    /// Override this in derived classes to provide actual flow estimation.
    /// The default implementation returns a zero-flow tensor (no motion).
    /// </summary>
    /// <param name="frame1">First frame [channels, height, width].</param>
    /// <param name="frame2">Second frame [channels, height, width].</param>
    /// <returns>Optical flow field [2, height, width] representing (dx, dy) displacement.</returns>
    public virtual Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        // Delegates to the library's RAFT implementation (Teed & Deng 2020) instead of returning zeros.
        //
        // The previous default returned `new Tensor<T>([2, height, width])` — an all-zero flow field,
        // i.e. "no motion" — and NO video super-resolution model overrode it. Every temporal-alignment
        // path in all 26 VSR models was therefore a no-op: warping a neighbouring frame by zero flow
        // just returns that frame unchanged, so the models were doing per-frame image upscaling while
        // their architectures claimed to exploit motion. Temporal alignment is the whole distinction
        // between video and image super-resolution.
        //
        // RAFT<T> already exists in this assembly (src/Video/Motion/RAFT.cs) and is the estimator the
        // VSR literature standardly builds on, so this reuses it rather than adding a second one.
        // Created lazily and cached: constructing it per call would allocate a full flow network on
        // every frame pair.
        _flowEstimator ??= new AiDotNet.Video.Motion.RAFT<T>();
        return _flowEstimator.EstimateFlow(frame1, frame2);
    }

    /// <summary>Lazily created RAFT estimator backing the default <see cref="EstimateFlow"/>.</summary>
    private AiDotNet.Video.Motion.RAFT<T>? _flowEstimator;

    /// <summary>
    /// Performs bilinear upsampling as a baseline or initialization.
    /// </summary>
    /// <param name="input">Input tensor [channels, height, width].</param>
    /// <param name="scale">Upscaling factor.</param>
    /// <returns>Upsampled tensor [channels, height*scale, width*scale].</returns>
    protected Tensor<T> BilinearUpsample(Tensor<T> input, int scale)
    {
        int channels = input.Shape[0];
        int height = input.Shape[1];
        int width = input.Shape[2];
        int outHeight = height * scale;
        int outWidth = width * scale;

        var output = new Tensor<T>([channels, outHeight, outWidth]);

        for (int c = 0; c < channels; c++)
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    double srcH = (oh + 0.5) / scale - 0.5;
                    double srcW = (ow + 0.5) / scale - 0.5;

                    int h0 = Math.Max(0, Math.Min((int)Math.Floor(srcH), height - 1));
                    int w0 = Math.Max(0, Math.Min((int)Math.Floor(srcW), width - 1));
                    int h1 = Math.Max(0, Math.Min(h0 + 1, height - 1));
                    int w1 = Math.Max(0, Math.Min(w0 + 1, width - 1));

                    double hWeight = srcH - Math.Floor(srcH);
                    double wWeight = srcW - Math.Floor(srcW);

                    double v00 = NumOps.ToDouble(input.Data.Span[c * height * width + h0 * width + w0]);
                    double v01 = NumOps.ToDouble(input.Data.Span[c * height * width + h0 * width + w1]);
                    double v10 = NumOps.ToDouble(input.Data.Span[c * height * width + h1 * width + w0]);
                    double v11 = NumOps.ToDouble(input.Data.Span[c * height * width + h1 * width + w1]);

                    double val = v00 * (1 - hWeight) * (1 - wWeight)
                               + v01 * (1 - hWeight) * wWeight
                               + v10 * hWeight * (1 - wWeight)
                               + v11 * hWeight * wWeight;

                    output.Data.Span[c * outHeight * outWidth + oh * outWidth + ow] = NumOps.FromDouble(val);
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return Upscale(input);
    }
}
