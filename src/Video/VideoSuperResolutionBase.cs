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
public abstract class VideoSuperResolutionBase<T> : VideoNeuralNetworkBase<T>
{
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
    protected virtual Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        int height = frame1.Shape[^2];
        int width = frame1.Shape[^1];
        return new Tensor<T>([2, height, width]);
    }

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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Upscale(input);
    }
}
