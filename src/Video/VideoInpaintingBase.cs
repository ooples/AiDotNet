using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for video inpainting models that fill in missing or masked regions in video sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video inpainting fills in missing, damaged, or unwanted regions in video while maintaining
/// temporal consistency. This base class provides:
///
/// - Binary mask handling for specifying regions to inpaint
/// - Temporal propagation utilities for consistent fills across frames
/// - Completion quality metrics (PSNR, SSIM within masked regions)
///
/// Derived classes implement specific architectures like STTN, FuseFormer, ProPainter, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video inpainting is like a smart "eraser" for video. You can mark
/// areas you want to remove (like a watermark, a person, or damage), and the model fills
/// those areas with realistic content that matches the surrounding video. It uses information
/// from other frames to figure out what should be there, making the result look natural and
/// consistent across the whole video.
/// </para>
/// </remarks>
public abstract class VideoInpaintingBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets whether this model supports temporal propagation for inpainting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Temporal propagation fills regions using information from neighboring frames,
    /// producing more consistent results than frame-by-frame inpainting.
    /// </para>
    /// </remarks>
    public bool SupportsTemporalPropagation { get; protected set; } = true;

    /// <summary>
    /// Gets the maximum supported mask ratio (fraction of frame that can be masked).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most models work well up to 30-50% mask coverage. Beyond that, quality degrades
    /// as there is insufficient context for reasonable fill-in.
    /// </para>
    /// </remarks>
    public double MaxMaskRatio { get; protected set; } = 0.5;

    /// <summary>
    /// Initializes a new instance of the VideoInpaintingBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VideoInpaintingBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Inpaints masked regions in a video sequence.
    /// </summary>
    /// <param name="frames">Input video frames [numFrames, channels, height, width].</param>
    /// <param name="masks">Binary masks [numFrames, 1, height, width] where 1 indicates regions to inpaint.</param>
    /// <returns>Inpainted video frames [numFrames, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in your video and a mask showing what to remove.
    /// The mask is a black and white image where white (1) marks the areas to fill in.
    /// The model will replace those areas with realistic content.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks);

    /// <summary>
    /// Propagates known pixel values from neighboring frames to fill masked regions.
    /// </summary>
    /// <param name="frames">Input frames with masked regions.</param>
    /// <param name="masks">Binary masks indicating regions to fill.</param>
    /// <param name="flows">Optical flow fields between consecutive frames.</param>
    /// <returns>Frames with propagated content in previously masked regions.</returns>
    protected virtual Tensor<T> PropagateTemporally(
        Tensor<T> frames,
        Tensor<T> masks,
        List<Tensor<T>> flows)
    {
        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        var result = new Tensor<T>(frames._shape);

        // Copy original frames
        for (int i = 0; i < frames.Length; i++)
        {
            result.Data.Span[i] = frames.Data.Span[i];
        }

        // Forward propagation: fill masked regions using previous frames
        for (int f = 1; f < numFrames; f++)
        {
            if (f - 1 < flows.Count)
            {
                var prevFrame = ExtractFrame(result, f - 1);
                var warped = WarpFeature(prevFrame, flows[f - 1]);

                int frameOffset = f * channels * height * width;
                int maskOffset = f * height * width;

                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            double maskVal = NumOps.ToDouble(masks.Data.Span[maskOffset + h * width + w]);
                            if (maskVal > 0.5)
                            {
                                int idx = frameOffset + c * height * width + h * width + w;
                                int warpIdx = c * height * width + h * width + w;
                                result.Data.Span[idx] = warped.Data.Span[warpIdx];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes PSNR only within masked regions for inpainting quality assessment.
    /// </summary>
    /// <param name="inpainted">Inpainted result.</param>
    /// <param name="groundTruth">Ground truth.</param>
    /// <param name="masks">Binary masks indicating inpainted regions.</param>
    /// <returns>PSNR value in decibels for masked regions only.</returns>
    public double ComputeMaskedPSNR(Tensor<T> inpainted, Tensor<T> groundTruth, Tensor<T> masks)
    {
        double sumSquaredError = 0;
        int maskedPixelCount = 0;

        int numFrames = inpainted.Shape[0];
        int channels = inpainted.Shape[1];
        int height = inpainted.Shape[2];
        int width = inpainted.Shape[3];

        for (int f = 0; f < numFrames; f++)
        {
            int maskOffset = f * height * width;
            int frameOffset = f * channels * height * width;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double maskVal = NumOps.ToDouble(masks.Data.Span[maskOffset + h * width + w]);
                    if (maskVal > 0.5)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            int idx = frameOffset + c * height * width + h * width + w;
                            double pred = NumOps.ToDouble(inpainted.Data.Span[idx]);
                            double gt = NumOps.ToDouble(groundTruth.Data.Span[idx]);
                            double diff = pred - gt;
                            sumSquaredError += diff * diff;
                        }
                        maskedPixelCount += channels;
                    }
                }
            }
        }

        if (maskedPixelCount == 0) return double.PositiveInfinity;

        double mse = sumSquaredError / maskedPixelCount;
        if (mse < 1e-10) return 100.0;

        return 10.0 * Math.Log10(1.0 / mse);
    }

    /// <summary>
    /// Builds the default single-channel hole mask <c>[n, 1, h, w]</c> used when no explicit mask is
    /// supplied — by both the generic inference path (<see cref="PredictCore"/>) and each model's
    /// training forward (<c>ForwardForTraining</c>). A centred rectangular hole covering roughly
    /// <see cref="MaxMaskRatio"/> of the frame area gives the mask channel a real, spatially-structured
    /// signal (1 = hole to fill, 0 = keep) that flows gradients through the encoder's mask-channel
    /// weights — replacing the all-zero mask that left the channel dead during training, so the model
    /// no longer trains as if it were doing plain identity reconstruction. It is deterministic (no
    /// per-call RNG), so training a fixed input still converges and inference reproduces the exact
    /// value space the model was trained in (the loss-reduction and Clone/determinism invariants
    /// hold). Callers that have a real mask supply it directly through
    /// <see cref="Inpaint(Tensor{T}, Tensor{T})"/>.
    /// </summary>
    /// <param name="n">Batch / frame count.</param>
    /// <param name="h">Frame height.</param>
    /// <param name="w">Frame width.</param>
    protected Tensor<T> CreateDefaultInpaintingMask(int n, int h, int w)
    {
        var mask = new Tensor<T>([n, 1, h, w]);
        // side = sqrt(ratio) so a centred square hole covers ~MaxMaskRatio of the area. Clamp the
        // ratio to [0, 1] by hand (no Math.Clamp — net471) and guarantee at least a 1px hole.
        double ratio = MaxMaskRatio;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;
        double side = Math.Sqrt(ratio);
        int holeH = Math.Max(1, (int)(h * side));
        int holeW = Math.Max(1, (int)(w * side));
        int top = (h - holeH) / 2;
        int left = (w - holeW) / 2;
        var span = mask.Data.Span;
        var one = NumOps.One;
        int plane = h * w;
        for (int b = 0; b < n; b++)
        {
            int baseOffset = b * plane;
            for (int y = top; y < top + holeH; y++)
            {
                int rowOffset = baseOffset + y * w;
                for (int x = left; x < left + holeW; x++)
                    span[rowOffset + x] = one;
            }
        }
        return mask;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Use the same default hole mask the training forward sees (CreateDefaultInpaintingMask) so
        // inference measures the model in the value space it was trained in. Callers with a real mask
        // use Inpaint directly.
        var mask = CreateDefaultInpaintingMask(input.Shape[0], input.Shape[2], input.Shape[3]);
        return Inpaint(input, mask);
    }
}
