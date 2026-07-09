using AiDotNet.Helpers;
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

    // Lazily-created RNG for the per-training-step synthetic mask (see CreateTrainingMask). Seeded from
    // the architecture seed when one is set (so seeded training runs are reproducible), else a
    // cryptographically-secure RNG per the project RNG policy.
    private Random? _trainingMaskRng;

    private Random TrainingMaskRng =>
        _trainingMaskRng ??= Architecture?.RandomSeed is int seed
            ? RandomHelper.CreateSeededRandom(seed)
            : RandomHelper.CreateSecureRandom();

    /// <summary>
    /// Builds a fresh, RANDOM single-channel hole mask <c>[n, 1, h, w]</c> for one training step — a
    /// randomly sized and positioned rectangular hole (1 = hole, 0 = keep) re-drawn on every call. This
    /// is the PyTorch video-inpainting training recipe (STTN / FuseFormer / E2FGVI / ProPainter train
    /// on random per-sample synthetic masks): because the mask varies every step the model cannot treat
    /// it as a constant shortcut and zero out its frame-channel weights, so training keeps using the
    /// frame content (the encoder's mask-channel weights are exercised with real gradient, and the model
    /// stays input-sensitive). Reproducible under a seeded architecture (see <see cref="TrainingMaskRng"/>).
    /// Inference uses the deterministic <see cref="CreateDefaultInpaintingMask"/> instead.
    /// </summary>
    /// <param name="n">Batch / frame count.</param>
    /// <param name="h">Frame height.</param>
    /// <param name="w">Frame width.</param>
    protected Tensor<T> CreateTrainingMask(int n, int h, int w)
    {
        var mask = new Tensor<T>([n, 1, h, w]);
        double ratio = MaxMaskRatio;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;
        double maxSide = Math.Sqrt(ratio);
        var rng = TrainingMaskRng;
        var span = mask.Data.Span;
        var one = NumOps.One;
        int plane = h * w;
        for (int b = 0; b < n; b++)
        {
            // Random box: 50-100% of the max side, at a random position — varies every training step.
            double side = maxSide * (0.5 + 0.5 * rng.NextDouble());
            int holeH = Math.Max(1, (int)(h * side));
            int holeW = Math.Max(1, (int)(w * side));
            if (holeH > h) holeH = h;
            if (holeW > w) holeW = w;
            int top = h > holeH ? rng.Next(h - holeH + 1) : 0;
            int left = w > holeW ? rng.Next(w - holeW + 1) : 0;
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

    /// <summary>
    /// Builds the deterministic single-channel hole mask <c>[n, 1, h, w]</c> used by the generic
    /// inference path (<see cref="PredictCore"/>) when no explicit mask is supplied: a centred
    /// rectangular hole covering roughly <see cref="MaxMaskRatio"/> of the frame area (1 = hole to fill,
    /// 0 = keep). It is deterministic (no per-call RNG) so inference is reproducible for the
    /// Clone/determinism invariants. Training uses the RANDOM <see cref="CreateTrainingMask"/> instead;
    /// callers that have a real mask supply it directly through <see cref="Inpaint(Tensor{T}, Tensor{T})"/>.
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

    // Frame-normalization scale last chosen by NormalizeInpaintFrames, so DenormalizeInpaintFrames
    // inverts the SAME mapping within a single forward. Defaults to 255 for a denormalize with no prior
    // normalize. Only ever set/read on the sequential normalize -> forward -> denormalize path.
    private double _frameNormalizationScale = 255.0;

    /// <summary>
    /// Scale-adaptive frame normalization for these mask-conditioned models. Divides [0, 255] pixel
    /// frames by 255, but passes frames that are already in [0, 1] through unchanged — so the frame
    /// signal stays on the same scale as the concatenated single-channel 0/1 mask. Dividing an already
    /// normalized frame by 255 drives it to ~0.004, which the 0/1 mask channel (~1.0) then numerically
    /// swamps through the network, collapsing the model to input-independent output. Records the applied
    /// scale so <see cref="DenormalizeInpaintFrames"/> applies the exact inverse.
    /// </summary>
    protected Tensor<T> NormalizeInpaintFrames(Tensor<T> frames)
    {
        _frameNormalizationScale = InferFrameScale(frames);
        return _frameNormalizationScale == 1.0
            ? frames
            : Engine.TensorDivideScalar(frames, NumOps.FromDouble(_frameNormalizationScale));
    }

    /// <summary>
    /// Inverse of <see cref="NormalizeInpaintFrames"/>: rescales the model output by the scale the
    /// matching normalize used and clamps to that scale's valid pixel range ([0, 1] or [0, 255]).
    /// </summary>
    protected Tensor<T> DenormalizeInpaintFrames(Tensor<T> frames)
    {
        double scale = _frameNormalizationScale;
        var restored = scale == 1.0
            ? frames
            : Engine.TensorMultiplyScalar(frames, NumOps.FromDouble(scale));
        return Engine.TensorClamp(restored, NumOps.Zero, NumOps.FromDouble(scale));
    }

    // Picks the normalization scale from the data: a max value clearly above [0, 1] means the frames are
    // in [0, 255] pixel space (scale 255); otherwise they are already normalized (scale 1, no-op). Only
    // selects a scalar — the actual divide/multiply stays a differentiable Engine op. Uses the vectorized
    // tensor max rather than a per-element scan so it stays negligible on the per-forward training path.
    private double InferFrameScale(Tensor<T> frames)
    {
        if (frames.Length == 0) return 1.0;
        return NumOps.ToDouble(frames.Max().maxVal) > 1.5 ? 255.0 : 1.0;
    }

    private bool _shapesProbed;

    /// <summary>
    /// Resolves lazy layer shapes for these mask-conditioned models. Their inference path
    /// (<see cref="PredictCore"/> -> <see cref="Inpaint(Tensor{T}, Tensor{T})"/>) concatenates a
    /// 1-channel mask before the encoder, so the lazy first conv must resolve to <c>InputDepth + 1</c> —
    /// not the <c>InputDepth</c> the base linear shape-walk infers from the architecture input shape.
    /// Probe the real inference forward once on a tiny dummy frame so callers that run before any real
    /// forward (GetParameters, serialization, Clone) resolve the encoder to the same depth training and
    /// inference feed it. The probe dispatches through the virtual <see cref="PredictCore"/>, so a model
    /// with its own PredictCore override is probed through its own path.
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_shapesProbed || Layers.Count == 0) return;
        _shapesProbed = true;
        int c = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        _ = PredictCore(new Tensor<T>([1, c, 32, 32]));
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
