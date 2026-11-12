using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.Metrics;

/// <summary>
/// Video Peak Signal-to-Noise Ratio (VPSNR) - Frame-averaged PSNR for video quality.
/// </summary>
/// <remarks>
/// <para>
/// VPSNR computes PSNR for each frame of a video and returns statistics (mean, min, max).
/// It's a straightforward extension of image PSNR to video sequences.
/// </para>
/// <para>
/// Typical values:
/// - &gt;40 dB: Excellent quality
/// - 30-40 dB: Good quality
/// - 20-30 dB: Acceptable quality
/// - &lt;20 dB: Poor quality
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VideoPSNR<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly PeakSignalToNoiseRatio<T> _framePsnr;

    /// <summary>
    /// Initializes a new instance of VideoPSNR.
    /// </summary>
    /// <param name="maxValue">Maximum pixel value (1.0 for normalized, 255 for 8-bit).</param>
    public VideoPSNR(T? maxValue = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _framePsnr = new PeakSignalToNoiseRatio<T>(maxValue);
    }

    /// <summary>
    /// Computes VPSNR between predicted and ground truth videos.
    /// </summary>
    /// <param name="predicted">Predicted video tensor [T, H, W, C] or [T, C, H, W].</param>
    /// <param name="groundTruth">Ground truth video tensor with same shape.</param>
    /// <param name="isChannelsFirst">True if format is TCHW, false if THWC.</param>
    /// <returns>Mean PSNR across all frames.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth, bool isChannelsFirst = false)
    {
        var stats = ComputeWithStats(predicted, groundTruth, isChannelsFirst);
        return stats.mean;
    }

    /// <summary>
    /// Computes VPSNR with detailed per-frame statistics.
    /// </summary>
    /// <param name="predicted">Predicted video tensor.</param>
    /// <param name="groundTruth">Ground truth video tensor.</param>
    /// <param name="isChannelsFirst">True if format is TCHW, false if THWC.</param>
    /// <returns>Tuple of (mean, min, max, per-frame values).</returns>
    public (T mean, T min, T max, T[] perFrame) ComputeWithStats(
        Tensor<T> predicted, Tensor<T> groundTruth, bool isChannelsFirst = false)
    {
        VideoShape.Validate(predicted, groundTruth);

        // A frame's PSNR is over all of its pixels, so the per-frame values do not depend on the channel layout.
        var frameScores = _framePsnr.ComputeBatch(predicted, groundTruth);

        var engine = AiDotNetEngine.Current;
        var scores = new Tensor<T>(frameScores, new[] { frameScores.Length });
        T mean = _numOps.Divide(engine.TensorSum(scores), _numOps.FromDouble(frameScores.Length));

        return (mean, engine.TensorMinValue(scores), engine.TensorMaxValue(scores), frameScores);
    }
}

/// <summary>
/// Shape checks shared by the video metrics.
/// </summary>
internal static class VideoShape
{
    /// <summary>
    /// Requires two 4D videos ([T, H, W, C] or [T, C, H, W]) of the same shape with at least one frame.
    /// </summary>
    internal static void Validate<T>(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        if (predicted.Shape.Length != 4 || groundTruth.Shape.Length != 4)
        {
            throw new ArgumentException("Video tensors must have 4 dimensions (T, H, W, C or T, C, H, W).");
        }

        if (!predicted._shape.SequenceEqual(groundTruth._shape))
        {
            throw new ArgumentException("Predicted and ground truth videos must have the same shape.");
        }

        if (predicted.Shape[0] == 0)
        {
            throw new ArgumentException("Video tensors must have at least one frame.");
        }
    }
}

/// <summary>
/// Video Structural Similarity (VSSIM) - Frame-averaged SSIM for video quality.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VideoSSIM<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly StructuralSimilarity<T> _frameSsim;

    /// <summary>
    /// Initializes a new instance of VideoSSIM.
    /// </summary>
    /// <param name="maxValue">Maximum pixel value.</param>
    /// <param name="k1">SSIM constant K1 (default 0.01).</param>
    /// <param name="k2">SSIM constant K2 (default 0.03).</param>
    public VideoSSIM(T? maxValue = null, double k1 = 0.01, double k2 = 0.03)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _frameSsim = new StructuralSimilarity<T>(maxValue, k1, k2);
    }

    /// <summary>
    /// Computes VSSIM between predicted and ground truth videos.
    /// </summary>
    /// <param name="predicted">Predicted video tensor [T, H, W, C] or [T, C, H, W] if isChannelsFirst is true.</param>
    /// <param name="groundTruth">Ground truth video tensor.</param>
    /// <param name="isChannelsFirst">If true, expects [T, C, H, W] format; otherwise [T, H, W, C].</param>
    /// <returns>Mean SSIM across all frames.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> groundTruth, bool isChannelsFirst = false)
    {
        VideoShape.Validate(predicted, groundTruth);

        // Mean over frames of the per-frame SSIM (each the mean over its channels).
        return _frameSsim.ComputeFrames(predicted, groundTruth, isChannelsFirst);
    }
}

/// <summary>
/// Temporal Consistency metric for evaluating video smoothness and coherence.
/// </summary>
/// <remarks>
/// <para>
/// Temporal consistency measures how smoothly content changes between consecutive frames.
/// It's crucial for video generation quality, as flickering and jittery artifacts
/// are often more distracting than per-frame quality issues.
/// </para>
/// <para>
/// Two main approaches:
/// - Pixel-level temporal difference (simple but sensitive to motion)
/// - Optical flow consistency (accounts for motion but more complex)
/// </para>
/// <para>
/// Values range from 0 to 1, where higher values indicate better temporal consistency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TemporalConsistency<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of TemporalConsistency calculator.
    /// </summary>
    public TemporalConsistency()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes temporal consistency using warped frame differences.
    /// </summary>
    /// <remarks>
    /// This measures how well content is preserved between frames, accounting for motion. Frame t is sampled
    /// bilinearly at (h + dy, w + dx), with coordinates clamped to the frame (grid_sample's border padding with
    /// align_corners = true), and compared with frame t + 1.
    /// </remarks>
    /// <param name="generatedVideo">Generated video tensor [T, H, W, C].</param>
    /// <param name="opticalFlow">Optical flow between frames [T-1, H, W, 2] (dx, dy).</param>
    /// <returns>Temporal consistency score (0-1, higher is better).</returns>
    public T ComputeWithFlow(Tensor<T> generatedVideo, Tensor<T> opticalFlow)
    {
        int numFrames = generatedVideo.Shape[0];
        int height = generatedVideo.Shape[1];
        int width = generatedVideo.Shape[2];
        int channels = generatedVideo.Shape[3];

        if (numFrames < 2)
        {
            return _numOps.One;
        }

        var engine = AiDotNetEngine.Current;
        int pairs = numFrames - 1;
        var current = engine.TensorSlice(generatedVideo, new[] { 0, 0, 0, 0 }, new[] { pairs, height, width, channels });
        var next = engine.TensorSlice(generatedVideo, new[] { 1, 0, 0, 0 }, new[] { pairs, height, width, channels });
        var dx = engine.TensorSlice(opticalFlow, new[] { 0, 0, 0, 0 }, new[] { pairs, height, width, 1 })
            .Reshape(new[] { pairs, height, width });
        var dy = engine.TensorSlice(opticalFlow, new[] { 0, 0, 0, 1 }, new[] { pairs, height, width, 1 })
            .Reshape(new[] { pairs, height, width });

        // Source coordinates of every output pixel, clamped to the frame.
        var rows = engine.TensorBroadcastTo(
            engine.TensorArange<T>(0, height, 1).Reshape(new[] { 1, height, 1 }), new[] { pairs, height, width });
        var cols = engine.TensorBroadcastTo(
            engine.TensorArange<T>(0, width, 1).Reshape(new[] { 1, 1, width }), new[] { pairs, height, width });
        var srcH = engine.TensorClamp(engine.TensorAdd(rows, dy), _numOps.Zero, _numOps.FromDouble(height - 1));
        var srcW = engine.TensorClamp(engine.TensorAdd(cols, dx), _numOps.Zero, _numOps.FromDouble(width - 1));

        var h0 = engine.TensorFloor(srcH);
        var w0 = engine.TensorFloor(srcW);
        var dh = engine.TensorSubtract(srcH, h0);
        var dw = engine.TensorSubtract(srcW, w0);
        var h0Index = h0.Cast<int>();
        var w0Index = w0.Cast<int>();
        var h1Index = engine.TensorMin(engine.TensorAddScalar(h0Index, 1), height - 1);
        var w1Index = engine.TensorMin(engine.TensorAddScalar(w0Index, 1), width - 1);

        // Flat pixel index (p * H + h) * W + w into current viewed as [pairs * H * W, C].
        var frameOffset = engine.TensorBroadcastTo(
            engine.TensorMultiplyScalar(engine.TensorArange<int>(0, pairs, 1), height * width)
                .Reshape(new[] { pairs, 1, 1 }),
            new[] { pairs, height, width });
        var pixels = current.Reshape(new[] { pairs * height * width, channels });

        var oneT = engine.TensorAddScalar(engine.TensorZerosLike(dh), _numOps.One);
        var oneMinusDh = engine.TensorSubtract(oneT, dh);
        var oneMinusDw = engine.TensorSubtract(oneT, dw);

        var warped = engine.TensorAdd(
            engine.TensorAdd(
                WeightedCorner(pixels, frameOffset, h0Index, w0Index, engine.TensorMultiply(oneMinusDh, oneMinusDw), width, channels),
                WeightedCorner(pixels, frameOffset, h0Index, w1Index, engine.TensorMultiply(oneMinusDh, dw), width, channels)),
            engine.TensorAdd(
                WeightedCorner(pixels, frameOffset, h1Index, w0Index, engine.TensorMultiply(dh, oneMinusDw), width, channels),
                WeightedCorner(pixels, frameOffset, h1Index, w1Index, engine.TensorMultiply(dh, dw), width, channels)));

        // Consistency per pair is 1 - min(1, 10 * MSE(warped, next)); the score is the mean over pairs.
        var difference = NormalizedFrameDifferences(warped, next);
        return _numOps.Subtract(_numOps.One, engine.TensorMean(difference));
    }

    /// <summary>
    /// Computes simple temporal consistency without optical flow.
    /// </summary>
    /// <remarks>
    /// This is a simpler metric that measures frame-to-frame differences directly.
    /// Less accurate for videos with motion but useful as a quick baseline.
    /// </remarks>
    /// <param name="generatedVideo">Generated video tensor [T, H, W, C].</param>
    /// <returns>Temporal consistency score (0-1, higher is better).</returns>
    public T ComputeSimple(Tensor<T> generatedVideo)
    {
        int numFrames = generatedVideo.Shape[0];
        int height = generatedVideo.Shape[1];
        int width = generatedVideo.Shape[2];
        int channels = generatedVideo.Shape[3];

        if (numFrames < 2)
        {
            return _numOps.One;
        }

        var engine = AiDotNetEngine.Current;
        int pairs = numFrames - 1;
        var first = engine.TensorSlice(generatedVideo, new[] { 0, 0, 0, 0 }, new[] { pairs, height, width, channels });
        var second = engine.TensorSlice(generatedVideo, new[] { 1, 0, 0, 0 }, new[] { pairs, height, width, channels });

        // Average normalised frame-to-frame difference, converted to consistency (lower difference = higher).
        T avgDiff = engine.TensorMean(NormalizedFrameDifferences(first, second));
        return _numOps.Subtract(_numOps.One, avgDiff);
    }

    /// <summary>
    /// Computes flicker metric (measures high-frequency temporal variations).
    /// </summary>
    /// <param name="generatedVideo">Generated video tensor [T, H, W, C].</param>
    /// <returns>Flicker score (0-1, lower is better - less flicker).</returns>
    public T ComputeFlicker(Tensor<T> generatedVideo)
    {
        int numFrames = generatedVideo.Shape[0];
        int height = generatedVideo.Shape[1];
        int width = generatedVideo.Shape[2];
        int channels = generatedVideo.Shape[3];

        if (numFrames < 3)
        {
            return _numOps.Zero;
        }

        var engine = AiDotNetEngine.Current;
        int triplets = numFrames - 2;
        var shape = new[] { triplets, height, width, channels };
        var prev = engine.TensorSlice(generatedVideo, new[] { 0, 0, 0, 0 }, shape);
        var curr = engine.TensorSlice(generatedVideo, new[] { 1, 0, 0, 0 }, shape);
        var next = engine.TensorSlice(generatedVideo, new[] { 2, 0, 0, 0 }, shape);

        // Flicker is the current frame's deviation from the interpolation of its neighbours:
        // per frame 1 - exp(-10 * MSE(curr, (prev + next) / 2)), averaged over frames.
        var expected = engine.TensorMultiplyScalar(engine.TensorAdd(prev, next), _numOps.FromDouble(0.5));
        var diff = engine.TensorSubtract(curr, expected);
        var mse = engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 1, 2, 3 }, false);
        var decay = engine.TensorExp(engine.TensorMultiplyScalar(mse, _numOps.FromDouble(-10.0)));
        var flicker = engine.TensorSubtract(engine.TensorAddScalar(engine.TensorZerosLike(decay), _numOps.One), decay);

        return engine.TensorMean(flicker);
    }

    /// <summary>
    /// One bilinear corner: the pixel at (h, w) of each pair's frame, weighted per output pixel.
    /// </summary>
    private static Tensor<T> WeightedCorner(Tensor<T> pixels, Tensor<int> frameOffset, Tensor<int> rowIndex,
        Tensor<int> colIndex, Tensor<T> weight, int width, int channels)
    {
        var engine = AiDotNetEngine.Current;
        var flatIndex = engine.TensorAdd(
            engine.TensorAdd(frameOffset, engine.TensorMultiplyScalar(rowIndex, width)), colIndex);
        int count = flatIndex.Length;
        var gathered = engine.TensorIndexSelect(pixels, flatIndex.Reshape(new[] { count }), 0);
        var weightPerChannel = engine.TensorBroadcastTo(weight.Reshape(new[] { count, 1 }), new[] { count, channels });
        return engine.TensorMultiply(gathered, weightPerChannel).Reshape(new[]
        {
            weight.Shape[0], weight.Shape[1], weight.Shape[2], channels
        });
    }

    /// <summary>
    /// Per-pair min(1, 10 * MSE) between two stacks of frames [P, H, W, C] (MSE saturates at 0.1).
    /// </summary>
    private Tensor<T> NormalizedFrameDifferences(Tensor<T> frames1, Tensor<T> frames2)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(frames1, frames2);
        var mse = engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 1, 2, 3 }, false);
        return engine.TensorMin(engine.TensorMultiplyScalar(mse, _numOps.FromDouble(10.0)), _numOps.One);
    }
}

/// <summary>
/// Video Quality Index (VQI) - A composite metric combining multiple video quality aspects.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VideoQualityIndex<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly VideoPSNR<T> _vpsnr;
    private readonly VideoSSIM<T> _vssim;
    private readonly TemporalConsistency<T> _temporal;

    /// <summary>
    /// Initializes a new instance of VideoQualityIndex.
    /// </summary>
    /// <param name="maxValue">Maximum pixel value for PSNR/SSIM calculations.</param>
    public VideoQualityIndex(T? maxValue = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _vpsnr = new VideoPSNR<T>(maxValue);
        _vssim = new VideoSSIM<T>(maxValue);
        _temporal = new TemporalConsistency<T>();
    }

    /// <summary>
    /// Computes comprehensive video quality metrics.
    /// </summary>
    /// <param name="predicted">Predicted video tensor [T, H, W, C].</param>
    /// <param name="groundTruth">Ground truth video tensor.</param>
    /// <returns>VideoQualityResult containing all metrics.</returns>
    public VideoQualityResult<T> Compute(Tensor<T> predicted, Tensor<T> groundTruth)
    {
        // Compute spatial quality metrics
        var psnrStats = _vpsnr.ComputeWithStats(predicted, groundTruth, false);
        T ssim = _vssim.Compute(predicted, groundTruth);

        // Compute temporal quality metrics
        T temporalConsistency = _temporal.ComputeSimple(predicted);
        T flicker = _temporal.ComputeFlicker(predicted);

        // Compute overall quality index (weighted combination)
        // Weights: PSNR (0.3), SSIM (0.4), Temporal (0.2), Anti-Flicker (0.1)
        T psnrNorm = MathHelper.Min(_numOps.One, _numOps.Divide(psnrStats.mean, _numOps.FromDouble(50.0)));
        T antiFlicker = _numOps.Subtract(_numOps.One, flicker); // Invert flicker

        T overallScore = _numOps.Add(
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(0.3), psnrNorm),
                _numOps.Multiply(_numOps.FromDouble(0.4), ssim)),
            _numOps.Add(
                _numOps.Multiply(_numOps.FromDouble(0.2), temporalConsistency),
                _numOps.Multiply(_numOps.FromDouble(0.1), antiFlicker)));

        return new VideoQualityResult<T>
        {
            MeanPSNR = psnrStats.mean,
            MinPSNR = psnrStats.min,
            MaxPSNR = psnrStats.max,
            PerFramePSNR = psnrStats.perFrame,
            MeanSSIM = ssim,
            TemporalConsistency = temporalConsistency,
            FlickerScore = flicker,
            OverallScore = overallScore
        };
    }
}

/// <summary>
/// Results from comprehensive video quality evaluation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class VideoQualityResult<T>
{
    /// <summary>
    /// Mean PSNR across all frames (dB).
    /// </summary>
    public required T MeanPSNR { get; set; }

    /// <summary>
    /// Minimum per-frame PSNR (dB).
    /// </summary>
    public required T MinPSNR { get; set; }

    /// <summary>
    /// Maximum per-frame PSNR (dB).
    /// </summary>
    public required T MaxPSNR { get; set; }

    /// <summary>
    /// PSNR values for each frame.
    /// </summary>
    public T[] PerFramePSNR { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Mean SSIM across all frames (0-1).
    /// </summary>
    public required T MeanSSIM { get; set; }

    /// <summary>
    /// Temporal consistency score (0-1, higher is better).
    /// </summary>
    public required T TemporalConsistency { get; set; }

    /// <summary>
    /// Flicker score (0-1, lower is better).
    /// </summary>
    public required T FlickerScore { get; set; }

    /// <summary>
    /// Overall composite quality score (0-1).
    /// </summary>
    public required T OverallScore { get; set; }
}
