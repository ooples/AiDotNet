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
        if (predicted.Shape.Length < 4 || groundTruth.Shape.Length < 4)
        {
            throw new ArgumentException("Video tensors must have at least 4 dimensions (T, H, W, C or T, C, H, W).");
        }

        if (!predicted.Shape.SequenceEqual(groundTruth.Shape))
        {
            throw new ArgumentException("Predicted and ground truth videos must have the same shape.");
        }

        int numFrames = predicted.Shape[0];
        if (numFrames == 0)
        {
            throw new ArgumentException("Video tensors must have at least one frame.");
        }
        var frameScores = new T[numFrames];

        int height, width, channels;
        if (isChannelsFirst)
        {
            channels = predicted.Shape[1];
            height = predicted.Shape[2];
            width = predicted.Shape[3];
        }
        else
        {
            height = predicted.Shape[1];
            width = predicted.Shape[2];
            channels = predicted.Shape[3];
        }

        for (int t = 0; t < numFrames; t++)
        {
            var predFrame = ExtractFrame(predicted, t, height, width, channels, isChannelsFirst);
            var gtFrame = ExtractFrame(groundTruth, t, height, width, channels, isChannelsFirst);

            frameScores[t] = _framePsnr.Compute(predFrame, gtFrame);
        }

        // Compute statistics
        T sum = _numOps.Zero;
        T minVal = frameScores[0];
        T maxVal = frameScores[0];

        foreach (T score in frameScores)
        {
            sum = _numOps.Add(sum, score);
            if (_numOps.LessThan(score, minVal)) minVal = score;
            if (_numOps.GreaterThan(score, maxVal)) maxVal = score;
        }

        T mean = _numOps.Divide(sum, _numOps.FromDouble(numFrames));

        return (mean, minVal, maxVal, frameScores);
    }

    private Tensor<T> ExtractFrame(Tensor<T> video, int frameIdx, int height, int width, int channels, bool isChannelsFirst)
    {
        var frame = new Tensor<T>(new[] { height, width, channels });
        int frameSize = height * width * channels;
        int offset = frameIdx * frameSize;

        if (isChannelsFirst)
        {
            // TCHW to HWC conversion
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int srcIdx = frameIdx * channels * height * width + c * height * width + h * width + w;
                        frame[h, w, c] = video.GetFlat(srcIdx);
                    }
                }
            }
        }
        else
        {
            // THWC - direct copy
            for (int i = 0; i < frameSize; i++)
            {
                frame.SetFlat(i, video.GetFlat(offset + i));
            }
        }

        return frame;
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
        if (predicted.Shape.Length < 4 || groundTruth.Shape.Length < 4)
        {
            throw new ArgumentException("Video tensors must have at least 4 dimensions (T, H, W, C or T, C, H, W).");
        }

        if (!predicted.Shape.SequenceEqual(groundTruth.Shape))
        {
            throw new ArgumentException("Predicted and ground truth videos must have the same shape.");
        }

        int numFrames = predicted.Shape[0];
        if (numFrames == 0)
        {
            throw new ArgumentException("Video tensors must have at least one frame.");
        }

        int height, width, channels;

        if (isChannelsFirst)
        {
            channels = predicted.Shape[1];
            height = predicted.Shape[2];
            width = predicted.Shape[3];
        }
        else
        {
            height = predicted.Shape[1];
            width = predicted.Shape[2];
            channels = predicted.Shape[3];
        }

        T sum = _numOps.Zero;

        for (int t = 0; t < numFrames; t++)
        {
            var predFrame = ExtractFrame(predicted, t, height, width, channels, isChannelsFirst);
            var gtFrame = ExtractFrame(groundTruth, t, height, width, channels, isChannelsFirst);

            T frameScore = _frameSsim.Compute(predFrame, gtFrame);
            sum = _numOps.Add(sum, frameScore);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(numFrames));
    }

    private Tensor<T> ExtractFrame(Tensor<T> video, int frameIdx, int height, int width, int channels, bool isChannelsFirst = false)
    {
        var frame = new Tensor<T>(new[] { height, width, channels });
        int frameSize = height * width * channels;
        int offset = frameIdx * frameSize;

        if (isChannelsFirst)
        {
            // TCHW to HWC conversion
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int srcIdx = frameIdx * channels * height * width + c * height * width + h * width + w;
                        frame[h, w, c] = video.GetFlat(srcIdx);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < frameSize; i++)
            {
                frame.SetFlat(i, video.GetFlat(offset + i));
            }
        }

        return frame;
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
    /// This measures how well content is preserved between frames, accounting for motion.
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

        T totalConsistency = _numOps.Zero;
        int validPairs = 0;

        for (int t = 0; t < numFrames - 1; t++)
        {
            // Warp frame t to frame t+1 using optical flow
            var warpedFrame = WarpFrame(generatedVideo, t, opticalFlow, t, height, width, channels);
            var nextFrame = ExtractFrame(generatedVideo, t + 1, height, width, channels);

            // Compute consistency as inverse of warped difference
            T diff = ComputeFrameDifference(warpedFrame, nextFrame);
            T consistency = _numOps.Subtract(_numOps.One, diff);

            totalConsistency = _numOps.Add(totalConsistency, consistency);
            validPairs++;
        }

        return _numOps.Divide(totalConsistency, _numOps.FromDouble(validPairs));
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

        T totalDiff = _numOps.Zero;
        int validPairs = 0;

        for (int t = 0; t < numFrames - 1; t++)
        {
            var frame1 = ExtractFrame(generatedVideo, t, height, width, channels);
            var frame2 = ExtractFrame(generatedVideo, t + 1, height, width, channels);

            T diff = ComputeFrameDifference(frame1, frame2);
            totalDiff = _numOps.Add(totalDiff, diff);
            validPairs++;
        }

        // Normalize by motion magnitude (higher motion = higher allowed difference)
        T avgDiff = _numOps.Divide(totalDiff, _numOps.FromDouble(validPairs));

        // Convert to consistency (lower difference = higher consistency)
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

        T totalFlicker = _numOps.Zero;
        int validTriplets = 0;

        for (int t = 1; t < numFrames - 1; t++)
        {
            var prevFrame = ExtractFrame(generatedVideo, t - 1, height, width, channels);
            var currFrame = ExtractFrame(generatedVideo, t, height, width, channels);
            var nextFrame = ExtractFrame(generatedVideo, t + 1, height, width, channels);

            // Flicker is detected when the current frame differs significantly from
            // both its neighbors in the same direction
            T flickerScore = ComputeFrameFlicker(prevFrame, currFrame, nextFrame);
            totalFlicker = _numOps.Add(totalFlicker, flickerScore);
            validTriplets++;
        }

        return _numOps.Divide(totalFlicker, _numOps.FromDouble(validTriplets));
    }

    /// <summary>
    /// Computes flicker for a single frame based on its neighbors.
    /// </summary>
    private T ComputeFrameFlicker(Tensor<T> prev, Tensor<T> curr, Tensor<T> next)
    {
        // Flicker occurs when current frame deviates from interpolation of neighbors
        // High value = high flicker
        T flickerSum = _numOps.Zero;
        T half = _numOps.FromDouble(0.5);

        for (int i = 0; i < curr.Length; i++)
        {
            // Expected value is interpolation between prev and next
            T expected = _numOps.Multiply(half, _numOps.Add(prev[i], next[i]));
            T diff = _numOps.Subtract(curr[i], expected);
            flickerSum = _numOps.Add(flickerSum, _numOps.Multiply(diff, diff));
        }

        // Normalize by number of pixels
        T mse = _numOps.Divide(flickerSum, _numOps.FromDouble(curr.Length));

        // Convert to 0-1 range using exponential decay
        double mseDouble = _numOps.ToDouble(mse);
        double flickerScore = 1.0 - Math.Exp(-mseDouble * 10);

        return _numOps.FromDouble(flickerScore);
    }

    /// <summary>
    /// Warps a frame using optical flow.
    /// </summary>
    private Tensor<T> WarpFrame(
        Tensor<T> video, int frameIdx,
        Tensor<T> flow, int flowIdx,
        int height, int width, int channels)
    {
        var frame = ExtractFrame(video, frameIdx, height, width, channels);
        var warped = new Tensor<T>(new[] { height, width, channels });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // Get flow displacement
                T dx = flow[flowIdx, h, w, 0];
                T dy = flow[flowIdx, h, w, 1];

                // Compute source coordinates
                double srcH = h + _numOps.ToDouble(dy);
                double srcW = w + _numOps.ToDouble(dx);

                // Bilinear interpolation
                for (int c = 0; c < channels; c++)
                {
                    warped[h, w, c] = BilinearSample(frame, srcH, srcW, c, height, width);
                }
            }
        }

        return warped;
    }

    /// <summary>
    /// Performs bilinear sampling from a frame.
    /// </summary>
    private T BilinearSample(Tensor<T> frame, double h, double w, int c, int height, int width)
    {
        // Clamp to valid range
        h = Math.Max(0, Math.Min(height - 1.001, h));
        w = Math.Max(0, Math.Min(width - 1.001, w));

        int h0 = (int)h;
        int w0 = (int)w;
        int h1 = Math.Min(h0 + 1, height - 1);
        int w1 = Math.Min(w0 + 1, width - 1);

        double dh = h - h0;
        double dw = w - w0;

        // Bilinear interpolation weights
        double w00 = (1 - dh) * (1 - dw);
        double w01 = (1 - dh) * dw;
        double w10 = dh * (1 - dw);
        double w11 = dh * dw;

        // Sample and interpolate
        double v00 = _numOps.ToDouble(frame[h0, w0, c]);
        double v01 = _numOps.ToDouble(frame[h0, w1, c]);
        double v10 = _numOps.ToDouble(frame[h1, w0, c]);
        double v11 = _numOps.ToDouble(frame[h1, w1, c]);

        double result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

        return _numOps.FromDouble(result);
    }

    private Tensor<T> ExtractFrame(Tensor<T> video, int frameIdx, int height, int width, int channels)
    {
        var frame = new Tensor<T>(new[] { height, width, channels });
        int frameSize = height * width * channels;
        int offset = frameIdx * frameSize;

        for (int i = 0; i < frameSize; i++)
        {
            frame.SetFlat(i, video.GetFlat(offset + i));
        }

        return frame;
    }

    private T ComputeFrameDifference(Tensor<T> frame1, Tensor<T> frame2)
    {
        T sumSq = _numOps.Zero;

        for (int i = 0; i < frame1.Length; i++)
        {
            T diff = _numOps.Subtract(frame1[i], frame2[i]);
            sumSq = _numOps.Add(sumSq, _numOps.Multiply(diff, diff));
        }

        // Return normalized MSE
        T mse = _numOps.Divide(sumSq, _numOps.FromDouble(frame1.Length));

        // Convert MSE to 0-1 range (saturates at MSE=0.1)
        double mseDouble = _numOps.ToDouble(mse);
        double normalized = Math.Min(1.0, mseDouble * 10);

        return _numOps.FromDouble(normalized);
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
        double psnrNorm = Math.Min(1.0, _numOps.ToDouble(psnrStats.mean) / 50.0);
        double ssimVal = _numOps.ToDouble(ssim);
        double tempVal = _numOps.ToDouble(temporalConsistency);
        double flickerVal = 1.0 - _numOps.ToDouble(flicker); // Invert flicker

        double overallScore = 0.3 * psnrNorm + 0.4 * ssimVal + 0.2 * tempVal + 0.1 * flickerVal;

        return new VideoQualityResult<T>
        {
            MeanPSNR = psnrStats.mean,
            MinPSNR = psnrStats.min,
            MaxPSNR = psnrStats.max,
            PerFramePSNR = psnrStats.perFrame,
            MeanSSIM = ssim,
            TemporalConsistency = temporalConsistency,
            FlickerScore = flicker,
            OverallScore = _numOps.FromDouble(overallScore)
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
    public T MeanPSNR { get; set; } = default!;

    /// <summary>
    /// Minimum per-frame PSNR (dB).
    /// </summary>
    public T MinPSNR { get; set; } = default!;

    /// <summary>
    /// Maximum per-frame PSNR (dB).
    /// </summary>
    public T MaxPSNR { get; set; } = default!;

    /// <summary>
    /// PSNR values for each frame.
    /// </summary>
    public T[] PerFramePSNR { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Mean SSIM across all frames (0-1).
    /// </summary>
    public T MeanSSIM { get; set; } = default!;

    /// <summary>
    /// Temporal consistency score (0-1, higher is better).
    /// </summary>
    public T TemporalConsistency { get; set; } = default!;

    /// <summary>
    /// Flicker score (0-1, lower is better).
    /// </summary>
    public T FlickerScore { get; set; } = default!;

    /// <summary>
    /// Overall composite quality score (0-1).
    /// </summary>
    public T OverallScore { get; set; } = default!;
}
