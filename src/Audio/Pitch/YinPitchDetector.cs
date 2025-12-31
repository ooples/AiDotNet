namespace AiDotNet.Audio.Pitch;

/// <summary>
/// YIN pitch detection algorithm implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// YIN is a widely-used pitch detection algorithm known for its accuracy and
/// relatively low computational cost. It was developed by de Cheveigné and Kawahara in 2002.
/// </para>
/// <para><b>For Beginners:</b> YIN finds pitch by looking at how similar a signal is to itself!
///
/// The key insight:
/// - A periodic (pitched) signal repeats itself
/// - If we shift the signal and compare it to the original, it should match at the period
///
/// How YIN works:
/// 1. Difference function: For each possible lag, calculate how different the signal
///    is from a shifted version of itself
/// 2. Cumulative mean normalization: Normalize the difference to handle varying energy
/// 3. Absolute threshold: Find the first lag where normalized difference is below threshold
/// 4. Parabolic interpolation: Refine the estimate for sub-sample accuracy
///
/// Why YIN is popular:
/// - More accurate than simple autocorrelation
/// - Fewer octave errors (detecting 2x or 0.5x the actual pitch)
/// - Works well for speech and musical instruments
/// - Relatively fast computation
///
/// Parameters:
/// - Threshold: How much similarity is required (lower = stricter)
/// - Frame size: Longer frames = lower min pitch, more latency
/// </para>
/// </remarks>
public class YinPitchDetector<T> : PitchDetectorBase<T>
{
    #region Configuration

    private readonly double _threshold;
    private readonly int _frameSize;

    #endregion

    /// <summary>
    /// Creates a YIN pitch detector with default parameters.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 44100).</param>
    /// <param name="minPitch">Minimum detectable pitch in Hz (default: 50).</param>
    /// <param name="maxPitch">Maximum detectable pitch in Hz (default: 2000).</param>
    /// <param name="threshold">Detection threshold 0-1 (default: 0.1).</param>
    /// <param name="frameSize">Frame size in samples (default: 2048).</param>
    public YinPitchDetector(
        int sampleRate = 44100,
        double minPitch = 50,
        double maxPitch = 2000,
        double threshold = 0.1,
        int frameSize = 2048)
        : base(sampleRate, minPitch, maxPitch)
    {
        _threshold = threshold;
        _frameSize = frameSize;
    }

    /// <inheritdoc/>
    protected override (double Pitch, double Confidence)? DetectPitchInternal(double[] frame)
    {
        int tauMax = Math.Min(frame.Length / 2, (int)(SampleRate / MinPitch));
        int tauMin = Math.Max(2, (int)(SampleRate / MaxPitch));

        // Step 1: Calculate difference function
        var difference = ComputeDifference(frame, tauMax);

        // Step 2: Cumulative mean normalized difference function
        var cmndf = ComputeCmndf(difference);

        // Step 3: Absolute threshold
        int tau = FindBestTau(cmndf, tauMin, tauMax);

        if (tau < 0)
        {
            // No pitch detected
            return null;
        }

        // Step 4: Parabolic interpolation
        double refinedTau = ParabolicInterpolation(cmndf, tau);

        // Calculate pitch and confidence
        double pitch = SampleRate / refinedTau;
        double confidence = 1.0 - cmndf[tau];

        // Validate pitch is in range
        if (pitch < MinPitch || pitch > MaxPitch)
        {
            return null;
        }

        return (pitch, confidence);
    }

    /// <summary>
    /// Computes the difference function d(tau).
    /// d(tau) = sum((x[j] - x[j+tau])^2) for j = 0..W-1
    /// </summary>
    private double[] ComputeDifference(double[] frame, int tauMax)
    {
        var diff = new double[tauMax];
        int W = Math.Min(frame.Length / 2, tauMax);

        diff[0] = 1.0; // By definition

        for (int tau = 1; tau < tauMax; tau++)
        {
            double sum = 0;
            for (int j = 0; j < W; j++)
            {
                double delta = frame[j] - frame[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        return diff;
    }

    /// <summary>
    /// Computes the cumulative mean normalized difference function.
    /// d'(tau) = d(tau) / ((1/tau) * sum(d(j)) for j=1..tau) if tau > 0, else 1
    /// </summary>
    private double[] ComputeCmndf(double[] difference)
    {
        var cmndf = new double[difference.Length];
        cmndf[0] = 1.0;

        double runningSum = 0;
        for (int tau = 1; tau < difference.Length; tau++)
        {
            runningSum += difference[tau];
            cmndf[tau] = difference[tau] / (runningSum / tau);
        }

        return cmndf;
    }

    /// <summary>
    /// Finds the best tau value using absolute threshold method.
    /// </summary>
    private int FindBestTau(double[] cmndf, int tauMin, int tauMax)
    {
        int tau = tauMin;

        // Find first tau below threshold
        while (tau < tauMax)
        {
            if (cmndf[tau] < _threshold)
            {
                // Find the minimum in this region
                while (tau + 1 < tauMax && cmndf[tau + 1] < cmndf[tau])
                {
                    tau++;
                }
                return tau;
            }
            tau++;
        }

        // If no tau below threshold, find the global minimum
        double minVal = double.MaxValue;
        int minTau = -1;
        for (int t = tauMin; t < tauMax; t++)
        {
            if (cmndf[t] < minVal)
            {
                minVal = cmndf[t];
                minTau = t;
            }
        }

        // Only return if reasonably confident
        if (minTau > 0 && minVal < 0.5)
        {
            return minTau;
        }

        return -1;
    }

    /// <summary>
    /// Applies parabolic interpolation for sub-sample accuracy.
    /// </summary>
    private double ParabolicInterpolation(double[] data, int tau)
    {
        if (tau <= 0 || tau >= data.Length - 1)
        {
            return tau;
        }

        double x0 = data[tau - 1];
        double x1 = data[tau];
        double x2 = data[tau + 1];

        // Parabolic fit: find the x position of the minimum
        double denominator = 2 * (x0 - 2 * x1 + x2);
        if (Math.Abs(denominator) < 1e-10)
        {
            return tau;
        }

        double adjustment = (x0 - x2) / denominator;
        return tau + adjustment;
    }
}
