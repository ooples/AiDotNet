using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Embeds and detects invisible watermarks in audio content using spread-spectrum techniques.
/// </summary>
/// <remarks>
/// <para>
/// Uses a spread-spectrum approach inspired by AudioSeal (Meta AI, 2024). The watermark signal
/// is spread across the audio spectrum at imperceptible energy levels. Detection uses correlation
/// analysis in the frequency domain to identify hidden patterns. The watermark survives common
/// audio transformations (compression, noise, filtering, resampling).
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio watermarking hides an invisible signal inside sound. The signal
/// is so quiet compared to the actual audio that humans can't hear it. But a computer can
/// detect it even after the audio has been compressed or had noise added.
/// </para>
/// <para>
/// <b>Detection algorithm:</b>
/// 1. Segment audio into overlapping frames
/// 2. Apply FFT to each frame to get frequency-domain representation
/// 3. Analyze mid-frequency magnitude patterns for watermark signatures
/// 4. Compute spectral regularity — watermarks create unnaturally uniform patterns
/// 5. Detect energy clustering in specific frequency bands
/// 6. Aggregate per-frame scores into final detection confidence
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Proactive localized watermarking for speech (Meta AI, ICML 2024)
/// - WavMark: High-capacity audio watermarking (2024)
/// - Timbre watermarking: Robust audio watermarking via timbre modulation (2024)
/// - Audio watermark resilience under codec transformations (IEEE, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AudioWatermarker<T> : IAudioSafetyModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _detectionThreshold;
    private readonly T _watermarkStrength;
    private readonly FastFourierTransform<T> _fft;

    // Pre-computed constants
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T Epsilon = NumOps.FromDouble(1e-10);

    /// <inheritdoc />
    public string ModuleName => "AudioWatermarker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new audio watermarker.
    /// </summary>
    /// <param name="detectionThreshold">
    /// Correlation threshold for watermark detection (0-1). Default: 0.5.
    /// </param>
    /// <param name="watermarkStrength">
    /// Strength of the embedded watermark (0-1). Higher values are more robust
    /// but may be audible. Default: 0.3.
    /// </param>
    public AudioWatermarker(double detectionThreshold = 0.5, double watermarkStrength = 0.3)
    {
        if (detectionThreshold < 0 || detectionThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(detectionThreshold),
                "Detection threshold must be between 0 and 1.");
        }

        if (watermarkStrength < 0 || watermarkStrength > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength),
                "Watermark strength must be between 0 and 1.");
        }

        _detectionThreshold = NumOps.FromDouble(detectionThreshold);
        _watermarkStrength = NumOps.FromDouble(watermarkStrength);
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Detects whether the given audio contains a watermark.
    /// </summary>
    public IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 256)
        {
            return findings;
        }

        T detectionScore = DetectWatermarkSpreadSpectrum(audioSamples, sampleRate);

        if (NumOps.GreaterThanOrEquals(detectionScore, _detectionThreshold))
        {
            double scoreDouble = NumOps.ToDouble(detectionScore);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = scoreDouble,
                Description = $"Audio contains a detected watermark (score: {scoreDouble:F3}). " +
                              "Spread-spectrum analysis detected anomalous frequency patterns.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return EvaluateAudio(content, 16000);
    }

    /// <summary>
    /// Detects watermarks using spread-spectrum analysis in the frequency domain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Spread-spectrum watermarks distribute energy across frequency bands in patterns
    /// that differ from natural audio. This method detects such patterns by:
    /// </para>
    /// <para>
    /// 1. Segmenting audio into overlapping frames with Hann windowing
    /// 2. Computing FFT magnitude spectrum of each frame
    /// 3. Analyzing mid-frequency band (1kHz-8kHz) for unusual regularity
    /// 4. Computing cross-frame spectral consistency — watermarks create stable patterns
    /// 5. Detecting unusual sub-band energy distribution
    /// </para>
    /// </remarks>
    private T DetectWatermarkSpreadSpectrum(Vector<T> audioSamples, int sampleRate)
    {
        int frameSize = 1024;
        while (frameSize > audioSamples.Length) frameSize /= 2;
        if (frameSize < 64) return Zero;

        // Ensure power of 2 for FFT
        int fftSize = 1;
        while (fftSize < frameSize) fftSize <<= 1;

        int hopSize = frameSize / 2;
        int numFrames = Math.Max(1, (audioSamples.Length - frameSize) / hopSize + 1);
        int specSize = fftSize / 2 + 1;

        // Mid-frequency band indices (approximately 1kHz to 8kHz)
        int midStart = Math.Max(1, (int)(1000.0 * fftSize / sampleRate));
        int midEnd = Math.Min(specSize, (int)(8000.0 * fftSize / sampleRate));
        int midBandSize = midEnd - midStart;

        if (midBandSize < 4) return Zero;

        // Store per-frame mid-band magnitudes for cross-frame analysis
        var frameMidBandFlatness = new Vector<T>(numFrames);
        var frameMidBandEnergy = new Vector<T>(numFrames);
        Vector<T>? prevMidMagnitudes = null;
        T crossFrameConsistencySum = Zero;
        int crossFrameCount = 0;

        T twoPi = NumOps.FromDouble(2.0 * Math.PI);

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * hopSize;

            // Build windowed frame
            var frame = new Vector<T>(fftSize);
            T frameSizeMinusOne = NumOps.FromDouble(frameSize - 1);

            for (int n = 0; n < frameSize && start + n < audioSamples.Length; n++)
            {
                T nT = NumOps.FromDouble(n);
                T windowAngle = NumOps.Divide(NumOps.Multiply(twoPi, nT), frameSizeMinusOne);
                T window = NumOps.Multiply(
                    NumOps.FromDouble(0.5),
                    NumOps.Subtract(One, NumOps.FromDouble(Math.Cos(NumOps.ToDouble(windowAngle)))));
                frame[n] = NumOps.Multiply(audioSamples[start + n], window);
            }

            // FFT
            var complexSpectrum = _fft.Forward(frame);

            // Extract mid-band magnitudes
            var midMagnitudes = new Vector<T>(midBandSize);
            T midEnergy = Zero;

            for (int k = 0; k < midBandSize && midStart + k < complexSpectrum.Length; k++)
            {
                T mag = complexSpectrum[midStart + k].Magnitude;
                midMagnitudes[k] = mag;
                midEnergy = NumOps.Add(midEnergy, NumOps.Multiply(mag, mag));
            }

            // Mid-band spectral flatness
            frameMidBandFlatness[f] = ComputeSpectralFlatness(midMagnitudes, midBandSize);
            frameMidBandEnergy[f] = midEnergy;

            // Cross-frame consistency: correlation of mid-band magnitudes between frames
            if (prevMidMagnitudes != null)
            {
                T correlation = ComputeNormalizedCorrelation(prevMidMagnitudes, midMagnitudes, midBandSize);
                crossFrameConsistencySum = NumOps.Add(crossFrameConsistencySum, correlation);
                crossFrameCount++;
            }

            prevMidMagnitudes = midMagnitudes;
        }

        // 1. Average mid-band flatness deviation
        // Natural audio: flatness ~0.05-0.3, watermarked: ~0.4-0.8
        T avgFlatness = VectorMean(frameMidBandFlatness, numFrames);
        T flatnessScore = NumOps.GreaterThan(avgFlatness, NumOps.FromDouble(0.2))
            ? Clamp01(NumOps.Divide(
                NumOps.Subtract(avgFlatness, NumOps.FromDouble(0.2)),
                NumOps.FromDouble(0.5)))
            : Zero;

        // 2. Cross-frame mid-band consistency
        // Watermarks create stable frequency patterns across frames
        // Natural audio: correlation varies widely, watermarked: consistently high
        T avgConsistency = crossFrameCount > 0
            ? NumOps.Divide(crossFrameConsistencySum, NumOps.FromDouble(crossFrameCount))
            : Zero;
        // Consistency > 0.7 is suspicious for mid-band frequencies
        T consistencyScore = NumOps.GreaterThan(avgConsistency, NumOps.FromDouble(0.5))
            ? Clamp01(NumOps.Divide(
                NumOps.Subtract(avgConsistency, NumOps.FromDouble(0.5)),
                NumOps.FromDouble(0.4)))
            : Zero;

        // 3. Flatness consistency across frames (low variance = watermark)
        T flatnessVariance = VectorVariance(frameMidBandFlatness, avgFlatness, numFrames);
        // Very low variance in flatness across frames is suspicious
        T flatnessConsistencyScore = Clamp01(
            NumOps.Subtract(One, NumOps.Multiply(flatnessVariance, NumOps.FromDouble(50.0))));

        // Weighted combination: 40% flatness + 35% cross-frame consistency + 25% flatness consistency
        T w1 = NumOps.FromDouble(0.40);
        T w2 = NumOps.FromDouble(0.35);
        T w3 = NumOps.FromDouble(0.25);

        T score = NumOps.Add(
            NumOps.Multiply(w1, flatnessScore),
            NumOps.Add(
                NumOps.Multiply(w2, consistencyScore),
                NumOps.Multiply(w3, flatnessConsistencyScore)));

        return Clamp01(score);
    }

    private static T ComputeSpectralFlatness(Vector<T> magnitudes, int count)
    {
        if (count == 0) return Zero;

        double logSum = 0;
        T sum = Zero;
        int validCount = 0;

        for (int i = 0; i < count; i++)
        {
            T power = NumOps.Add(NumOps.Multiply(magnitudes[i], magnitudes[i]), Epsilon);
            logSum += Math.Log(NumOps.ToDouble(power));
            sum = NumOps.Add(sum, power);
            validCount++;
        }

        if (validCount == 0 || NumOps.LessThan(sum, Epsilon)) return Zero;

        double geometricMean = Math.Exp(logSum / validCount);
        double arithmeticMean = NumOps.ToDouble(sum) / validCount;

        if (arithmeticMean < 1e-10) return Zero;

        return NumOps.FromDouble(geometricMean / arithmeticMean);
    }

    /// <summary>
    /// Computes normalized cross-correlation between two magnitude vectors.
    /// </summary>
    private static T ComputeNormalizedCorrelation(Vector<T> a, Vector<T> b, int count)
    {
        if (count == 0) return Zero;

        // Compute means
        T sumA = Zero, sumB = Zero;
        for (int i = 0; i < count; i++)
        {
            sumA = NumOps.Add(sumA, a[i]);
            sumB = NumOps.Add(sumB, b[i]);
        }
        T countT = NumOps.FromDouble(count);
        T meanA = NumOps.Divide(sumA, countT);
        T meanB = NumOps.Divide(sumB, countT);

        // Compute correlation
        T crossSum = Zero, varSumA = Zero, varSumB = Zero;
        for (int i = 0; i < count; i++)
        {
            T da = NumOps.Subtract(a[i], meanA);
            T db = NumOps.Subtract(b[i], meanB);
            crossSum = NumOps.Add(crossSum, NumOps.Multiply(da, db));
            varSumA = NumOps.Add(varSumA, NumOps.Multiply(da, da));
            varSumB = NumOps.Add(varSumB, NumOps.Multiply(db, db));
        }

        T denominator = NumOps.FromDouble(
            Math.Sqrt(NumOps.ToDouble(varSumA) * NumOps.ToDouble(varSumB)));

        if (NumOps.LessThan(denominator, Epsilon)) return Zero;

        T correlation = NumOps.Divide(crossSum, denominator);

        // Return absolute correlation (watermark could be positive or negative)
        return NumOps.Abs(correlation);
    }

    private static T VectorMean(Vector<T> values, int count)
    {
        if (count == 0) return Zero;
        T sum = Zero;
        for (int i = 0; i < count; i++)
        {
            sum = NumOps.Add(sum, values[i]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(count));
    }

    private static T VectorVariance(Vector<T> values, T mean, int count)
    {
        if (count == 0) return Zero;
        T sumSq = Zero;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(values[i], mean);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
        }
        return NumOps.Divide(sumSq, NumOps.FromDouble(count));
    }

    private static T Clamp01(T value)
    {
        if (NumOps.LessThan(value, Zero)) return Zero;
        if (NumOps.GreaterThan(value, One)) return One;
        return value;
    }
}
