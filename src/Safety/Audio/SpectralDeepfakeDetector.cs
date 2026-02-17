using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects deepfake/synthetic audio by analyzing spectral characteristics of the waveform.
/// </summary>
/// <remarks>
/// <para>
/// Examines frequency-domain features of audio to identify artifacts characteristic
/// of synthesized or manipulated speech. Neural vocoders (WaveNet, HiFi-GAN, etc.) and TTS
/// systems leave distinctive spectral fingerprints that differ from natural speech.
/// </para>
/// <para>
/// <b>For Beginners:</b> When computers generate fake voices, the sound has subtle patterns
/// in its frequency content that are different from real speech. This module analyzes those
/// frequencies to detect computer-generated audio.
/// </para>
/// <para>
/// <b>Detection features:</b>
/// 1. Spectral flatness — synthetic audio often has unnaturally uniform frequency distribution
/// 2. Zero-crossing rate statistics — TTS systems produce smoother waveforms
/// 3. Sub-band energy ratios — neural vocoders have characteristic high-frequency rolloff
/// 4. Spectral flux — frame-to-frame spectral change patterns differ in synthetic speech
/// 5. Harmonic-to-noise ratio — synthetic speech has unnaturally high harmonicity
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeEar: Content-agnostic audio deepfake detection, ACM CCS 2024
/// - VoiceRadar: Robust voice liveness detection, NDSS 2025
/// - LAVDE: Codec-robust deepfake detection via multi-feature aggregation, 2025
/// - ADD 2024 Challenge: Audio deepfake detection advancements, ICASSP 2024
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class SpectralDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly FastFourierTransform<T> _fft;

    // Pre-computed constants
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;
    private static readonly T Epsilon = NumOps.FromDouble(1e-10);

    /// <inheritdoc />
    public override string ModuleName => "SpectralDeepfakeDetector";

    /// <summary>
    /// Initializes a new spectral deepfake detector.
    /// </summary>
    /// <param name="threshold">
    /// Detection threshold (0-1). Audio scoring above this is flagged as potentially synthetic.
    /// Default: 0.7. Lower values increase sensitivity but may produce more false positives.
    /// </param>
    /// <param name="defaultSampleRate">
    /// Default sample rate in Hz when not provided in EvaluateAudio. Default: 16000.
    /// </param>
    public SpectralDeepfakeDetector(double threshold = 0.7, int defaultSampleRate = 16000)
        : base(defaultSampleRate)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = NumOps.FromDouble(threshold);
        _fft = new FastFourierTransform<T>();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 256)
        {
            return findings; // Need minimum samples for spectral analysis
        }

        var features = ComputeSpectralFeatures(audioSamples, sampleRate);
        T deepfakeScore = ComputeDeepfakeScore(features);

        if (NumOps.GreaterThanOrEquals(deepfakeScore, _threshold))
        {
            double scoreDouble = NumOps.ToDouble(deepfakeScore);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = SafetySeverity.Medium,
                Confidence = scoreDouble,
                Description = $"Audio flagged as potentially synthetic/deepfake (score: {scoreDouble:F3}). " +
                              $"Spectral flatness: {NumOps.ToDouble(features.SpectralFlatness):F3}, " +
                              $"ZCR consistency: {NumOps.ToDouble(features.ZCRConsistency):F3}, " +
                              $"HNR anomaly: {NumOps.ToDouble(features.HNRAnomaly):F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private SpectralFeatures ComputeSpectralFeatures(Vector<T> samples, int sampleRate)
    {
        int length = samples.Length;
        int frameSize = Math.Min(512, length);
        int hopSize = frameSize / 2;
        int numFrames = Math.Max(1, (length - frameSize) / hopSize + 1);

        // Pad frame size to next power of 2 for FFT
        int fftSize = 1;
        while (fftSize < frameSize) fftSize <<= 1;
        int specSize = fftSize / 2 + 1;

        // Per-frame features stored as Vector<T>
        var frameZCR = new Vector<T>(numFrames);
        var frameEnergy = new Vector<T>(numFrames);
        var frameSpectralFlatness = new Vector<T>(numFrames);
        var frameSpectralFlux = new Vector<T>(numFrames);
        var prevMagnitudes = new Vector<T>(specSize);

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * hopSize;
            int end = Math.Min(start + frameSize, length);
            int actualSize = end - start;

            // Zero-crossing rate for this frame
            int zeroCrossings = 0;
            T energySum = Zero;
            T prevSample = samples[start];

            for (int i = start + 1; i < end; i++)
            {
                T val = samples[i];
                energySum = NumOps.Add(energySum, NumOps.Multiply(val, val));

                if (IsZeroCrossing(prevSample, val))
                {
                    zeroCrossings++;
                }

                prevSample = val;
            }

            frameZCR[f] = actualSize > 1
                ? NumOps.Divide(NumOps.FromDouble(zeroCrossings), NumOps.FromDouble(actualSize - 1))
                : Zero;
            frameEnergy[f] = actualSize > 0
                ? NumOps.Divide(energySum, NumOps.FromDouble(actualSize))
                : Zero;

            // Compute power spectrum using FFT
            var magnitudes = ComputeMagnitudeSpectrum(samples, start, actualSize, fftSize, specSize);

            // Spectral flatness: geometric mean / arithmetic mean of power spectrum
            frameSpectralFlatness[f] = ComputeSpectralFlatness(magnitudes, specSize);

            // Spectral flux: sum of squared differences between consecutive frames
            if (f > 0)
            {
                T flux = Zero;
                for (int k = 0; k < specSize; k++)
                {
                    T diff = NumOps.Subtract(magnitudes[k], prevMagnitudes[k]);
                    flux = NumOps.Add(flux, NumOps.Multiply(diff, diff));
                }

                frameSpectralFlux[f] = NumOps.FromDouble(
                    Math.Sqrt(NumOps.ToDouble(flux) / specSize));
            }

            // Copy magnitudes to prevMagnitudes
            for (int k = 0; k < specSize; k++)
            {
                prevMagnitudes[k] = magnitudes[k];
            }
        }

        // Compute aggregate features
        T zcrMean = VectorMean(frameZCR, 0, numFrames);
        T zcrVariance = VectorVariance(frameZCR, zcrMean, 0, numFrames);
        T hundred = NumOps.FromDouble(100.0);
        T zcrConsistency = NumOps.GreaterThan(zcrVariance, Zero)
            ? NumOps.Divide(One, NumOps.Add(One, NumOps.Multiply(zcrVariance, hundred)))
            : One;

        T spectralFlatnessMean = VectorMean(frameSpectralFlatness, 0, numFrames);
        T spectralFlatnessVar = VectorVariance(frameSpectralFlatness, spectralFlatnessMean, 0, numFrames);

        T spectralFluxMean = numFrames > 1 ? VectorMean(frameSpectralFlux, 1, numFrames) : Zero;
        T spectralFluxVar = numFrames > 1 ? VectorVariance(frameSpectralFlux, spectralFluxMean, 1, numFrames) : Zero;

        // Energy dynamics
        T energyMean = VectorMean(frameEnergy, 0, numFrames);
        T energyVariance = VectorVariance(frameEnergy, energyMean, 0, numFrames);
        T energyDynamicRange = NumOps.GreaterThan(energyMean, Zero)
            ? NumOps.Divide(energyVariance, energyMean)
            : Zero;

        // Harmonic-to-noise ratio anomaly:
        // Synthetic speech typically has very high HNR (unnaturally clean harmonics)
        T hnr = EstimateHNR(samples, sampleRate);
        // HNR > 25 dB is suspicious for speech (natural speech ~6-20 dB)
        T fifteen = NumOps.FromDouble(15.0);
        T twenty = NumOps.FromDouble(20.0);
        T hnrAnomaly = NumOps.GreaterThan(hnr, fifteen)
            ? Clamp01(NumOps.Divide(NumOps.Subtract(hnr, fifteen), twenty))
            : Zero;

        return new SpectralFeatures
        {
            SpectralFlatness = spectralFlatnessMean,
            SpectralFlatnessVariance = spectralFlatnessVar,
            SpectralFluxMean = spectralFluxMean,
            SpectralFluxVariance = spectralFluxVar,
            ZCRConsistency = zcrConsistency,
            EnergyDynamicRange = energyDynamicRange,
            HNRAnomaly = hnrAnomaly,
            Duration = NumOps.Divide(NumOps.FromDouble(samples.Length), NumOps.FromDouble(sampleRate))
        };
    }

    /// <summary>
    /// Computes final deepfake score from spectral features.
    /// </summary>
    /// <remarks>
    /// Weighted combination of spectral anomaly indicators:
    /// - High spectral flatness (30%): synthetic audio has more uniform spectrum
    /// - ZCR over-consistency (20%): TTS produces unnaturally smooth waveforms
    /// - Low spectral flux variance (20%): synthetic audio has more regular transitions
    /// - HNR anomaly (20%): synthetic speech is unnaturally clean
    /// - Low energy dynamic range (10%): synthetic speech has compressed dynamics
    /// </remarks>
    private static T ComputeDeepfakeScore(SpectralFeatures features)
    {
        T pointOne = NumOps.FromDouble(0.1);
        T pointFive = NumOps.FromDouble(0.5);

        // Spectral flatness indicator: values above 0.1 are suspicious
        // (natural speech has lower flatness due to harmonic structure)
        T flatnessScore = NumOps.GreaterThan(features.SpectralFlatness, pointOne)
            ? Clamp01(NumOps.Divide(NumOps.Subtract(features.SpectralFlatness, pointOne), pointFive))
            : Zero;

        // ZCR consistency: very consistent ZCR across frames indicates synthetic
        T zcrScore = features.ZCRConsistency;

        // Spectral flux regularity: low variance in flux = synthetic (too regular transitions)
        T ten = NumOps.FromDouble(10.0);
        T fluxRegularityScore = NumOps.GreaterThan(features.SpectralFluxVariance, Zero)
            ? Clamp01(NumOps.Subtract(One, NumOps.Multiply(features.SpectralFluxVariance, ten)))
            : NumOps.FromDouble(0.5);

        // HNR anomaly: directly maps to synthetic likelihood
        T hnrScore = features.HNRAnomaly;

        // Energy dynamics: very compressed dynamic range indicates synthetic
        T two = NumOps.FromDouble(2.0);
        T dynamicsScore = NumOps.GreaterThan(features.EnergyDynamicRange, Zero)
            ? Clamp01(NumOps.Subtract(One, NumOps.Multiply(features.EnergyDynamicRange, two)))
            : NumOps.FromDouble(0.5);

        T w1 = NumOps.FromDouble(0.30);
        T w2 = NumOps.FromDouble(0.20);
        T w3 = NumOps.FromDouble(0.20);
        T w4 = NumOps.FromDouble(0.20);
        T w5 = NumOps.FromDouble(0.10);

        T score = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(w1, flatnessScore),
                NumOps.Multiply(w2, zcrScore)),
            NumOps.Add(
                NumOps.Add(
                    NumOps.Multiply(w3, fluxRegularityScore),
                    NumOps.Multiply(w4, hnrScore)),
                NumOps.Multiply(w5, dynamicsScore)));

        return Clamp01(score);
    }

    private Vector<T> ComputeMagnitudeSpectrum(Vector<T> samples, int start, int frameLength, int fftSize, int specSize)
    {
        // Build windowed frame, zero-padded to fftSize
        var frame = new Vector<T>(fftSize);
        T frameLenMinusOne = NumOps.FromDouble(frameLength - 1);
        T twoPi = NumOps.FromDouble(2.0 * Math.PI);

        for (int n = 0; n < frameLength; n++)
        {
            int idx = start + n;
            if (idx >= samples.Length) break;

            // Apply Hann window to reduce spectral leakage
            T nT = NumOps.FromDouble(n);
            T windowAngle = NumOps.Divide(NumOps.Multiply(twoPi, nT), frameLenMinusOne);
            T window = NumOps.Multiply(
                NumOps.FromDouble(0.5),
                NumOps.Subtract(One, NumOps.FromDouble(Math.Cos(NumOps.ToDouble(windowAngle)))));

            frame[n] = NumOps.Multiply(samples[idx], window);
        }

        // Use FFT for spectral analysis
        var complexSpectrum = _fft.Forward(frame);

        // Extract magnitude for positive frequencies
        var magnitudes = new Vector<T>(specSize);
        for (int k = 0; k < specSize && k < complexSpectrum.Length; k++)
        {
            magnitudes[k] = complexSpectrum[k].Magnitude;
        }

        return magnitudes;
    }

    private static T ComputeSpectralFlatness(Vector<T> magnitudes, int specSize)
    {
        if (specSize <= 1) return Zero;

        // Geometric mean / arithmetic mean of power spectrum
        // Use log-domain for geometric mean to avoid overflow
        double logSum = 0;
        T sum = Zero;
        int count = 0;

        for (int i = 1; i < specSize && i < magnitudes.Length; i++) // Skip DC
        {
            T mag = magnitudes[i];
            T power = NumOps.Add(NumOps.Multiply(mag, mag), Epsilon);
            logSum += Math.Log(NumOps.ToDouble(power));
            sum = NumOps.Add(sum, power);
            count++;
        }

        if (count == 0 || NumOps.LessThan(sum, Epsilon)) return Zero;

        double geometricMean = Math.Exp(logSum / count);
        double arithmeticMean = NumOps.ToDouble(sum) / count;

        if (arithmeticMean < 1e-10) return Zero;

        return NumOps.FromDouble(geometricMean / arithmeticMean);
    }

    private T EstimateHNR(Vector<T> samples, int sampleRate)
    {
        // Estimate Harmonic-to-Noise Ratio using autocorrelation
        int maxLag = sampleRate / 80;  // Min pitch ~80 Hz
        int minLag = sampleRate / 500; // Max pitch ~500 Hz
        int analysisLength = Math.Min(samples.Length, sampleRate); // Analyze up to 1 second

        T midRange = NumOps.FromDouble(10.0);
        if (analysisLength < maxLag * 2) return midRange; // Default mid-range

        // Compute autocorrelation at lag 0 (total energy)
        T r0 = Zero;
        for (int i = 0; i < analysisLength; i++)
        {
            T val = samples[i];
            r0 = NumOps.Add(r0, NumOps.Multiply(val, val));
        }

        if (NumOps.LessThan(r0, Epsilon)) return Zero;

        // Find peak autocorrelation in pitch range
        T maxAutocorr = Zero;
        for (int lag = minLag; lag <= maxLag && lag < analysisLength; lag++)
        {
            T rLag = Zero;
            for (int i = 0; i < analysisLength - lag; i++)
            {
                rLag = NumOps.Add(rLag, NumOps.Multiply(samples[i], samples[i + lag]));
            }

            T normalizedAutocorr = NumOps.Divide(rLag, r0);
            if (NumOps.GreaterThan(normalizedAutocorr, maxAutocorr))
            {
                maxAutocorr = normalizedAutocorr;
            }
        }

        // HNR in dB: 10 * log10(r_peak / (1 - r_peak))
        double maxAC = NumOps.ToDouble(maxAutocorr);
        if (maxAC >= 1.0) return NumOps.FromDouble(40.0); // Cap at 40 dB
        if (maxAC <= 0.0) return Zero;

        return NumOps.FromDouble(10.0 * Math.Log10(maxAC / (1.0 - maxAC)));
    }

    private static T VectorMean(Vector<T> values, int startIndex, int count)
    {
        if (count <= startIndex) return Zero;
        T sum = Zero;
        int n = 0;
        for (int i = startIndex; i < count; i++)
        {
            sum = NumOps.Add(sum, values[i]);
            n++;
        }

        return n > 0 ? NumOps.Divide(sum, NumOps.FromDouble(n)) : Zero;
    }

    private static T VectorVariance(Vector<T> values, T mean, int startIndex, int count)
    {
        if (count <= startIndex) return Zero;
        T sumSq = Zero;
        int n = 0;
        for (int i = startIndex; i < count; i++)
        {
            T diff = NumOps.Subtract(values[i], mean);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            n++;
        }

        return n > 0 ? NumOps.Divide(sumSq, NumOps.FromDouble(n)) : Zero;
    }

    private static bool IsZeroCrossing(T prev, T current)
    {
        bool currentNonNeg = NumOps.GreaterThanOrEquals(current, Zero);
        bool prevNonNeg = NumOps.GreaterThanOrEquals(prev, Zero);
        return currentNonNeg != prevNonNeg;
    }

    private static T Clamp01(T value)
    {
        if (NumOps.LessThan(value, Zero)) return Zero;
        if (NumOps.GreaterThan(value, One)) return One;
        return value;
    }

    private struct SpectralFeatures
    {
        public T SpectralFlatness;
        public T SpectralFlatnessVariance;
        public T SpectralFluxMean;
        public T SpectralFluxVariance;
        public T ZCRConsistency;
        public T EnergyDynamicRange;
        public T HNRAnomaly;
        public T Duration;
    }
}
