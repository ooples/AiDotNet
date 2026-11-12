using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SafeEar: Content Privacy-Preserving Audio Deepfake Detection",
    "https://dl.acm.org/doi/10.1145/3658644.3690291",
    Year = 2024,
    Authors = "Various")]
internal class SpectralDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _threshold;

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
        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;
        int length = samples.Length;
        int frameSize = Math.Min(512, length);
        int hopSize = frameSize / 2;
        int numFrames = Math.Max(1, (length - frameSize) / hopSize + 1);

        // Pad frame size to next power of 2 for FFT
        int fftSize = 1;
        while (fftSize < frameSize) fftSize <<= 1;
        int specSize = fftSize / 2 + 1;

        // Per-frame features, every frame at once: frames are [numFrames, frameSize]
        // (frame f starts at f * hopSize and always fits inside the signal).
        var frames = SafetyTensorOps<T>.Frames(SafetyTensorOps<T>.AsTensor(samples), frameSize, hopSize, numFrames);
        var axis1 = new[] { 1 };

        // Zero-crossing rate and energy (the energy sum starts at the frame's second sample)
        var frameZCR = frameSize > 1
            ? engine.TensorDivideScalar(SafetyTensorOps<T>.ZeroCrossings(frames, nanIsNegative: true),
                NumOps.FromDouble(frameSize - 1))
            : Tensor<T>.CreateZeros(numFrames);
        var frameEnergy = frameSize > 1
            ? engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorSquare(engine.TensorNarrow(frames, 1, 1, frameSize - 1)), axis1, false),
                NumOps.FromDouble(frameSize))
            : Tensor<T>.CreateZeros(numFrames);

        // Compute power spectrum using FFT
        var magnitudes = ComputeMagnitudeSpectrum(frames, fftSize, specSize);

        // Spectral flatness: geometric mean / arithmetic mean of power spectrum
        var frameSpectralFlatness = ComputeSpectralFlatness(magnitudes, specSize);

        // Spectral flux: RMS of the differences between consecutive frames (frames 1..)
        var frameSpectralFlux = numFrames > 1
            ? engine.TensorSqrt(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorSquare(engine.TensorDiff(magnitudes, 1, 0)), axis1, false),
                NumOps.FromDouble(specSize)))
            : Tensor<T>.CreateZeros(0);

        // Compute aggregate features
        var (zcrMean, zcrVariance) = MeanVariance(frameZCR, 0);
        T hundred = NumOps.FromDouble(100.0);
        T zcrConsistency = NumOps.GreaterThan(zcrVariance, Zero)
            ? NumOps.Divide(One, NumOps.Add(One, NumOps.Multiply(zcrVariance, hundred)))
            : One;

        var (spectralFlatnessMean, spectralFlatnessVar) = MeanVariance(frameSpectralFlatness, 0);

        var (spectralFluxMean, spectralFluxVar) = numFrames > 1
            ? MeanVariance(frameSpectralFlux, 0)
            : (Zero, Zero);

        // Energy dynamics
        var (energyMean, energyVariance) = MeanVariance(frameEnergy, 0);
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

        T term1 = NumOps.Multiply(w1, flatnessScore);
        T term2 = NumOps.Multiply(w2, zcrScore);
        T term3 = NumOps.Multiply(w3, fluxRegularityScore);
        T term4 = NumOps.Multiply(w4, hnrScore);
        T term5 = NumOps.Multiply(w5, dynamicsScore);

        T score = NumOps.Add(NumOps.Add(NumOps.Add(term1, term2), NumOps.Add(term3, term4)), term5);

        return Clamp01(score);
    }

    private static Tensor<T> ComputeMagnitudeSpectrum(Tensor<T> frames, int fftSize, int specSize)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int frameCount = frames.Shape[0];
        int frameLength = frames.Shape[1];

        // Apply Hann window to reduce spectral leakage, zero-padded to fftSize
        var windowed = engine.TensorMultiply(frames, SafetyTensorOps<T>.SymmetricHann(frameLength));
        if (fftSize > frameLength)
        {
            windowed = engine.TensorConcatenate(new[]
            {
                windowed, Tensor<T>.CreateZeros(frameCount, fftSize - frameLength)
            }, 1);
        }

        // Magnitude for positive frequencies
        var magnitudes = SafetyTensorOps<T>.FftMagnitude(windowed);
        return engine.TensorNarrow(magnitudes, 1, 0, Math.Min(specSize, fftSize));
    }

    private static Tensor<T> ComputeSpectralFlatness(Tensor<T> magnitudes, int specSize)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int frameCount = magnitudes.Shape[0];
        int count = Math.Min(specSize, magnitudes.Shape[1]) - 1;
        if (specSize <= 1 || count <= 0) return Tensor<T>.CreateZeros(frameCount);

        // Geometric mean / arithmetic mean of power spectrum (skip DC).
        // Use log-domain for geometric mean to avoid overflow
        var axis1 = new[] { 1 };
        var power = engine.TensorAddScalar(
            engine.TensorSquare(engine.TensorNarrow(magnitudes, 1, 1, count)), Epsilon);
        var logSum = engine.ReduceSum(engine.TensorLog(power), axis1, false);
        var sum = engine.ReduceSum(power, axis1, false);
        T countT = NumOps.FromDouble(count);

        var geometricMean = engine.TensorExp(engine.TensorDivideScalar(logSum, countT));
        var arithmeticMean = engine.TensorDivideScalar(sum, countT);
        var valid = SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.Not(engine.TensorLessThan(sum, Epsilon)),
            SafetyTensorOps<T>.Not(SafetyTensorOps<T>.Lt(arithmeticMean, 1e-10)));
        var safeMean = SafetyTensorOps<T>.Where(valid, arithmeticMean, 1.0);
        return SafetyTensorOps<T>.Where(valid, engine.TensorDivide(geometricMean, safeMean), 0.0);
    }

    private static T EstimateHNR(Vector<T> samples, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;

        // Estimate Harmonic-to-Noise Ratio using autocorrelation
        int maxLag = sampleRate / 80;  // Min pitch ~80 Hz
        int minLag = sampleRate / 500; // Max pitch ~500 Hz
        int analysisLength = Math.Min(samples.Length, sampleRate); // Analyze up to 1 second

        T midRange = NumOps.FromDouble(10.0);
        if (analysisLength < maxLag * 2) return midRange; // Default mid-range

        // Compute autocorrelation at lag 0 (total energy)
        var window = engine.Reshape(
            engine.TensorNarrow(SafetyTensorOps<T>.AsTensor(samples), 0, 0, analysisLength),
            new[] { 1, analysisLength });
        T r0 = engine.TensorSumOfSquares(window);

        if (NumOps.LessThan(r0, Epsilon)) return Zero;

        // Find peak autocorrelation in pitch range (first strictly larger positive value)
        int lagEnd = Math.Min(maxLag + 1, analysisLength);
        if (minLag >= lagEnd) return Zero;
        SafetyTensorOps<T>.LaggedProducts(window, minLag, lagEnd, out var rLag, out _, out _);
        var normalizedAutocorr = engine.TensorDivideScalar(rLag, r0);
        var (best, _) = SafetyTensorOps<T>.FirstStrictMax(normalizedAutocorr, 0.0);
        T maxAutocorr = best.GetFlat(0);

        // HNR in dB: 10 * log10(r_peak / (1 - r_peak))
        if (NumOps.GreaterThanOrEquals(maxAutocorr, One)) return NumOps.FromDouble(40.0); // Cap at 40 dB
        if (NumOps.LessThanOrEquals(maxAutocorr, Zero)) return Zero;

        T ratio = NumOps.Divide(maxAutocorr, NumOps.Subtract(One, maxAutocorr));
        return NumOps.Multiply(NumOps.FromDouble(10.0 / Math.Log(10.0)), NumOps.Log(ratio));
    }

    private static (T Mean, T Variance) MeanVariance(Tensor<T> values, int startIndex)
    {
        int count = values.Length;
        if (count <= startIndex) return (Zero, Zero);
        var slice = startIndex == 0
            ? values
            : SafetyTensorOps<T>.Engine.TensorNarrow(values, 0, startIndex, count - startIndex);
        return SafetyTensorOps<T>.MeanVariance(slice);
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
