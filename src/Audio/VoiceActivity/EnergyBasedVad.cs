using System.Numerics;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Simple energy-based voice activity detector (algorithmic, no neural network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is a basic VAD that detects speech based on signal energy (loudness).
/// It combines multiple features for more robust detection:
/// - Short-time energy
/// - Zero-crossing rate
/// - Spectral flatness
/// </para>
/// <para><b>For Beginners:</b> This is the simplest type of VAD:
///
/// Basic idea: Speech is louder than silence!
/// - Compute the "energy" (sum of squared samples) for each frame
/// - If energy exceeds a threshold, it's probably speech
///
/// Enhanced features used here:
/// 1. Energy: How loud is the signal?
/// 2. Zero-crossings: How often does the signal cross zero?
///    - Speech: Medium zero-crossings (voiced sounds)
///    - Noise: High zero-crossings (random noise)
/// 3. Spectral flatness: Is it tonal or noisy?
///    - Speech: Low flatness (has harmonic structure)
///    - Noise: High flatness (random spectrum)
///
/// Pros:
/// - Very fast (no neural network)
/// - Low latency
/// - Works well in quiet environments
///
/// Cons:
/// - Struggles with background noise
/// - May trigger on loud non-speech sounds
/// - Requires threshold tuning for different environments
///
/// For better noise robustness, use neural network-based VAD like SileroVad.
/// </para>
/// </remarks>
public class EnergyBasedVad<T> : VoiceActivityDetectorBase<T>
{
    #region Configuration

    private readonly double _energyWeight;
    private readonly double _zcrWeight;
    private readonly double _flatnessWeight;
    private readonly bool _adaptiveThreshold;

    #endregion

    #region Adaptive State

    private double _noiseFloor;
    private double _signalPeak;
    private readonly double _adaptationRate;

    #endregion

    /// <summary>
    /// Creates an energy-based VAD with default parameters.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 16000).</param>
    /// <param name="frameSize">Frame size in samples (default: 480 = 30ms at 16kHz).</param>
    /// <param name="threshold">Detection threshold 0-1 (default: 0.5).</param>
    /// <param name="energyWeight">Weight for energy feature (default: 0.5).</param>
    /// <param name="zcrWeight">Weight for zero-crossing rate (default: 0.25).</param>
    /// <param name="flatnessWeight">Weight for spectral flatness (default: 0.25).</param>
    /// <param name="adaptiveThreshold">Enable adaptive threshold (default: true).</param>
    /// <param name="minSpeechDurationMs">Minimum speech duration (default: 250ms).</param>
    /// <param name="minSilenceDurationMs">Minimum silence duration (default: 300ms).</param>
    public EnergyBasedVad(
        int sampleRate = 16000,
        int frameSize = 480,
        double threshold = 0.5,
        double energyWeight = 0.5,
        double zcrWeight = 0.25,
        double flatnessWeight = 0.25,
        bool adaptiveThreshold = true,
        int minSpeechDurationMs = 250,
        int minSilenceDurationMs = 300)
        : base(sampleRate, frameSize, threshold, minSpeechDurationMs, minSilenceDurationMs)
    {
        _energyWeight = energyWeight;
        _zcrWeight = zcrWeight;
        _flatnessWeight = flatnessWeight;
        _adaptiveThreshold = adaptiveThreshold;
        _adaptationRate = 0.01;

        // Initialize adaptive thresholds
        _noiseFloor = 0.001;
        _signalPeak = 0.1;
    }

    /// <inheritdoc/>
    protected override T ComputeFrameProbability(T[] frame)
    {
        // Compute features
        double energy = ComputeEnergy(frame);
        double zcr = ComputeZeroCrossingRate(frame);
        double flatness = ComputeSpectralFlatness(frame);

        // Update adaptive thresholds
        if (_adaptiveThreshold)
        {
            UpdateAdaptiveThresholds(energy);
        }

        // Normalize features to 0-1 range
        double energyScore = NormalizeEnergy(energy);
        double zcrScore = NormalizeZcr(zcr);
        double flatnessScore = 1.0 - flatness; // Invert: speech has low flatness

        // Combine features
        double combinedScore = _energyWeight * energyScore +
                              _zcrWeight * zcrScore +
                              _flatnessWeight * flatnessScore;

        // Clamp to 0-1
        combinedScore = Math.Max(0.0, Math.Min(1.0, combinedScore));

        return NumOps.FromDouble(combinedScore);
    }

    #region Feature Computation

    private double ComputeEnergy(T[] frame)
    {
        double sum = 0;
        for (int i = 0; i < frame.Length; i++)
        {
            double val = NumOps.ToDouble(frame[i]);
            sum += val * val;
        }
        return sum / frame.Length;
    }

    private double ComputeZeroCrossingRate(T[] frame)
    {
        int crossings = 0;
        for (int i = 1; i < frame.Length; i++)
        {
            double prev = NumOps.ToDouble(frame[i - 1]);
            double curr = NumOps.ToDouble(frame[i]);
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
            {
                crossings++;
            }
        }
        return (double)crossings / (frame.Length - 1);
    }

    private double ComputeSpectralFlatness(T[] frame)
    {
        // Simple spectral flatness using FFT magnitudes
        int fftSize = 256;

        // Prepare frame data for FFT (zero-padded if needed)
        var frameData = new double[fftSize];
        int copyLen = Math.Min(frame.Length, fftSize);
        for (int i = 0; i < copyLen; i++)
        {
            frameData[i] = NumOps.ToDouble(frame[i]);
        }

        // Use FftSharp for O(N log N) FFT
        Complex[] spectrum = FftSharp.FFT.Forward(frameData);

        // Extract magnitude spectrum for positive frequencies
        var magnitudes = new double[fftSize / 2];
        for (int k = 0; k < fftSize / 2; k++)
        {
            magnitudes[k] = spectrum[k].Magnitude + 1e-10;
        }

        // Spectral flatness = geometric mean / arithmetic mean
        double logSum = 0;
        double sum = 0;
        for (int i = 0; i < magnitudes.Length; i++)
        {
            logSum += Math.Log(magnitudes[i]);
            sum += magnitudes[i];
        }

        double geometricMean = Math.Exp(logSum / magnitudes.Length);
        double arithmeticMean = sum / magnitudes.Length;

        return geometricMean / arithmeticMean;
    }

    #endregion

    #region Adaptive Threshold

    private void UpdateAdaptiveThresholds(double energy)
    {
        // Slowly adapt noise floor and signal peak
        if (energy < _noiseFloor * 2)
        {
            // Likely noise - update noise floor
            _noiseFloor = _noiseFloor * (1 - _adaptationRate) + energy * _adaptationRate;
        }
        else if (energy > _signalPeak * 0.5)
        {
            // Likely signal - update signal peak
            _signalPeak = _signalPeak * (1 - _adaptationRate) + energy * _adaptationRate;
        }

        // Ensure reasonable bounds
        _noiseFloor = Math.Max(_noiseFloor, 1e-8);
        _signalPeak = Math.Max(_signalPeak, _noiseFloor * 10);
    }

    private double NormalizeEnergy(double energy)
    {
        // Normalize between noise floor and signal peak
        double range = _signalPeak - _noiseFloor;
        if (range <= 0) return 0;

        double normalized = (energy - _noiseFloor) / range;
        return Math.Max(0, Math.Min(1, normalized));
    }

    private double NormalizeZcr(double zcr)
    {
        // Speech typically has ZCR between 0.05 and 0.25
        // Too low = probably silence, too high = probably noise
        if (zcr < 0.02) return 0;
        if (zcr > 0.35) return 0.3; // High ZCR might be fricatives

        // Peak around 0.15
        double score = 1.0 - Math.Abs(zcr - 0.15) / 0.2;
        return Math.Max(0, Math.Min(1, score));
    }

    #endregion

    /// <summary>
    /// Resets the VAD state including adaptive thresholds.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _noiseFloor = 0.001;
        _signalPeak = 0.1;
    }
}
