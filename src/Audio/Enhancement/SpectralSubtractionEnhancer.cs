namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Audio enhancer using spectral subtraction for noise reduction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spectral subtraction is a classic noise reduction technique that:
/// 1. Estimates the noise spectrum during silent periods
/// 2. Subtracts the noise spectrum from the noisy signal spectrum
/// 3. Reconstructs the cleaned signal
/// </para>
/// <para><b>For Beginners:</b> Think of it like this:
///
/// Imagine you're in a cafe trying to hear a friend:
/// - Noisy signal = friend's voice + cafe background noise
/// - If we know what the cafe sounds like alone (noise estimate)
/// - We can "subtract" the cafe sound to hear just the friend
///
/// Advantages:
/// - Simple and fast
/// - Low latency (good for real-time)
/// - Works well for stationary noise (AC hum, fan noise)
///
/// Limitations:
/// - Can introduce "musical noise" artifacts (twinkling sounds)
/// - Struggles with non-stationary noise (traffic, other speakers)
/// - May reduce speech quality if over-applied
///
/// This implementation includes:
/// - Adaptive noise estimation
/// - Spectral flooring (prevents negative magnitudes)
/// - Smoothing to reduce musical noise
/// </para>
/// </remarks>
public class SpectralSubtractionEnhancer<T> : AudioEnhancerBase<T>
{
    #region Configuration

    /// <summary>
    /// Over-subtraction factor (1.0 = exact subtraction, higher = more aggressive).
    /// </summary>
    private readonly double _alpha;

    /// <summary>
    /// Spectral floor factor (prevents complete zeroing of bins).
    /// </summary>
    private readonly double _beta;

    /// <summary>
    /// Smoothing factor for noise estimate updates.
    /// </summary>
    private readonly double _smoothingFactor;

    /// <summary>
    /// Whether to use adaptive noise estimation.
    /// </summary>
    private readonly bool _adaptiveNoiseEstimation;

    #endregion

    #region State

    /// <summary>
    /// Running noise estimate.
    /// </summary>
    private T[]? _runningNoiseEstimate;

    /// <summary>
    /// Previous frame magnitudes for smoothing.
    /// </summary>
    private T[]? _previousMagnitudes;

    #endregion

    /// <summary>
    /// Initializes a new SpectralSubtractionEnhancer with default parameters.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 16000 Hz).</param>
    /// <param name="fftSize">FFT size (default: 512).</param>
    /// <param name="hopSize">Hop size (default: 128).</param>
    /// <param name="alpha">Over-subtraction factor (default: 2.0).</param>
    /// <param name="beta">Spectral floor factor (default: 0.01).</param>
    /// <param name="smoothingFactor">Noise estimate smoothing (default: 0.98).</param>
    /// <param name="adaptiveNoiseEstimation">Enable adaptive noise tracking (default: true).</param>
    /// <param name="enhancementStrength">Overall enhancement strength 0-1 (default: 0.7).</param>
    public SpectralSubtractionEnhancer(
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 128,
        double alpha = 2.0,
        double beta = 0.01,
        double smoothingFactor = 0.98,
        bool adaptiveNoiseEstimation = true,
        double enhancementStrength = 0.7)
        : base(sampleRate, 1, fftSize, hopSize, enhancementStrength)
    {
        _alpha = alpha;
        _beta = beta;
        _smoothingFactor = smoothingFactor;
        _adaptiveNoiseEstimation = adaptiveNoiseEstimation;

        InitializeState();
    }

    private void InitializeState()
    {
        int numBins = _fftSize / 2 + 1;
        _runningNoiseEstimate = new T[numBins];
        _previousMagnitudes = new T[numBins];
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        InitializeState();
    }

    /// <inheritdoc/>
    protected override T[] ProcessSpectralFrame(T[] magnitudes, T[] phases)
    {
        if (_runningNoiseEstimate is null || _previousMagnitudes is null)
            InitializeState();

        var enhanced = new T[magnitudes.Length];
        var strength = EnhancementStrength;

        // Use provided noise profile or running estimate
        var noiseEstimate = _noiseProfile ?? _runningNoiseEstimate!;

        // Calculate frame energy for VAD
        double frameEnergy = 0;
        for (int i = 0; i < magnitudes.Length; i++)
        {
            frameEnergy += NumOps.ToDouble(magnitudes[i]) * NumOps.ToDouble(magnitudes[i]);
        }
        frameEnergy = Math.Sqrt(frameEnergy / magnitudes.Length);

        // Simple VAD: update noise estimate during low-energy frames
        if (_adaptiveNoiseEstimation && frameEnergy < GetNoiseThreshold())
        {
            UpdateNoiseEstimate(magnitudes);
        }

        // Apply spectral subtraction
        for (int i = 0; i < magnitudes.Length; i++)
        {
            double mag = NumOps.ToDouble(magnitudes[i]);
            double noise = NumOps.ToDouble(noiseEstimate[i]);

            // Spectral subtraction with over-subtraction
            double subtracted = mag * mag - _alpha * strength * noise * noise;

            // Spectral flooring
            double floor = _beta * mag * mag;
            subtracted = Math.Max(subtracted, floor);

            // Take square root and apply smoothing
            double enhancedMag = Math.Sqrt(subtracted);

            // Temporal smoothing to reduce musical noise
            double prevMag = NumOps.ToDouble(_previousMagnitudes![i]);
            enhancedMag = 0.3 * prevMag + 0.7 * enhancedMag;

            enhanced[i] = NumOps.FromDouble(enhancedMag);
            _previousMagnitudes[i] = enhanced[i];
        }

        return enhanced;
    }

    /// <summary>
    /// Updates the running noise estimate with new frame.
    /// </summary>
    private void UpdateNoiseEstimate(T[] magnitudes)
    {
        if (_runningNoiseEstimate is null) return;

        for (int i = 0; i < magnitudes.Length; i++)
        {
            double current = NumOps.ToDouble(_runningNoiseEstimate[i]);
            double newMag = NumOps.ToDouble(magnitudes[i]);

            // Exponential smoothing
            double updated = _smoothingFactor * current + (1 - _smoothingFactor) * newMag;
            _runningNoiseEstimate[i] = NumOps.FromDouble(updated);
        }
    }

    /// <summary>
    /// Gets the threshold for considering a frame as noise-only.
    /// </summary>
    private double GetNoiseThreshold()
    {
        if (_runningNoiseEstimate is null) return 0.01;

        double sum = 0;
        for (int i = 0; i < _runningNoiseEstimate.Length; i++)
        {
            sum += NumOps.ToDouble(_runningNoiseEstimate[i]);
        }
        return sum / _runningNoiseEstimate.Length * 2.0;
    }

}
