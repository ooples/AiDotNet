namespace AiDotNet.Audio.Effects;

/// <summary>
/// Dynamic range compressor effect.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A compressor reduces the dynamic range of audio by attenuating signals
/// that exceed a threshold. This makes quiet parts louder relative to loud parts.
/// </para>
/// <para><b>For Beginners:</b> A compressor is like an automatic volume control!
///
/// The problem it solves:
/// - Singer gets too close to mic → TOO LOUD
/// - Singer backs away → too quiet
/// - Drums hit hard → peaks clip
///
/// What a compressor does:
/// - Watches the signal level
/// - When it exceeds the threshold, it turns down the volume
/// - The ratio controls how much it turns down (4:1 means 4dB over becomes 1dB over)
///
/// Key parameters:
/// - Threshold: Level above which compression starts (dB)
/// - Ratio: How aggressively to compress (2:1 = gentle, 20:1 = limiting)
/// - Attack: How quickly compression kicks in (ms)
/// - Release: How quickly compression releases (ms)
/// - Makeup Gain: Boost to compensate for reduced volume (dB)
///
/// Common uses:
/// - Vocals: Even out dynamics (ratio 3:1 to 6:1)
/// - Drums: Punch and sustain (ratio 4:1 to 8:1)
/// - Bass: Consistent level (ratio 4:1 to 6:1)
/// - Master bus: Glue the mix together (ratio 2:1 to 4:1)
/// - Podcasts: Keep voice at consistent level
/// </para>
/// </remarks>
public class Compressor<T> : AudioEffectBase<T>
{
    #region Configuration

    private double _threshold;
    private double _ratio;
    private double _attackMs;
    private double _releaseMs;
    private double _makeupGainDb;
    private double _kneeDb;

    #endregion

    #region State

    private double _envelope;
    private double _attackCoef;
    private double _releaseCoef;

    #endregion

    /// <inheritdoc/>
    public override string Name => "Compressor";

    /// <summary>
    /// Creates a compressor with default broadcast settings.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 44100).</param>
    /// <param name="thresholdDb">Compression threshold in dB (default: -20).</param>
    /// <param name="ratio">Compression ratio (default: 4).</param>
    /// <param name="attackMs">Attack time in ms (default: 10).</param>
    /// <param name="releaseMs">Release time in ms (default: 100).</param>
    /// <param name="makeupGainDb">Makeup gain in dB (default: 0).</param>
    /// <param name="kneeDb">Soft knee width in dB (default: 6).</param>
    /// <param name="mix">Dry/wet mix (default: 1.0).</param>
    public Compressor(
        int sampleRate = 44100,
        double thresholdDb = -20.0,
        double ratio = 4.0,
        double attackMs = 10.0,
        double releaseMs = 100.0,
        double makeupGainDb = 0.0,
        double kneeDb = 6.0,
        double mix = 1.0)
        : base(sampleRate, mix)
    {
        // Initialize parameters
        AddParameter("threshold", "Threshold", NumOps.FromDouble(-60), NumOps.FromDouble(0), NumOps.FromDouble(thresholdDb), "dB", "Level above which compression begins");
        AddParameter("ratio", "Ratio", NumOps.FromDouble(1), NumOps.FromDouble(20), NumOps.FromDouble(ratio), ":1", "Compression ratio");
        AddParameter("attack", "Attack", NumOps.FromDouble(0.1), NumOps.FromDouble(100), NumOps.FromDouble(attackMs), "ms", "Attack time");
        AddParameter("release", "Release", NumOps.FromDouble(10), NumOps.FromDouble(1000), NumOps.FromDouble(releaseMs), "ms", "Release time");
        AddParameter("makeup", "Makeup Gain", NumOps.FromDouble(0), NumOps.FromDouble(24), NumOps.FromDouble(makeupGainDb), "dB", "Makeup gain");
        AddParameter("knee", "Knee", NumOps.FromDouble(0), NumOps.FromDouble(12), NumOps.FromDouble(kneeDb), "dB", "Soft knee width");

        _threshold = thresholdDb;
        _ratio = ratio;
        _attackMs = attackMs;
        _releaseMs = releaseMs;
        _makeupGainDb = makeupGainDb;
        _kneeDb = kneeDb;

        CalculateCoefficients();
        Reset();
    }

    private void CalculateCoefficients()
    {
        // Time constants for envelope follower
        _attackCoef = Math.Exp(-1.0 / (SampleRate * _attackMs / 1000.0));
        _releaseCoef = Math.Exp(-1.0 / (SampleRate * _releaseMs / 1000.0));
    }

    /// <inheritdoc/>
    protected override void OnParameterChanged(string name, T value)
    {
        double val = NumOps.ToDouble(value);
        switch (name)
        {
            case "threshold": _threshold = val; break;
            case "ratio": _ratio = val; break;
            case "attack": _attackMs = val; CalculateCoefficients(); break;
            case "release": _releaseMs = val; CalculateCoefficients(); break;
            case "makeup": _makeupGainDb = val; break;
            case "knee": _kneeDb = val; break;
        }
    }

    /// <inheritdoc/>
    protected override T ProcessSampleInternal(T input)
    {
        double sample = NumOps.ToDouble(input);

        // Get input level in dB
        double inputLevel = Math.Abs(sample);
        double inputDb = LinearToDb(inputLevel);

        // Update envelope (peak detector)
        double targetEnvelope = inputDb;
        if (targetEnvelope > _envelope)
        {
            _envelope = _attackCoef * _envelope + (1 - _attackCoef) * targetEnvelope;
        }
        else
        {
            _envelope = _releaseCoef * _envelope + (1 - _releaseCoef) * targetEnvelope;
        }

        // Calculate gain reduction
        double gainReduction = ComputeGainReduction(_envelope);

        // Apply gain reduction and makeup gain
        double outputDb = inputDb - gainReduction + _makeupGainDb;
        double outputLevel = DbToLinear(outputDb);

        // Preserve sign
        double output = sample >= 0 ? outputLevel : -outputLevel;

        return NumOps.FromDouble(output);
    }

    private double ComputeGainReduction(double inputDb)
    {
        // Below threshold: no reduction
        if (inputDb < _threshold - _kneeDb / 2)
        {
            return 0;
        }

        // Above threshold + knee: full compression
        if (inputDb > _threshold + _kneeDb / 2)
        {
            double overThreshold = inputDb - _threshold;
            return overThreshold * (1 - 1 / _ratio);
        }

        // In the knee: gradual transition
        double kneeInput = inputDb - _threshold + _kneeDb / 2;
        double kneeOutput = (kneeInput * kneeInput) / (2 * _kneeDb);
        return kneeOutput * (1 - 1 / _ratio);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        _envelope = -96; // Very quiet starting envelope
    }

    /// <summary>
    /// Gets the current gain reduction in dB.
    /// </summary>
    /// <returns>Current gain reduction (negative value).</returns>
    public double GetGainReduction()
    {
        return -ComputeGainReduction(_envelope);
    }
}
