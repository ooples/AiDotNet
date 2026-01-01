namespace AiDotNet.Audio.Effects;

/// <summary>
/// Multi-band parametric equalizer effect.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A parametric EQ allows precise control over frequency response with
/// adjustable frequency, gain, and bandwidth (Q) for each band.
/// </para>
/// <para><b>For Beginners:</b> An equalizer adjusts the volume of different frequencies!
///
/// Think of it like a graphic equalizer on a stereo:
/// - Bass slider controls low frequencies (20-200 Hz)
/// - Mid slider controls middle frequencies (200-4000 Hz)
/// - Treble slider controls high frequencies (4000-20000 Hz)
///
/// A PARAMETRIC EQ is more flexible:
/// - Frequency: Which frequency to adjust (e.g., 1000 Hz)
/// - Gain: How much to boost/cut (e.g., +6 dB)
/// - Q (bandwidth): How wide the adjustment is (narrow = surgical, wide = gentle)
///
/// Filter types:
/// - Low Shelf: Boosts/cuts all frequencies below a point
/// - High Shelf: Boosts/cuts all frequencies above a point
/// - Peak (Bell): Boosts/cuts around a specific frequency
/// - Low Pass: Removes frequencies above a point
/// - High Pass: Removes frequencies below a point
///
/// Common EQ moves:
/// - Cut 200-400 Hz: Reduce muddiness
/// - Boost 3-5 kHz: Add presence to vocals
/// - Cut 2-4 kHz: Reduce harshness
/// - Boost 10+ kHz: Add "air" and sparkle
/// - High pass at 80 Hz: Remove rumble from vocals
/// </para>
/// </remarks>
public class ParametricEqualizer<T> : AudioEffectBase<T>
{
    #region Configuration

    /// <summary>
    /// EQ bands.
    /// </summary>
    private readonly List<EqBand<T>> _bands = [];

    #endregion

    /// <inheritdoc/>
    public override string Name => "Parametric EQ";

    /// <summary>
    /// Gets the EQ bands.
    /// </summary>
    public IReadOnlyList<EqBand<T>> Bands => _bands;

    /// <summary>
    /// Creates a parametric EQ with a default 5-band configuration.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 44100).</param>
    /// <param name="mix">Dry/wet mix (default: 1.0).</param>
    public ParametricEqualizer(int sampleRate = 44100, double mix = 1.0)
        : base(sampleRate, mix)
    {
        // Create default 5-band EQ
        AddBand(80, 0, 0.7, EqFilterType.LowShelf);      // Low shelf
        AddBand(300, 0, 1.0, EqFilterType.Peak);         // Low-mid
        AddBand(1000, 0, 1.0, EqFilterType.Peak);        // Mid
        AddBand(3500, 0, 1.0, EqFilterType.Peak);        // High-mid
        AddBand(10000, 0, 0.7, EqFilterType.HighShelf);  // High shelf
    }

    /// <summary>
    /// Adds an EQ band.
    /// </summary>
    /// <param name="frequency">Center/corner frequency in Hz.</param>
    /// <param name="gainDb">Gain in dB (-24 to +24).</param>
    /// <param name="q">Q factor (bandwidth, 0.1 to 10).</param>
    /// <param name="filterType">Type of filter.</param>
    public void AddBand(double frequency, double gainDb, double q, EqFilterType filterType)
    {
        var band = new EqBand<T>(NumOps, SampleRate, frequency, gainDb, q, filterType);
        _bands.Add(band);
    }

    /// <summary>
    /// Removes an EQ band by index.
    /// </summary>
    /// <param name="index">Band index.</param>
    public void RemoveBand(int index)
    {
        if (index >= 0 && index < _bands.Count)
        {
            _bands.RemoveAt(index);
        }
    }

    /// <summary>
    /// Sets band parameters.
    /// </summary>
    /// <param name="bandIndex">Band index.</param>
    /// <param name="frequency">Center frequency in Hz.</param>
    /// <param name="gainDb">Gain in dB.</param>
    /// <param name="q">Q factor.</param>
    public void SetBand(int bandIndex, double frequency, double gainDb, double q)
    {
        if (bandIndex >= 0 && bandIndex < _bands.Count)
        {
            _bands[bandIndex].SetParameters(frequency, gainDb, q);
        }
    }

    /// <inheritdoc/>
    protected override T ProcessSampleInternal(T input)
    {
        T output = input;

        // Process through each band in series
        foreach (var band in _bands)
        {
            output = band.Process(output);
        }

        return output;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        foreach (var band in _bands)
        {
            band.Reset();
        }
    }
}

/// <summary>
/// Types of EQ filters.
/// </summary>
public enum EqFilterType
{
    /// <summary>Peak/Bell filter.</summary>
    Peak,
    /// <summary>Low shelf filter.</summary>
    LowShelf,
    /// <summary>High shelf filter.</summary>
    HighShelf,
    /// <summary>Low pass filter.</summary>
    LowPass,
    /// <summary>High pass filter.</summary>
    HighPass,
    /// <summary>Band pass filter.</summary>
    BandPass,
    /// <summary>Notch filter.</summary>
    Notch
}

/// <summary>
/// Represents a single EQ band with biquad filter.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class EqBand<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;

    /// <summary>
    /// Center/corner frequency in Hz.
    /// </summary>
    public double Frequency { get; private set; }

    /// <summary>
    /// Gain in dB.
    /// </summary>
    public double GainDb { get; private set; }

    /// <summary>
    /// Q factor (bandwidth).
    /// </summary>
    public double Q { get; private set; }

    /// <summary>
    /// Filter type.
    /// </summary>
    public EqFilterType FilterType { get; private set; }

    // Biquad coefficients
    private double _a0, _a1, _a2, _b0, _b1, _b2;

    // Filter state
    private double _x1, _x2, _y1, _y2;

    /// <summary>
    /// Creates a new EQ band.
    /// </summary>
    public EqBand(INumericOperations<T> numOps, int sampleRate, double frequency, double gainDb, double q, EqFilterType filterType)
    {
        _numOps = numOps;
        _sampleRate = sampleRate;
        FilterType = filterType;
        SetParameters(frequency, gainDb, q);
    }

    /// <summary>
    /// Sets the band parameters and recalculates coefficients.
    /// </summary>
    public void SetParameters(double frequency, double gainDb, double q)
    {
        Frequency = Math.Max(20, Math.Min(20000, frequency));
        GainDb = Math.Max(-24, Math.Min(24, gainDb));
        Q = Math.Max(0.1, Math.Min(10, q));

        CalculateCoefficients();
    }

    private void CalculateCoefficients()
    {
        double omega = 2 * Math.PI * Frequency / _sampleRate;
        double sinOmega = Math.Sin(omega);
        double cosOmega = Math.Cos(omega);
        double alpha = sinOmega / (2 * Q);
        double A = Math.Pow(10, GainDb / 40);

        switch (FilterType)
        {
            case EqFilterType.Peak:
                _b0 = 1 + alpha * A;
                _b1 = -2 * cosOmega;
                _b2 = 1 - alpha * A;
                _a0 = 1 + alpha / A;
                _a1 = -2 * cosOmega;
                _a2 = 1 - alpha / A;
                break;

            case EqFilterType.LowShelf:
                double sqrtA = Math.Sqrt(A);
                _b0 = A * ((A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha);
                _b1 = 2 * A * ((A - 1) - (A + 1) * cosOmega);
                _b2 = A * ((A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha);
                _a0 = (A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha;
                _a1 = -2 * ((A - 1) + (A + 1) * cosOmega);
                _a2 = (A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha;
                break;

            case EqFilterType.HighShelf:
                sqrtA = Math.Sqrt(A);
                _b0 = A * ((A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha);
                _b1 = -2 * A * ((A - 1) + (A + 1) * cosOmega);
                _b2 = A * ((A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha);
                _a0 = (A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha;
                _a1 = 2 * ((A - 1) - (A + 1) * cosOmega);
                _a2 = (A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha;
                break;

            case EqFilterType.LowPass:
                _b0 = (1 - cosOmega) / 2;
                _b1 = 1 - cosOmega;
                _b2 = (1 - cosOmega) / 2;
                _a0 = 1 + alpha;
                _a1 = -2 * cosOmega;
                _a2 = 1 - alpha;
                break;

            case EqFilterType.HighPass:
                _b0 = (1 + cosOmega) / 2;
                _b1 = -(1 + cosOmega);
                _b2 = (1 + cosOmega) / 2;
                _a0 = 1 + alpha;
                _a1 = -2 * cosOmega;
                _a2 = 1 - alpha;
                break;

            case EqFilterType.BandPass:
                _b0 = alpha;
                _b1 = 0;
                _b2 = -alpha;
                _a0 = 1 + alpha;
                _a1 = -2 * cosOmega;
                _a2 = 1 - alpha;
                break;

            case EqFilterType.Notch:
                _b0 = 1;
                _b1 = -2 * cosOmega;
                _b2 = 1;
                _a0 = 1 + alpha;
                _a1 = -2 * cosOmega;
                _a2 = 1 - alpha;
                break;
        }

        // Normalize coefficients
        _b0 /= _a0;
        _b1 /= _a0;
        _b2 /= _a0;
        _a1 /= _a0;
        _a2 /= _a0;
    }

    /// <summary>
    /// Processes a sample through the biquad filter.
    /// </summary>
    public T Process(T input)
    {
        double x0 = _numOps.ToDouble(input);

        // Biquad filter: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        double y0 = _b0 * x0 + _b1 * _x1 + _b2 * _x2 - _a1 * _y1 - _a2 * _y2;

        // Update state
        _x2 = _x1;
        _x1 = x0;
        _y2 = _y1;
        _y1 = y0;

        return _numOps.FromDouble(y0);
    }

    /// <summary>
    /// Resets the filter state.
    /// </summary>
    public void Reset()
    {
        _x1 = _x2 = _y1 = _y2 = 0;
    }
}
