namespace AiDotNet.Audio.Effects;

/// <summary>
/// Algorithmic reverb effect using Schroeder-Moorer structure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reverb simulates the acoustic reflections of a physical space,
/// adding depth and ambience to dry recordings.
/// </para>
/// <para><b>For Beginners:</b> Reverb makes things sound like they're in a room!
///
/// What reverb is:
/// - Sound bounces off walls, ceiling, floor
/// - These reflections blend together
/// - Creates a sense of space and depth
///
/// Key parameters:
/// - Room Size: How big the virtual room is
/// - Decay Time: How long reverb rings (small room = short, hall = long)
/// - Pre-delay: Time before reverb starts (sense of room distance)
/// - Damping: How much high frequencies are absorbed
/// - Wet/Dry: Balance between original and reverb signal
///
/// Types of reverb sounds:
/// - Room: Small, intimate (0.2-0.5s decay)
/// - Hall: Large concert hall (1-2s decay)
/// - Cathedral: Huge, ethereal (3-6s decay)
/// - Plate: Artificial, bright (classic 80s sound)
/// - Spring: Metallic, twangy (guitar amps)
///
/// This implementation uses:
/// - Allpass filters for diffusion (smears the early reflections)
/// - Comb filters for resonance (creates the decay tail)
/// - Low-pass filter for damping (natural high-frequency absorption)
/// </para>
/// </remarks>
public class Reverb<T> : AudioEffectBase<T>
{
    #region Configuration

    private double _roomSize;
    private double _decayTime;
    private double _preDelayMs;
    private double _damping;
    private double _diffusion;

    #endregion

    #region Delay Lines

    // Allpass filters for diffusion
    private readonly AllpassFilter[] _allpassFilters;
    private readonly int[] _allpassDelays = [556, 441, 341, 225];

    // Comb filters for resonance
    private readonly CombFilter[] _combFilters;
    private readonly int[] _combDelays = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];

    // Pre-delay buffer
    private double[] _preDelayBuffer;
    private int _preDelayIndex;
    private int _preDelaySamples;

    #endregion

    /// <inheritdoc/>
    public override string Name => "Reverb";

    /// <inheritdoc/>
    public override int TailSamples => (int)(_decayTime * SampleRate);

    /// <summary>
    /// Creates a reverb effect with room-style defaults.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate (default: 44100).</param>
    /// <param name="roomSize">Room size 0-1 (default: 0.5).</param>
    /// <param name="decayTime">Decay time in seconds (default: 1.5).</param>
    /// <param name="preDelayMs">Pre-delay in milliseconds (default: 20).</param>
    /// <param name="damping">High frequency damping 0-1 (default: 0.5).</param>
    /// <param name="diffusion">Diffusion amount 0-1 (default: 0.7).</param>
    /// <param name="mix">Dry/wet mix (default: 0.3).</param>
    public Reverb(
        int sampleRate = 44100,
        double roomSize = 0.5,
        double decayTime = 1.5,
        double preDelayMs = 20.0,
        double damping = 0.5,
        double diffusion = 0.7,
        double mix = 0.3)
        : base(sampleRate, mix)
    {
        // Initialize parameters
        AddParameter("roomSize", "Room Size", NumOps.FromDouble(0), NumOps.FromDouble(1), NumOps.FromDouble(roomSize), "", "Size of virtual room");
        AddParameter("decayTime", "Decay Time", NumOps.FromDouble(0.1), NumOps.FromDouble(10), NumOps.FromDouble(decayTime), "s", "Reverb decay time");
        AddParameter("preDelay", "Pre-Delay", NumOps.FromDouble(0), NumOps.FromDouble(100), NumOps.FromDouble(preDelayMs), "ms", "Initial delay before reverb");
        AddParameter("damping", "Damping", NumOps.FromDouble(0), NumOps.FromDouble(1), NumOps.FromDouble(damping), "", "High frequency absorption");
        AddParameter("diffusion", "Diffusion", NumOps.FromDouble(0), NumOps.FromDouble(1), NumOps.FromDouble(diffusion), "", "Reverb density");

        _roomSize = roomSize;
        _decayTime = decayTime;
        _preDelayMs = preDelayMs;
        _damping = damping;
        _diffusion = diffusion;

        // Scale delays for sample rate (reference is 44100 Hz)
        double sampleRateScale = sampleRate / 44100.0;

        // Initialize allpass filters
        _allpassFilters = new AllpassFilter[_allpassDelays.Length];
        for (int i = 0; i < _allpassDelays.Length; i++)
        {
            int delaySize = (int)(_allpassDelays[i] * sampleRateScale * (0.5 + roomSize * 0.5));
            _allpassFilters[i] = new AllpassFilter(delaySize, diffusion * 0.5);
        }

        // Initialize comb filters
        _combFilters = new CombFilter[_combDelays.Length];
        for (int i = 0; i < _combDelays.Length; i++)
        {
            int delaySize = (int)(_combDelays[i] * sampleRateScale * (0.5 + roomSize * 0.5));
            double feedback = CalculateCombFeedback(delaySize);
            _combFilters[i] = new CombFilter(delaySize, feedback, damping);
        }

        // Initialize pre-delay
        _preDelaySamples = (int)(preDelayMs * sampleRate / 1000.0);
        _preDelayBuffer = new double[Math.Max(1, _preDelaySamples)];
        _preDelayIndex = 0;
    }

    private double CalculateCombFeedback(int delaySamples)
    {
        // Calculate feedback to achieve desired decay time
        // RT60 = -3 * delayTime / log10(feedback)
        if (delaySamples <= 0) return 0;

        double delayTime = (double)delaySamples / SampleRate;
        double feedback = Math.Pow(10, -3 * delayTime / _decayTime);
        return Math.Min(0.98, feedback);
    }

    /// <inheritdoc/>
    protected override void OnParameterChanged(string name, T value)
    {
        double val = NumOps.ToDouble(value);
        switch (name)
        {
            case "roomSize": _roomSize = val; break;
            case "decayTime":
                _decayTime = val;
                // Recalculate comb filter feedback
                for (int i = 0; i < _combFilters.Length; i++)
                {
                    double feedback = CalculateCombFeedback(_combFilters[i].DelaySize);
                    _combFilters[i].Feedback = feedback;
                }
                break;
            case "preDelay":
                _preDelayMs = val;
                _preDelaySamples = (int)(val * SampleRate / 1000.0);
                _preDelayBuffer = new double[Math.Max(1, _preDelaySamples)];
                _preDelayIndex = 0;
                break;
            case "damping":
                _damping = val;
                foreach (var comb in _combFilters)
                    comb.Damping = val;
                break;
            case "diffusion":
                _diffusion = val;
                foreach (var allpass in _allpassFilters)
                    allpass.Coefficient = val * 0.5;
                break;
        }
    }

    /// <inheritdoc/>
    protected override T ProcessSampleInternal(T input)
    {
        double sample = NumOps.ToDouble(input);

        // Apply pre-delay
        double preDelayed = sample;
        if (_preDelaySamples > 0)
        {
            preDelayed = _preDelayBuffer[_preDelayIndex];
            _preDelayBuffer[_preDelayIndex] = sample;
            _preDelayIndex = (_preDelayIndex + 1) % _preDelayBuffer.Length;
        }

        // Process through comb filters in parallel
        double combSum = 0;
        foreach (var comb in _combFilters)
        {
            combSum += comb.Process(preDelayed);
        }
        combSum /= _combFilters.Length;

        // Process through allpass filters in series
        double output = combSum;
        foreach (var allpass in _allpassFilters)
        {
            output = allpass.Process(output);
        }

        return NumOps.FromDouble(output);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        foreach (var comb in _combFilters)
            comb.Reset();
        foreach (var allpass in _allpassFilters)
            allpass.Reset();

        Array.Clear(_preDelayBuffer, 0, _preDelayBuffer.Length);
        _preDelayIndex = 0;
    }

    #region Helper Classes

    private class CombFilter
    {
        private readonly double[] _buffer;
        private int _index;
        private double _filterStore;

        public int DelaySize => _buffer.Length;
        public double Feedback { get; set; }
        public double Damping { get; set; }

        public CombFilter(int delaySize, double feedback, double damping)
        {
            _buffer = new double[delaySize];
            Feedback = feedback;
            Damping = damping;
            _filterStore = 0;
        }

        public double Process(double input)
        {
            double output = _buffer[_index];

            // Low-pass filter for damping
            _filterStore = output * (1 - Damping) + _filterStore * Damping;

            // Feedback
            _buffer[_index] = input + _filterStore * Feedback;

            _index = (_index + 1) % _buffer.Length;
            return output;
        }

        public void Reset()
        {
            Array.Clear(_buffer, 0, _buffer.Length);
            _filterStore = 0;
            _index = 0;
        }
    }

    private class AllpassFilter
    {
        private readonly double[] _buffer;
        private int _index;

        public double Coefficient { get; set; }

        public AllpassFilter(int delaySize, double coefficient)
        {
            _buffer = new double[delaySize];
            Coefficient = coefficient;
        }

        public double Process(double input)
        {
            double delayed = _buffer[_index];
            double output = -input + delayed;

            _buffer[_index] = input + delayed * Coefficient;

            _index = (_index + 1) % _buffer.Length;
            return output;
        }

        public void Reset()
        {
            Array.Clear(_buffer, 0, _buffer.Length);
            _index = 0;
        }
    }

    #endregion
}
