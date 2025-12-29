using AiDotNet.Interfaces;

namespace AiDotNet.Audio.Pitch;

/// <summary>
/// Base class for pitch detection implementations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class PitchDetectorBase<T> : IPitchDetector<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    #region IPitchDetector Properties

    /// <inheritdoc/>
    public int SampleRate { get; protected set; }

    /// <inheritdoc/>
    public double MinPitch { get; set; }

    /// <inheritdoc/>
    public double MaxPitch { get; set; }

    #endregion

    #region Constants

    /// <summary>
    /// Reference frequency for A4 (440 Hz standard).
    /// </summary>
    protected const double A4Frequency = 440.0;

    /// <summary>
    /// MIDI note number for A4.
    /// </summary>
    protected const int A4MidiNote = 69;

    /// <summary>
    /// Note names for display.
    /// </summary>
    protected static readonly string[] NoteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    #endregion

    /// <summary>
    /// Initializes a new PitchDetectorBase.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate.</param>
    /// <param name="minPitch">Minimum detectable pitch in Hz.</param>
    /// <param name="maxPitch">Maximum detectable pitch in Hz.</param>
    protected PitchDetectorBase(int sampleRate = 44100, double minPitch = 50, double maxPitch = 2000)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        SampleRate = sampleRate;
        MinPitch = minPitch;
        MaxPitch = maxPitch;
    }

    #region Abstract Methods

    /// <summary>
    /// Detects pitch from audio frame data.
    /// </summary>
    /// <param name="frame">Audio frame samples.</param>
    /// <returns>Pitch in Hz and confidence, or null if unvoiced.</returns>
    protected abstract (double Pitch, double Confidence)? DetectPitchInternal(double[] frame);

    #endregion

    #region IPitchDetector Implementation

    /// <inheritdoc/>
    public virtual (bool HasPitch, T Pitch) DetectPitch(Tensor<T> audioFrame)
    {
        var result = DetectPitchWithConfidence(audioFrame);
        if (result.HasValue)
        {
            return (true, result.Value.Pitch);
        }
        return (false, NumOps.Zero);
    }

    /// <inheritdoc/>
    public virtual (T Pitch, T Confidence)? DetectPitchWithConfidence(Tensor<T> audioFrame)
    {
        var samples = audioFrame.ToVector().ToArray();
        var doubleSamples = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            doubleSamples[i] = NumOps.ToDouble(samples[i]);
        }

        var result = DetectPitchInternal(doubleSamples);
        if (result.HasValue)
        {
            return (NumOps.FromDouble(result.Value.Pitch), NumOps.FromDouble(result.Value.Confidence));
        }
        return null;
    }

    /// <inheritdoc/>
    public virtual T[] ExtractPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        var detailed = ExtractDetailedPitchContour(audio, hopSizeMs);
        var pitches = new T[detailed.Count];

        for (int i = 0; i < detailed.Count; i++)
        {
            pitches[i] = detailed[i].IsVoiced
                ? detailed[i].Pitch
                : NumOps.FromDouble(0);
        }

        return pitches;
    }

    /// <inheritdoc/>
    public virtual IReadOnlyList<PitchFrame<T>> ExtractDetailedPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        var samples = audio.ToVector().ToArray();
        var results = new List<PitchFrame<T>>();

        int hopSamples = hopSizeMs * SampleRate / 1000;
        int frameSamples = Math.Max(hopSamples * 2, (int)(SampleRate / MinPitch) * 2);

        for (int i = 0; i + frameSamples <= samples.Length; i += hopSamples)
        {
            var frame = new double[frameSamples];
            for (int j = 0; j < frameSamples; j++)
            {
                frame[j] = NumOps.ToDouble(samples[i + j]);
            }

            var result = DetectPitchInternal(frame);
            double time = (double)i / SampleRate;

            if (result.HasValue)
            {
                results.Add(new PitchFrame<T>
                {
                    Time = time,
                    Pitch = NumOps.FromDouble(result.Value.Pitch),
                    Confidence = NumOps.FromDouble(result.Value.Confidence),
                    IsVoiced = true
                });
            }
            else
            {
                results.Add(new PitchFrame<T>
                {
                    Time = time,
                    Pitch = NumOps.FromDouble(0),
                    Confidence = NumOps.FromDouble(0),
                    IsVoiced = false
                });
            }
        }

        return results;
    }

    /// <inheritdoc/>
    public virtual double PitchToMidi(T pitchHz)
    {
        double freq = NumOps.ToDouble(pitchHz);
        if (freq <= 0) return 0;

        // MIDI note = 69 + 12 * log2(freq / 440)
        // Using Math.Log(x) / Math.Log(2) for net471 compatibility
        return A4MidiNote + 12 * (Math.Log(freq / A4Frequency) / Math.Log(2));
    }

    /// <inheritdoc/>
    public virtual T MidiToPitch(double midiNote)
    {
        // freq = 440 * 2^((midi - 69) / 12)
        double freq = A4Frequency * Math.Pow(2, (midiNote - A4MidiNote) / 12.0);
        return NumOps.FromDouble(freq);
    }

    /// <inheritdoc/>
    public virtual string PitchToNoteName(T pitchHz)
    {
        double midi = PitchToMidi(pitchHz);
        if (midi <= 0) return "---";

        int midiRounded = (int)Math.Round(midi);
        int noteIndex = midiRounded % 12;
        int octave = (midiRounded / 12) - 1;

        return $"{NoteNames[noteIndex]}{octave}";
    }

    /// <inheritdoc/>
    public virtual double GetCentsDeviation(T pitchHz)
    {
        double midi = PitchToMidi(pitchHz);
        double midiRounded = Math.Round(midi);
        return (midi - midiRounded) * 100;
    }

    #endregion
}
