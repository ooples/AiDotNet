using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for pitch (fundamental frequency) detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Pitch detection finds the fundamental frequency (F0) of periodic signals.
/// This is essential for music analysis and speech processing.
/// </para>
/// <para><b>For Beginners:</b> Pitch is what makes a note sound "high" or "low".
///
/// Technical definition:
/// - Pitch = perceived frequency of a sound
/// - F0 (fundamental frequency) = the lowest frequency component
/// - Measured in Hz (cycles per second)
///
/// Human voice pitch ranges:
/// - Bass: 80-300 Hz
/// - Baritone: 100-400 Hz
/// - Tenor: 130-500 Hz
/// - Alto: 175-700 Hz
/// - Soprano: 250-1000 Hz
///
/// Applications:
/// - Auto-tune / pitch correction (T-Pain effect)
/// - Music transcription (audio to sheet music)
/// - Karaoke scoring
/// - Speech therapy (monitoring pitch for dysphonia)
/// - Voice training for singing or public speaking
/// - Lie detection (pitch changes under stress)
///
/// Common algorithms:
/// - YIN: Fast, accurate for monophonic audio
/// - PYIN: Probabilistic YIN (handles uncertainty)
/// - CREPE: Neural network approach (most accurate)
/// - Autocorrelation: Classic signal processing method
/// </para>
/// </remarks>
public interface IPitchDetector<T>
{
    /// <summary>
    /// Gets the sample rate this detector operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets or sets the minimum detectable pitch in Hz.
    /// </summary>
    double MinPitch { get; set; }

    /// <summary>
    /// Gets or sets the maximum detectable pitch in Hz.
    /// </summary>
    double MaxPitch { get; set; }

    /// <summary>
    /// Detects the pitch of an audio frame.
    /// </summary>
    /// <param name="audioFrame">Audio frame to analyze.</param>
    /// <returns>Result with detected pitch in Hz, or HasPitch=false if no pitch detected (silence/noise).</returns>
    (bool HasPitch, T Pitch) DetectPitch(Tensor<T> audioFrame);

    /// <summary>
    /// Detects pitch with confidence score.
    /// </summary>
    /// <param name="audioFrame">Audio frame to analyze.</param>
    /// <returns>Pitch in Hz and confidence (0-1), or null if unvoiced.</returns>
    (T Pitch, T Confidence)? DetectPitchWithConfidence(Tensor<T> audioFrame);

    /// <summary>
    /// Extracts pitch contour from audio (F0 over time).
    /// </summary>
    /// <param name="audio">Full audio recording.</param>
    /// <param name="hopSizeMs">Time step between pitch estimates in milliseconds.</param>
    /// <returns>Array of pitch values (Hz), with 0 or NaN for unvoiced frames.</returns>
    T[] ExtractPitchContour(Tensor<T> audio, int hopSizeMs = 10);

    /// <summary>
    /// Extracts pitch contour with voicing information.
    /// </summary>
    /// <param name="audio">Full audio recording.</param>
    /// <param name="hopSizeMs">Time step in milliseconds.</param>
    /// <returns>Array of (pitch, confidence, isVoiced) tuples.</returns>
    IReadOnlyList<PitchFrame<T>> ExtractDetailedPitchContour(Tensor<T> audio, int hopSizeMs = 10);

    /// <summary>
    /// Converts pitch in Hz to MIDI note number.
    /// </summary>
    /// <param name="pitchHz">Pitch in Hz.</param>
    /// <returns>MIDI note number (69 = A4 = 440 Hz).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> MIDI notes are numbered 0-127.
    /// Middle C is 60, A4 (440 Hz) is 69.
    /// Each note is one semitone apart.
    /// </para>
    /// </remarks>
    double PitchToMidi(T pitchHz);

    /// <summary>
    /// Converts MIDI note number to pitch in Hz.
    /// </summary>
    /// <param name="midiNote">MIDI note number.</param>
    /// <returns>Pitch in Hz.</returns>
    T MidiToPitch(double midiNote);

    /// <summary>
    /// Gets the note name for a pitch.
    /// </summary>
    /// <param name="pitchHz">Pitch in Hz.</param>
    /// <returns>Note name like "A4", "C#5", "Bb3".</returns>
    string PitchToNoteName(T pitchHz);

    /// <summary>
    /// Calculates cents deviation from nearest note.
    /// </summary>
    /// <param name="pitchHz">Pitch in Hz.</param>
    /// <returns>Cents deviation (-50 to +50, where 100 cents = 1 semitone).</returns>
    /// <remarks>
    /// Used for tuning. 0 cents = perfectly in tune.
    /// Positive = sharp, Negative = flat.
    /// </remarks>
    double GetCentsDeviation(T pitchHz);
}

/// <summary>
/// Represents a single pitch detection frame.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class PitchFrame<T>
{
    /// <summary>
    /// Gets the timestamp in seconds.
    /// </summary>
    public required double Time { get; init; }

    /// <summary>
    /// Gets the detected pitch in Hz.
    /// </summary>
    public required T Pitch { get; init; }

    /// <summary>
    /// Gets the detection confidence (0-1).
    /// </summary>
    public required T Confidence { get; init; }

    /// <summary>
    /// Gets whether this frame is voiced (contains pitched content).
    /// </summary>
    public required bool IsVoiced { get; init; }
}
