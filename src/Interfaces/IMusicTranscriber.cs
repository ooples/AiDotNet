using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for automatic music transcription (audio to notes).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Music transcription converts audio recordings into symbolic note representations (like MIDI).
/// It detects what notes are played, when they start and end, and optionally their velocity.
/// </para>
/// <para><b>For Beginners:</b> Music transcription is like having a computer "listen" to music
/// and write down the notes. The output is similar to sheet music data:
/// - Which note is playing (e.g., C4, A#3)
/// - When each note starts and stops
/// - How loud each note is (velocity)
///
/// This is used for:
/// - Converting recordings to MIDI files
/// - Music education (showing what notes are played)
/// - Music analysis and research
/// - Karaoke systems
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MusicTranscriber")]
public interface IMusicTranscriber<T>
{
    /// <summary>
    /// Gets the number of MIDI notes this transcriber can detect.
    /// </summary>
    int NumMidiNotes { get; }

    /// <summary>
    /// Gets the MIDI offset (lowest MIDI note number, e.g., 21 for A0).
    /// </summary>
    int MidiOffset { get; }

    /// <summary>
    /// Transcribes audio into a list of detected notes.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>List of detected notes with timing, pitch, and velocity information.</returns>
    IReadOnlyList<TranscribedNote<T>> Transcribe(Tensor<T> audio);

    /// <summary>
    /// Transcribes audio asynchronously.
    /// </summary>
    Task<IReadOnlyList<TranscribedNote<T>>> TranscribeAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets frame-level note activations (piano roll representation).
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of shape [num_frames, num_midi_notes] with activation probabilities.</returns>
    Tensor<T> GetFrameActivations(Tensor<T> audio);

    /// <summary>
    /// Gets frame-level onset activations.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of shape [num_frames, num_midi_notes] with onset probabilities.</returns>
    Tensor<T> GetOnsetActivations(Tensor<T> audio);

    /// <summary>
    /// Extracts notes from frame and onset activations using post-processing.
    /// </summary>
    /// <param name="frameActivations">Frame-level note activations.</param>
    /// <param name="onsetActivations">Frame-level onset activations.</param>
    /// <param name="frameThreshold">Threshold for note activity.</param>
    /// <param name="onsetThreshold">Threshold for onset detection.</param>
    /// <returns>List of transcribed notes.</returns>
    IReadOnlyList<TranscribedNote<T>> ExtractNotes(
        Tensor<T> frameActivations, Tensor<T> onsetActivations,
        double frameThreshold = 0.5, double onsetThreshold = 0.5);
}

/// <summary>
/// Represents a single transcribed musical note.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class TranscribedNote<T>
{
    /// <summary>
    /// Gets the start time in seconds.
    /// </summary>
    public required double StartTime { get; init; }

    /// <summary>
    /// Gets the end time in seconds.
    /// </summary>
    public required double EndTime { get; init; }

    /// <summary>
    /// Gets the MIDI note number (21-108 for piano).
    /// </summary>
    public required int MidiNote { get; init; }

    /// <summary>
    /// Gets the confidence/velocity of the note (0-1).
    /// </summary>
    public required T Confidence { get; init; }

    /// <summary>
    /// Gets the pitch in Hz.
    /// </summary>
    public double PitchHz => 440.0 * Math.Pow(2.0, (MidiNote - 69) / 12.0);

    /// <summary>
    /// Gets the note name (e.g., "C4", "A#3").
    /// </summary>
    public string NoteName
    {
        get
        {
            string[] names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
            int octave = (MidiNote / 12) - 1;
            int noteIndex = MidiNote % 12;
            return $"{names[noteIndex]}{octave}";
        }
    }

    /// <summary>
    /// Gets the duration of the note in seconds.
    /// </summary>
    public double Duration => EndTime - StartTime;
}
