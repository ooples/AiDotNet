using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the Basic Pitch multi-pitch detection model.
/// </summary>
/// <remarks>
/// <para>
/// Basic Pitch (Bittner et al., 2022) from Spotify is a lightweight neural network for
/// polyphonic music transcription. It detects note onsets, offsets, and pitch for multiple
/// simultaneous notes, producing MIDI-like output from audio. Unlike monophonic pitch detectors
/// (CREPE), Basic Pitch handles chords and polyphonic music.
/// </para>
/// <para>
/// <b>For Beginners:</b> Basic Pitch turns audio of music into "sheet music" data. It can
/// detect multiple notes playing at the same time (like a piano chord), when each note starts
/// and stops, and what pitch each note is. The output is similar to MIDI - a list of
/// (start_time, end_time, pitch, velocity) for every detected note.
/// </para>
/// </remarks>
public class BasicPitchOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the number of harmonically-stacked CQT bins.
    /// </summary>
    public int NumHarmonicBins { get; set; } = 264;

    /// <summary>
    /// Gets or sets the number of frequency bins per octave.
    /// </summary>
    public int BinsPerOctave { get; set; } = 36;

    /// <summary>
    /// Gets or sets the hop length for the CQT in samples.
    /// </summary>
    public int HopLength { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of harmonics to stack.
    /// </summary>
    public int NumHarmonics { get; set; } = 8;

    #endregion

    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of MIDI notes to predict (88 piano keys by default, A0 to C8).
    /// </summary>
    public int NumMidiNotes { get; set; } = 88;

    /// <summary>
    /// Gets or sets the lowest MIDI note (A0 = 21).
    /// </summary>
    public int MidiOffset { get; set; } = 21;

    /// <summary>
    /// Gets or sets the number of convolutional filters in the encoder.
    /// </summary>
    public int EncoderFilters { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of convolutional layers in the encoder.
    /// </summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of output heads (note, onset, contour).
    /// </summary>
    public int NumOutputHeads { get; set; } = 3;

    #endregion

    #region Inference

    /// <summary>
    /// Gets or sets the onset detection threshold.
    /// </summary>
    public double OnsetThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the note (frame) detection threshold.
    /// </summary>
    public double NoteThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum note duration in seconds.
    /// </summary>
    public double MinNoteDurationSec { get; set; } = 0.0580;

    /// <summary>
    /// Gets or sets the minimum frequency in Hz for note detection.
    /// </summary>
    public double MinFrequencyHz { get; set; } = 27.5;

    /// <summary>
    /// Gets or sets the maximum frequency in Hz for note detection.
    /// </summary>
    public double MaxFrequencyHz { get; set; } = 4186.01;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
