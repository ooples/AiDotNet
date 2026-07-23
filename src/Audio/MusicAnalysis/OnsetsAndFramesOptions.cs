using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the Onsets and Frames piano transcription model.
/// </summary>
/// <remarks>
/// <para>
/// Onsets and Frames (Hawthorne et al., 2018, Google Magenta) jointly predicts onsets and
/// frame-level note activations for automatic piano transcription. The model uses a CNN front-end
/// with bidirectional LSTMs, and was trained on the MAESTRO dataset. It achieves frame-level
/// note F1 of ~90% on piano recordings.
/// </para>
/// <para>
/// <b>For Beginners:</b> Onsets and Frames listens to piano music and writes down which notes
/// are being played and when. It detects two things: "onsets" (when a key is pressed) and
/// "frames" (which keys are held down at each moment). By combining both, it produces accurate
/// note-by-note transcriptions of piano recordings.
/// </para>
/// </remarks>
public class OnsetsAndFramesOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 229;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the minimum frequency for the mel filterbank.
    /// </summary>
    public double FMin { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the maximum frequency for the mel filterbank.
    /// </summary>
    public double FMax { get; set; } = 8000.0;

    #endregion

    #region Model Architecture

    /// <summary>
    /// Gets or sets the number of MIDI notes (88 piano keys: A0=21 to C8=108).
    /// </summary>
    public int NumMidiNotes { get; set; } = 88;

    /// <summary>
    /// Gets or sets the lowest MIDI note (A0 = 21).
    /// </summary>
    public int MidiOffset { get; set; } = 21;

    /// <summary>
    /// Gets or sets the acoustic model CNN feature dimension.
    /// </summary>
    public int AcousticModelDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the bidirectional LSTM hidden size.
    /// </summary>
    public int LstmHiddenSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of LSTM layers.
    /// </summary>
    public int NumLstmLayers { get; set; } = 2;

    #endregion

    #region Inference

    /// <summary>
    /// Gets or sets the onset detection threshold.
    /// </summary>
    public double OnsetThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the frame (note held) detection threshold.
    /// </summary>
    public double FrameThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum note duration in seconds.
    /// </summary>
    public double MinNoteDurationSec { get; set; } = 0.0;

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
    public double LearningRate { get; set; } = 6e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.2;

    #endregion
}
