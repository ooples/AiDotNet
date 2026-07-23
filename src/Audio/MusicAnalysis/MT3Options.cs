using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the MT3 (Multi-Track Music Transcription) model.
/// </summary>
/// <remarks>
/// <para>
/// MT3 (Gardner et al., 2022, Google) is a Transformer-based model that transcribes polyphonic
/// audio into MIDI across multiple instruments simultaneously. It uses a T5-style encoder-decoder
/// architecture with spectrogram input and tokenized MIDI output.
/// </para>
/// <para>
/// <b>For Beginners:</b> MT3 listens to a full song with multiple instruments and writes out
/// the sheet music (as MIDI) for each instrument separately. It can tell which notes the piano
/// is playing while also transcribing the guitar, drums, and bass at the same time.
/// </para>
/// </remarks>
public class MT3Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 512;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 128;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 8;

    /// <summary>Gets or sets the decoder hidden dimension.</summary>
    public int DecoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the maximum number of instruments.</summary>
    public int MaxInstruments { get; set; } = 128;

    /// <summary>Gets or sets the MIDI vocabulary size (tokens).</summary>
    public int VocabSize { get; set; } = 6000;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
