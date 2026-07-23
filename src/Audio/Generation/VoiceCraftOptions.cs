using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the VoiceCraft speech editing and generation model.
/// </summary>
/// <remarks>
/// <para>
/// VoiceCraft (Peng et al., 2024) is a neural codec language model for speech editing and
/// zero-shot TTS. It uses a token rearrangement procedure with causal masking that enables
/// both editing existing speech (replacing/inserting words) and generating new speech from
/// a short prompt, achieving high naturalness and speaker similarity.
/// </para>
/// <para>
/// <b>For Beginners:</b> VoiceCraft can do two amazing things: (1) edit speech like you
/// edit text - change specific words in a recording while keeping the speaker's voice, and
/// (2) clone a voice from a few seconds of audio and generate new speech. It's like having
/// a "find and replace" for spoken words, plus a voice cloning tool.
/// </para>
/// </remarks>
public class VoiceCraftOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the maximum generation duration in seconds.</summary>
    public double MaxDurationSeconds { get; set; } = 30.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model hidden dimension.</summary>
    public int HiddenDim { get; set; } = 2048;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumLayers { get; set; } = 16;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 16;

    /// <summary>Gets or sets the codec codebook size.</summary>
    public int CodebookSize { get; set; } = 2048;

    /// <summary>Gets or sets the number of codec quantizers.</summary>
    public int NumQuantizers { get; set; } = 4;

    /// <summary>Gets or sets the codec embedding dimension per quantizer.</summary>
    public int CodecEmbeddingDim { get; set; } = 128;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 80;

    #endregion

    #region Speech Editing

    /// <summary>Gets or sets the context window size in seconds for speech editing.</summary>
    public double EditContextSeconds { get; set; } = 3.0;

    /// <summary>Gets or sets the masking ratio for causal masking during editing.</summary>
    public double MaskRatio { get; set; } = 0.5;

    #endregion

    #region Generation

    /// <summary>Gets or sets the temperature for sampling.</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Gets or sets the codec frame rate (frames per second).</summary>
    public int CodecFrameRate { get; set; } = 75;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
