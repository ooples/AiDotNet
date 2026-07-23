using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the VALL-E zero-shot TTS model.
/// </summary>
/// <remarks>
/// <para>
/// VALL-E (Wang et al., 2023, Microsoft) treats text-to-speech as a language modeling problem
/// using discrete audio codes from EnCodec. A 3-second enrollment recording is enough to
/// synthesize speech in the speaker's voice. It uses an autoregressive (AR) model for the
/// first codebook layer and a non-autoregressive (NAR) model for remaining layers.
/// </para>
/// <para>
/// <b>For Beginners:</b> VALL-E can hear someone speak for 3 seconds and then generate
/// new speech that sounds just like them. It works by converting speech into "audio words"
/// (codec tokens) and then using a language model - the same kind of AI behind ChatGPT -
/// to predict what tokens come next, given a text prompt and the speaker's voice sample.
/// </para>
/// </remarks>
public class VALLEOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the maximum generation duration in seconds.</summary>
    public double MaxDurationSeconds { get; set; } = 30.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the AR model hidden dimension.</summary>
    public int ARHiddenDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of AR transformer layers.</summary>
    public int NumARLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of AR attention heads.</summary>
    public int NumARHeads { get; set; } = 16;

    /// <summary>Gets or sets the NAR model hidden dimension.</summary>
    public int NARHiddenDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of NAR transformer layers.</summary>
    public int NumNARLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of NAR attention heads.</summary>
    public int NumNARHeads { get; set; } = 16;

    /// <summary>Gets or sets the phoneme vocabulary size.</summary>
    public int PhonemeVocabSize { get; set; } = 512;

    /// <summary>Gets or sets the codec codebook size (from EnCodec).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the number of codec quantizer layers (8 for EnCodec).</summary>
    public int NumCodebooks { get; set; } = 8;

    #endregion

    #region Generation

    /// <summary>Gets or sets the temperature for AR sampling.</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Gets or sets the minimum enrollment audio duration in seconds.</summary>
    public double MinEnrollmentSeconds { get; set; } = 3.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
