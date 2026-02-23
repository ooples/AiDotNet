using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the StyleTTS 2 text-to-speech model.
/// </summary>
/// <remarks>
/// <para>
/// StyleTTS 2 (Li et al., 2023) uses diffusion models for style transfer and achieves
/// human-level naturalness on single-speaker synthesis (MOS 4.16 on LJSpeech). It
/// disentangles speech into content and style, allowing fine-grained control over
/// prosody, speaking rate, and emotion through style vectors.
/// </para>
/// <para>
/// <b>For Beginners:</b> StyleTTS 2 is one of the most natural-sounding TTS models.
/// It works by separating what is said (content) from how it is said (style). You can
/// change the speaking style by providing a reference audio clip, or let the model
/// generate a natural style automatically. It uses a diffusion model (similar to image
/// generators like DALL-E) to create realistic-sounding prosody.
/// </para>
/// </remarks>
public class StyleTTS2Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the hop length for spectrogram computation.</summary>
    public int HopLength { get; set; } = 256;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the text encoder hidden dimension.</summary>
    public int TextEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of text encoder layers.</summary>
    public int NumTextEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the style encoder dimension.</summary>
    public int StyleDim { get; set; } = 128;

    /// <summary>Gets or sets the prosody predictor hidden dimension.</summary>
    public int ProsodyDim { get; set; } = 512;

    /// <summary>Gets or sets the number of diffusion steps for style generation.</summary>
    /// <remarks>
    /// More steps produce more natural prosody but are slower.
    /// - 5-10: Fast inference with good quality
    /// - 20-50: High quality for offline synthesis
    /// </remarks>
    public int NumDiffusionSteps { get; set; } = 10;

    /// <summary>Gets or sets the speaker embedding dimension for multi-speaker models.</summary>
    public int SpeakerEmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets whether the model is multi-speaker.</summary>
    public bool IsMultiSpeaker { get; set; }

    /// <summary>Gets or sets the number of attention heads in the text encoder.</summary>
    public int NumAttentionHeads { get; set; } = 8;

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

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
