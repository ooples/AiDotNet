using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the CosyVoice2 model.
/// </summary>
/// <remarks>
/// <para>
/// CosyVoice2 (Du et al., 2024, Alibaba) is a scalable streaming TTS model that achieves
/// natural-sounding speech with very low latency. It uses a finite scalar quantization (FSQ)
/// codec with a flow-matching decoder and supports zero-shot voice cloning, cross-lingual
/// synthesis, and emotion control.
/// </para>
/// <para>
/// <b>For Beginners:</b> CosyVoice2 converts text to natural-sounding speech that can
/// clone any voice from just a few seconds of reference audio. It's fast enough for
/// real-time applications like voice assistants and audiobooks.
/// </para>
/// </remarks>
public class CosyVoice2Options : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the text encoder dimension.</summary>
    public int TextEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of text encoder layers.</summary>
    public int NumTextEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the decoder dimension.</summary>
    public int DecoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of mel bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the speaker embedding dimension.</summary>
    public int SpeakerEmbeddingDim { get; set; } = 192;

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
