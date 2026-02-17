using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the Fish Speech TTS model.
/// </summary>
/// <remarks>
/// <para>
/// Fish Speech (Fish Audio, 2024) is an open-source multilingual TTS system that uses a
/// dual-AR architecture with grouped finite scalar quantization (GFSQ). It supports zero-shot
/// voice cloning from a few seconds of reference audio and generates natural-sounding speech
/// in multiple languages with very low latency suitable for real-time streaming.
/// </para>
/// <para>
/// <b>For Beginners:</b> Fish Speech is a fast, open-source text-to-speech system. Give it
/// a few seconds of someone's voice and some text, and it will speak the text in that person's
/// voice. It works in many languages and is fast enough for live conversations. Think of it
/// as an open-source alternative to commercial voice cloning services.
/// </para>
/// </remarks>
public class FishSpeechOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the maximum generation duration in seconds.</summary>
    public double MaxDurationSeconds { get; set; } = 60.0;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the semantic language model dimension.</summary>
    public int SemanticDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of semantic transformer layers.</summary>
    public int NumSemanticLayers { get; set; } = 24;

    /// <summary>Gets or sets the number of semantic attention heads.</summary>
    public int NumSemanticHeads { get; set; } = 16;

    /// <summary>Gets or sets the VQGAN vocoder hidden dimension.</summary>
    public int VocoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of VQGAN vocoder layers.</summary>
    public int NumVocoderLayers { get; set; } = 8;

    /// <summary>Gets or sets the GFSQ codebook size.</summary>
    public int CodebookSize { get; set; } = 8192;

    /// <summary>Gets or sets the number of GFSQ groups.</summary>
    public int NumGroups { get; set; } = 8;

    /// <summary>Gets or sets the text token vocabulary size.</summary>
    public int TextVocabSize { get; set; } = 32000;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 128;

    #endregion

    #region Generation

    /// <summary>Gets or sets the temperature for sampling.</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Gets or sets the top-p (nucleus) sampling parameter.</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>Gets or sets the repetition penalty factor.</summary>
    public double RepetitionPenalty { get; set; } = 1.2;

    /// <summary>Gets or sets the minimum reference audio duration in seconds for voice cloning.</summary>
    public double MinReferenceSeconds { get; set; } = 3.0;

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
