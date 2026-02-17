using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the Canary model.
/// </summary>
/// <remarks>
/// <para>
/// Canary (NVIDIA, 2024) is a multilingual speech recognition model based on the Fast Conformer
/// encoder with a multi-task decoder. It supports automatic speech recognition (ASR), speech
/// translation (ST), and language identification across many languages, using a single unified
/// architecture with task-specific prompting.
/// </para>
/// <para>
/// <b>For Beginners:</b> Canary is like a multilingual transcription assistant. It can listen
/// to speech in many languages and either transcribe it (write down what was said) or translate
/// it into another language, all with a single model.
/// </para>
/// </remarks>
public class CanaryOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "1b";

    /// <summary>Gets or sets the encoder dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 24;

    /// <summary>Gets or sets the decoder dimension.</summary>
    public int DecoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the subsampling factor.</summary>
    public int SubsamplingFactor { get; set; } = 8;

    /// <summary>Gets or sets the vocabulary size.</summary>
    public int VocabSize { get; set; } = 32128;

    #endregion

    #region Recognition

    /// <summary>Gets or sets the beam width for decoding.</summary>
    public int BeamWidth { get; set; } = 5;

    /// <summary>Gets or sets the maximum output tokens.</summary>
    public int MaxOutputTokens { get; set; } = 512;

    /// <summary>Gets or sets the supported languages.</summary>
    public string[] SupportedLanguages { get; set; } = ["en", "de", "es", "fr"];

    /// <summary>Gets or sets the target language for translation.</summary>
    public string TargetLanguage { get; set; } = "en";

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
