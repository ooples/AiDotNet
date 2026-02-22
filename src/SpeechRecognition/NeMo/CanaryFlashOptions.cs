using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>Options for Canary-Flash (NVIDIA NeMo, 2025): lightweight multilingual ASR+translation with hybrid CTC/attention.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CanaryFlash model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class CanaryFlashOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CanaryFlashOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CanaryFlashOptions(CanaryFlashOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        SupportedLanguagesList = other.SupportedLanguagesList;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 1024;
    public int DecoderDim { get; set; } = 1024;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumDecoderLayers { get; set; } = 6;
    public int NumAttentionHeads { get; set; } = 16;
    public int NumMels { get; set; } = 128;
    public int VocabSize { get; set; } = 32128;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
    public string[] SupportedLanguagesList { get; set; } = new[] { "en", "de", "es", "fr", "it", "pt", "nl", "ja", "ko", "zh", "hi", "ar", "ru", "uk", "pl", "sv", "fi", "no", "da", "tr" };
}
