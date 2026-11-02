using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>Options for Faster-Whisper (SYSTRAN/CTranslate2, 2023): CTranslate2-optimized Whisper with int8 quantization.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FasterWhisper model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class FasterWhisperOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FasterWhisperOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FasterWhisperOptions(FasterWhisperOptions other)
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
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        ComputeType = other.ComputeType;
        BeamSize = other.BeamSize;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 1280;
    /// <summary>Gets or sets the hidden width of the decoder.</summary>
    public int DecoderDim { get; set; } = 1280;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 32;
    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 32;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 20;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 51866;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 448;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.0;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Compute type for quantization (int8, float16, float32).</summary>
    public string ComputeType { get; set; } = "int8";
    /// <summary>Number of parallel transcription beams.</summary>
    public int BeamSize { get; set; } = 5;
}
