using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>Options for OLMoASR: Whisper's encoder-decoder trained on the open OLMoASR-Mix.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released OLMoASR-large.en-v2 (1.5B parameters): Whisper large's geometry
/// (80-band log-mels over 30-second windows, width 1,280, 20 heads, 32 encoder and 32 decoder layers) with the
/// English-only models' 51,864-token GPT-2 vocabulary. The smaller releases are tiny.en (384, 6 heads, 4 + 4 layers),
/// base.en (512, 8, 6 + 6), small.en (768, 12, 12 + 12) and medium.en (1,024, 16, 24 + 24).</para>
/// </remarks>
public class OLMoASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OLMoASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OLMoASROptions(OLMoASROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        NumMels = other.NumMels;
        ModelDim = other.ModelDim;
        NumAttentionHeads = other.NumAttentionHeads;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Log-mel bands (80, as every English-only Whisper geometry).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>The encoder's and decoder's width (1,280 for large).</summary>
    public int ModelDim { get; set; } = 1280;
    /// <summary>Attention heads in the encoder and the decoder (20 for large).</summary>
    public int NumAttentionHeads { get; set; } = 20;
    /// <summary>Encoder layers (32 for large).</summary>
    public int NumEncoderLayers { get; set; } = 32;
    /// <summary>Decoder layers (32 for large).</summary>
    public int NumDecoderLayers { get; set; } = 32;
    /// <summary>The vocabulary (51,864, the English-only models' GPT-2 encoding with Whisper's special tokens). A smaller
    /// vocabulary has its own special tokens at its top: end of text, start of transcript and no timestamps.</summary>
    public int VocabSize { get; set; } = 51864;
    /// <summary>The longest transcript generated, in tokens (224, half the decoder's 448-token context, as Whisper's
    /// <c>sample_len</c>).</summary>
    public int MaxTextLength { get; set; } = 224;
    /// <summary>Gets or sets the path to the model weights.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"en"</c>.</summary>
    public string Language { get; set; } = "en";
    /// <summary>Optional vocabulary mapping token ids to text when the vocabulary is not Whisper's English one.</summary>
    public string[]? Vocabulary { get; set; }
}
