using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the Fast Conformer speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// Fast Conformer (Rekesh et al., 2023, NVIDIA NeMo) is an optimized Conformer variant
/// with 8x depthwise-separable convolution downsampling in the front-end, reducing the
/// sequence length early and enabling efficient processing of long audio. Combined with
/// multi-blank CTC or RNN-T, it achieves 2.4x speedup over standard Conformer with no
/// accuracy loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> Fast Conformer is NVIDIA's speed-optimized version of the Conformer.
/// It compresses audio early on (8x downsampling) so the expensive transformer layers process
/// much shorter sequences. Think of it as reading a summary instead of the full book - same
/// information, much faster processing.
/// </para>
/// </remarks>
public class FastConformerOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the language code.</summary>
    public string Language { get; set; } = "en";

    #endregion

    #region Architecture

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 17;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the convolution kernel size.</summary>
    public int ConvKernelSize { get; set; } = 9;

    /// <summary>Gets or sets the front-end downsampling factor.</summary>
    public int DownsampleFactor { get; set; } = 8;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>Gets or sets the vocabulary size.</summary>
    public int VocabSize { get; set; } = 5000;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Decoding

    /// <summary>Gets or sets the model variant ("small", "medium", "large").</summary>
    public string Variant { get; set; } = "medium";

    /// <summary>Gets or sets the CTC vocabulary (characters or BPE tokens).</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

    private static string[] GetDefaultVocabulary()
    {
        return new[]
        {
            "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z", "'", " "
        };
    }

    #endregion
}
