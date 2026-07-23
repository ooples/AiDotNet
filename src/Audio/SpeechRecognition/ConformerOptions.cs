using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the Conformer speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// The Conformer (Gulati et al., 2020, Google) combines convolution and self-attention to
/// capture both local and global audio dependencies. It achieves state-of-the-art on
/// LibriSpeech (WER 1.9%/3.9% test-clean/other with LM) and is now the dominant encoder
/// architecture for production ASR systems (used in Google, NVIDIA NeMo, etc.).
/// </para>
/// <para>
/// <b>For Beginners:</b> The Conformer is the go-to architecture for modern speech recognition.
/// It outperforms both pure Transformer and pure CNN encoders by using the best of both:
/// - Convolution captures local patterns like phonemes (individual sounds)
/// - Self-attention captures long-range context (e.g., sentence-level meaning)
/// The result is an encoder that understands both fine-grained and global speech patterns.
/// </para>
/// </remarks>
public class ConformerOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the maximum audio length in seconds.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "medium", "large").</summary>
    /// <remarks>
    /// <para>
    /// - "small": 16 layers, 256 dim (fast, good for edge/streaming)
    /// - "medium": 18 layers, 512 dim (balanced)
    /// - "large": 18 layers, 512 dim, 8 heads (best accuracy)
    /// </para>
    /// </remarks>
    public string Variant { get; set; } = "medium";

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of Conformer encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 18;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the feed-forward expansion factor.</summary>
    /// <remarks>
    /// The feed-forward dimension is <c>EncoderDim * FeedForwardExpansionFactor</c>.
    /// Conformer uses half-step residual (the feed-forward output is scaled by 0.5).
    /// </remarks>
    public int FeedForwardExpansionFactor { get; set; } = 4;

    /// <summary>Gets or sets the convolution kernel size in the Conformer block.</summary>
    /// <remarks>
    /// The depthwise convolution in the convolution module captures local context.
    /// Kernel size 31 covers roughly 310 ms of audio at the typical 10 ms frame stride,
    /// which spans a phoneme or short syllable.
    /// </remarks>
    public int ConvKernelSize { get; set; } = 31;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the subsampling factor for the encoder front-end.</summary>
    /// <remarks>
    /// Conv-subsampling typically uses stride-2 convolution twice to reduce the frame rate
    /// by 4x (from 10 ms to 40 ms). This reduces compute in the encoder layers.
    /// </remarks>
    public int SubsamplingFactor { get; set; } = 4;

    /// <summary>Gets or sets the vocabulary size for the CTC output head.</summary>
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

    /// <summary>Gets or sets the label smoothing factor for CTC loss.</summary>
    public double LabelSmoothing { get; set; } = 0.1;

    /// <summary>Gets or sets the warmup steps for the Noam learning-rate schedule.</summary>
    public int WarmupSteps { get; set; } = 25000;

    #endregion

    #region Decoding

    /// <summary>Gets or sets the CTC vocabulary (characters or BPE tokens).</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

    /// <summary>Gets or sets the default language code.</summary>
    public string Language { get; set; } = "en";

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
