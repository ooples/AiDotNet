using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the Zipformer speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// Zipformer (Yao et al., 2023, Next-gen Kaldi) is a more efficient variant of the Conformer
/// with temporal downsampling, BiasNorm instead of LayerNorm, and SwooshR/SwooshL activations.
/// It uses a U-Net-like structure with different time resolutions at different encoder stacks,
/// achieving better accuracy with fewer parameters than standard Conformer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Zipformer is an improved version of the Conformer that's both faster
/// and more accurate. It processes speech at different "zoom levels" - some parts look at fine
/// details (individual sounds) and other parts look at the bigger picture (whole words and phrases).
/// This makes it one of the most efficient speech encoders available.
/// </para>
/// </remarks>
public class ZipformerOptions : ModelOptions
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

    /// <summary>Gets or sets the encoder dimensions at each stack (U-Net style).</summary>
    public int[] EncoderDims { get; set; } = [192, 256, 384, 512, 384, 256];

    /// <summary>Gets or sets the number of layers per encoder stack.</summary>
    public int[] NumLayersPerStack { get; set; } = [2, 2, 3, 4, 3, 2];

    /// <summary>Gets or sets the attention heads per stack.</summary>
    public int[] NumHeadsPerStack { get; set; } = [4, 4, 4, 8, 4, 4];

    /// <summary>Gets or sets the downsampling factors per stack.</summary>
    public int[] DownsampleFactors { get; set; } = [1, 2, 4, 8, 4, 2];

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
    public double LearningRate { get; set; } = 4.5e-2;

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
