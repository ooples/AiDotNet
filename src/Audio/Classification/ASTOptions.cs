using AiDotNet.Audio.Features;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the AST (Audio Spectrogram Transformer) models, <see cref="AST{T}"/> and
/// <see cref="Fingerprinting.ASTModel{T}"/>, which run one network.
/// </summary>
/// <remarks>
/// <para>
/// AST (Gong, Chung and Glass, Interspeech 2021) applies a Vision Transformer to audio. The defaults are the paper's
/// AudioSet configuration: 128-bin Kaldi filterbanks of 10-second clips (1024 frames), 16×16 patches with an overlap of
/// 6, the 768-wide 12-layer DeiT body with the class and distillation tokens averaged, inputs normalized to mean 0 and
/// standard deviation 0.5, and training with mixup and SpecAugment.
/// </para>
/// <para>
/// <b>For Beginners:</b> AST treats an audio spectrogram as an image. It cuts the spectrogram into overlapping
/// patches, then uses a Transformer to relate them, and scores every sound label independently.
///
/// Example usage:
/// <code>
/// var ast = new AST&lt;float&gt;(new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 160000, outputSize: 527));
/// var result = ast.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)</item>
/// <item>Repository: https://github.com/YuanGongND/ast</item>
/// </list>
/// </para>
/// </remarks>
public class ASTOptions : ModelOptions
{
    public ASTOptions()
    {
    }

    public ASTOptions(ASTOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        NumMels = other.NumMels;
        TargetLength = other.TargetLength;
        Window = other.Window;
        DatasetMean = other.DatasetMean;
        DatasetStd = other.DatasetStd;
        PatchSize = other.PatchSize;
        FrequencyStride = other.FrequencyStride;
        TimeStride = other.TimeStride;
        EmbeddingDim = other.EmbeddingDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        MixupRate = other.MixupRate;
        FrequencyMaskLength = other.FrequencyMaskLength;
        TimeMaskLength = other.TimeMaskLength;
        Threshold = other.Threshold;
        WindowSize = other.WindowSize;
        WindowOverlap = other.WindowOverlap;
        CustomLabels = other.CustomLabels?.ToArray();
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
    }

    #region Audio Front End

    /// <summary>Gets or sets the audio sample rate in Hz (16 kHz in the paper).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel bins of the Kaldi filterbank (128, §2.1).</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the number of filterbank frames the network takes (1024: 10 s at the 10 ms frame shift).
    /// Shorter clips are zero-padded at the end and longer ones cut, as the official data loader does.</summary>
    public int TargetLength { get; set; } = 1024;

    /// <summary>Gets or sets the filterbank's analysis window. The paper states a 25 ms Hamming window (§2.1); the
    /// official code and its released checkpoints use Hanning (<see cref="WindowType.Hann"/>).</summary>
    public WindowType Window { get; set; } = WindowType.Hamming;

    /// <summary>Gets or sets the dataset filterbank mean the input is normalized by (AudioSet's, from the official
    /// recipe). With <see cref="DatasetStd"/> the input becomes (x − mean) / (2·std): mean 0, standard deviation 0.5
    /// (§2.2).</summary>
    public double DatasetMean { get; set; } = -4.2677393;

    /// <summary>Gets or sets the dataset filterbank standard deviation (AudioSet's).</summary>
    public double DatasetStd { get; set; } = 4.5689974;

    #endregion

    #region Patch Embedding

    /// <summary>Gets or sets the square patch size (16).</summary>
    public int PatchSize { get; set; } = 16;

    /// <summary>Gets or sets the patch stride along frequency (10: an overlap of 6, §2.1).</summary>
    public int FrequencyStride { get; set; } = 10;

    /// <summary>Gets or sets the patch stride along time (10).</summary>
    public int TimeStride { get; set; } = 10;

    #endregion

    #region Transformer Architecture

    /// <summary>Gets or sets the Transformer width (768).</summary>
    public int EmbeddingDim { get; set; } = 768;

    /// <summary>Gets or sets the number of Transformer blocks (12).</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads (12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Gets or sets the MLP width of each block (3072, four times the width).</summary>
    public int FeedForwardDim { get; set; } = 3072;

    #endregion

    #region Training Augmentation

    /// <summary>Gets or sets the probability that a training clip is mixed with another (0.5, §3.1.1). The two
    /// waveforms are mixed with λ ~ Beta(10, 10) and so are their labels, as the official data loader does; the
    /// partner is drawn from the training batch.</summary>
    public double MixupRate { get; set; } = 0.5;

    /// <summary>Gets or sets the maximum frequency-mask length in mel bins (48, §3.1.1).</summary>
    public int FrequencyMaskLength { get; set; } = 48;

    /// <summary>Gets or sets the maximum time-mask length in frames (192, §3.1.1).</summary>
    public int TimeMaskLength { get; set; } = 192;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold for event detection.
    /// </summary>
    /// <remarks>
    /// <para>Events with confidence below this threshold are not reported.
    /// AST uses sigmoid activation for multi-label AudioSet classification.</para>
    /// <para><b>For Beginners:</b> How confident the model must be to report a sound.</para>
    /// </remarks>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds for event detection.
    /// </summary>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio (0-1) for event detection.
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels. If null, uses AudioSet-527 labels.
    /// </summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have a pre-trained model file, specify its path here
    /// for fast inference without training.</para>
    /// </remarks>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
