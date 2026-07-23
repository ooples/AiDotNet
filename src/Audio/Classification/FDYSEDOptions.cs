using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the FDY-SED (Frequency Dynamic Sound Event Detection) model.
/// </summary>
/// <remarks>
/// <para>
/// FDY-SED (Nam et al., ICASSP 2022) introduces frequency-dynamic convolutions for sound event
/// detection, where convolution kernels are dynamically generated based on the frequency band being
/// processed. This allows the model to apply different processing strategies to different frequency
/// ranges (e.g., bass frequencies need different patterns than treble). FDY-SED achieves DCASE
/// challenge-winning results on the DESED dataset for domestic sound event detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> FDY-SED is like having a team of specialized listeners, where each
/// listener is an expert at hearing different pitch ranges:
///
/// - One listener is good at hearing low rumbles (bass) like washing machines or traffic
/// - Another is good at mid-range sounds like speech or dog barks
/// - Another specializes in high-pitched sounds like bird chirps or alarms
///
/// Instead of using the same "ear" for all frequencies (like a standard CNN), FDY-SED adapts
/// its filters based on which frequency range it's analyzing. This makes it more accurate at
/// detecting sounds that have very different spectral characteristics.
/// </para>
/// </remarks>
public class FDYSEDOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between FFT frames.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the minimum frequency for mel filterbank.</summary>
    public int FMin { get; set; } = 0;

    /// <summary>Gets or sets the maximum frequency for mel filterbank.</summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region FDY Architecture

    /// <summary>Gets or sets the model variant ("small", "base", "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the number of CNN channels per block.</summary>
    public int[] CNNChannels { get; set; } = [16, 32, 64, 128, 256];

    /// <summary>Gets or sets the embedding dimension after CNN blocks.</summary>
    public int EmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the number of frequency dynamic convolution groups.</summary>
    /// <remarks>
    /// <para>The frequency axis is divided into this many groups, each getting its own
    /// dynamically generated convolution kernel.</para>
    /// </remarks>
    public int NumFrequencyGroups { get; set; } = 4;

    /// <summary>Gets or sets the frequency dynamic kernel generation hidden size.</summary>
    public int FDYKernelHiddenSize { get; set; } = 64;

    /// <summary>Gets or sets the RNN hidden size for temporal modeling.</summary>
    public int RNNHiddenSize { get; set; } = 128;

    /// <summary>Gets or sets the number of RNN layers.</summary>
    public int NumRNNLayers { get; set; } = 2;

    #endregion

    #region Detection

    /// <summary>Gets or sets the confidence threshold for event detection.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Gets or sets the window size in seconds for detection.</summary>
    public double DetectionWindowSize { get; set; } = 10.0;

    /// <summary>Gets or sets the window overlap ratio (0-1).</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Gets or sets custom event labels. If null, uses AudioSet-527 labels.</summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to a pre-trained ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>Gets or sets the label smoothing factor.</summary>
    public double LabelSmoothing { get; set; } = 0.0;

    /// <summary>Gets or sets the mean teacher consistency weight for semi-supervised training.</summary>
    /// <remarks>
    /// <para>FDY-SED uses mean teacher training for semi-supervised learning on DESED,
    /// where only a fraction of clips have strong annotations.</para>
    /// </remarks>
    public double MeanTeacherWeight { get; set; } = 2.0;

    #endregion
}
