using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the AudioSep (Audio Separation with Natural Language Queries) model.
/// </summary>
/// <remarks>
/// <para>
/// AudioSep (Liu et al., ICML 2024) is a foundation model for open-vocabulary audio separation
/// and sound event detection. Unlike traditional SED models that use fixed label sets, AudioSep
/// can detect and separate any sound described by natural language. It uses CLAP (Contrastive
/// Language-Audio Pretraining) embeddings to condition a separation network, enabling queries like
/// "separate the dog barking from the traffic noise" or "detect the sound of glass breaking."
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioSep is like having a smart audio assistant that understands
/// natural language. Instead of being limited to a fixed set of 527 sound categories:
///
/// - You can ask: "Find the sound of a baby crying" and it will detect/separate it
/// - You can ask: "Extract the bird chirping" from a noisy recording
/// - You can describe any sound in your own words, and AudioSep understands
///
/// This is possible because AudioSep combines two powerful ideas:
/// 1. <b>CLAP</b>: A model that understands the relationship between sounds and text descriptions
/// 2. <b>Separation network</b>: A U-Net that extracts the described sound from the mixture
///
/// AudioSep achieves state-of-the-art results on both sound separation and sound event detection
/// benchmarks, making it one of the most versatile audio models available.
/// </para>
/// </remarks>
public class AudioSepOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 32000;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between FFT frames.</summary>
    public int HopLength { get; set; } = 320;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the minimum frequency for mel filterbank.</summary>
    public int FMin { get; set; } = 0;

    /// <summary>Gets or sets the maximum frequency for mel filterbank.</summary>
    public int FMax { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base", "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the CLAP embedding dimension.</summary>
    /// <remarks>
    /// <para>CLAP produces 512-dimensional embeddings that encode the semantic meaning
    /// of both audio and text in the same space.</para>
    /// </remarks>
    public int CLAPEmbeddingDim { get; set; } = 512;

    /// <summary>Gets or sets the separation network hidden dimension.</summary>
    public int SeparationDim { get; set; } = 256;

    /// <summary>Gets or sets the number of separation network layers.</summary>
    public int NumSeparationLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads in the separator.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the U-Net encoder channels.</summary>
    public int[] EncoderChannels { get; set; } = [32, 64, 128, 256];

    #endregion

    #region Detection

    /// <summary>Gets or sets the confidence threshold for event detection.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Gets or sets the window size in seconds for detection.</summary>
    public double DetectionWindowSize { get; set; } = 10.0;

    /// <summary>Gets or sets the window overlap ratio (0-1).</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Gets or sets custom event labels. If null, uses AudioSet-527 labels.</summary>
    /// <remarks>
    /// <para>AudioSep can detect any sound described by text, but for standard SED evaluation
    /// it uses the AudioSet-527 label set. Custom labels can be provided for specific use cases.</para>
    /// </remarks>
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
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
