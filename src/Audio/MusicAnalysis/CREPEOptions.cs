using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the CREPE (Convolutional Representation for Pitch Estimation) model.
/// </summary>
/// <remarks>
/// <para>
/// CREPE (Kim et al., 2018) is a deep learning model for monophonic pitch detection. It uses
/// a convolutional architecture trained on synthesized audio to predict pitch with high accuracy,
/// outperforming traditional methods (YIN, pYIN) especially on noisy or challenging audio.
/// </para>
/// <para>
/// <b>For Beginners:</b> CREPE detects the pitch (how high or low a note sounds) from audio.
/// It works by analyzing small windows of audio through a neural network that outputs which
/// frequency (pitch) is most likely. The model uses 360 pitch bins spanning from C1 (32.7 Hz)
/// to B7 (1975.5 Hz) with 20-cent resolution.
/// </para>
/// </remarks>
public class CREPEOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the input frame size in samples (CREPE uses 1024 samples at 16 kHz).
    /// </summary>
    public int FrameSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region Model Architecture

    /// <summary>
    /// Gets or sets the model capacity variant.
    /// </summary>
    /// <remarks>
    /// Available variants: "tiny" (4), "small" (8), "medium" (16), "large" (24), "full" (32).
    /// The number is the capacity multiplier for filter counts.
    /// </remarks>
    public string Variant { get; set; } = "full";

    /// <summary>
    /// Gets or sets the capacity multiplier for convolutional filter counts.
    /// </summary>
    public int CapacityMultiplier { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of pitch bins (20-cent resolution from C1 to B7).
    /// </summary>
    public int NumBins { get; set; } = 360;

    /// <summary>
    /// Gets or sets the minimum frequency in Hz (C1).
    /// </summary>
    public double MinFrequency { get; set; } = 32.70;

    /// <summary>
    /// Gets or sets the maximum frequency in Hz (B7).
    /// </summary>
    public double MaxFrequency { get; set; } = 1975.53;

    /// <summary>
    /// Gets or sets the cents per bin resolution.
    /// </summary>
    public double CentsPerBin { get; set; } = 20.0;

    #endregion

    #region Inference

    /// <summary>
    /// Gets or sets the voicing confidence threshold.
    /// </summary>
    /// <remarks>
    /// Frames with confidence below this threshold are considered unvoiced.
    /// </remarks>
    public double VoicingThreshold { get; set; } = 0.21;

    /// <summary>
    /// Gets or sets whether to use Viterbi decoding for pitch smoothing.
    /// </summary>
    public bool UseViterbiDecoding { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.25;

    #endregion
}
