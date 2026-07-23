using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the AudioLDM Classifier model.
/// </summary>
/// <remarks>
/// <para>
/// AudioLDM Classifier (Liu et al., 2023) repurposes the latent representations from the
/// AudioLDM diffusion model for audio classification. By extracting intermediate features
/// from the AudioLDM U-Net, it achieves strong classification performance by leveraging
/// the rich audio representations learned during generative pre-training.
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLDM was originally built to generate audio from text, but
/// it turns out the internal features it learned are also great for recognizing audio.
/// This classifier uses those learned features to identify sounds, similar to how image
/// generation models can also be used for image recognition.
/// </para>
/// </remarks>
public class AudioLDMClassifierOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the hop length.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Gets or sets the minimum frequency for mel filterbank.</summary>
    public int FMin { get; set; } = 50;

    /// <summary>Gets or sets the maximum frequency for mel filterbank.</summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the latent dimension from AudioLDM.</summary>
    public int LatentDim { get; set; } = 256;

    /// <summary>Gets or sets the classifier hidden dimension.</summary>
    public int ClassifierDim { get; set; } = 512;

    /// <summary>Gets or sets the number of classifier layers.</summary>
    public int NumClassifierLayers { get; set; } = 3;

    #endregion

    #region Detection

    /// <summary>Gets or sets the confidence threshold for event detection.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Gets or sets the window size in seconds for detection.</summary>
    public double DetectionWindowSize { get; set; } = 10.0;

    /// <summary>Gets or sets the window overlap ratio (0-1).</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Gets or sets custom event labels.</summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
