using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the Matcha-TTS model.
/// </summary>
/// <remarks>
/// <para>
/// Matcha-TTS (Mehta et al., 2024) is a fast, lightweight TTS model based on conditional
/// flow matching (OT-CFM). Unlike diffusion-based TTS that requires many denoising steps,
/// Matcha-TTS generates high-quality mel-spectrograms in just 2-4 steps, achieving 10x
/// faster synthesis than Grad-TTS while maintaining comparable quality (MOS 4.04 on LJSpeech).
/// </para>
/// <para>
/// <b>For Beginners:</b> Matcha-TTS is like a fast artist who can sketch a beautiful
/// portrait in just a few strokes. Other models (like Grad-TTS) need many small steps to
/// gradually refine the output, but Matcha-TTS takes a more direct path from noise to
/// speech - like drawing a straight line instead of a winding road.
///
/// Key advantages:
/// - Very fast: Only 2-4 synthesis steps (vs 50-1000 for diffusion models)
/// - Lightweight: Smaller model with fewer parameters
/// - High quality: Comparable to slower diffusion models
/// - Memory efficient: Uses optimal transport for efficient training
/// </para>
/// </remarks>
public class MatchaTTSOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the hop length for spectrogram alignment.</summary>
    public int HopLength { get; set; } = 256;

    #endregion

    #region Text Encoder

    /// <summary>Gets or sets the text encoder hidden dimension.</summary>
    public int TextEncoderDim { get; set; } = 192;

    /// <summary>Gets or sets the number of text encoder layers.</summary>
    public int NumTextEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads in the text encoder.</summary>
    public int NumTextEncoderHeads { get; set; } = 2;

    /// <summary>Gets or sets the phoneme vocabulary size.</summary>
    public int PhonemeVocabSize { get; set; } = 178;

    #endregion

    #region Flow Matching Decoder

    /// <summary>Gets or sets the model variant ("small", "base").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the decoder hidden dimension.</summary>
    public int DecoderDim { get; set; } = 256;

    /// <summary>Gets or sets the number of decoder layers (U-Net blocks).</summary>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>Gets or sets the number of ODE solver steps for synthesis.</summary>
    /// <remarks>
    /// Matcha-TTS uses very few steps (2-4) compared to diffusion models (50-1000).
    /// More steps = slightly better quality but slower.
    /// </remarks>
    public int NumSynthesisSteps { get; set; } = 4;

    /// <summary>Gets or sets the temperature for flow matching sampling.</summary>
    public double Temperature { get; set; } = 0.667;

    #endregion

    #region Duration Predictor

    /// <summary>Gets or sets the duration predictor hidden dimension.</summary>
    public int DurationPredictorDim { get; set; } = 256;

    /// <summary>Gets or sets the number of duration predictor layers.</summary>
    public int NumDurationPredictorLayers { get; set; } = 2;

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
