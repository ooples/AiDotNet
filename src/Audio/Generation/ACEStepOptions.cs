using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the ACE-Step model.
/// </summary>
/// <remarks>
/// <para>
/// ACE-Step (2024) is an Accelerated Consistency-Enhanced music generation model that uses
/// consistency training to generate high-quality music in very few diffusion steps (1-4 steps
/// vs 50-100 for standard diffusion). It achieves real-time music generation while maintaining
/// quality comparable to multi-step models.
/// </para>
/// <para>
/// <b>For Beginners:</b> ACE-Step generates music from text descriptions super fast.
/// While most AI music generators need many steps (like painting layer by layer), ACE-Step
/// can create music in just 1-4 steps, making it fast enough for real-time use.
/// </para>
/// </remarks>
public class ACEStepOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the number of output channels.</summary>
    public int NumChannels { get; set; } = 2;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the latent dimension.</summary>
    public int LatentDim { get; set; } = 128;

    /// <summary>Gets or sets the U-Net hidden dimension.</summary>
    public int UNetDim { get; set; } = 512;

    /// <summary>Gets or sets the number of U-Net layers.</summary>
    public int NumUNetLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of inference steps.</summary>
    public int NumSteps { get; set; } = 4;

    /// <summary>Gets or sets the text encoder dimension.</summary>
    public int TextEncoderDim { get; set; } = 768;

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
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
