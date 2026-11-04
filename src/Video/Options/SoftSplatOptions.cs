using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.SoftSplat{T}"/>, softmax splatting for video
/// frame interpolation (Niklaus and Liu, CVPR 2020).
/// </summary>
/// <remarks>
/// <para>
/// The widths are the release's (sniklaus/softmax-splatting, run.py): the PWC-Net flow estimator, the three-level
/// importance-metric U-Net and the three-stream GridNet synthesis. Training follows the paper, which adopts the
/// training of context-aware synthesis (Niklaus and Liu, CVPR 2018): AdaMax at 1e-3 on the five-level Laplacian loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoftSplat estimates motion both ways between the two frames, pushes each frame's feature
/// pyramid forward along its motion to the in-between time (blending pixels that land on the same spot by a learned
/// importance), and then lets a grid of convolutions paint the in-between frame from the pushed features.
/// </para>
/// </remarks>
public class SoftSplatOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SoftSplatOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SoftSplatOptions(SoftSplatOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        FlowChannels = (int[])other.FlowChannels.Clone();
        FlowDecoderChannels = (int[])other.FlowDecoderChannels.Clone();
        MetricChannels = (int[])other.MetricChannels.Clone();
        SynthesisChannels = (int[])other.SynthesisChannels.Clone();
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        LaplacianLevels = other.LaplacianLevels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the flow estimator's six pyramid widths, 1/2 to 1/64 size (16, 32, 64, 96, 128, 192).</summary>
    public int[] FlowChannels { get; set; } = { 16, 32, 64, 96, 128, 192 };

    /// <summary>Gets or sets the widths of each flow decoder's five hidden convolutions (128, 128, 96, 64, 32).</summary>
    public int[] FlowDecoderChannels { get; set; } = { 128, 128, 96, 64, 32 };

    /// <summary>Gets or sets the importance-metric U-Net's four row widths (16, 32, 64, 96); the first exceeds 4.</summary>
    public int[] MetricChannels { get; set; } = { 16, 32, 64, 96 };

    /// <summary>Gets or sets the feature pyramid's and the GridNet's three stream widths (32, 64, 96).</summary>
    public int[] SynthesisChannels { get; set; } = { 32, 64, 96 };

    #endregion

    #region Training

    /// <summary>Gets or sets AdaMax's learning rate (1e-3).</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets AdaMax's first-moment decay (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdaMax's infinity-norm decay (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Gets or sets the Laplacian loss's pyramid levels, the last being the low-pass residual (5).</summary>
    public int LaplacianLevels { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
