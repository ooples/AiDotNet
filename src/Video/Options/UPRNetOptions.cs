using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for UPR-Net unified pyramid recurrent network.
/// </summary>
/// <remarks>
/// <para>
/// UPR-Net (Jin et al., 2023) recurrently reuses one feature pyramid, one
/// partial-correlation motion estimator, and one synthesis network across image-pyramid levels.
/// The recurrence is across levels; the released architecture has no ConvLSTM and no per-level
/// copies of the motion or synthesis weights.
/// </para>
/// <para>
/// <b>For Beginners:</b> UPR-Net combines motion estimation and frame creation into a single
/// efficient network that processes images at multiple scales. At each scale, it repeatedly
/// refines its predictions until they're good enough, like iterating on a drawing.
/// </para>
/// </remarks>
public class UPRNetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public UPRNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UPRNetOptions(UPRNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumPyramidLevels = other.NumPyramidLevels;
        NumLevelsSkipped = other.NumLevelsSkipped;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Gets or sets the number of pyramid levels. Default 3 — the base UPR-Net trains with a
    /// 3-level image pyramid (Jin et al., CVPR 2023, §4.1: "3-level image pyramids during training …
    /// sufficient to capture the motions on Vimeo90K"; the base model is a lightweight ~1.7M params).
    /// The "unified" design lets inference run MORE levels for high-resolution motion, but 3 is the
    /// paper's training/base configuration.
    /// </summary>
    public int NumPyramidLevels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of finest pyramid levels whose motion estimation is skipped.
    /// Zero is the paper's base-model default.
    /// </summary>
    public int NumLevelsSkipped { get; set; }

    /// <summary>Validates paper-architecture options.</summary>
    public void Validate()
    {
        if (NumPyramidLevels < 1)
            throw new ArgumentOutOfRangeException(nameof(NumPyramidLevels));
        if (NumLevelsSkipped < 0 || NumLevelsSkipped > NumPyramidLevels)
            throw new ArgumentOutOfRangeException(nameof(NumLevelsSkipped));
    }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
