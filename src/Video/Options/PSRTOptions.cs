using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the PSRT progressive spatio-temporal alignment model.
/// </summary>
/// <remarks>
/// <para>
/// PSRT (Shi et al., 2022) uses progressive window-based spatio-temporal attention:
/// - Spatio-temporal attention blocks (STABs): joint spatial and temporal attention within
///   3D windows (height x width x time) for aligned multi-frame feature fusion
/// - Progressive alignment: a coarse-to-fine encoder-decoder structure where early layers
///   capture large motions and later layers refine sub-pixel alignment
/// - Window-based attention: limits attention to local spatio-temporal windows for
///   computational efficiency while shifted windows enable cross-window communication
/// - Temporal mutual attention: cross-attention between reference and supporting frames
/// </para>
/// <para>
/// <b>For Beginners:</b> PSRT aligns video frames step by step, starting with big motion
/// corrections and progressively refining small details. It uses "windows" (small patches)
/// in both space and time to efficiently find corresponding regions across frames, similar
/// to how Swin Transformer works but extended to video.
/// </para>
/// </remarks>
public class PSRTOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public PSRTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PSRTOptions(PSRTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumSTABs = other.NumSTABs;
        ScaleFactor = other.ScaleFactor;
        WindowSize = other.WindowSize;
        TemporalRadius = other.TemporalRadius;
        NumHeads = other.NumHeads;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of spatio-temporal attention blocks (STABs).</summary>
    /// <remarks>The paper uses 6 STABs in the base configuration.</remarks>
    public int NumSTABs { get; set; } = 6;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the spatial window size for window-based attention.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the temporal window radius (number of neighboring frames).</summary>
    /// <remarks>Total temporal window is 2 * TemporalRadius + 1 frames.</remarks>
    public int TemporalRadius { get; set; } = 3;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

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
