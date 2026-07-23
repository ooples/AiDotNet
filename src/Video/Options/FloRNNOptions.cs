using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for FloRNN optical-flow-guided recurrent video denoising.
/// </summary>
/// <remarks>
/// <para>
/// FloRNN (Li et al., AAAI 2022) uses optical flow to guide recurrent denoising:
/// - Flow-guided alignment: warps previous hidden states using estimated optical flow before
///   feeding them to the recurrent unit, ensuring temporal features align with current frame
/// - Recurrent architecture: ConvLSTM/ConvGRU processes temporally aligned features,
///   accumulating clean signal over time while averaging out random noise
/// - Occlusion-aware gating: learned gates suppress features from occluded regions where
///   flow-based alignment is unreliable, preventing ghosting artifacts
/// - Multi-scale processing: operates at multiple spatial scales to handle both fine noise
///   patterns and large noisy regions
/// </para>
/// <para>
/// <b>For Beginners:</b> FloRNN removes noise from video by tracking how objects move
/// between frames (optical flow) and using that to align information from previous frames.
/// This lets it average out random noise while keeping real details sharp.
/// </para>
/// </remarks>
public class FloRNNOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FloRNNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FloRNNOptions(FloRNNOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumRecurrentLayers = other.NumRecurrentLayers;
        HiddenDim = other.HiddenDim;
        NumFlowScales = other.NumFlowScales;
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

    /// <summary>Gets or sets the number of recurrent layers.</summary>
    public int NumRecurrentLayers { get; set; } = 3;

    /// <summary>Gets or sets the hidden dimension for recurrent cells.</summary>
    public int HiddenDim { get; set; } = 64;

    /// <summary>Gets or sets the number of flow estimation scales.</summary>
    public int NumFlowScales { get; set; } = 3;

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
