using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for FloRNN (Li et al., ECCV 2022), a unidirectional video denoiser whose look-ahead
/// recurrent module mimics a backward one.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the authors' sRGB training script (github.com/nagejacob/FloRNN, train_models/sRGB_train.py):
/// 64 feature channels, 15 residual blocks per recurrent module, a look-ahead of 3 frames, a border of 0.1 of each
/// side for the forward-warping, and Adam at 1e-4 (the script then divides it by 10 for the last of 12 epochs).
/// </para>
/// <para>
/// <b>For Beginners:</b> FloRNN removes noise from video one frame at a time but peeks a few frames ahead. A
/// forward recurrent module carries what it has learned from past frames; a look-ahead module runs ahead and its
/// features are warped back along the motion so the current frame also benefits from the future.
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

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ForwardCount = other.ForwardCount;
        BorderRatio = other.BorderRatio;
        FlowPyramidLevels = other.FlowPyramidLevels;
        NoiseLevel = other.NoiseLevel;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
    }

    #region Architecture

    /// <summary>Gets or sets the feature channels of both recurrent modules' hidden states (64).</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks in each recurrent module (15 in the authors' sRGB script).</summary>
    public int NumResBlocks { get; set; } = 15;

    /// <summary>
    /// Gets or sets how many frames the look-ahead module runs ahead of the output frame (3 in the authors' sRGB
    /// script).
    /// </summary>
    public int ForwardCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets the zero border, as a fraction of each side, around a hidden state before it is forward-warped;
    /// features splatted past the frame edge land there and are restored when the state is warped back (0.1).
    /// </summary>
    public double BorderRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the pyramid levels of the frozen SPyNet flow estimator that stands in for the paper's
    /// pretrained PWC-Net.
    /// </summary>
    public int FlowPyramidLevels { get; set; } = 5;

    /// <summary>
    /// Gets or sets the noise standard deviation (in the frames' own units) of the noise-level map that
    /// inference feeds; null estimates it from the frames. Training uses the RMS of noisy minus clean, the
    /// σ the authors' noise was drawn with. Default: 20/255, the authors' validation noise.
    /// </summary>
    public double? NoiseLevel { get; set; } = 20.0 / 255.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
