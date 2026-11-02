using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for PSRT-recurrent, the patch-alignment recurrent video super-resolution transformer.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Shi et al., NeurIPS 2022, and the authors' release (XPixelGroup/RethinkVSRAlignment,
/// psrt_recurrent_arch.py and options/4126_PSRTRecurrent_mix_precision_REDS_600K_N16.yml): 64-channel reconstruction,
/// 120-channel embeddings, three RSTBs of six window-attention layers with 6 heads over 3x8x8 windows, MLP ratio 2,
/// stochastic depth up to 0.1, 16-frame training sequences, Charbonnier loss (eps 1e-12 under the square root), Adam
/// (0.9, 0.99) at 2e-4 with SPyNet at 0.125 times the rate and frozen for the first 5,000 iterations, cosine-annealed
/// to 1e-7 over 600,000 iterations, and an exponential moving average of the weights (decay 0.999) for inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> PSRT walks through a video forward and backward in time. To line up the previous frame with
/// the current one it moves whole 8x8 tiles by their average motion, then lets attention fill in the details.
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

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        EmbedDim = other.EmbedDim;
        Depths = (int[])other.Depths.Clone();
        NumHeads = other.NumHeads;
        WindowSize = other.WindowSize;
        MlpRatio = other.MlpRatio;
        DropPathRate = other.DropPathRate;
        NumFrames = other.NumFrames;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        SpyNetLearningRateMultiplier = other.SpyNetLearningRateMultiplier;
        FlowFreezeIterations = other.FlowFreezeIterations;
        MinLearningRate = other.MinLearningRate;
        CosineIterations = other.CosineIterations;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
        EmaMomentum = other.EmaMomentum;
    }

    #region Architecture

    /// <summary>Gets or sets the width of the reconstruction convolutions (64).</summary>
    public int MidChannels { get; set; } = 64;

    /// <summary>Gets or sets the embedding width of every alignment transformer (120).</summary>
    public int EmbedDim { get; set; } = 120;

    /// <summary>Gets or sets the window-attention layers of each RSTB in an alignment transformer ([6, 6, 6]).</summary>
    public int[] Depths { get; set; } = { 6, 6, 6 };

    /// <summary>Gets or sets the window-attention heads (6).</summary>
    public int NumHeads { get; set; } = 6;

    /// <summary>Gets or sets the spatial window size (8; windows are 3x8x8, the three aligned frames).</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the MLP width as a multiple of the embedding (2).</summary>
    public double MlpRatio { get; set; } = 2.0;

    /// <summary>Gets or sets the largest stochastic-depth rate, reached by the last layer of each transformer (0.1).</summary>
    public double DropPathRate { get; set; } = 0.1;

    /// <summary>Gets or sets the frames of a training sequence (16, the REDS recipe).</summary>
    public int NumFrames { get; set; } = 16;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's starting learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the multiplier of SPyNet's learning rate (0.125, i.e. 2.5e-5).</summary>
    public double SpyNetLearningRateMultiplier { get; set; } = 0.125;

    /// <summary>Gets or sets the iterations SPyNet stays frozen at the start of training (5,000).</summary>
    public int FlowFreezeIterations { get; set; } = 5000;

    /// <summary>Gets or sets the floor of the cosine schedule (1e-7).</summary>
    public double MinLearningRate { get; set; } = 1e-7;

    /// <summary>Gets or sets the iterations of the cosine schedule (600,000).</summary>
    public int CosineIterations { get; set; } = 600000;

    /// <summary>
    /// Gets or sets the Charbonnier ε; BasicSR's loss is sqrt(d² + 1e-12), so ε = 1e-6.
    /// </summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the momentum of the weights' moving average used for inference (0.001, ema_decay 0.999); null
    /// disables it and inference uses the trained weights.
    /// </summary>
    public double? EmaMomentum { get; set; } = 0.001;

    #endregion
}
