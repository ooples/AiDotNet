using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Video.FrameInterpolation.Figan{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults come from van Amersfoort et al., "Frame Interpolation with Multi-Scale Deep Loss Functions
/// and Generative Adversarial Networks" (arXiv:1711.06045): <c>J = 3</c> scales, 32-channel six-layer
/// generator modules with 3x3 kernels, an eight-block discriminator starting at 32 filters, Adam at
/// 1e-4, batch 128, 128x128 crops.
/// </para>
/// <para>
/// This replaces <c>TDPNetOptions</c>, whose <c>NumDiffBlocks</c>, <c>NumHeads</c> and
/// <c>DifferenceThreshold</c> configured a "difference-aware attention" mechanism with no basis in any
/// paper. They are gone rather than renamed.
/// </para>
/// <para><b>For Beginners:</b> These control how many levels of detail the motion estimate is refined
/// through, how wide the small convolutional networks are, and how strongly the critic influences
/// training.</para>
/// </remarks>
public class FiganOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the number of scales J in the coarse-to-fine flow recursion. Default 3.
    /// </summary>
    /// <remarks>
    /// The paper's value. Each additional scale halves the resolution again, so the coarsest level of a
    /// 128x128 crop at J = 3 is 16x16.
    /// </remarks>
    public int NumScales { get; set; } = 3;

    /// <summary>
    /// Gets or sets the hidden channel width of every generator module. Default 32.
    /// </summary>
    /// <remarks>
    /// Table 1: layer 1 maps <c>N_i -> 32</c>, layers 2-5 keep <c>32 -> 32</c>, layer 6 maps
    /// <c>32 -> N_o</c>.
    /// </remarks>
    public int NumFeatures { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of convolutional layers per generator module. Default 6.
    /// </summary>
    public int LayersPerModule { get; set; } = 6;

    /// <summary>
    /// Gets or sets the convolution kernel size. Default 3.
    /// </summary>
    public int KernelSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the discriminator's initial filter count. Default 32.
    /// </summary>
    /// <remarks>
    /// "32 filters initially, 8 blocks of convolution, batch normalization and leaky ReLU with
    /// alternating strides of 2 and 1. Features doubled at stride-2 blocks."
    /// </remarks>
    public int DiscriminatorFilters { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of discriminator blocks. Default 8.
    /// </summary>
    public int DiscriminatorBlocks { get; set; } = 8;

    /// <summary>
    /// Gets or sets the negative slope of the discriminator's leaky ReLU. Default 0.2.
    /// </summary>
    /// <remarks>
    /// The paper names leaky ReLU without giving a slope; 0.2 is the standard GAN value and is recorded
    /// here as a convention rather than a published figure.
    /// </remarks>
    public double LeakyReluSlope { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the learning rate. Default 1e-4, the paper's value.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the training crop size. Default 128, the paper's value.
    /// </summary>
    public int CropSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dropout rate. Default 0 — the paper does not use dropout.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets an optional ONNX model path for inference-only use.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
