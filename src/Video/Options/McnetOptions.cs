using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Video.Prediction.Mcnet{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Villegas, Yang, Hong, Lin and Lee, "Decomposing Motion and Content for Natural
/// Video Sequence Prediction" (ICLR 2017, arXiv:1706.08033): learning rate 1e-4, alpha = 1,
/// beta = 0.02 (KTH) or 0.001 (UCF-101), lambda = 1, p = 2, and 10 input frames on KTH / 4 on UCF-101.
/// </para>
/// <para>
/// This replaces <c>DRVIOptions</c>. Its <c>NumContentBlocks</c> and <c>NumMotionBlocks</c> survive in
/// spirit because MCnet genuinely has two encoder pathways, but the model they configured was an
/// interpolator, not a predictor.
/// </para>
/// <para><b>For Beginners:</b> These control how deep each of the two pathways is, how many past frames
/// the model watches, how many future frames it predicts, and how much the critic network influences
/// training.</para>
/// </remarks>
public class McnetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the feature width of the encoders and decoder. Default 64.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Gets or sets the depth of the content pathway (<c>f_cont</c>). Default 4.
    /// </summary>
    public int NumContentBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the depth of the motion pathway (<c>f_dyn</c>). Default 4.
    /// </summary>
    public int NumMotionBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the decoder depth (<c>g_dec</c>). Default 4.
    /// </summary>
    public int NumDecoderBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of scales at which residual features are communicated
    /// (<c>r_t^l</c>). Default 3.
    /// </summary>
    /// <remarks>
    /// The paper communicates motion-content features into the decoder at EVERY scale, not only at the
    /// bottleneck, so this is a genuine architectural parameter rather than a tuning knob.
    /// </remarks>
    public int NumScales { get; set; } = 3;

    /// <summary>
    /// Gets or sets how many past frames the model observes. Default 10, the paper's KTH setting
    /// (UCF-101 uses 4).
    /// </summary>
    /// <remarks>
    /// At least two are required: the motion pathway consumes differences <c>x_t - x_{t-1}</c>, which a
    /// single frame cannot produce.
    /// </remarks>
    public int NumInputFrames { get; set; } = 10;

    /// <summary>
    /// Gets or sets how many future frames to predict. Default 1.
    /// </summary>
    public int NumPredictedFrames { get; set; } = 1;

    /// <summary>
    /// Gets or sets alpha, the image-loss weight. Default 1, the paper's value.
    /// </summary>
    public double ImageLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets beta, the adversarial weight. Default 0.02 — the paper's KTH value; it uses 0.001
    /// for UCF-101.
    /// </summary>
    /// <remarks>
    /// Dataset-dependent in the paper, so there is no single "correct" value. 0.02 is recorded here as
    /// the KTH figure rather than presented as universal.
    /// </remarks>
    public double AdversarialLossWeight { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets lambda, the gradient-difference exponent. Default 1, the paper's value.
    /// </summary>
    public double GradientLossExponent { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets p, the pixel-loss norm. Default 2, the paper's value.
    /// </summary>
    public int PixelLossNorm { get; set; } = 2;

    /// <summary>
    /// Gets or sets the learning rate. Default 1e-4, the paper's value.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate. Default 0 — the paper does not use dropout.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets an optional ONNX model path for inference-only use.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
