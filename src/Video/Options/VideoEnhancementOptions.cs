using AiDotNet.Video.Interfaces;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for video enhancement models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extends <see cref="VideoModelOptions{T}"/> with enhancement-specific settings
/// like scale factor, temporal smoothing, and loss function configuration.
/// </para>
/// <para>
/// <b>For Beginners:</b> These options control how your video enhancement model works.
/// The most important options are:
/// - ScaleFactor: How much to upscale the video (2x, 4x, etc.)
/// - EnhancementType: What kind of enhancement to apply
/// - UseTemporalConsistency: Whether to keep frames looking consistent over time
///
/// Example:
/// <code>
/// var options = new VideoEnhancementOptions&lt;double&gt;
/// {
///     ScaleFactor = 4,
///     EnhancementType = VideoEnhancementType.SuperResolution,
///     UseTemporalConsistency = true
/// };
/// var model = new VideoSuperResolution&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class VideoEnhancementOptions<T> : VideoModelOptions<T>
{
    #region Enhancement Configuration

    /// <summary>
    /// Gets or sets the spatial scale factor for upscaling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 2 (2x upscaling).
    /// Common values: 2, 4, 8. Higher values produce larger output but require more computation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how much bigger the output video will be.
    /// - ScaleFactor = 2: 720x480 becomes 1440x960
    /// - ScaleFactor = 4: 720x480 becomes 2880x1920
    /// </para>
    /// </remarks>
    public int? ScaleFactor { get; set; }

    /// <summary>
    /// Gets or sets the temporal scale factor for frame rate increase.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1 (no frame rate change).
    /// Set to 2 for 2x frame rate, 4 for 4x, etc.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This adds extra frames to make the video smoother.
    /// - TemporalScaleFactor = 2: 30fps becomes 60fps
    /// - TemporalScaleFactor = 4: 30fps becomes 120fps
    /// </para>
    /// </remarks>
    public int? TemporalScaleFactor { get; set; }

    /// <summary>
    /// Gets or sets the type of enhancement to perform.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: SuperResolution.
    /// Different enhancement types use different model architectures internally.
    /// </para>
    /// </remarks>
    public VideoEnhancementType? EnhancementType { get; set; }

    /// <summary>
    /// Gets or sets whether to enforce temporal consistency between frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// When enabled, the model uses recurrent connections or temporal attention
    /// to ensure frames look consistent over time (no flickering).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Keep this true! It prevents annoying flickering
    /// artifacts in the output video. Only disable for single-frame processing.
    /// </para>
    /// </remarks>
    public bool? UseTemporalConsistency { get; set; }

    /// <summary>
    /// Gets or sets the number of recurrent iterations for temporal models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 3.
    /// More iterations can improve quality but increase processing time.
    /// </para>
    /// </remarks>
    public int? RecurrentIterations { get; set; }

    #endregion

    #region Loss Function Configuration

    /// <summary>
    /// Gets or sets the weight for perceptual loss (VGG-based).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1.0.
    /// Perceptual loss encourages visually pleasing results even if they differ
    /// pixel-by-pixel from the target. Set to 0 to disable.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Perceptual loss makes the output look good to humans,
    /// even if it's not mathematically identical to the target. It's important
    /// for realistic-looking results.
    /// </para>
    /// </remarks>
    public double? PerceptualLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight for pixel-wise L1 loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1.0.
    /// L1 loss ensures pixel accuracy. Higher weight = more accurate but potentially blurrier.
    /// </para>
    /// </remarks>
    public double? L1LossWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight for adversarial (GAN) loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.1.
    /// Adversarial loss encourages realistic textures and details.
    /// Set to 0 to disable GAN training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> GAN loss uses a "discriminator" network to push
    /// the model to generate more realistic-looking results. It adds some
    /// training complexity but produces sharper outputs.
    /// </para>
    /// </remarks>
    public double? AdversarialLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight for temporal consistency loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.5.
    /// Penalizes flickering between frames. Only used when UseTemporalConsistency is true.
    /// </para>
    /// </remarks>
    public double? TemporalLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the weight for flow-warping loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.2.
    /// Ensures motion-consistent enhancement by warping frames using optical flow.
    /// </para>
    /// </remarks>
    public double? FlowLossWeight { get; set; }

    #endregion

    #region Architecture Options

    /// <summary>
    /// Gets or sets the number of residual blocks in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 16 for 2x scale, 23 for 4x scale.
    /// More blocks = higher quality but slower.
    /// </para>
    /// </remarks>
    public int? NumResidualBlocks { get; set; }

    /// <summary>
    /// Gets or sets the number of feature channels in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 64.
    /// More channels = more capacity but more memory usage.
    /// </para>
    /// </remarks>
    public int? NumFeatureChannels { get; set; }

    /// <summary>
    /// Gets or sets whether to use attention mechanisms in residual blocks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// Attention helps the model focus on important regions but adds computation.
    /// </para>
    /// </remarks>
    public bool? UseAttention { get; set; }

    /// <summary>
    /// Gets or sets whether to use bidirectional propagation for temporal models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// Bidirectional models look at both past and future frames, improving quality
    /// but requiring more memory and preventing real-time processing.
    /// </para>
    /// </remarks>
    public bool? UseBidirectional { get; set; }

    #endregion

    #region Default Value Accessors

    /// <summary>
    /// Gets the effective scale factor with default fallback.
    /// </summary>
    internal int EffectiveScaleFactor => ScaleFactor ?? 2;

    /// <summary>
    /// Gets the effective temporal scale factor with default fallback.
    /// </summary>
    internal int EffectiveTemporalScaleFactor => TemporalScaleFactor ?? 1;

    /// <summary>
    /// Gets the effective enhancement type with default fallback.
    /// </summary>
    internal VideoEnhancementType EffectiveEnhancementType => EnhancementType ?? VideoEnhancementType.SuperResolution;

    /// <summary>
    /// Gets the effective temporal consistency setting with default fallback.
    /// </summary>
    internal bool EffectiveUseTemporalConsistency => UseTemporalConsistency ?? true;

    /// <summary>
    /// Gets the effective recurrent iterations with default fallback.
    /// </summary>
    internal int EffectiveRecurrentIterations => RecurrentIterations ?? 3;

    /// <summary>
    /// Gets the effective perceptual loss weight with default fallback.
    /// </summary>
    internal double EffectivePerceptualLossWeight => PerceptualLossWeight ?? 1.0;

    /// <summary>
    /// Gets the effective L1 loss weight with default fallback.
    /// </summary>
    internal double EffectiveL1LossWeight => L1LossWeight ?? 1.0;

    /// <summary>
    /// Gets the effective adversarial loss weight with default fallback.
    /// </summary>
    internal double EffectiveAdversarialLossWeight => AdversarialLossWeight ?? 0.1;

    /// <summary>
    /// Gets the effective temporal loss weight with default fallback.
    /// </summary>
    internal double EffectiveTemporalLossWeight => TemporalLossWeight ?? 0.5;

    /// <summary>
    /// Gets the effective flow loss weight with default fallback.
    /// </summary>
    internal double EffectiveFlowLossWeight => FlowLossWeight ?? 0.2;

    /// <summary>
    /// Gets the effective number of residual blocks with default fallback.
    /// </summary>
    internal int EffectiveNumResidualBlocks => NumResidualBlocks ?? (EffectiveScaleFactor >= 4 ? 23 : 16);

    /// <summary>
    /// Gets the effective number of feature channels with default fallback.
    /// </summary>
    internal int EffectiveNumFeatureChannels => NumFeatureChannels ?? 64;

    /// <summary>
    /// Gets the effective attention setting with default fallback.
    /// </summary>
    internal bool EffectiveUseAttention => UseAttention ?? true;

    /// <summary>
    /// Gets the effective bidirectional setting with default fallback.
    /// </summary>
    internal bool EffectiveUseBidirectional => UseBidirectional ?? true;

    #endregion
}
