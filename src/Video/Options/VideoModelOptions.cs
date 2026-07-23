namespace AiDotNet.Video.Options;

/// <summary>
/// Base configuration options for all video AI models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All options use nullable properties with industry-standard defaults applied internally.
/// This allows zero-configuration usage while still enabling full customization.
/// </para>
/// <para>
/// <b>For Beginners:</b> You don't need to set any of these options to get started!
/// The model will use sensible defaults for everything. Only change options if you
/// know you need something different.
///
/// Example - Using defaults:
/// <code>
/// var model = new VideoSuperResolution&lt;double&gt;();  // All defaults
/// </code>
///
/// Example - Customizing some options:
/// <code>
/// var model = new VideoSuperResolution&lt;double&gt;(new VideoEnhancementOptions&lt;double&gt;
/// {
///     ScaleFactor = 4,      // Custom: 4x upscaling
///     LearningRate = 0.001  // Custom: faster learning
///     // Everything else uses defaults
/// });
/// </code>
/// </para>
/// </remarks>
public class VideoModelOptions<T>
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the hidden dimension size for the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 768 (industry standard for vision transformers).
    /// Larger values can learn more complex patterns but use more memory.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how much "thinking capacity" the model has.
    /// The default works well for most cases. Only increase if you have lots of GPU memory
    /// and need to learn very complex patterns.
    /// </para>
    /// </remarks>
    public int? HiddenDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads for transformer-based models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 12 (standard for medium-sized vision models).
    /// Must evenly divide HiddenDimension.
    /// </para>
    /// </remarks>
    public int? NumAttentionHeads { get; set; }

    /// <summary>
    /// Gets or sets the number of transformer/encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 12 (standard for medium-sized models).
    /// More layers can learn more complex patterns but train slower.
    /// </para>
    /// </remarks>
    public int? NumLayers { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.1 (10% dropout).
    /// Helps prevent overfitting. Set to 0 to disable.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly ignores some neurons during training,
    /// which helps the model generalize better to new data instead of memorizing training data.
    /// </para>
    /// </remarks>
    public double? DropoutRate { get; set; }

    #endregion

    #region Input Configuration

    /// <summary>
    /// Gets or sets the expected number of input frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 16 (standard for video models).
    /// Models process this many frames at once.
    /// </para>
    /// </remarks>
    public int? NumFrames { get; set; }

    /// <summary>
    /// Gets or sets the expected input height in pixels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 224 (standard for many vision models).
    /// Input videos will be resized to this height if different.
    /// </para>
    /// </remarks>
    public int? InputHeight { get; set; }

    /// <summary>
    /// Gets or sets the expected input width in pixels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 224 (standard for many vision models).
    /// Input videos will be resized to this width if different.
    /// </para>
    /// </remarks>
    public int? InputWidth { get; set; }

    /// <summary>
    /// Gets or sets the number of input channels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 3 (RGB color).
    /// Use 1 for grayscale, 4 for RGBA.
    /// </para>
    /// </remarks>
    public int? InputChannels { get; set; }

    #endregion

    #region Training Configuration

    /// <summary>
    /// Gets or sets the learning rate for optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.0001 (standard for Adam optimizer with video models).
    /// Lower values train more slowly but may find better solutions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate controls how big the adjustments are
    /// during training. Too high = unstable training, too low = slow training.
    /// The default is a good starting point.
    /// </para>
    /// </remarks>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: null (auto-detect based on available memory).
    /// Larger batches train faster but use more memory.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many videos the model looks at before
    /// making an adjustment. Leave as null to let the system figure out the best value.
    /// </para>
    /// </remarks>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the weight decay for L2 regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.01 (standard for vision transformers).
    /// Helps prevent overfitting by penalizing large weights.
    /// </para>
    /// </remarks>
    public double? WeightDecay { get; set; }

    /// <summary>
    /// Gets or sets whether to use gradient clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// Prevents exploding gradients during training.
    /// </para>
    /// </remarks>
    public bool? UseGradientClipping { get; set; }

    /// <summary>
    /// Gets or sets the maximum gradient norm for clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1.0.
    /// Only used when UseGradientClipping is true.
    /// </para>
    /// </remarks>
    public double? MaxGradientNorm { get; set; }

    #endregion

    #region Hardware Configuration

    /// <summary>
    /// Gets or sets whether to use GPU acceleration if available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: null (auto-detect GPU availability).
    /// Set to false to force CPU-only execution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> GPUs are much faster for video AI. Leave this as null
    /// to automatically use a GPU if one is available.
    /// </para>
    /// </remarks>
    public bool? UseGpu { get; set; }

    /// <summary>
    /// Gets or sets whether to use mixed precision (FP16) for faster computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: null (auto-detect based on GPU capabilities).
    /// Mixed precision can be 2x faster on supported GPUs.
    /// </para>
    /// </remarks>
    public bool? UseMixedPrecision { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: null (random seed each time).
    /// Set to a specific value to get reproducible results.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    #endregion

    #region Default Value Accessors

    /// <summary>
    /// Gets the effective hidden dimension with default fallback.
    /// </summary>
    internal int EffectiveHiddenDimension => HiddenDimension ?? 768;

    /// <summary>
    /// Gets the effective number of attention heads with default fallback.
    /// </summary>
    internal int EffectiveNumAttentionHeads => NumAttentionHeads ?? 12;

    /// <summary>
    /// Gets the effective number of layers with default fallback.
    /// </summary>
    internal int EffectiveNumLayers => NumLayers ?? 12;

    /// <summary>
    /// Gets the effective dropout rate with default fallback.
    /// </summary>
    internal double EffectiveDropoutRate => DropoutRate ?? 0.1;

    /// <summary>
    /// Gets the effective number of frames with default fallback.
    /// </summary>
    internal int EffectiveNumFrames => NumFrames ?? 16;

    /// <summary>
    /// Gets the effective input height with default fallback.
    /// </summary>
    internal int EffectiveInputHeight => InputHeight ?? 224;

    /// <summary>
    /// Gets the effective input width with default fallback.
    /// </summary>
    internal int EffectiveInputWidth => InputWidth ?? 224;

    /// <summary>
    /// Gets the effective number of input channels with default fallback.
    /// </summary>
    internal int EffectiveInputChannels => InputChannels ?? 3;

    /// <summary>
    /// Gets the effective learning rate with default fallback.
    /// </summary>
    internal double EffectiveLearningRate => LearningRate ?? 0.0001;

    /// <summary>
    /// Gets the effective weight decay with default fallback.
    /// </summary>
    internal double EffectiveWeightDecay => WeightDecay ?? 0.01;

    /// <summary>
    /// Gets the effective gradient clipping setting with default fallback.
    /// </summary>
    internal bool EffectiveUseGradientClipping => UseGradientClipping ?? true;

    /// <summary>
    /// Gets the effective max gradient norm with default fallback.
    /// </summary>
    internal double EffectiveMaxGradientNorm => MaxGradientNorm ?? 1.0;

    #endregion
}
