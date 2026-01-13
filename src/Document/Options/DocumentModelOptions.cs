namespace AiDotNet.Document.Options;

/// <summary>
/// Base configuration options for all document AI models.
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
/// var model = new LayoutLMv3&lt;double&gt;();  // All defaults
/// </code>
///
/// Example - Customizing some options:
/// <code>
/// var model = new LayoutLMv3&lt;double&gt;(new DocumentModelOptions&lt;double&gt;
/// {
///     ImageSize = 384,       // Custom: larger images
///     LearningRate = 0.001   // Custom: faster learning
///     // Everything else uses defaults
/// });
/// </code>
/// </para>
/// </remarks>
public class DocumentModelOptions<T>
{
    #region Model Architecture

    /// <summary>
    /// Gets or sets the hidden dimension size for the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 768 (industry standard for transformer-based document models).
    /// Larger values can learn more complex patterns but use more memory.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how much "thinking capacity" the model has.
    /// The default works well for most cases. Only increase if you have lots of GPU memory
    /// and need to learn very complex document patterns.
    /// </para>
    /// </remarks>
    public int? HiddenDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads for transformer-based models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 12 (standard for medium-sized document models).
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
    /// </remarks>
    public double? DropoutRate { get; set; }

    #endregion

    #region Input Configuration

    /// <summary>
    /// Gets or sets the expected input image size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 224 (standard ViT size). Common alternatives: 384, 448, 512, 768.
    /// Larger sizes can capture more detail but use more memory.
    /// </para>
    /// </remarks>
    public int? ImageSize { get; set; }

    /// <summary>
    /// Gets or sets the maximum text sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 512 (standard for most document models).
    /// Increase for documents with more text content.
    /// </para>
    /// </remarks>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the patch size for vision transformers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 16.
    /// Smaller patches capture finer detail but increase computation.
    /// </para>
    /// </remarks>
    public int? PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the number of input channels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 3 (RGB color).
    /// Use 1 for grayscale documents.
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
    /// Default: 5e-5 (standard for fine-tuning transformers).
    /// Lower values train more slowly but may find better solutions.
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
    /// </remarks>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the weight decay for L2 regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 0.01 (standard for document transformers).
    /// Helps prevent overfitting by penalizing large weights.
    /// </para>
    /// </remarks>
    public double? WeightDecay { get; set; }

    /// <summary>
    /// Gets or sets the maximum gradient norm for clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1.0.
    /// Prevents exploding gradients during training.
    /// </para>
    /// </remarks>
    public double? MaxGradientNorm { get; set; }

    /// <summary>
    /// Gets or sets whether to use gradient clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// Helps stabilize training.
    /// </para>
    /// </remarks>
    public bool? UseGradientClipping { get; set; }

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

    #region Document-Specific Settings

    /// <summary>
    /// Gets or sets whether to include 2D position embeddings for layout information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true for layout-aware models.
    /// 2D position embeddings encode the spatial location of text elements.
    /// </para>
    /// </remarks>
    public bool? Use2DPositionEmbeddings { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize bounding box coordinates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: true.
    /// Normalizes coordinates to 0-1000 range for model input.
    /// </para>
    /// </remarks>
    public bool? NormalizeBoundingBoxes { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of layout elements to process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 200.
    /// Limits the number of text boxes/layout regions processed per document.
    /// </para>
    /// </remarks>
    public int? MaxLayoutElements { get; set; }

    /// <summary>
    /// Gets or sets the vocabulary size for text tokenization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 30522 (BERT vocabulary size).
    /// Use 50265 for RoBERTa-based models.
    /// </para>
    /// </remarks>
    public int? VocabularySize { get; set; }

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
    /// Gets the effective image size with default fallback.
    /// </summary>
    internal int EffectiveImageSize => ImageSize ?? 224;

    /// <summary>
    /// Gets the effective max sequence length with default fallback.
    /// </summary>
    internal int EffectiveMaxSequenceLength => MaxSequenceLength ?? 512;

    /// <summary>
    /// Gets the effective patch size with default fallback.
    /// </summary>
    internal int EffectivePatchSize => PatchSize ?? 16;

    /// <summary>
    /// Gets the effective number of input channels with default fallback.
    /// </summary>
    internal int EffectiveInputChannels => InputChannels ?? 3;

    /// <summary>
    /// Gets the effective learning rate with default fallback.
    /// </summary>
    internal double EffectiveLearningRate => LearningRate ?? 5e-5;

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

    /// <summary>
    /// Gets the effective 2D position embeddings setting with default fallback.
    /// </summary>
    internal bool EffectiveUse2DPositionEmbeddings => Use2DPositionEmbeddings ?? true;

    /// <summary>
    /// Gets the effective bounding box normalization setting with default fallback.
    /// </summary>
    internal bool EffectiveNormalizeBoundingBoxes => NormalizeBoundingBoxes ?? true;

    /// <summary>
    /// Gets the effective max layout elements with default fallback.
    /// </summary>
    internal int EffectiveMaxLayoutElements => MaxLayoutElements ?? 200;

    /// <summary>
    /// Gets the effective vocabulary size with default fallback.
    /// </summary>
    internal int EffectiveVocabularySize => VocabularySize ?? 30522;

    #endregion
}
