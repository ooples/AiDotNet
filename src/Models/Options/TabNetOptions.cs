namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabNet, an attention-based interpretable deep learning model for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabNet uses sequential attention to choose which features to reason from at each decision step,
/// enabling interpretable feature selection while achieving performance competitive with gradient boosting.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabNet is a neural network specifically designed for tables of data
/// (like spreadsheets or databases). Unlike traditional neural networks that use all features
/// at once, TabNet learns to focus on the most important features for each prediction.
///
/// Key advantages of TabNet:
/// - **Interpretable**: You can see which features the model uses for each prediction
/// - **Feature Selection**: Automatically selects relevant features at each step
/// - **No Preprocessing**: Works directly with numerical and categorical data
/// - **Competitive Performance**: Often matches or beats gradient boosting methods
///
/// Example usage:
/// <code>
/// var options = new TabNetOptions&lt;double&gt;
/// {
///     NumDecisionSteps = 5,
///     FeatureDimension = 64,
///     OutputDimension = 64,
///     RelaxationFactor = 1.5
/// };
/// var tabnet = new TabNet&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
public class TabNetOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of sequential decision steps.
    /// </summary>
    /// <value>The number of decision steps, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// Each decision step selects a subset of features to process. More steps allow the model
    /// to capture more complex feature interactions but increase computation and risk of overfitting.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of decision steps as rounds of feature selection.
    /// At each step, the model asks "which features should I focus on next?"
    ///
    /// - Fewer steps (3-5): Faster training, simpler patterns
    /// - More steps (7-10): Can capture complex interactions but slower
    ///
    /// Start with 5 steps and adjust based on your data complexity.
    /// </para>
    /// </remarks>
    public int NumDecisionSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the dimension of the feature transformation (also known as HiddenDimension).
    /// </summary>
    /// <value>The feature dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para>
    /// This controls the width of the feature transformation layers. Larger values
    /// allow learning more complex representations but require more data and computation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like the "width" of the internal processing.
    /// Larger values mean the model can learn more complex patterns.
    ///
    /// Common values: 8, 16, 32, 64, 128, 256
    /// - Small datasets: 8-32
    /// - Medium datasets: 32-64
    /// - Large datasets: 64-256
    /// </para>
    /// </remarks>
    public int FeatureDimension { get; set; } = 64;

    /// <summary>
    /// Alias for FeatureDimension for backward compatibility.
    /// </summary>
    public int HiddenDimension
    {
        get => FeatureDimension;
        set => FeatureDimension = value;
    }

    /// <summary>
    /// Gets or sets the dimension of the output at each decision step.
    /// </summary>
    /// <value>The output dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para>
    /// Controls the size of the representation passed to the final prediction layer.
    /// Often set equal to FeatureDimension.
    /// </para>
    /// </remarks>
    public int OutputDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the relaxation factor (gamma) for attention mask reuse.
    /// </summary>
    /// <value>The relaxation factor, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// Controls how much the model can reuse features across decision steps.
    /// A value of 1.0 means strict feature reuse constraint; higher values allow more reuse.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls whether the model can look at the same features
    /// multiple times across decision steps.
    ///
    /// - gamma = 1.0: Each feature can only be used once (strictest)
    /// - gamma = 1.5: Default, moderate feature reuse allowed
    /// - gamma &gt; 2.0: Features can be reused more freely
    ///
    /// Higher values can improve performance but reduce interpretability.
    /// </para>
    /// </remarks>
    public double RelaxationFactor { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the sparsity coefficient for the attention masks.
    /// </summary>
    /// <value>The sparsity coefficient, defaulting to 1e-3.</value>
    /// <remarks>
    /// <para>
    /// Controls the entropy penalty on attention masks. Higher values encourage
    /// sparser attention (fewer features selected per step).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This encourages the model to be "picky" about features.
    ///
    /// - Higher values: More sparse attention, fewer features used per step
    /// - Lower values: Less sparse, more features used
    ///
    /// Sparsity improves interpretability (clearer feature selection) but may
    /// reduce model capacity. Typical range: 1e-5 to 1e-2.
    /// </para>
    /// </remarks>
    public double SparsityCoefficient { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the momentum for batch normalization.
    /// </summary>
    /// <value>The batch normalization momentum, defaulting to 0.02.</value>
    public double BatchNormalizationMomentum { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the virtual batch size for Ghost Batch Normalization.
    /// </summary>
    /// <value>The virtual batch size, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// Ghost Batch Normalization splits the actual batch into virtual batches
    /// to provide regularization. Smaller virtual batches add more noise/regularization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Ghost Batch Normalization is a regularization technique
    /// that helps prevent overfitting by adding controlled noise during training.
    ///
    /// The virtual batch size determines how "chunky" this noise is:
    /// - Smaller values (32-64): More regularization, helps with smaller datasets
    /// - Larger values (128-256): Less regularization, better for larger datasets
    ///
    /// Should typically be smaller than your actual batch size.
    /// </para>
    /// </remarks>
    public int VirtualBatchSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of shared layers in the feature transformer.
    /// </summary>
    /// <value>The number of shared layers, defaulting to 2.</value>
    public int NumSharedLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decision-step-specific layers.
    /// </summary>
    /// <value>The number of step-specific layers, defaulting to 2.</value>
    public int NumStepSpecificLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the epsilon value for numerical stability in batch normalization.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-5.</value>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use pre-training (encoder-decoder) mode.
    /// </summary>
    /// <value>True to enable pre-training mode; otherwise, false. Defaults to false.</value>
    /// <remarks>
    /// <para>
    /// TabNet supports self-supervised pre-training by reconstructing masked features.
    /// This can improve performance when labeled data is limited.
    /// </para>
    /// </remarks>
    public bool EnablePreTraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the masking ratio for pre-training.
    /// </summary>
    /// <value>The masking ratio, defaulting to 0.3.</value>
    public double PreTrainingMaskingRatio { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0 (no dropout).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout randomly turns off neurons during training
    /// to prevent overfitting. Higher values add more regularization.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to clip gradients.
    /// </summary>
    /// <value>True to enable gradient clipping; otherwise, false. Defaults to true.</value>
    public bool EnableGradientClipping { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for clipping.
    /// </summary>
    /// <value>The maximum gradient norm, defaulting to 2.0.</value>
    public double MaxGradientNorm { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the embedding dimension for categorical features.
    /// </summary>
    /// <value>The categorical embedding dimension, defaulting to 1 (simple embedding).</value>
    public int CategoricalEmbeddingDimension { get; set; } = 1;

    /// <summary>
    /// Creates a copy of the options.
    /// </summary>
    /// <returns>A new TabNetOptions instance with the same values.</returns>
    public TabNetOptions<T> Clone()
    {
        return new TabNetOptions<T>
        {
            NumDecisionSteps = NumDecisionSteps,
            FeatureDimension = FeatureDimension,
            OutputDimension = OutputDimension,
            RelaxationFactor = RelaxationFactor,
            SparsityCoefficient = SparsityCoefficient,
            BatchNormalizationMomentum = BatchNormalizationMomentum,
            VirtualBatchSize = VirtualBatchSize,
            NumSharedLayers = NumSharedLayers,
            NumStepSpecificLayers = NumStepSpecificLayers,
            Epsilon = Epsilon,
            EnablePreTraining = EnablePreTraining,
            PreTrainingMaskingRatio = PreTrainingMaskingRatio,
            DropoutRate = DropoutRate,
            EnableGradientClipping = EnableGradientClipping,
            MaxGradientNorm = MaxGradientNorm,
            CategoricalEmbeddingDimension = CategoricalEmbeddingDimension
        };
    }
}
