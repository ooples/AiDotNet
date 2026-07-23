using AiDotNet.ActivationFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Mambular (State Space Models for Tabular Data).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Mambular applies the Mamba (State Space Model) architecture to tabular data,
/// treating features as a sequence and using selective state spaces for processing.
/// </para>
/// <para>
/// <b>For Beginners:</b> Mambular treats your features like a sequence:
///
/// - **State Space Models**: An efficient alternative to transformers
/// - **Linear complexity**: Scales better than attention with many features
/// - **Selective mechanism**: Learns which features to remember/forget
///
/// Example:
/// <code>
/// var options = new MambularOptions&lt;double&gt;
/// {
///     StateDimension = 16,
///     NumLayers = 4,
///     ExpansionFactor = 2
/// };
/// var model = new MambularClassifier&lt;double&gt;(numFeatures: 50, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "Mambular: A Sequential Model for Tabular Deep Learning" (2024)
/// </para>
/// </remarks>
public class MambularOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the embedding dimension for features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 32.</value>
    public int EmbeddingDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the state dimension for the SSM.
    /// </summary>
    /// <value>The state dimension, defaulting to 16.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The state dimension controls how much information
    /// the model can "remember" as it processes features sequentially.
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of Mamba layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the expansion factor for the inner dimension.
    /// </summary>
    /// <value>The expansion factor, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inner processing dimension is
    /// EmbeddingDimension * ExpansionFactor.
    /// </para>
    /// </remarks>
    public int ExpansionFactor { get; set; } = 2;

    /// <summary>
    /// Gets the inner dimension.
    /// </summary>
    public int InnerDimension => EmbeddingDimension * ExpansionFactor;

    /// <summary>
    /// Gets or sets the convolution kernel size.
    /// </summary>
    /// <value>The kernel size, defaulting to 4.</value>
    public int ConvKernelSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the hidden dimensions for the MLP head.
    /// </summary>
    /// <value>Array of hidden dimensions, defaulting to [64, 32].</value>
    public int[] MLPHiddenDimensions { get; set; } = [64, 32];

    /// <summary>
    /// Gets or sets the cardinalities of categorical features.
    /// </summary>
    /// <value>Array of cardinalities, or null if no categorical features.</value>
    public int[]? CategoricalCardinalities { get; set; }

    /// <summary>
    /// Gets or sets the initialization scale for parameters.
    /// </summary>
    /// <value>The initialization scale, defaulting to 0.02.</value>
    public double InitScale { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the delta (discretization step) range minimum.
    /// </summary>
    /// <value>The minimum delta, defaulting to 0.001.</value>
    public double DeltaMin { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the delta (discretization step) range maximum.
    /// </summary>
    /// <value>The maximum delta, defaulting to 0.1.</value>
    public double DeltaMax { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use bidirectional processing.
    /// </summary>
    /// <value>True for bidirectional; false for unidirectional. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bidirectional processing looks at features
    /// from both left-to-right and right-to-left, capturing more context.
    /// </para>
    /// </remarks>
    public bool UseBidirectional { get; set; } = true;

    /// <summary>
    /// Gets or sets the hidden layer activation function for the MLP head.
    /// </summary>
    /// <value>The activation function, defaulting to ReLU.</value>
    public IActivationFunction<T>? HiddenActivation { get; set; } = new ReLUActivation<T>();

    /// <summary>
    /// Gets or sets the hidden layer vector activation function (alternative to scalar activation).
    /// </summary>
    /// <value>The vector activation function, or null to use scalar activation.</value>
    public IVectorActivationFunction<T>? HiddenVectorActivation { get; set; }
}
