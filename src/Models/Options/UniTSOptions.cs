using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for UniTS (Unified Time Series Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// UniTS is a unified architecture for multiple time series tasks including
/// forecasting, classification, anomaly detection, and imputation using
/// a single pretrained model.
/// </para>
/// <para><b>For Beginners:</b> UniTS is designed to be a universal time series model:
///
/// <b>The Key Insight:</b>
/// Different time series tasks share common patterns. Instead of training
/// separate models, UniTS learns a unified representation that works for all tasks.
///
/// <b>Supported Tasks:</b>
/// 1. <b>Forecasting:</b> Predict future values
/// 2. <b>Classification:</b> Categorize entire time series
/// 3. <b>Anomaly Detection:</b> Identify unusual patterns
/// 4. <b>Imputation:</b> Fill in missing values
///
/// <b>Architecture:</b>
/// - Multi-scale temporal convolution for local patterns
/// - Transformer layers for global dependencies
/// - Task-specific heads for different outputs
/// - Shared backbone pretrained on diverse datasets
///
/// <b>Advantages:</b>
/// - One model for multiple tasks (transfer learning)
/// - Strong zero-shot performance on new domains
/// - Efficient inference (shared computation)
/// </para>
/// <para>
/// <b>Reference:</b> Gao et al., "UniTS: A Unified Multi-Task Time Series Model", 2024.
/// https://arxiv.org/abs/2403.00131
/// </para>
/// </remarks>
public class UniTSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UniTSOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default UniTS configuration for
    /// multi-task time series processing.
    /// </para>
    /// </remarks>
    public UniTSOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UniTSOptions(UniTSOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        ConvKernelSizes = other.ConvKernelSizes;
        DropoutRate = other.DropoutRate;
        TaskType = other.TaskType;
        NumClasses = other.NumClasses;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict (for forecasting task).
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the hidden dimension.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// different patterns simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the convolution kernel sizes for multi-scale temporal convolution.
    /// </summary>
    /// <value>Array of kernel sizes, defaulting to [3, 5, 7].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-scale convolutions capture patterns at different
    /// time scales. Smaller kernels capture fine-grained patterns, larger kernels
    /// capture broader trends.
    /// </para>
    /// </remarks>
    public int[] ConvKernelSizes { get; set; } = new[] { 3, 5, 7 };

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the task type.
    /// </summary>
    /// <value>The task type, defaulting to "forecasting".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specifies which task to perform:
    /// - "forecasting": Predict future values
    /// - "classification": Categorize time series
    /// - "anomaly": Detect anomalies
    /// - "imputation": Fill missing values
    /// </para>
    /// </remarks>
    public string TaskType { get; set; } = "forecasting";

    /// <summary>
    /// Gets or sets the number of classes for classification task.
    /// </summary>
    /// <value>The number of classes, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only used when TaskType is "classification".
    /// </para>
    /// </remarks>
    public int NumClasses { get; set; } = 2;
}
