using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Temporal Fusion Transformer (TFT) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Temporal Fusion Transformer is a state-of-the-art deep learning architecture for multi-horizon forecasting.
/// It combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.
/// TFT uses self-attention mechanisms to learn temporal relationships at different scales and integrates
/// static metadata, time-varying known inputs, and time-varying unknown inputs.
/// </para>
/// <para><b>For Beginners:</b> TFT is an advanced neural network designed specifically for forecasting
/// that can handle multiple types of input data:
/// - Static features (e.g., store location, product category) that don't change over time
/// - Known future inputs (e.g., holidays, promotions) that we know ahead of time
/// - Unknown inputs (e.g., past sales) that we can only observe historically
///
/// The model uses "attention" mechanisms to focus on the most relevant time periods and features,
/// making it both accurate and interpretable.
/// </para>
/// </remarks>
public class TemporalFusionTransformerOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalFusionTransformerOptions{T}"/> class.
    /// </summary>
    public TemporalFusionTransformerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public TemporalFusionTransformerOptions(TemporalFusionTransformerOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenSize = other.HiddenSize;
        NumAttentionHeads = other.NumAttentionHeads;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        QuantileLevels = other.QuantileLevels != null ? (double[])other.QuantileLevels.Clone() : null;
        UseVariableSelection = other.UseVariableSelection;
        StaticCovariateSize = other.StaticCovariateSize;
        TimeVaryingKnownSize = other.TimeVaryingKnownSize;
        TimeVaryingUnknownSize = other.TimeVaryingUnknownSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (number of historical time steps used as input).
    /// </summary>
    /// <value>The lookback window size, defaulting to 24.</value>
    public int LookbackWindow { get; set; } = 24;

    /// <summary>
    /// Gets or sets the forecast horizon (number of future time steps to predict).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 6.</value>
    public int ForecastHorizon { get; set; } = 6;

    /// <summary>
    /// Gets or sets the hidden state size for the model.
    /// </summary>
    /// <value>The hidden state size, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls the capacity of the model's internal representations.
    /// Larger values allow the model to capture more complex patterns but require more memory and computation.
    /// </para>
    /// </remarks>
    public int HiddenSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of attention heads in the multi-head attention mechanism.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads allow the model to focus on different aspects
    /// of the time series simultaneously. More heads can capture more diverse patterns.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 2.</value>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly ignores some neurons during training
    /// to prevent overfitting. A value of 0.1 means 10% of neurons are ignored in each training step.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the quantile levels for probabilistic forecasting.
    /// </summary>
    /// <value>Array of quantile levels, defaulting to [0.1, 0.5, 0.9].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Quantile forecasting provides prediction intervals.
    /// For example, [0.1, 0.5, 0.9] gives you the 10th percentile (pessimistic),
    /// median (most likely), and 90th percentile (optimistic) predictions.
    /// </para>
    /// </remarks>
    public double[] QuantileLevels { get; set; } = new double[] { 0.1, 0.5, 0.9 };

    /// <summary>
    /// Gets or sets whether to use variable selection networks.
    /// </summary>
    /// <value>True to use variable selection, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Variable selection automatically determines which
    /// input features are most important for making predictions, improving both
    /// accuracy and interpretability.
    /// </para>
    /// </remarks>
    public bool UseVariableSelection { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of static covariates (features that don't change over time).
    /// </summary>
    /// <value>The number of static covariates, defaulting to 0.</value>
    public int StaticCovariateSize { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of time-varying known inputs (future values that are known).
    /// </summary>
    /// <value>The number of time-varying known inputs, defaulting to 0.</value>
    public int TimeVaryingKnownSize { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of time-varying unknown inputs (past observations only).
    /// </summary>
    /// <value>The number of time-varying unknown inputs, defaulting to 1.</value>
    public int TimeVaryingUnknownSize { get; set; } = 1;
}
