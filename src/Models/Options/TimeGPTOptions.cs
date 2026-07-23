using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeGPT-style time series forecasting model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimeGPT represents a GPT-style architecture adapted for time series forecasting,
/// featuring large-scale pre-training on diverse time series data with zero-shot
/// and few-shot forecasting capabilities.
/// </para>
/// <para><b>For Beginners:</b> TimeGPT brings GPT-like capabilities to time series:
///
/// <b>The Key Insight:</b>
/// Just as GPT was trained on internet-scale text data to become a general-purpose
/// language model, TimeGPT is trained on millions of diverse time series to become
/// a general-purpose forecasting model.
///
/// <b>Core Features:</b>
/// 1. <b>Large-scale Pre-training:</b> Trained on millions of time series
/// 2. <b>Zero-shot Forecasting:</b> No training needed for new data
/// 3. <b>Uncertainty Quantification:</b> Provides prediction intervals
/// 4. <b>Multi-horizon:</b> Forecasts at any horizon
///
/// <b>Architecture:</b>
/// - Positional encoding for temporal information
/// - Multi-head self-attention for pattern recognition
/// - Large transformer backbone
/// - Conformal prediction for uncertainty
///
/// <b>Advantages:</b>
/// - Works out-of-the-box on new time series
/// - No hyperparameter tuning required
/// - Handles various frequencies and domains
/// - Production-ready forecasting API style
/// </para>
/// <para>
/// <b>Reference:</b> Garza et al., "TimeGPT-1", 2023.
/// https://arxiv.org/abs/2310.03589
/// </para>
/// </remarks>
public class TimeGPTOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TimeGPTOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TimeGPT configuration optimized
    /// for zero-shot time series forecasting.
    /// </para>
    /// </remarks>
    public TimeGPTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TimeGPTOptions(TimeGPTOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        DropoutRate = other.DropoutRate;
        UseConformalPrediction = other.UseConformalPrediction;
        ConfidenceLevel = other.ConfidenceLevel;
        FineTuningSteps = other.FineTuningSteps;
        FineTuningLearningRate = other.FineTuningLearningRate;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>The context length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// TimeGPT uses a flexible context window that adapts to available history.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// TimeGPT can forecast any horizon, but accuracy typically decreases with longer horizons.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the hidden dimension size.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 1024.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the model.
    /// Larger dimensions can capture more complex patterns but require more memory.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// TimeGPT uses a large number of layers for its foundation model capabilities.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of heads, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// different patterns simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// For zero-shot inference, dropout is typically disabled (0.0).
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use conformal prediction for uncertainty quantification.
    /// </summary>
    /// <value>True to use conformal prediction; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Conformal prediction provides statistically rigorous
    /// prediction intervals. It guarantees that the true value falls within the interval
    /// with the specified confidence level.
    /// </para>
    /// </remarks>
    public bool UseConformalPrediction { get; set; } = true;

    /// <summary>
    /// Gets or sets the confidence level for prediction intervals.
    /// </summary>
    /// <value>The confidence level, defaulting to 0.90 (90%).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The probability that the true value falls within
    /// the prediction interval. Common values: 0.80 (80%), 0.90 (90%), 0.95 (95%).
    /// Higher confidence = wider intervals.
    /// </para>
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.90;

    /// <summary>
    /// Gets or sets the number of fine-tuning steps for domain adaptation.
    /// </summary>
    /// <value>The number of fine-tuning steps, defaulting to 0 (no fine-tuning).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> While TimeGPT works well zero-shot, you can optionally
    /// fine-tune it on your specific data for better performance. A few hundred steps
    /// is usually sufficient.
    /// </para>
    /// </remarks>
    public int FineTuningSteps { get; set; } = 0;

    /// <summary>
    /// Gets or sets the learning rate for fine-tuning.
    /// </summary>
    /// <value>The fine-tuning learning rate, defaulting to 1e-5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When fine-tuning, use a small learning rate to
    /// preserve the pre-trained knowledge while adapting to your domain.
    /// </para>
    /// </remarks>
    public double FineTuningLearningRate { get; set; } = 1e-5;
}
