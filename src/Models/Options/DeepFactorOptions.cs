using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the DeepFactor (Deep Factor Model) for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DeepFactor combines factor modeling with deep learning for multivariate time series.
/// It decomposes time series into global factors (shared patterns) and local components
/// (series-specific behavior), learning both through neural networks.
/// </para>
/// <para><b>For Beginners:</b> DeepFactor is designed for forecasting many related time series:
///
/// <b>What is Factor Modeling?</b>
/// Factor models assume observed variables are driven by hidden "factors":
/// - Global factors: Market-wide patterns (economy, weather, trends)
/// - Factor loadings: How much each series is affected by each factor
/// - Local component: Series-specific noise and behavior
///
/// <b>Example - Retail Sales:</b>
/// Factors might represent:
/// - F1: Overall economic conditions (affects all stores)
/// - F2: Holiday shopping season (affects all stores differently)
/// - F3: Regional weather (affects nearby stores similarly)
/// Each store's sales = (loading1 * F1) + (loading2 * F2) + (loading3 * F3) + local
///
/// <b>Why "Deep" Factor?</b>
/// Traditional factor models use linear relationships.
/// DeepFactor uses neural networks to:
/// - Learn non-linear factor dynamics
/// - Automatically discover factor structure
/// - Capture complex cross-series dependencies
///
/// <b>Benefits:</b>
/// - Captures shared patterns across many time series efficiently
/// - Reduces overfitting when series are related
/// - Works well for hierarchical forecasting (stores in regions, products in categories)
/// - Interpretable through factor analysis
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Deep Factors for Forecasting", 2019.
/// https://arxiv.org/abs/1905.12417
/// </para>
/// </remarks>
public class DeepFactorOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DeepFactorOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default DeepFactor configuration suitable for
    /// multivariate time series with shared patterns.
    /// </para>
    /// </remarks>
    public DeepFactorOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of existing options.
    /// </para>
    /// </remarks>
    public DeepFactorOptions(DeepFactorOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        NumFactors = other.NumFactors;
        FactorHiddenDimension = other.FactorHiddenDimension;
        LocalHiddenDimension = other.LocalHiddenDimension;
        NumFactorLayers = other.NumFactorLayers;
        NumLocalLayers = other.NumLocalLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (input sequence length).
    /// </summary>
    /// <value>The lookback window, defaulting to 168.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of latent factors.
    /// </summary>
    /// <value>The number of factors, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many global patterns to learn.
    /// More factors can capture more complex relationships but may overfit.
    /// Start with 5-20 factors for most applications.
    ///
    /// Rule of thumb: sqrt(number of series) is often a good starting point.
    /// </para>
    /// </remarks>
    public int NumFactors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the hidden dimension for the factor model RNN.
    /// </summary>
    /// <value>The factor hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The capacity of the network that generates factors.
    /// Larger values can learn more complex factor dynamics.
    /// </para>
    /// </remarks>
    public int FactorHiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension for the local model.
    /// </summary>
    /// <value>The local hidden dimension, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The capacity of the network that handles
    /// series-specific patterns. Usually smaller than factor dimension since
    /// most variation should be captured by factors.
    /// </para>
    /// </remarks>
    public int LocalHiddenDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of layers in the factor model.
    /// </summary>
    /// <value>The number of factor layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deeper factor models can capture
    /// more complex global patterns.
    /// </para>
    /// </remarks>
    public int NumFactorLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of layers in the local model.
    /// </summary>
    /// <value>The number of local layers, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The local model is usually simpler
    /// since it only needs to capture series-specific residual patterns.
    /// </para>
    /// </remarks>
    public int NumLocalLayers { get; set; } = 1;

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
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big of a step to take when updating the model.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to go through the entire training dataset.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples to process at once.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;
}
