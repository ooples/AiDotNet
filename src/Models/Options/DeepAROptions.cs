using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the DeepAR (Deep Autoregressive) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting methodology based on autoregressive recurrent neural networks.
/// Unlike traditional methods that provide point forecasts, DeepAR produces probabilistic forecasts
/// that include prediction intervals. It's particularly effective for:
/// - Handling multiple related time series simultaneously
/// - Cold-start problems (forecasting for new items with limited history)
/// - Capturing complex seasonal patterns and trends
/// - Quantifying forecast uncertainty
/// </para>
/// <para><b>For Beginners:</b> DeepAR is an advanced forecasting model that not only predicts
/// what will happen, but also how confident it is in those predictions. Instead of saying
/// "sales will be exactly 100 units," it might say "sales will likely be between 80 and 120 units,
/// with 100 being most probable."
///
/// This is especially useful when:
/// - You need to plan for worst-case and best-case scenarios
/// - You have many related time series (e.g., sales across many stores)
/// - You have some series with very little historical data
///
/// The "autoregressive" part means it uses its own predictions as inputs for future predictions,
/// and "deep" refers to the use of deep neural networks (specifically, LSTM networks).
/// </para>
/// </remarks>
public class DeepAROptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DeepAROptions{T}"/> class.
    /// </summary>
    public DeepAROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public DeepAROptions(DeepAROptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenSize = other.HiddenSize;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        NumSamples = other.NumSamples;
        LikelihoodType = other.LikelihoodType;
        CovariateSize = other.CovariateSize;
        EmbeddingDimension = other.EmbeddingDimension;
    }

    /// <summary>
    /// Gets or sets the lookback window size (context length).
    /// </summary>
    /// <value>The lookback window, defaulting to 30.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at before
    /// making a prediction. For daily data, 30 would mean looking at the past month.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 30;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 7.</value>
    public int ForecastHorizon { get; set; } = 7;

    /// <summary>
    /// Gets or sets the hidden state size of the LSTM layers.
    /// </summary>
    /// <value>The hidden size, defaulting to 40.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The capacity of the model's memory. Larger values
    /// allow the model to remember more complex patterns but require more training data.
    /// </para>
    /// </remarks>
    public int HiddenSize { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of LSTM layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 2.</value>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// ignoring some neurons during training. A value of 0.1 means 10% are ignored.
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
    /// Gets or sets the number of samples to draw for probabilistic forecasts.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> DeepAR generates multiple possible futures (samples)
    /// to create prediction intervals. More samples give more accurate probability
    /// estimates but take longer to compute. 100 is a good balance.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets the likelihood distribution type.
    /// </summary>
    /// <value>The likelihood type, defaulting to "Gaussian".</value>
    /// <remarks>
    /// <para>
    /// Supported values: "Gaussian", "StudentT", "NegativeBinomial"
    /// - Gaussian: For continuous data that can be negative (e.g., temperature, stock returns)
    /// - StudentT: For continuous data with heavy tails (outliers)
    /// - NegativeBinomial: For count data (e.g., number of sales, website visits)
    /// </para>
    /// <para><b>For Beginners:</b> This determines what kind of randomness the model assumes
    /// in your data. Choose based on your data type:
    /// - Gaussian (Normal): Most common, works for temperatures, prices, etc.
    /// - StudentT: When you have occasional extreme outliers
    /// - NegativeBinomial: When counting things (must be non-negative integers)
    /// </para>
    /// </remarks>
    public string LikelihoodType { get; set; } = "Gaussian";

    /// <summary>
    /// Gets or sets the number of covariates (external features).
    /// </summary>
    /// <value>The covariate size, defaulting to 0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Covariates are additional features that might influence
    /// your forecast, like holidays, promotions, weather, etc. Set this to the number
    /// of such features you want to include.
    /// </para>
    /// </remarks>
    public int CovariateSize { get; set; } = 0;

    /// <summary>
    /// Gets or sets the embedding dimension for categorical features.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have categorical features (like store ID,
    /// product category), embeddings convert them into numerical representations
    /// that the model can understand. This sets how many dimensions to use.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 10;
}
