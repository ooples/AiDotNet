using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Base interface for all financial AI models in AiDotNet.Finance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> with financial-specific capabilities,
/// including support for both ONNX inference and native trainable implementations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Financial AI models in this library follow a dual-mode pattern:
///
/// - <b>Native Mode:</b> Uses pure C# neural network layers for training and inference.
///   This mode supports gradient computation, parameter updates, and full training capabilities.
///   Use this when you need to train models from scratch or fine-tune on your data.
///
/// - <b>ONNX Mode:</b> Loads pretrained models in ONNX format for fast inference only.
///   This mode is optimized for production deployment where you don't need training.
///   Use this when deploying pretrained models for prediction.
///
/// All financial models share common capabilities:
/// - Time series forecasting
/// - Uncertainty quantification (prediction intervals)
/// - Financial metrics computation (Sharpe ratio, drawdown, etc.)
/// - Integration with the AiDotNet ecosystem (serialization, checkpointing, etc.)
/// </para>
/// </remarks>
public interface IFinancialModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode allows training and uses pure C# layers.
    /// ONNX mode loads a pre-trained model for inference only.
    /// </para>
    /// </remarks>
    bool UseNativeMode { get; }

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Training is only supported in native mode. ONNX mode provides inference-only capabilities.
    /// </para>
    /// </remarks>
    bool SupportsTraining { get; }

    /// <summary>
    /// Gets the model's expected input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Financial models typically process sequences of historical data.
    /// This property indicates how many time steps the model expects as input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If the model has a sequence length of 96, it means
    /// the model looks at the last 96 time periods to make predictions.
    /// </para>
    /// </remarks>
    int SequenceLength { get; }

    /// <summary>
    /// Gets the model's prediction horizon (number of future time steps to forecast).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the prediction horizon is 24, the model forecasts
    /// the next 24 time periods into the future.
    /// </para>
    /// </remarks>
    int PredictionHorizon { get; }

    /// <summary>
    /// Gets the number of input features (variables) the model expects.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multivariate financial models can process multiple variables simultaneously,
    /// such as price, volume, and various technical indicators.
    /// </para>
    /// </remarks>
    int NumFeatures { get; }

    /// <summary>
    /// Generates forecasts with uncertainty quantification.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <param name="quantiles">Optional quantiles for prediction intervals (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>
    /// Forecast tensor. If quantiles are provided, shape is [batch_size, prediction_horizon, num_quantiles].
    /// Otherwise, shape is [batch_size, prediction_horizon, 1] for point forecasts.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uncertainty quantification tells you not just the predicted value,
    /// but also how confident the model is. For example, quantiles [0.1, 0.5, 0.9] give you:
    /// - 10th percentile (lower bound with 80% confidence)
    /// - 50th percentile (median prediction)
    /// - 90th percentile (upper bound with 80% confidence)
    /// </para>
    /// </remarks>
    Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null);

    /// <summary>
    /// Gets the name of the model architecture.
    /// </summary>
    string ModelName { get; }

    /// <summary>
    /// Gets financial-specific metrics from the model.
    /// </summary>
    /// <returns>Dictionary containing financial metrics like MAE, RMSE, MAPE, etc.</returns>
    Dictionary<string, T> GetFinancialMetrics();
}
