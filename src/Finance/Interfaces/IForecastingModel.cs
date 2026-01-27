namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for time series forecasting models in the Finance module.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IFinancialModel{T}"/> with capabilities specific to
/// time series forecasting, including multi-step prediction, lookback configuration,
/// and support for both univariate and multivariate forecasting.
/// </para>
/// <para>
/// <b>For Beginners:</b> Forecasting models predict future values based on historical patterns.
///
/// Key concepts:
/// - <b>Lookback window:</b> How far back in time the model looks to make predictions.
/// - <b>Prediction horizon:</b> How far into the future the model predicts.
/// - <b>Univariate:</b> Forecasting a single variable (e.g., stock price).
/// - <b>Multivariate:</b> Forecasting using multiple variables (e.g., price, volume, indicators).
///
/// Example use cases:
/// - Stock price prediction
/// - Sales forecasting
/// - Demand planning
/// - Energy load forecasting
/// - Cryptocurrency price prediction
/// </para>
/// </remarks>
public interface IForecastingModel<T> : IFinancialModel<T>
{
    /// <summary>
    /// Gets the patch size for patch-based models (like PatchTST).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Patch-based models divide the input sequence into patches (segments) for processing.
    /// This improves efficiency and enables the model to capture local patterns.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A patch is like a "window within the window". If the sequence
    /// length is 96 and patch size is 16, the model processes 6 patches of 16 time steps each.
    /// Returns 0 if the model doesn't use patches.
    /// </para>
    /// </remarks>
    int PatchSize { get; }

    /// <summary>
    /// Gets the stride between consecutive patches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The stride determines how much the patch window moves between consecutive patches.
    /// A stride equal to patch size means non-overlapping patches; a smaller stride means
    /// overlapping patches.
    /// </para>
    /// </remarks>
    int Stride { get; }

    /// <summary>
    /// Generates multi-step forecasts iteratively (autoregressive forecasting).
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of future steps to predict.</param>
    /// <returns>Tensor containing predictions for all requested steps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoregressive forecasting works like this:
    /// 1. Predict the next step using current data
    /// 2. Add that prediction to the input
    /// 3. Predict the step after that
    /// 4. Repeat until all steps are predicted
    ///
    /// This allows generating longer forecasts than the model's native prediction horizon,
    /// but accuracy typically decreases for distant predictions.
    /// </para>
    /// </remarks>
    Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps);

    /// <summary>
    /// Evaluates the model's forecasting performance on test data.
    /// </summary>
    /// <param name="inputs">Test input sequences.</param>
    /// <param name="targets">Actual target values.</param>
    /// <returns>Dictionary containing forecasting metrics (MAE, RMSE, MAPE, etc.).</returns>
    Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets);

    /// <summary>
    /// Gets whether this model supports channel-independent (CI) forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Channel-independent models process each variable independently, sharing parameters
    /// across channels. This can improve generalization for multivariate forecasting.
    /// </para>
    /// </remarks>
    bool IsChannelIndependent { get; }

    /// <summary>
    /// Applies instance normalization during inference for distribution shift handling.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Financial data often experiences "distribution shift" where
    /// the statistical properties change over time (e.g., higher volatility in a crash).
    /// Instance normalization helps the model adapt to these changes during inference.
    /// </para>
    /// </remarks>
    Tensor<T> ApplyInstanceNormalization(Tensor<T> input);
}
