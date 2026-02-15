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
[AiDotNet.Configuration.YamlConfigurable("ForecastingModel")]
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
    /// <b>For Beginners:</b> Sometimes you need to predict further into the future than the model's
    /// native prediction horizon. Autoregressive forecasting solves this by:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item>Predict the next N steps using current data (where N is the prediction horizon)</item>
    /// <item>Add those predictions to the input data</item>
    /// <item>Predict the next N steps after that</item>
    /// <item>Repeat until all requested steps are predicted</item>
    /// </list>
    /// </para>
    /// <para>
    /// This allows generating longer forecasts than the model's native prediction horizon,
    /// but accuracy typically decreases for distant predictions because the model is
    /// building on its own (potentially incorrect) earlier predictions.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Model's native prediction horizon is 24 steps
    /// // But we need to predict 100 steps ahead
    /// var longForecast = model.AutoregressiveForecast(input, steps: 100);
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps);

    /// <summary>
    /// Evaluates the model's forecasting performance on test data.
    /// </summary>
    /// <param name="inputs">Test input sequences.</param>
    /// <param name="targets">Actual target values.</param>
    /// <returns>Dictionary containing forecasting metrics (MAE, RMSE, MAPE, etc.).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training a model, you need to know how well it performs
    /// on data it hasn't seen before. This method calculates common error metrics:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>MAE</b>: Mean Absolute Error - average of absolute differences between
    /// predictions and actual values. Easy to interpret - it's in the same units as your data.</item>
    /// <item><b>RMSE</b>: Root Mean Squared Error - emphasizes larger errors more than MAE.
    /// Also in the same units as your data.</item>
    /// <item><b>MAPE</b>: Mean Absolute Percentage Error - error expressed as a percentage,
    /// useful for comparing across different scales.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Lower values are better for all these metrics.
    /// </para>
    /// </remarks>
    Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets);

    /// <summary>
    /// Gets whether this model supports channel-independent (CI) forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In multivariate time series (data with multiple variables like
    /// price, volume, and indicators), there are two main approaches:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Channel-independent:</b> Each variable is processed separately using the same
    /// model weights. This often improves generalization because the model learns patterns
    /// that work across all variables.</item>
    /// <item><b>Channel-dependent:</b> All variables are processed together, allowing the model
    /// to learn relationships between variables. This can capture interactions but may overfit.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Research has shown that channel-independent models often perform better for forecasting tasks.
    /// </para>
    /// </remarks>
    bool IsChannelIndependent { get; }

    /// <summary>
    /// Applies instance normalization during inference for distribution shift handling.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor with zero mean and unit variance per instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Financial and time series data often experience "distribution shift"
    /// where the statistical properties change over time. For example:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item>Stock prices might average $100 one year and $500 the next</item>
    /// <item>Volatility increases during market crashes</item>
    /// <item>Sales patterns change due to seasonal trends or global events</item>
    /// </list>
    /// </para>
    /// <para>
    /// Instance normalization helps the model adapt to these changes by:
    /// <list type="number">
    /// <item>Calculating the mean and standard deviation of each input sequence</item>
    /// <item>Normalizing the data to have zero mean and unit variance</item>
    /// <item>Processing the normalized data through the model</item>
    /// </list>
    /// </para>
    /// <para>
    /// This makes the model more robust to changes in the data distribution.
    /// </para>
    /// </remarks>
    Tensor<T> ApplyInstanceNormalization(Tensor<T> input);
}
