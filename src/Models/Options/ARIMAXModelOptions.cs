namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) time series forecasting model.
/// </summary>
/// <typeparam name="T">The data type of the time series values.</typeparam>
/// <remarks>
/// <para>
/// ARIMAX extends the ARIMA model by incorporating external (exogenous) variables that may influence the time series.
/// This allows the model to account for known external factors when making predictions.
/// </para>
/// <para><b>For Beginners:</b> ARIMAX is like ARIMA (which predicts future values based on past patterns) but with an 
/// added superpower: it can also consider outside factors that might affect your data. For example, when predicting 
/// ice cream sales, ARIMA would only look at past sales patterns, but ARIMAX could also consider temperature data. 
/// Think of it as a weather forecaster who not only looks at past weather patterns but also considers upcoming events 
/// like a hurricane forming in the ocean that will likely affect the forecast.</para>
/// </remarks>
public class ARIMAXModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the AutoRegressive (AR) component.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order specifies how many previous time steps are used to predict the current value.
    /// For example, AROrder=1 means the model uses only the immediately preceding value.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past data points the model looks at to make predictions.
    /// With the default value of 1, the model only considers what happened in the previous time period. If you set it to 2,
    /// it would look at the last two time periods, and so on. Think of it like predicting tomorrow's sales based on
    /// today's sales (AROrder=1) versus considering both today and yesterday (AROrder=2). Higher values can capture more complex
    /// patterns but might make the model unnecessarily complicated for simple data.</para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the Moving Average (MA) component.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order specifies how many previous forecast errors are used in the model.
    /// For example, MAOrder=1 means the model incorporates the error from the previous prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past prediction errors the model considers.
    /// With the default value of 1, the model adjusts based on how wrong its most recent prediction was.
    /// Think of it like a weather forecaster who not only looks at yesterday's weather but also considers how wrong
    /// their previous forecast was, learning from their mistakes. If yesterday's prediction was too high, they might
    /// adjust today's prediction downward a bit to compensate.</para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of differencing (Integration).
    /// </summary>
    /// <value>The differencing order, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// The differencing order specifies how many times the data is differenced to achieve stationarity.
    /// Differencing helps remove trends and seasonal patterns from the data. A value of 0 means no differencing is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how many times the model subtracts consecutive values to remove trends.
    /// The default value of 0 means the model works with the raw values directly. If set to 1, the model would work with the 
    /// changes between consecutive values rather than the raw values themselves. For example, instead of using temperatures 
    /// like [70°F, 72°F, 75°F], it would use the differences [2°F, 3°F]. This helps the model focus on how values are changing 
    /// rather than their absolute values, which is useful when data has an upward or downward trend. Think of it like focusing 
    /// on how much warmer or cooler it gets each day rather than the actual temperature.</para>
    /// </remarks>
    public int DifferenceOrder { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of exogenous (external) variables to include in the model.
    /// </summary>
    /// <value>The number of exogenous variables, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// Exogenous variables are external factors that may influence the time series but are not influenced by it.
    /// This parameter specifies how many such variables the model should consider.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how many outside factors the model will consider when making predictions.
    /// The default value of 1 means the model will use one external factor (like temperature when predicting ice cream sales).
    /// These are factors that affect your data but aren't affected by it. For example, when predicting retail sales, 
    /// external factors might include holidays, weather, or marketing campaigns. Including these can significantly improve 
    /// predictions when your data is influenced by known external conditions.</para>
    /// </remarks>
    public int ExogenousVariables { get; set; } = 1;

    /// <summary>
    /// Gets or sets the matrix decomposition method used for solving the model's equations.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to Cholesky.</value>
    /// <remarks>
    /// <para>
    /// Different matrix decomposition methods offer trade-offs between computational efficiency, numerical stability,
    /// and applicability to different types of matrices. Cholesky decomposition is efficient for positive definite matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical setting that determines the mathematical method used to solve 
    /// the equations in the model. The default Cholesky method is a good balance of speed and accuracy for most cases. 
    /// Think of it like choosing which calculator to use - some calculators are better for certain types of math problems. 
    /// Unless you have specific knowledge about the mathematical properties of your data, it's usually best to leave this 
    /// at the default setting.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
