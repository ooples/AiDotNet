namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dynamic Regression with ARIMA Errors, a powerful time series forecasting method
/// that combines regression with time series error correction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Dynamic Regression with ARIMA Errors (sometimes called RegARIMA) is a hybrid forecasting approach that uses
/// external variables (regressors) to explain the main trend while modeling the residual errors with ARIMA
/// (AutoRegressive Integrated Moving Average) to capture temporal patterns not explained by the regressors.
/// </para>
/// <para><b>For Beginners:</b> This is a forecasting method that combines two powerful techniques. First, it uses
/// regression to find relationships between your target variable (what you're trying to predict) and other variables
/// that might influence it (like temperature affecting ice cream sales). Then, it analyzes the errors in that
/// prediction to find patterns over time (like seasonal trends or cycles). By combining both approaches, it often
/// produces more accurate forecasts than either method alone. Think of it as not just predicting based on related
/// factors, but also learning from its own mistakes.</para>
/// </remarks>
public class DynamicRegressionWithARIMAErrorsOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of external regressor variables to use in the model.
    /// </summary>
    /// <value>The number of external regressors, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// External regressors are independent variables that help explain the behavior of the target variable.
    /// This setting determines how many such variables will be included in the regression component of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many outside factors your model will consider when making
    /// predictions. With the default value of 1, the model will use one external factor (like temperature when
    /// predicting ice cream sales). If you have multiple factors that might influence your prediction (like
    /// temperature, day of week, and whether it's a holiday), you would increase this number accordingly.</para>
    /// </remarks>
    public int ExternalRegressors { get; set; } = 1;

    /// <summary>
    /// Gets or sets the AutoRegressive (AR) order for the ARIMA component.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order (p) specifies how many previous time steps of the error term are used to predict the current error.
    /// Higher values capture longer-term dependencies but increase model complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how far back in time the model looks when analyzing its errors.
    /// With the default value of 1, the model considers errors from one time step ago to help predict the current value.
    /// For example, if yesterday's forecast was too high, the model might adjust today's forecast downward.
    /// Higher values (like 2 or 3) would make the model consider patterns from multiple previous time periods.</para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the Moving Average (MA) order for the ARIMA component.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order (q) specifies how many previous random shock terms are used in the prediction equation.
    /// It captures the short-term effects of random events on the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how the model handles unexpected events or "shocks" in your data.
    /// With the default value of 1, the model remembers the surprise factor from one time period ago.
    /// For example, if there was an unexpected spike in sales yesterday, the model accounts for how that surprise
    /// might affect today's prediction. Higher values make the model consider the lingering effects of surprises
    /// from multiple previous time periods.</para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the differencing order for the ARIMA component.
    /// </summary>
    /// <value>The differencing order, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// The differencing order (d) specifies how many times the data is differenced to achieve stationarity.
    /// Differencing helps remove trends and seasonality by computing the differences between consecutive observations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines whether and how much the model transforms your data to remove
    /// upward or downward trends. With the default value of 0, no differencing is applied. A value of 1 means the
    /// model will work with the changes between consecutive values rather than the raw values themselves.
    /// This is useful when your data has a strong trend (like consistently increasing sales year over year).
    /// Think of it as focusing on the rate of change rather than the absolute values.</para>
    /// </remarks>
    public int DifferenceOrder { get; set; } = 0;

    /// <summary>
    /// Gets or sets the matrix decomposition method used for solving the regression equations.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to LU decomposition.</value>
    /// <remarks>
    /// <para>
    /// Different matrix decomposition methods offer trade-offs between numerical stability, computational efficiency,
    /// and accuracy. LU decomposition is a good general-purpose choice for many problems.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical setting that controls how the math behind the model is solved.
    /// The default (LU decomposition) works well for most situations. You typically won't need to change this unless
    /// you're dealing with special cases like highly correlated variables or numerical precision issues.
    /// Think of it as selecting which mathematical technique the computer uses to solve the equations in your model.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;

    /// <summary>
    /// Gets or sets the regularization method used to prevent overfitting in the regression component.
    /// </summary>
    /// <value>The regularization method, defaulting to null (no regularization).</value>
    /// <remarks>
    /// <para>
    /// Regularization adds penalties to the model's complexity to prevent it from fitting noise in the training data.
    /// Common regularization methods include Ridge (L2), Lasso (L1), and ElasticNet (combination of L1 and L2).
    /// </para>
    /// <para><b>For Beginners:</b> This controls whether and how the model prevents itself from becoming too complex
    /// and "memorizing" your training data instead of learning general patterns. Without regularization (the default),
    /// the model might perform very well on your training data but poorly on new data. Adding regularization helps the
    /// model generalize better to new situations. Think of it as encouraging the model to find simpler explanations
    /// rather than complicated ones that might just be capturing noise in your data.</para>
    /// </remarks>
    public IRegularization<T, Matrix<T>, Vector<T>>? Regularization { get; set; }
}
