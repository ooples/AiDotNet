namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model,
/// which is used for analyzing and forecasting volatility in time series data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// GARCH models are specialized time series models designed to capture volatility clustering in financial
/// and economic data. Unlike standard time series models that focus on predicting the mean value, GARCH
/// models specifically model how the variance (volatility) of a time series changes over time, accounting
/// for periods of high and low volatility.
/// </para>
/// <para><b>For Beginners:</b> Think of GARCH as a specialized tool for predicting how much something will
/// fluctuate or vary in the future, rather than predicting its exact value. It's particularly useful for
/// financial data like stock prices, where you might want to know not just whether a price will go up or down,
/// but how stable or volatile it will be. For example, GARCH can help predict whether tomorrow's stock price
/// will likely stay close to today's price or might swing dramatically in either direction. This is valuable
/// for risk management and option pricing in finance. The model works by recognizing that periods of high
/// volatility often cluster together (if today is volatile, tomorrow is likely to be volatile too).</para>
/// </remarks>
public class GARCHModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the ARCH order (p), which determines how many past squared errors are used to model current volatility.
    /// </summary>
    /// <value>The ARCH order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The ARCH component of the model captures how recent shocks (unexpected changes) affect current volatility.
    /// The order (p) specifies how many past squared errors are included in the model. Higher values allow the
    /// model to capture more complex patterns in how past shocks influence current volatility.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past surprises or shocks the model considers
    /// when predicting today's volatility. With the default value of 1, the model only looks at yesterday's surprise
    /// (how far the actual value was from what was expected). Think of it like judging how nervous someone might be
    /// based on recent surprises they've experienced. If you set this higher (like 2 or 3), the model will consider
    /// surprises from multiple previous days, which might be more accurate but makes the model more complex. For most
    /// applications, a value of 1 or 2 works well, capturing the idea that recent surprises affect current volatility.</para>
    /// </remarks>
    public int ARCHOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the GARCH order (q), which determines how many past volatility values are used to model current volatility.
    /// </summary>
    /// <value>The GARCH order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The GARCH component of the model captures how past volatility levels affect current volatility.
    /// The order (q) specifies how many past volatility values are included in the model. Higher values allow
    /// the model to capture more persistent patterns in volatility over time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past volatility levels the model considers
    /// when predicting today's volatility. With the default value of 1, the model only looks at yesterday's volatility.
    /// Think of it like judging how rough the sea will be today based on how rough it was yesterday. If you set this
    /// higher (like 2 or 3), the model will consider volatility from multiple previous days, which might capture
    /// longer-lasting patterns. The combination of ARCH (recent surprises) and GARCH (recent volatility levels)
    /// helps the model predict whether things will be stable or unstable in the near future. A GARCH(1,1) model
    /// (both orders set to 1) is often sufficient for many applications.</para>
    /// </remarks>
    public int GARCHOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum number of iterations allowed during parameter estimation.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// GARCH models are typically estimated using iterative numerical optimization methods. This parameter
    /// limits how many iterations the optimization algorithm will perform before stopping, preventing
    /// excessively long computation times.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how long the model will spend trying to find the best
    /// parameters. With the default value of 1000, the model will try up to 1000 rounds of refinement before
    /// stopping, even if it hasn't found the perfect solution. Think of it like setting a time limit on a
    /// treasure hunt - at some point, you need to stop searching even if you haven't found the absolute best
    /// treasure. For simple datasets, the model might find good parameters much sooner than 1000 iterations.
    /// You might increase this for complex datasets where the model needs more time to find good parameters,
    /// or decrease it if you need faster results and can accept slightly less optimal parameters.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for parameter estimation.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how small the change in parameter estimates needs to be between iterations
    /// for the optimization algorithm to consider that it has converged to a solution. Smaller values require
    /// more precise convergence but may increase computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the model needs to be before it decides
    /// it's found a good solution. With the default value of 0.000001 (one millionth), the model will stop
    /// refining its parameters when the improvements between attempts become very tiny. Think of it like
    /// deciding when a drawing is "finished" - at some point, additional pencil strokes make such a small
    /// difference that it's not worth continuing. A smaller value (like 0.0000001) would make the model more
    /// precise but might take longer to finish. A larger value (like 0.0001) would make the model finish faster
    /// but with potentially less optimal parameters. The default is a good balance for most applications.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the model used to predict the mean of the time series before modeling volatility.
    /// </summary>
    /// <value>The mean model, defaulting to null (which uses a simple mean).</value>
    /// <remarks>
    /// <para>
    /// GARCH models focus on modeling the variance (volatility) of a time series, but they typically need
    /// a separate model to handle the mean prediction. This property allows you to specify a different time
    /// series model (like ARIMA) to predict the mean values before the GARCH model analyzes the residuals
    /// for volatility patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you choose another model to predict the actual values
    /// before GARCH predicts the volatility. By default (when null), the model uses a simple average. Think
    /// of it like a two-step process: first, predict what value you expect (using this mean model), then
    /// predict how far off that prediction might be (using GARCH). For example, you might use an ARIMA model
    /// here to predict stock prices, and then GARCH to predict how volatile those prices will be. This combined
    /// approach (often called ARIMA-GARCH) is common in financial forecasting because it handles both the
    /// direction and the uncertainty of the forecast.</para>
    /// </remarks>
    public ITimeSeriesModel<T>? MeanModel { get; set; }
}
