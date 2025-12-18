namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Transfer Function models, which model the dynamic relationship
/// between input (exogenous) and output (endogenous) time series.
/// </summary>
/// <remarks>
/// <para>
/// Transfer Function models extend ARIMA (AutoRegressive Integrated Moving Average) models by incorporating 
/// the effects of one or more input (exogenous) time series on an output (endogenous) time series. They are 
/// particularly useful for modeling systems where there is a known causal relationship between input and output 
/// variables, with possible lagged effects. The transfer function component captures how changes in the input 
/// series affect the output series over time, while the ARIMA component models the autocorrelation structure 
/// of the output series. These models are widely used in fields such as economics, engineering, and environmental 
/// science for applications like dynamic system modeling, intervention analysis, and forecasting with leading 
/// indicators. This class provides configuration options for controlling the order of various components in the 
/// Transfer Function model.
/// </para>
/// <para><b>For Beginners:</b> Transfer Function models help you understand how one time series affects another over time.
/// 
/// When analyzing relationships between time series:
/// - Simple regression assumes immediate effects (X affects Y at the same time point)
/// - But in reality, effects often occur with delays (X affects Y after several time periods)
/// - The effect might also be distributed over multiple time periods
/// 
/// Transfer Function models solve this by:
/// - Capturing how input variables affect output variables over time
/// - Modeling both immediate and delayed effects
/// - Accounting for the autocorrelation in the output series itself
/// - Combining elements of regression and time series analysis
/// 
/// This approach is useful for:
/// - Understanding how marketing campaigns affect sales over time
/// - Modeling how temperature changes affect energy consumption
/// - Analyzing how policy changes impact economic indicators
/// 
/// This class lets you configure the structure of the Transfer Function model.
/// </para>
/// </remarks>
public class TransferFunctionOptions<T, TInput, TOutput> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the AutoRegressive (AR) component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the order of the AutoRegressive (AR) component in the Transfer Function model. 
    /// The AR component models the dependency of the current value of the output series on its own past values. 
    /// An AR order of p means that the current value depends on the p previous values. A higher order allows the 
    /// model to capture more complex autocorrelation patterns but increases the risk of overfitting and the 
    /// number of parameters to estimate. The default value of 1 provides a simple AR component suitable for many 
    /// applications, capturing first-order autocorrelation. The optimal value depends on the autocorrelation 
    /// structure of the output series after accounting for the effects of the input series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past values of the output series influence the current value.
    /// 
    /// The AutoRegressive (AR) order:
    /// - Determines how many previous values of the output series affect the current value
    /// - Helps the model capture patterns where past values predict future values
    /// - Higher values can model more complex temporal dependencies
    /// 
    /// The default value of 1 means:
    /// - The model considers the immediate previous value of the output series
    /// - This is sufficient for many time series with simple autocorrelation patterns
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 3): Can capture more complex patterns of autocorrelation
    /// - Lower values (e.g., 0): Ignore autocorrelation in the output series
    /// 
    /// When to adjust this value:
    /// - Increase it when the output series shows significant autocorrelation at multiple lags
    /// - Decrease it or set to 0 when the output series doesn't show significant autocorrelation
    /// - Consider using autocorrelation and partial autocorrelation functions to guide selection
    /// 
    /// For example, in quarterly economic data that shows seasonal patterns,
    /// you might increase this to 4 to capture year-over-year effects.
    /// </para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the Moving Average (MA) component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the order of the Moving Average (MA) component in the Transfer Function model. 
    /// The MA component models the dependency of the current value of the output series on past random shocks 
    /// or errors. An MA order of q means that the current value depends on the q previous random shocks. A 
    /// higher order allows the model to capture more complex patterns in the error terms but increases the risk 
    /// of overfitting and the number of parameters to estimate. The default value of 1 provides a simple MA 
    /// component suitable for many applications, capturing first-order effects of random shocks. The optimal 
    /// value depends on the structure of the error terms after accounting for the AR component and the effects 
    /// of the input series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past random shocks influence the current value.
    /// 
    /// The Moving Average (MA) order:
    /// - Determines how many previous random shocks (errors) affect the current value
    /// - Helps the model capture the lingering effects of unexpected events
    /// - Higher values can model more complex patterns in the error terms
    /// 
    /// The default value of 1 means:
    /// - The model considers the immediate previous random shock
    /// - This is sufficient for many time series with simple error structures
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 3): Can capture more complex patterns in the error terms
    /// - Lower values (e.g., 0): Ignore the effects of past random shocks
    /// 
    /// When to adjust this value:
    /// - Increase it when the autocorrelation of residuals shows significant patterns
    /// - Decrease it or set to 0 when the residuals appear to be white noise
    /// - Consider using autocorrelation of residuals to guide selection
    /// 
    /// For example, in financial time series where shocks often have lingering effects,
    /// you might increase this to 2 or 3 to capture these patterns.
    /// </para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the input lag in the transfer function.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum lag of the input series that affects the output series in the Transfer 
    /// Function model. An input lag order of r means that the current value of the output series can be affected 
    /// by the input series values from the current time point up to r time points in the past. A higher order 
    /// allows the model to capture longer-term effects of the input series but increases the number of parameters 
    /// to estimate. The default value of 1 allows for immediate and one-period lagged effects, suitable for many 
    /// applications where the input series has a relatively quick impact on the output series. The optimal value 
    /// depends on the expected time delay between changes in the input series and their effects on the output series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past values of the input series can affect the output series.
    /// 
    /// The input lag order:
    /// - Determines how many previous values of the input series can affect the current output
    /// - Helps the model capture delayed effects between input and output
    /// - Higher values allow for longer-term effects to be modeled
    /// 
    /// The default value of 1 means:
    /// - The model considers the current and immediate previous value of the input series
    /// - This is appropriate when input variables have a quick effect on the output
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 3 or 4): Can capture effects that take several time periods to manifest
    /// - Lower values (e.g., 0): Only immediate effects are considered
    /// 
    /// When to adjust this value:
    /// - Increase it when you expect the input series to have delayed effects on the output
    /// - Decrease it when you expect only immediate or short-term effects
    /// - Consider domain knowledge about the expected time delay between cause and effect
    /// 
    /// For example, in marketing analysis where advertising might affect sales for
    /// several months, you might increase this to 3-6 to capture these delayed effects.
    /// </para>
    /// </remarks>
    public int InputLagOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the output lag in the transfer function.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the order of the denominator polynomial in the transfer function, which determines 
    /// how the effect of the input series on the output series depends on past values of the output series itself. 
    /// An output lag order of s means that the effect of the input series on the current output depends on the s 
    /// previous values of the output. A higher order allows the model to capture more complex dynamic relationships 
    /// but increases the risk of overfitting and the number of parameters to estimate. The default value of 1 
    /// provides a simple dynamic relationship suitable for many applications. The optimal value depends on the 
    /// complexity of the dynamic relationship between the input and output series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the effect of inputs depends on past outputs.
    /// 
    /// The output lag order:
    /// - Determines how the effect of inputs depends on previous output values
    /// - Helps model feedback mechanisms where effects compound or diminish over time
    /// - Higher values allow for more complex dynamic relationships
    /// 
    /// The default value of 1 means:
    /// - The effect of inputs depends on the immediate previous output value
    /// - This creates a simple dynamic relationship suitable for many systems
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 3): Can model more complex feedback mechanisms
    /// - Lower values (e.g., 0): Effects of inputs don't depend on previous outputs
    /// 
    /// When to adjust this value:
    /// - Increase it when modeling systems with complex feedback mechanisms
    /// - Decrease it or set to 0 for simpler systems with more direct effects
    /// - Consider domain knowledge about how the system responds to inputs over time
    /// 
    /// For example, in economic systems where effects often compound over time,
    /// you might increase this to 2 to capture how the impact of policy changes
    /// depends on the previous state of the economy.
    /// </para>
    /// </remarks>
    public int OutputLagOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the optimizer used for parameter estimation.
    /// </summary>
    /// <value>An instance of a class implementing IOptimizer&lt;T&gt;, or null to use the default optimizer.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the optimization algorithm used to estimate the parameters of the Transfer Function 
    /// model. Different optimizers have different strengths and weaknesses in terms of convergence speed, ability 
    /// to escape local optima, and sensitivity to initial conditions. The default value of null indicates that a 
    /// default optimizer appropriate for the specific model implementation should be used. Common optimization 
    /// algorithms for Transfer Function models include maximum likelihood estimation, least squares, and variants 
    /// of gradient descent. The optimal choice depends on the specific characteristics of the data and the desired 
    /// trade-off between estimation accuracy and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the algorithm used to find the best model parameters.
    /// 
    /// The optimizer:
    /// - Controls how the model finds the optimal parameter values
    /// - Different optimizers have different strengths and weaknesses
    /// - Can significantly affect both the quality of results and computation time
    /// 
    /// The default value of null means:
    /// - The system will choose an appropriate default optimizer
    /// - This is suitable for most applications
    /// 
    /// Common optimizer types include:
    /// - Gradient-based: Fast but may get stuck in local optima
    /// - Evolutionary: Better at finding global optima but slower
    /// - Bayesian: Efficient for expensive objective functions
    /// 
    /// When to adjust this value:
    /// - Specify a different optimizer when the default doesn't converge well
    /// - Choose based on whether you prioritize speed or solution quality
    /// - This is an advanced setting that most users can leave as default
    /// 
    /// For example, if the default optimizer struggles with your complex model,
    /// you might specify a more robust optimizer that's less likely to get
    /// stuck in local optima.
    /// </para>
    /// </remarks>
    public IOptimizer<T, TInput, TOutput>? Optimizer { get; set; }
}
