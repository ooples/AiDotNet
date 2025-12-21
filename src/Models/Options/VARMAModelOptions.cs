namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Vector Autoregressive Moving Average (VARMA) models, which extend VAR models
/// by incorporating moving average terms.
/// </summary>
/// <remarks>
/// <para>
/// Vector Autoregressive Moving Average (VARMA) models extend Vector Autoregressive (VAR) models by incorporating 
/// moving average (MA) terms. While VAR models express each variable as a linear function of past values of itself 
/// and past values of other variables, VARMA models also include past error terms. This additional flexibility can 
/// lead to more parsimonious models and better forecasting performance, especially when the true data generating 
/// process includes moving average components. VARMA models are particularly useful for modeling and forecasting 
/// multiple interrelated time series, capturing both the autoregressive and moving average dynamics in the system. 
/// This class inherits from VARModelOptions and adds parameters specific to the moving average component of VARMA 
/// models.
/// </para>
/// <para><b>For Beginners:</b> VARMA models extend VAR models by including both past values and past errors.
/// 
/// When modeling multiple related time series:
/// - VAR models use past values of all variables to predict future values
/// - VARMA models add another component: past prediction errors
/// 
/// This additional component:
/// - Captures patterns in the errors or "shocks" to the system
/// - Can lead to more accurate and efficient models
/// - Is particularly useful when the effects of shocks persist for multiple periods
/// 
/// Think of it like this:
/// - VAR: "Tomorrow's values depend on today's and yesterday's values"
/// - VARMA: "Tomorrow's values depend on today's and yesterday's values, plus how wrong our recent predictions were"
/// 
/// This class lets you configure the moving average component of VARMA models,
/// while inheriting all the configuration options for VAR models.
/// </para>
/// </remarks>
public class VARMAModelOptions<T> : VARModelOptions<T>
{
    /// <summary>
    /// Gets or sets the lag order for the Moving Average (MA) component.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lag order for the Moving Average (MA) component in the VARMA model. The MA 
    /// lag order, often denoted as q, determines how many past error terms (or "shocks") are included in the 
    /// model for each variable. An MA lag of q means that the current values depend on the error terms from the 
    /// current time point up to q time points in the past. A higher order allows the model to capture more 
    /// complex patterns in the error terms but increases the number of parameters to estimate and the risk of 
    /// overfitting. The default value of 1 provides a simple MA component suitable for many applications, 
    /// capturing first-order effects of past errors. The optimal value depends on the autocorrelation structure 
    /// of the residuals from a pure VAR model and the desired trade-off between model complexity and fit.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many past prediction errors influence the current values.
    /// 
    /// The Moving Average (MA) lag:
    /// - Determines how many previous prediction errors affect the current values
    /// - Helps the model capture the lingering effects of unexpected events or "shocks"
    /// - Higher values can model more complex patterns in how shocks persist
    /// 
    /// The default value of 1 means:
    /// - The model considers the immediate previous prediction error
    /// - This is sufficient for many time series with simple error structures
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2 or 3): Can capture more complex patterns in how shocks persist
    /// - Lower values (e.g., 0): Effectively turns the model into a standard VAR model
    /// 
    /// When to adjust this value:
    /// - Increase it when the autocorrelation of residuals from a VAR model shows significant patterns
    /// - Keep at 1 for a simple VARMA model
    /// - Set to 0 if you want a pure VAR model
    /// - Consider using information criteria (AIC, BIC) to select the optimal order
    /// 
    /// For example, in financial time series where shocks often have lingering effects,
    /// you might increase this to 2 to better capture how market shocks persist over time.
    /// </para>
    /// </remarks>
    public int MaLag { get; set; } = 1;
}
