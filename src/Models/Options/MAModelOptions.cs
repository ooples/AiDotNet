namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Moving Average (MA) models, which are used to analyze time series data
/// by modeling the error terms as a linear combination of previous error terms.
/// </summary>
/// <remarks>
/// <para>
/// The Moving Average (MA) model is a fundamental component in time series analysis and forecasting.
/// Unlike autoregressive (AR) models that express the current value as a function of past values,
/// MA models express the current value as a function of past forecast errors (also called shocks or innovations).
/// This makes MA models particularly effective at capturing short-term, irregular patterns in time series data.
/// The model is defined by its order (q), which determines how many past error terms are included in the model.
/// </para>
/// <para><b>For Beginners:</b> A Moving Average (MA) model helps predict future values in a time series
/// (like daily temperatures, stock prices, or website traffic) based on recent unexpected changes or "surprises."
/// 
/// Imagine you're trying to predict tomorrow's temperature:
/// - An MA model doesn't just look at yesterday's actual temperature
/// - Instead, it focuses on the recent "surprises" - the differences between what was predicted and what actually happened
/// - It assumes that these recent surprises contain useful information about what might happen next
/// 
/// For example:
/// - If the weather has been consistently 2 degrees warmer than predicted for the past few days
/// - An MA model would adjust tomorrow's forecast to account for this pattern of surprises
/// 
/// This approach is particularly useful for data that has short-term patterns or fluctuations.
/// This class allows you to configure how the MA model will be built and optimized.
/// </para>
/// </remarks>
public class MAModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the Moving Average model, which determines how many past error terms are included.
    /// </summary>
    /// <value>The MA order (q), defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order, commonly denoted as q, specifies how many lagged error terms are included in the model.
    /// For example, an MA(1) model includes only the most recent error term, while an MA(2) includes the two
    /// most recent error terms. Higher order models can capture more complex patterns in the data but require
    /// more parameters to be estimated, increasing the risk of overfitting and computational complexity.
    /// The appropriate order typically depends on the structure of the autocorrelation in the error terms and
    /// can be informed by examining the autocorrelation function (ACF) of the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many past "surprises" the model considers.
    /// 
    /// Think of it like determining how far back you look for patterns:
    /// - MAOrder = 1: Only yesterday's surprise matters (default)
    /// - MAOrder = 2: Both yesterday's and the day before's surprises matter
    /// - MAOrder = 3: The last three days' surprises matter
    /// 
    /// The default value of 1 is often a good starting point because:
    /// - It captures the most immediate effects
    /// - It's less likely to overfit to random fluctuations
    /// - It's easier to estimate reliably
    /// 
    /// You might want to increase this value if:
    /// - Your data shows patterns that persist for several time periods
    /// - Statistical tests (like looking at autocorrelation) suggest longer-term relationships
    /// - You have enough data to reliably estimate more parameters
    /// 
    /// Be cautious with higher orders, as they require more data to estimate accurately and can lead
    /// to more complex models that might not generalize well to new data.
    /// </para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate that controls the step size in each iteration of the optimization process.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines how large of a step the optimization algorithm takes in the direction
    /// of the gradient during each iteration when estimating the MA model parameters. A higher learning
    /// rate allows for faster convergence but risks overshooting the optimal solution or causing instability.
    /// A lower learning rate provides more stability but may require more iterations to converge and
    /// risks getting stuck in local optima. For MA models, the optimization typically involves minimizing
    /// the sum of squared errors between observed values and the model's predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the model adjusts its parameters
    /// during training.
    /// 
    /// Imagine you're trying to find the lowest point in a valley while blindfolded:
    /// - The learning rate determines your step size as you move downhill
    /// - A large learning rate (like 0.1) means taking big steps
    /// - A small learning rate (like 0.001) means taking tiny steps
    /// 
    /// The default value of 0.01 provides a balance between:
    /// - Speed: Higher values help the model learn faster
    /// - Stability: Lower values help prevent the model from "overshooting" the best answer
    /// 
    /// If your model's parameters seem to be fluctuating wildly during training, try decreasing this value.
    /// If training is proceeding very slowly, try increasing it.
    /// 
    /// Finding the right learning rate often requires experimentation - too high and the training might
    /// become unstable, too low and it might take too long to converge.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when estimating the MA model parameters. The algorithm may terminate earlier if convergence is
    /// achieved based on the tolerance value. In the context of MA models, each iteration involves
    /// computing the model errors based on the current parameter estimates and then updating those
    /// parameters based on the gradient of the error function.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many attempts the algorithm gets to
    /// find the optimal parameters.
    /// 
    /// Think of it like trying to tune a radio:
    /// - Each "iteration" is one adjustment of the dial
    /// - You keep adjusting until you get clear reception or run out of patience
    /// - This parameter is your "patience limit"
    /// 
    /// The default value of 1000 is sufficient for many problems. You might need to increase it if:
    /// - You have a complex time series with many patterns
    /// - The model consistently stops due to reaching the maximum iterations without converging
    /// - You're working with a low learning rate that requires more iterations
    /// 
    /// You might want to decrease it if:
    /// - Training is taking too long and you want faster results
    /// - You're doing initial experimentation and don't need perfect results
    /// 
    /// Note that the algorithm might stop before reaching this maximum if it determines that
    /// additional iterations won't significantly improve the model (based on the Tolerance setting).
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the optimization algorithm should stop.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged.
    /// The algorithm will stop when the improvement in the objective function (typically the sum
    /// of squared errors) between consecutive iterations falls below this tolerance value.
    /// A smaller tolerance requires more precision in the solution, potentially leading to better model
    /// performance but requiring more iterations, while a larger tolerance allows for earlier
    /// termination but might result in less optimal parameter estimates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the model needs to be before
    /// it decides it's "good enough."
    /// 
    /// Imagine you're weighing ingredients for a recipe:
    /// - How precise do you need to be? Within 1 gram? 0.1 gram? 0.01 gram?
    /// - This setting determines when to stop fine-tuning the model's parameters
    /// 
    /// The default value of 0.000001 (one millionth) means:
    /// - If an iteration improves the model's performance by less than one millionth
    /// - The algorithm will decide it's "good enough" and stop
    /// 
    /// This is quite a precise setting, appropriate for many time series applications where
    /// high accuracy is desirable.
    /// 
    /// You might want to increase this value (make it less strict, like 1e-4) if:
    /// - Training is taking too long
    /// - You're doing preliminary exploration
    /// - You don't need extremely precise parameter estimates
    /// 
    /// You might want to decrease this value (make it more strict, like 1e-8) if:
    /// - You need very precise forecasts
    /// - Your application is highly sensitive to small errors
    /// - You have the computational resources for longer training
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;
}
